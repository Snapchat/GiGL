/*
 * Note: This is a proof of concept implementation and the code needs some work to be production ready.
 */

package libs.task.pureSparkV2

import common.types.pb_wrappers.GbmlConfigPbWrapper
import common.types.pb_wrappers.ResourceConfigPbWrapper
import common.userDefinedAggregators.RnnUDAF
import common.utils.Cacher
import common.utils.GiGLComponents
import common.utils.NumCores
import common.utils.SlottedJoiner
import common.utils.SparkSessionEntry.getActiveSparkSession
import libs.task.SubgraphSamplerTask
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.{functions => F}
import snapchat.research.gbml.graph_schema.Edge
import snapchat.research.gbml.graph_schema.Graph
import snapchat.research.gbml.graph_schema.Node
import snapchat.research.gbml.training_samples_schema.RootedNodeNeighborhood
import upickle.default._

import scala.collection.mutable.HashMap
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.Set

case class EgoNetGenFlags(
  node_bq_table: String,
  edge_bq_table: String,
  from_node_id_column: String,
  to_node_id_column: String)

object EgoNetGeneration {
  // val FORWARD_DIRECTION_EDGE  = 1
  // val BACKWARD_DIRECTION_EDGE = -1
  val DEFAULT_EDGE_TYPE = 0
  val DEFAULT_NODE_TYPE = 0

  def biDirectionalizeTable(sampledEdgeTable: String): DataFrame = {

    val spark = getActiveSparkSession()

    // Create a bidirectional edge table.
    spark.sql(
      s"""
        SELECT DISTINCT * FROM (
          SELECT 
            dst_node_id as src_node_id,
            src_node_id as dst_node_id,
            ${DEFAULT_EDGE_TYPE} as edge_type,
            edge_features
          FROM ${sampledEdgeTable}
          UNION
          SELECT
            src_node_id as src_node_id,
            dst_node_id as dst_node_id,
            ${DEFAULT_EDGE_TYPE} as edge_type,
            edge_features as edge_features
          FROM ${sampledEdgeTable}
        )
      """,
    ) // TODO: (svij-sc) - Distinct will slow down as # of features scale - need to think of better solution here

    // // Create a bidirectional edge table.
    // // For now, we use "edge_type" to denote the direction of the edge, as it helps reuse constructs defined
    // // in GiGL already. TODO: (svij-sc) Think of better design here; it could be reasonable to handle
    // // the bidirectional edges this way too i.e. `-1*edge_type_num`
    // // First filter sampleEdgeTable
    // spark.sql(
    //   s"""
    //     SELECT DISTINCT * FROM (
    //       SELECT
    //         dst_node_id as src_node_id,
    //         src_node_id as dst_node_id,
    //         ${BACKWARD_DIRECTION_EDGE} as edge_type,
    //         edge_features
    //       FROM ${sampledEdgeTable}
    //       UNION
    //       SELECT
    //         src_node_id as src_node_id,
    //         dst_node_id as dst_node_id,
    //         ${FORWARD_DIRECTION_EDGE} as edge_type,
    //         edge_features as edge_features
    //       FROM ${sampledEdgeTable}
    //     )
    //   """,
    // ) // TODO: (svij-sc) - Distinct will slow down as # of features scale - need to think of better solution here
  }

  def generateEgoNodes(
    spark: SparkSession,
    rawEdgeDF: DataFrame,
    numOptimalPartitions: Int,
    numSlots: Int,
    sampleN: Int,
    numHops: Int,
    outputDir: String,
  ): Unit = {
    val rawEdgeTable = "raw_edge_Table"
    rawEdgeDF.createOrReplaceTempView(rawEdgeTable)
    // Sample sampleN src nodes for each dst node
    val sampledEdgeDF = spark.sql(s"""
      SELECT
        dst_node_id,
        src_node_id,
        edge_features
      FROM (
        SELECT
          dst_node_id,
          src_node_id,
          edge_features,
          ROW_NUMBER() OVER (PARTITION BY dst_node_id ORDER BY rand()) as row_num
        FROM ${rawEdgeTable}
      )
      WHERE row_num <= ${sampleN}
      """)
    val sampledEdgeTable = "sampled_edge_table"
    Cacher.createDiskPartitionedTable(
      df = sampledEdgeDF,
      targetPartitionNum = numOptimalPartitions,
      repartitionOnColumn = "dst_node_id",
      outputTableName = sampledEdgeTable,
    )

    // Create a bidirectional edge table.
    // For now, we use "edge_type" to denote the direction of the edge, as it helps reuse constructs defined
    // in GiGL already. TODO: (svij-sc) Think of better design here; it could be reasonable to handle
    // the bidirectional edges this way too i.e. `-1*edge_type_num`
    val bidirectionalEdgeDF = EgoNetGeneration.biDirectionalizeTable(sampledEdgeTable)
    bidirectionalEdgeDF.show()
    val bidirectionalEdgeTable = "bidirectional_edge_table"
    Cacher.createDiskPartitionedTable(
      df = bidirectionalEdgeDF,
      targetPartitionNum = numOptimalPartitions,
      repartitionOnColumn = "dst_node_id",
      outputTableName = bidirectionalEdgeTable,
    )
    spark.sql(s"DROP TABLE IF EXISTS ${sampledEdgeTable}")
    val neighborsTable = "neighbors_table"
    // Perform a group by on the dst_node_id to get the neighbor lists of each dst_node
    val joinedEdgeDF: DataFrame = if (numHops == 2) {
      // To conform to GiGL's design, we need to have the node features for each of the src_node_id and dst_node_id
      // Thus, we just add empty columns that confirm to the schema as expected by the RNN UDAF
      spark.sql(s"""
      SELECT
        src_node_id as src_node_id,
        dst_node_id as dst_node_id,
        edge_features as edge_features,
        edge_type as edge_type,
        cast(ARRAY() as array<float>) as src_node_features,
        cast(ARRAY() as array<float>) as dst_node_features
      FROM ${bidirectionalEdgeTable}
      """)
    } else if (numHops == 3) {
      // We hijack the use of node_features (GiGL feature) to store the neighbors of nodes
      // TODO: (svij-sc) - This is a hacky way to do this, we should refactor this to be more explicit
      val neighborsDF = spark.sql(s"""
        SELECT
          dst_node_id as node_id,
          cast(collect_list(src_node_id) as array<float>) as node_features
        FROM ${bidirectionalEdgeTable}
        GROUP BY dst_node_id
      """)
      Cacher.createDiskPartitionedTable(
        df = neighborsDF,
        targetPartitionNum = numOptimalPartitions,
        repartitionOnColumn = "node_id",
        outputTableName = neighborsTable,
      )
      // Join the neighbors_table with the bidirectional_edge_table to get the node features for each of
      // src_node_id and dst_node_id in the bidirectional_edge_table
      val df = spark.sql(s"""
        WITH dst_node_feature_added_table as (
          SELECT
            bidirectional_edge_table.src_node_id as src_node_id,
            bidirectional_edge_table.dst_node_id as dst_node_id,
            bidirectional_edge_table.edge_features as edge_features,
            bidirectional_edge_table.edge_type as edge_type,
            neighbors_table.node_features as dst_node_features
          FROM
            ${bidirectionalEdgeTable}
          JOIN
            ${neighborsTable}
          ON
            bidirectional_edge_table.dst_node_id = neighbors_table.node_id
        )
        SELECT
          dst_node_feature_added_table.src_node_id as src_node_id,
          dst_node_feature_added_table.dst_node_id as dst_node_id,
          dst_node_feature_added_table.edge_features as edge_features,
          dst_node_feature_added_table.dst_node_features as dst_node_features,
          dst_node_feature_added_table.edge_type as edge_type,
          neighbors_table.node_features as src_node_features
        FROM
          dst_node_feature_added_table
        JOIN
          ${neighborsTable}
        ON
          dst_node_feature_added_table.src_node_id = neighbors_table.node_id
        """)
      df
    } else {
      throw new IllegalArgumentException("Only 2 or 3 hops supported")
    }

    val dstPartitionedRawEdgeTable = "dstPartitionedRawEdgeTable"
    Cacher.createDiskPartitionedTable(
      df = joinedEdgeDF,
      targetPartitionNum = numOptimalPartitions,
      repartitionOnColumn = "dst_node_id",
      outputTableName = dstPartitionedRawEdgeTable,
    )
    spark.sql(f"DROP TABLE IF EXISTS ${bidirectionalEdgeTable}")
    spark.sql(f"DROP TABLE IF EXISTS ${neighborsTable}")

    val srcPartitionedRawEdgeTableName = "srcPartitionedRawEdgeTable"
    // TODO: (svij-sc) - Explore if we really need to persist the input table dstPartitionedRawEdgeTable
    val leftSlottedDF = if (numSlots <= 1) {
      spark.table(dstPartitionedRawEdgeTable)
    } else {
      SlottedJoiner.computeSlotsOnDataframe(
        df = spark.table(dstPartitionedRawEdgeTable),
        columnToComputeSlotOn = "dst_node_id",
        numSlots = numSlots,
      )
    }

    Cacher.createDiskPartitionedTable(
      df = leftSlottedDF,
      targetPartitionNum = numOptimalPartitions,
      repartitionOnColumn = "src_node_id",
      outputTableName = srcPartitionedRawEdgeTableName,
    )

    val postProcessFn = (rnn: RootedNodeNeighborhood) => {
      // Since we added 1 hop neighbors as node features, we use this post process function
      // to take the formed neighbors and add them to edges. Subsequently we remove the unecessary node features.
      val finalEdgeIds           = new HashMap[Int, Set[Int]]() // dstNodeId -> (srcNodeId, ...)
      val finalNodeIds: Set[Int] = Set()

      for (node <- rnn.neighborhood.get.nodes) {
        val dstNodeId: Int = node.nodeId
        finalNodeIds += dstNodeId
        // Each feature value is a srcNodeId in Float
        for (nodeId <- node.featureValues) {
          val srcNodeId: Int = nodeId.toInt
          finalNodeIds += srcNodeId
          if (!finalEdgeIds.contains(dstNodeId)) {
            finalEdgeIds(dstNodeId) = Set(srcNodeId)
          } else {
            finalEdgeIds(dstNodeId) += srcNodeId
          }
        }
      }
      val finalNodes = new ListBuffer[Node]()
      for (nodeId <- finalNodeIds) {
        finalNodes += Node(
          nodeId = nodeId,
          condensedNodeType = Some(DEFAULT_NODE_TYPE),
          featureValues = Array[Float](),
        )
      }
      val finalEdges = new ListBuffer[Edge]()
      for ((dstNodeId, srcNodeIds) <- finalEdgeIds) {
        for (srcNodeId <- srcNodeIds) {
          finalEdges += Edge(
            srcNodeId = srcNodeId,
            dstNodeId = dstNodeId,
            condensedEdgeType = Some(DEFAULT_EDGE_TYPE),
            featureValues = Array[Float](),
          )
        }
      }
      val finalRootNode = Node(
        nodeId = rnn.rootNode.get.nodeId,
        condensedNodeType = Some(DEFAULT_NODE_TYPE),
        featureValues = Array[Float](),
      )
      RootedNodeNeighborhood(
        rootNode = Some(finalRootNode),
        neighborhood = Some(
          Graph(
            nodes = finalNodes.toSeq,
            edges = finalEdges.toSeq,
          ),
        ),
      )
    }

    // Join the tables using slotted joins
    spark.udf.register(
      "rnnUDAF",
      F.udaf(new RnnUDAF(sampleN = Some(sampleN), postProcessFn = Some(postProcessFn))),
    )
    /* Schema:
     * _root_node_id: NodeId,
     * _root_node_type: NodeType,
     * _root_node_features: Seq[Float],
     * _1_hop_node_id: NodeId,
     * _1_hop_node_type: NodeType,
     * _1_hop_node_features: Seq[Float],
     * _1_hop_edge_features: Seq[Float],
     * _1_hop_edge_type: CondensedEdgeType,
     * _2_hop_node_id: NodeId,
     * _2_hop_node_type: NodeType,
     * _2_hop_node_features: Seq[Float],
     * _2_hop_edge_features: Seq[Float],
     * _2_hop_edge_type: CondensedEdgeType
     */
    // TODO: (svij-sc) - We can pipe node type in later if needed. We would then likely also group by _root_node_type
    val sqlJoin = (f"""
      SELECT
          rnnUDAF(
            ${srcPartitionedRawEdgeTableName}.dst_node_id,
            ${DEFAULT_NODE_TYPE},
            ${srcPartitionedRawEdgeTableName}.dst_node_features,
            ${srcPartitionedRawEdgeTableName}.src_node_id,
            ${DEFAULT_NODE_TYPE},
            ${srcPartitionedRawEdgeTableName}.src_node_features,
            ${srcPartitionedRawEdgeTableName}.edge_features,
            ${srcPartitionedRawEdgeTableName}.edge_type,
            ${dstPartitionedRawEdgeTable}.src_node_id,
            ${DEFAULT_NODE_TYPE},
            ${dstPartitionedRawEdgeTable}.src_node_features,
            ${dstPartitionedRawEdgeTable}.edge_features,
            ${dstPartitionedRawEdgeTable}.edge_type
          ) as result
      FROM
          ${srcPartitionedRawEdgeTableName}
      LEFT JOIN
          ${dstPartitionedRawEdgeTable}
      ON
          ${srcPartitionedRawEdgeTableName}.src_node_id = ${dstPartitionedRawEdgeTable}.dst_node_id
      GROUP BY
          ${srcPartitionedRawEdgeTableName}.dst_node_id
      """)
    if (numSlots <= 1) {
      val df = spark.sql(sqlJoin)
      write_tfrecord_to_dir(df, outputDir)
    } else {
      val joinIterator: Iterator[DataFrame] = SlottedJoiner.performSlottedJoinOnLeftSlottedTable(
        sql = sqlJoin,
        numSlots = numSlots,
        leftSlottedTableName = srcPartitionedRawEdgeTableName,
        rightTableName = dstPartitionedRawEdgeTable,
      )
      var slotNum = 0
      for (df <- joinIterator) {
        println(s"Slot joining on ${slotNum}")
        println(s"Writing to outputDir: ${outputDir}")
        write_tfrecord_to_dir(df, outputDir)
        println(s"Slot joining on ${slotNum} - FINISHED")
        slotNum += 1
        // Setting to blocking=true caused some observed slowdowns and failires; thus we let system unpersist
        // That being said, we are unsure if the system is actually unpersisting anything here since we observe
        // a memory leak i.e. HDFS usage keeps growing each iteration
        df.unpersist()
      }
    }
  }

  def write_tfrecord_to_dir(
    df: DataFrame,
    outputDir: String,
  ): Unit = {
    df.write
      .mode("append")
      .format("tfrecord")
      .option(
        "recordType",
        "ByteArray",
      )
      .save(outputDir)
  }

}

class EgoNetGeneration(
  gbmlConfigWrapper: GbmlConfigPbWrapper,
  giglResourceConfigWrapper: ResourceConfigPbWrapper,
  jobName: String)
    extends SubgraphSamplerTask(gbmlConfigWrapper) {

  val numVCPUs: Int = NumCores.getNumVCPUs(
    giglResourceConfigWrapper = giglResourceConfigWrapper,
    component = GiGLComponents.SubgraphSampler,
  )
  // TODO: (svij-sc) This needs to be less for earlier tables i.e. bidirectional_edge_table and neighbors_table
  // The files generated for these tables are <5mb
  // Whereas the files generated for 	dstpartitionedrawedgetable and srcpartitionedrawedgetable are ~ 200-300Mb
  // If anything for later tables which we actually join on we might want to explore growing the size.
  val DEFAULT_NUM_OPTIMAL_PARTITIONS: Int = numVCPUs * 5
  val DEFAULT_NUM_SLOTS                   = 10
  val DEFAULT_SAMPLE_N =
    gbmlConfigWrapper.datasetConfigPb.subgraphSamplerConfig.get.numNeighborsToSample
  val DEFAULT_NUM_HOPS = gbmlConfigWrapper.datasetConfigPb.subgraphSamplerConfig.get.numHops

  // macroRW is a method provided by uPickle that can generate a instances of
  // ReadWriter automatically for case classes, using the information about its fields.
  implicit val egoNetGenFlagsRW: ReadWriter[EgoNetGenFlags] = macroRW

  def fetchEdgeTableFromBq(
    tableName: String,
    fromNodeIdColumn: String,
    toNodeIdColumn: String,
  ): DataFrame = {

    /**
      * Fetches the edge table from BigQuery and returns a DataFrame
      * All the edge features are cast to float type and returned as an array
      */
    val edgeDF = spark.read.format("bigquery").option("table", tableName).load()
    val featColNames = edgeDF.columns.filterNot(colName =>
      colName.equals(fromNodeIdColumn) || colName.equals(toNodeIdColumn),
    )

    edgeDF.createOrReplaceTempView("edge_table_raw")
    // We cast to expected data format to catch any errors early
    // If we do not have this here we will need to do explicit cast later when converting DS/DF to protobuf message anyways.
    val featsStructColQuery =
      featColNames.map(colName => s"cast(coalesce(${colName}, 0) as float)").mkString(", ")
    spark.sql(s"""
      SELECT
        cast(${fromNodeIdColumn} as INTEGER) AS src_node_id,
        cast(${toNodeIdColumn} as INTEGER) AS dst_node_id,
        cast(array(${featsStructColQuery}) as array<float>) AS edge_features
      FROM edge_table_raw
      """)
  }

  def run(): Unit = {
    println("Running EgoNetGeneration")
    val outputDir =
      gbmlConfigWrapper.sharedConfigPb.flattenedGraphMetadata.get.getNodeAnchorBasedLinkPredictionOutput.tfrecordUriPrefix + jobName + "/"
    val numComputeMachines: Int = giglResourceConfigWrapper.subgraphSamplerConfig.numReplicas
    val experimentalFlags: Map[String, String] =
      gbmlConfigWrapper.datasetConfigPb.subgraphSamplerConfig.get.experimentalFlags
    assert(
      experimentalFlags.contains("compute_ego_net") && experimentalFlags(
        "compute_ego_net",
      ) == "True",
    )

    // Only BQ READ supported right now
    // TODO: (svij-sc) This will need refactoring, just templating for now so we can test the flows
    val bq_table_input_str = experimentalFlags("bq_table_input")
    val numSlots: Integer  = experimentalFlags.getOrElse("num_slots", s"${DEFAULT_NUM_SLOTS}").toInt
    val flags: EgoNetGenFlags = read[EgoNetGenFlags](bq_table_input_str)
    val nodeTableName         = flags.node_bq_table
    val edgeTableName         = flags.edge_bq_table
    val fromNodeIdColumn      = flags.from_node_id_column
    val toNodeIdColumn        = flags.to_node_id_column

    println(s"""
    Running EgoNetGeneration Job w/ 
      nodeTableName: ${nodeTableName}, 
      edgeTableName: ${edgeTableName}, 
      fromNodeIdColumn: ${fromNodeIdColumn}, 
      toNodeIdColumn: ${toNodeIdColumn}
    """)

    val rawEdgeDF = fetchEdgeTableFromBq(
      tableName = edgeTableName,
      fromNodeIdColumn = fromNodeIdColumn,
      toNodeIdColumn = toNodeIdColumn,
    ) // table: src_node_id: int, dst_node_id: int, edge_features: array<float>
    EgoNetGeneration.generateEgoNodes(
      spark = spark,
      rawEdgeDF = rawEdgeDF,
      numOptimalPartitions = DEFAULT_NUM_OPTIMAL_PARTITIONS,
      numSlots = numSlots,
      sampleN = DEFAULT_SAMPLE_N,
      numHops = DEFAULT_NUM_HOPS,
      outputDir = outputDir,
    )
  }
}
