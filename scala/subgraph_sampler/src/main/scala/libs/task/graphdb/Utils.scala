package libs.task.graphdb

import com.typesafe.scalalogging.LazyLogging
import common.src.main.scala.types.EdgeUsageType
import common.types.GraphTypes.CondensedNodeType
import common.types.GraphTypes.NodeType
import common.types.Metapath
import common.types.pb_wrappers.GbmlConfigPbWrapper
import common.types.pb_wrappers.GraphMetadataPbWrapper
import common.utils.SparkSessionEntry.getActiveSparkSession
import common.utils.SparkSessionEntry.getNumCurrentShufflePartitions
import common.utils.TFRecordIO.RecordTypes
import common.utils.TFRecordIO.readDataframeFromTfrecord
import org.apache.spark.sql.Column
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{functions => F}
import org.apache.spark.storage.StorageLevel
import scalapb.spark.Implicits._
import snapchat.research.gbml.graph_schema.EdgeType
import snapchat.research.gbml.preprocessed_metadata.PreprocessedMetadata
import snapchat.research.gbml.training_samples_schema.RootedNodeNeighborhood

import java.util.UUID.randomUUID
import scala.collection.mutable
import scala.collection.mutable.ListBuffer

object Utils extends Serializable with LazyLogging {
  val spark: SparkSession = getActiveSparkSession
  val samplingSeed: Int   = 42
  // var lockCallToMethod: Boolean = false

  private def generateUniqueSuffix: String = {

    /** create a unique suffix for table view names, upon initialization of this calss
     */
    val uniqueTempViewSuffix = randomUUID.toString.replace("-", "_")
    "_" + uniqueTempViewSuffix
  }

  val uniqueTempViewSuffix: String = generateUniqueSuffix

  def loadNodeDataframeIntoSparkSql(
    condensedNodeTypes: Seq[Int],
    gbmlConfigWrapper: GbmlConfigPbWrapper,
  ): String = {

    /** For a given condensed_node_type, loads the node feature/type dataframe with columns:
     * _node_id: Integer, _node_features: Array(), _condensed_node_type: Integer. Returns VEIW name
     * associated with the node dataframe.
     */
    val hydratedNodeDFs: Seq[DataFrame] = condensedNodeTypes.map { condensedNodeType =>
      /** For a given condensed_node_type, loads the node feature/type dataframe with columns:
       * _node_id: Integer, _node_features: Array(), _condensed_node_type: Integer. Returns VIEW name
       * associated with the node dataframe.
       */
      val nodeMetadataOutputPb: PreprocessedMetadata.NodeMetadataOutput =
        gbmlConfigWrapper.preprocessedMetadataWrapper
          .getPreprocessedMetadataForCondensedNodeType(condensedNodeType = condensedNodeType)

      val featureKeys: List[String] = nodeMetadataOutputPb.featureKeys.toList
      val nodeIdKey: String         = nodeMetadataOutputPb.nodeIdKey
      val nodeDfUri: String         = nodeMetadataOutputPb.tfrecordUriPrefix

      val rawNodeDF: DataFrame = readDataframeFromTfrecord(
        uri = nodeDfUri,
        recordType = RecordTypes.ExampleRecordType,
      )

      val preprocessedNodeDF = rawNodeDF.withColumn(
        nodeIdKey,
        rawNodeDF(nodeIdKey) cast (IntegerType),
      )

      val preprocessedNodeVIEW: String = "preprocessedNodeDF" + uniqueTempViewSuffix
      preprocessedNodeDF.createOrReplaceTempView(
        preprocessedNodeVIEW,
      )

      var colDtypeMap = collection.mutable.Map[String, String]()
      preprocessedNodeDF.dtypes.foreach { case (columnName, columnType) =>
        colDtypeMap += (columnName -> columnType)
      }

      var columnWithOperations = new ListBuffer[String]()
      featureKeys.foreach { columnName =>
        if (colDtypeMap(columnName).startsWith("ArrayType")) { // cast to array<float>
          columnWithOperations += s"cast(${columnName} as array<float>)"
        } else { // convert + cast to array<float>
          columnWithOperations += s"cast(array(${columnName}) as array<float>)"
        }
      }
      val hydratedNodeDF: DataFrame = spark.sql(f"""
      SELECT /*+ REPARTITION(_node_id) */
        ${nodeIdKey} AS _node_id,
        flatten(array(${columnWithOperations.mkString(", ")})) AS _node_features,
        ${condensedNodeType} AS _condensed_node_type
      FROM ${preprocessedNodeVIEW}
        """)
      getActiveSparkSession.catalog.dropTempView(preprocessedNodeVIEW)
      hydratedNodeDF
    }

    val hydratedNodeDF = hydratedNodeDFs.reduce(_ union _)
    val hydratedNodeVIEW: String =
      "hydratedNodeDF" + uniqueTempViewSuffix
    hydratedNodeDF.createOrReplaceTempView(
      hydratedNodeVIEW,
    )
    val cachedHydratedNodeVIEW = applyCachingToDf(
      dfVIEW = hydratedNodeVIEW,
      forceTriggerCaching = true,
      withRepartition = false,
    )
    cachedHydratedNodeVIEW
  }

  // TODO (yliu2-sc) in HGS we'll have to load multiple types, to add
  def loadEdgeDataframeIntoSparkSql(
    condensedEdgeTypes: Seq[Int],
    edgeUsageType: EdgeUsageType.EdgeUsageType = EdgeUsageType.MAIN,
    gbmlConfigWrapper: GbmlConfigPbWrapper,
  ): String = {

    /** For a given condensed_edge_type loads edge dataframe with columns: _from: Int, _to: Int,
     * _condensed_edge_type: Int, _edge_features: Array()(if features are provided) (used for
     * hydration)
     * edgeUsageType: EdgeUsageType.EdgeUsageType = EdgeUsageType.MAIN, EdgeUsageType.POS, EdgeUsageType.NEG
     */
    val hydratedEdgeDFs = condensedEdgeTypes.map { condensedEdgeType =>
      val edgeMetadataOutputPb: PreprocessedMetadata.EdgeMetadataOutput =
        gbmlConfigWrapper.preprocessedMetadataWrapper
          .getPreprocessedMetadataForCondensedEdgeType(condensedEdgeType = condensedEdgeType)

      val edgeInfo = if (edgeUsageType == EdgeUsageType.MAIN) {
        edgeMetadataOutputPb.mainEdgeInfo
      } else if (edgeUsageType == EdgeUsageType.POS) {
        edgeMetadataOutputPb.positiveEdgeInfo
      } else if (edgeUsageType == EdgeUsageType.NEG) {
        edgeMetadataOutputPb.negativeEdgeInfo
      } else {
        throw new IllegalArgumentException(
          "edgeUsageType in function loadEdgeDataframeIntoSparkSql must be one of " +
            s"{${EdgeUsageType.MAIN}, ${EdgeUsageType.POS}, ${EdgeUsageType.NEG}}. " +
            s"Got: ${edgeUsageType}",
        )
      }
      println(
        s"loadEdgeDataframeIntoSparkSql - edgeUsageType: ${edgeUsageType}, edgeInfo: ${edgeInfo}, " +
          s"tfrecordUriPrefix: ${edgeInfo.get.tfrecordUriPrefix}",
      )

      val srcNodeIdKey: String = edgeMetadataOutputPb.srcNodeIdKey
      val dstNodeIdKey: String = edgeMetadataOutputPb.dstNodeIdKey
      val edgeDfUri: String    = edgeInfo.get.tfrecordUriPrefix
      val hasEdgeFeatures: Boolean =
        edgeInfo.get.featureKeys.nonEmpty // hasEdgeFeatures can be inferred from preprocessed metadata
      val edgeFeatureKeys: List[String] = edgeInfo.get.featureKeys.toList

      val rawEdgeDF: DataFrame = readDataframeFromTfrecord(
        uri = edgeDfUri,
        recordType = RecordTypes.ExampleRecordType,
      )

      val colsToProcess: Array[Column] = rawEdgeDF.columns.map(c => {
        if (c.contains(srcNodeIdKey) || c.contains(dstNodeIdKey)) F.col(c).cast(IntegerType)
        else F.col(c)
      })
      val preprocessedEdgeDF = rawEdgeDF.select(colsToProcess: _*) // See NOTE1
      val preprocessedEdgeVIEW: String =
        f"preprocessed${condensedEdgeType}${edgeUsageType}EdgeDF" + uniqueTempViewSuffix
      preprocessedEdgeDF.createOrReplaceTempView(preprocessedEdgeVIEW)
      // rename df-edge columns and add edge types and features columns
      val hydratedEdgeDF = if (hasEdgeFeatures == true) {
        // below handles cases where we have array of edge features
        var colDtypeMap = collection.mutable.Map[String, String]()
        preprocessedEdgeDF.dtypes.foreach { case (columnName, columnType) =>
          colDtypeMap += (columnName -> columnType)
        }
        var columnWithOperations = new ListBuffer[String]()
        edgeFeatureKeys.foreach { columnName =>
          if (colDtypeMap(columnName).startsWith("ArrayType")) { // cast to array<float>
            columnWithOperations += s"cast(${columnName} as array<float>)"
          } else { // convert + cast to array<float>
            columnWithOperations += s"cast(array(${columnName}) as array<float>)"
          }
        }
        spark.sql(f"""
          SELECT
            $srcNodeIdKey AS _from,
            $dstNodeIdKey AS _to,
            ${condensedEdgeType} AS _condensed_edge_type,
            flatten(array(${columnWithOperations.mkString(", ")})) AS _edge_features
          FROM
            ${preprocessedEdgeVIEW}
          """)
      } else {
        val emptyEdgeFeature: Seq[Float] =
          Seq.empty[Float] // @spark: explicitly assigns empty list to edge features -- to avoid errors during protobuf conversion
        // NOTE: writing below using SQL leads to conflicts between SQL types and Scala types.
        // That's why below is written in Dataframe API, this way
        // conversions between spark.sql types and scala types are handeled under the hood
        spark
          .table(preprocessedEdgeVIEW)
          .select(
            F.col(srcNodeIdKey).alias("_from"),
            F.col(dstNodeIdKey).alias("_to"),
            F.lit(condensedEdgeType).alias("_condensed_edge_type"),
            F.typedLit(emptyEdgeFeature).alias("_edge_features"),
          )
      }
      getActiveSparkSession.catalog.dropTempView(preprocessedEdgeVIEW)
      hydratedEdgeDF
    }
    val hydratedEdgeDF = hydratedEdgeDFs.reduce(_ union _)
    var hydratedEdgeVIEW: String =
      f"hydrated${edgeUsageType}EdgeDF" + uniqueTempViewSuffix
    hydratedEdgeDF.createOrReplaceTempView(
      hydratedEdgeVIEW,
    )

    // @spark: we cache hydratedEdgeDF here to trim DAG and avoid running same stages again and again,
    // saves at least 1hr of runtime for MAU
    val cachedHydratedEdgeVIEW = applyCachingToDf(
      dfVIEW = hydratedEdgeVIEW,
      forceTriggerCaching = true,
      withRepartition = true,
      repartitionFactor = 1,
      colNameForRepartition = "_from",
    )

    cachedHydratedEdgeVIEW
  }

  def loadUnhydratedEdgeDataframeIntoSparkSql(
    hydratedEdgeVIEW: String,
    edgeUsageType: EdgeUsageType.EdgeUsageType = EdgeUsageType.MAIN,
  ): String = {

    /** Loads edge dataframe with columns: _src_node: Int, _dst_node: Int (used for sampling)
     * @param hydratedEdgeVIEW with columns:
     * _from: Int, src node id
     * _to: Int, dst node id
     * _condensed_edge_type: Int
     * _edge_features: Array, icluding all features
     */
    val unhydratedEdgeVIEW: String = f"unhydrated${edgeUsageType}EdgeDF" + uniqueTempViewSuffix
    val unhydratedEdgeDF = spark.sql(f"""
        SELECT
          _from AS _src_node,
          _to AS _dst_node
        FROM
          ${hydratedEdgeVIEW}
        """)

    unhydratedEdgeDF.createOrReplaceTempView(unhydratedEdgeVIEW)
    unhydratedEdgeVIEW
  }

  def castToRootedNodeNeighborhoodProtoSchema(dfVIEW: String): String = {

    /** Creates desirable schema for RootedNodeNeighborhood defined in training_samples_schema.proto. Returns rnnVIEW
     * with columns: root_node: Node, neighborhood: Graph
     */
    val rnnDF = spark.sql(f"""
                SELECT
                  struct(
                    _root_node AS node_id,
                    _condensed_node_type AS condensed_node_type,
                    _node_features AS feature_values
                  ) AS root_node,
                  struct(
                    _neighbor_nodes AS nodes,
                    _neighbor_edges AS edges
                  ) AS neighborhood
                FROM ${dfVIEW}
                """)
    val rnnVIEW = "rnnDF" + uniqueTempViewSuffix
    rnnDF.createOrReplaceTempView(rnnVIEW)
    rnnVIEW
  }

  def downsampleNumberOfNodes(
    numSamples: Int,
    dfVIEW: String,
  ): String = {

    /**
     * Downsamples number of root nodes to numSamples
     * @param numSamples number of root nodes to sample
     * @param dfVIEW
     * @param rootNodeColName
     */
    // @spark: we use RAND to sample uniformely, but this is not deterministic,which is not problematic in this case.
    // in case we wish to enforce determinism, we can use sort and take first numSamples
    val downsampledDF = spark.sql(f"""
          SELECT
            *
          FROM
            ${dfVIEW}
          LIMIT
            ${numSamples}
          """)
    //   (yliu2) using below df.sample can be fast(similar to above query) but we need to do a count which take time
    //    val num_nodes: Long = spark.table(dfVIEW).count() //this might have different effects if the df is cached or not
    //    val ratio: Double = numSamples.toDouble / num_nodes.toDouble
    //    println(s"Downsampling with ratio ${ratio}")
    //    val downsampledDF =
    //      spark.table(dfVIEW).sample(withReplacement = false, fraction = ratio, seed = samplingSeed)

    val downsampledVIEW = "downsampledDF" + uniqueTempViewSuffix
    downsampledDF.createOrReplaceTempView(downsampledVIEW)

    //    if we don 't force sequential execution nor break the parallelization we will have different samples.
    //    So to be extra safe we need another cache here
    val cachedDownSampledVIEW = applyCachingToDf(
      dfVIEW = downsampledVIEW,
      forceTriggerCaching = true,
      withRepartition = false,
    )
    cachedDownSampledVIEW
  }

  def applyCachingToDf(
    dfVIEW: String,
    forceTriggerCaching: Boolean,
    withRepartition: Boolean,
    repartitionFactor: Int = -1,
    colNameForRepartition: String = "",
    storageLevel: StorageLevel = StorageLevel.DISK_ONLY,
  ): String = {

    /**
     * Used for performance optimization and more importantly for accuracy of sampling.
     * @param dfVIEW associated with the DatafFrame we want to cache
     * @param triggerCachingViaCount whether to apply caching by triggering an action, ie df.count(). If false, another action such as df.write() will trigger cache.
     * @param withRepartition whether to repartition the df associate with subgraph dfVIEW. If there are spills in final stages, set this to true
     * @param repartitionFactor should be determined based on graph size.
     * @param colNameForRepartition name of column (col that has root nodes, or node ids)
     */
    //  @spark: StorageLevel=DISK_ONLY, if data is cached in MEMORY instead of DISK, it leads to faster spill to disk during shuffles. Since both cached data and shuffle data compete for Storage memory of executors. As long as local SSD is attached to each worker, StorageLevel=DISK_ONLY is fast enough.
    // NOTE: repartition MUST be done before caching, since repartitioning clears corresponding cached data
    val df = spark.table(dfVIEW)
    val cachedDF = if (withRepartition) {
      val repartitionedDF = applyRepartitionToDataFrame(
        df = df,
        colName = colNameForRepartition,
        repartitionFactor = repartitionFactor,
      )
      repartitionedDF.persist(storageLevel)
    } else { df.persist(storageLevel) }

    if (forceTriggerCaching == true) {
      // use count as action, to trigger cache completely (ie distribute/materialize data on worker nodes).
      // Count is not expensive since it does NOT accumulate data in the driver node, but rather in workers node. [Execution time for count+cache (mau data): 5 min]
      cachedDF.count()
    }

    val cachedDfVIEW = "cached_" + dfVIEW + uniqueTempViewSuffix
    cachedDF.createOrReplaceTempView(cachedDfVIEW)
    cachedDfVIEW
  }

  def applyRepartitionToDataFrame(
    df: DataFrame,
    colName: String,
    repartitionFactor: Int,
  ): DataFrame = {

    /**
     * @spark: used for repartitioning dataframes in intermediate stages of the Spark job
     * to avoid spills as data size grows.
     * e.g., curNumShufflePartitions = 300 and we have initial partition size 200Mb
     * stages proceed, and data size for each partition grows to 5000Mb, spills or OOM error happen
     * in this case we repartition by a factor eg 10, so numPartitions = 300 * 10 = 3000
     * and data size for each partition will be 5000Mb/10= 500 Mb and spill/OOMs will be mitigated
     * NOTE: repartition clears cached data, first repartition then cache if required
     * @param colName: the column name that joins are performed on
     */
    val numShufflePartitions = getNumCurrentShufflePartitions(spark = spark)
    val repartitionedDF = df.repartition(
      numPartitions = numShufflePartitions * repartitionFactor,
      partitionExprs = F.col(colName),
    )
    repartitionedDF
  }

  def applyRepartitionToDataFrameVIEW(
    dfVIEW: String,
    colName: String,
    repartitionFactor: Int,
  ): String = {

    /**
     * @spark: used for repartitioning dataframes in intermediate stages of the Spark job
     *         to avoid spills as data size grows.
     *         e.g., curNumShufflePartitions = 300 and we have initial partition size 200Mb
     *         stages proceed, and data size for each partition grows to 5000Mb, spills or OOM error happen
     *         in this case we repartition by a factor eg 10, so numPartitions = 300 * 10 = 3000
     *         and data size for each partition will be 5000Mb/10= 500 Mb and spill/OOMs will be mitigated
     *         NOTE: repartition clears cached data, first repartition then cache if required
     * @param colName : the column name that joins are performed on
     */
    val df                   = spark.table(dfVIEW)
    val numShufflePartitions = getNumCurrentShufflePartitions(spark = spark)
    val repartitionedDF = df.repartition(
      numPartitions = numShufflePartitions * repartitionFactor,
      partitionExprs = F.col(colName),
    )
    val repartitionedDfVIEW = "repartitioned_" + dfVIEW + uniqueTempViewSuffix
    repartitionedDF.createOrReplaceTempView(repartitionedDfVIEW)

    repartitionedDfVIEW
  }

  def applyCoalesceToDataFrameVIEW(
    dfVIEW: String,
    colName: String,
    coalesceFactor: Int,
  ): String = {

    /**
     * @spark: used for coalescing dataframes in intermediate stages of the Spark job
     *         to avoid small file problems.
     * @param colName : the column name that joins are performed on
     * @param coalesceFactor : the number of partitions to coalesce to
     *                       ex.10 then new # of partitions will be getCurrentShufflePartitionsNum(spark) / 10
     */
    val df                   = spark.table(dfVIEW)
    val numShufflePartitions = getNumCurrentShufflePartitions(spark = spark)
    val coalescedDF          = df.coalesce(numShufflePartitions / coalesceFactor)
    val coalescedDfVIEW      = "coalesced_" + dfVIEW + uniqueTempViewSuffix
    coalescedDF.createOrReplaceTempView(coalescedDfVIEW)

    coalescedDfVIEW
  }

  // we do a travesal and get all possible metapaths
  // metapaths return the relation of the EdgTypes
  // TODO (yliu2-sc) can return EdgeType directly instead
  def getMetapaths(
    nodeTypes: Seq[NodeType],
    graphMetadataPbWrapper: GraphMetadataPbWrapper,
    numHops: Int,
  ): Seq[Metapath] = {
    val edgeTypes                = graphMetadataPbWrapper.edgeTypes
    val dstNodeTypeToEdgeTypeMap = graphMetadataPbWrapper.dstNodeTypeToEdgeTypeMap

    val queue     = mutable.Queue[(List[EdgeType], Int)]()
    val metapaths = mutable.ListBuffer[Metapath]()

    nodeTypes
      .flatMap(nodeType => dstNodeTypeToEdgeTypeMap.get(nodeType))
      .foreach(edgeTypeSeq => edgeTypeSeq.foreach(edgeType => queue.enqueue((List(edgeType), 1))))

    while (queue.nonEmpty) {
      val (edgeTypeList, hop) = queue.dequeue()
      val currEdgeType        = edgeTypeList.last
      if (hop == numHops) {
        metapaths += Metapath(edgeTypeList.map(_.relation))
      } else {
        dstNodeTypeToEdgeTypeMap
          .get(currEdgeType.srcNodeType)
          .getOrElse(Seq())
          .foreach(edgeType => {
            queue.enqueue((edgeTypeList :+ edgeType, hop + 1))
          })
      }
    }
    println(f"Metapaths for NodeTypes ${nodeTypes} : " + metapaths.toList)
    metapaths.toList
  }

  def loadNodeIds(
    hydratedNodeVIEW: String,
    condensedNodeType: CondensedNodeType,
    numPartitions: Int,
    numNodeIds: Int,
  ): Dataset[Int] = {
    val nodeIdsDF = spark.sql(
      f"""
        SELECT
          _node_id
        FROM ${hydratedNodeVIEW}
        WHERE _condensed_node_type = ${condensedNodeType}
        """,
    )

    val nodeIdsDS: Dataset[Int] = if (numNodeIds > 0) {
      nodeIdsDF.limit(numNodeIds).as[Int]
    } else {
      nodeIdsDF.as[Int]
    }

    println("Number of partitions: " + nodeIdsDS.rdd.getNumPartitions)

    // TODO (yliu2-sc) repartition to number of total cores or user provide
    val repartitionedNodeIdsDS = nodeIdsDS.repartition(numPartitions)
    println(
      "Number of partitions after repartition: " + repartitionedNodeIdsDS.rdd.getNumPartitions,
    )
    //    repartitionedNodeIdsDS

    // TODO (yliu2-sc) probably don't need to cache this, since we're only using it once
    val cachedNodeIdsDS = repartitionedNodeIdsDS.persist(StorageLevel.MEMORY_ONLY)
    println(
      f"NodeIdsDS count for node type ${condensedNodeType}: ${cachedNodeIdsDS.count()}",
    ) // force trigger action to cache
    cachedNodeIdsDS
  }

  def getAnchorNodeTypes(supervisionEdgeTypes: Seq[EdgeType]): Seq[NodeType] = {
    // dstNodeType since we consider in-edges
    supervisionEdgeTypes.map(_.dstNodeType)
  }

  def hydrateRnn(
    rnnDS: Dataset[RootedNodeNeighborhood],
    hydratedNodeVIEW: String,
    hydratedEdgeVIEW: String,
    shouldHydrateRootNode: Boolean = true,
    shuoldCacheResult: Boolean = true,
  ): Dataset[RootedNodeNeighborhood] = {
    /*
    Given a dataset of RootedNodeNeighborhoods, hydrate the node and edge features.
    Parmeter options to cache the result, and hydrate the root nodes.

    This function first explode nodes and edges separately for RootedNodeNeighborhood's neighborhood graphs, join their features,
    then finally join them back together into RootedNodeNeighborhood format

    @param rnnDS: Spark Dataset of type RootedNodeNeighborhood
    @param hydrateNodeVIEW: Dataframe VIEW of hydrated nodes with _node_id, _condensed_node_type, _feature_values as columns
    @param hydratedEdgeVIEW: Dataframe VIEW of hydrate edges with _from, _to, _condensed_edge_type, _feature_values as columns
    @param shouldHydrateRootNode: To hydrate root node if not hydrated already
    @param shuoldCacheResult: Cache the result
    @return: Spark Dataset of type RootedNodeNeighborhood
     */
    val rnnDF     = rnnDS.toDF()
    val rnnDFVIEW = "rnnDF" + uniqueTempViewSuffix
    rnnDF.createOrReplaceTempView(rnnDFVIEW)

    val rnnHydrateStartTime = System.currentTimeMillis()
    // First, join the node features
    val rnnNodeHydratedDF = spark.sql(
      f"""
          SELECT
            root_node,
            collect_list(
              struct(
                node.node_id AS node_id,
                node.condensed_node_type AS condensed_node_type,
                coalesce(nodeFeatures._node_features, array()) AS feature_values
               )
            ) AS neighbor_nodes
          FROM (
            SELECT root_node, explode(neighborhood.nodes) as node
            FROM ${rnnDFVIEW}
          )
          LEFT JOIN ${hydratedNodeVIEW} as nodeFeatures
          ON node.node_id = nodeFeatures._node_id AND node.condensed_node_type = nodeFeatures._condensed_node_type
          GROUP BY root_node
        """,
    )
    val rnnNodeHydratedVIEW = "rnnNodeHydratedDF" + uniqueTempViewSuffix
    rnnNodeHydratedDF.createOrReplaceTempView(rnnNodeHydratedVIEW)

    // Next, join the edge features
    val rnnEdgeHydratedDF = spark.sql(
      f"""
        SELECT
          root_node,
          collect_list(
            struct(
              edge.src_node_id AS src_node_id,
              edge.dst_node_id AS dst_node_id,
              edge.condensed_edge_type AS condensed_edge_type,
              coalesce(edgeFeatures._edge_features, array()) AS feature_values
            )
          ) AS neighbor_edges
        FROM (
          SELECT root_node, explode(neighborhood.edges) as edge
          FROM ${rnnDFVIEW}
        )
        LEFT JOIN ${hydratedEdgeVIEW} as edgeFeatures
        ON edge.src_node_id = edgeFeatures._from AND edge.dst_node_id = edgeFeatures._to AND edge.condensed_edge_type = edgeFeatures._condensed_edge_type
        GROUP BY root_node
        """,
    )
    val rnnEdgeHydratedVIEW = "rnnEdgeHydratedDF" + uniqueTempViewSuffix
    rnnEdgeHydratedDF.createOrReplaceTempView(rnnEdgeHydratedVIEW)

    // Finally, join the node and edge features back to neighborhood, with root node features
    var hydratedRnnDF =
      if (shouldHydrateRootNode) {
        spark.sql(
          f"""
          SELECT
            struct(
              hydratedNeighborhoodNodes.root_node.node_id AS node_id,
              hydratedNeighborhoodNodes.root_node.condensed_node_type AS condensed_node_type,
              rootNodeFeatures._node_features AS feature_values
            ) as root_node,
            struct (
              hydratedNeighborhoodNodes.neighbor_nodes AS nodes,
              hydratedNeighborhoodEdges.neighbor_edges AS edges
            ) as neighborhood
          FROM ${rnnNodeHydratedVIEW} AS hydratedNeighborhoodNodes
          JOIN ${rnnEdgeHydratedVIEW} AS hydratedNeighborhoodEdges
          ON hydratedNeighborhoodNodes.root_node = hydratedNeighborhoodEdges.root_node
          JOIN ${hydratedNodeVIEW} as rootNodeFeatures
          ON hydratedNeighborhoodNodes.root_node.node_id = rootNodeFeatures._node_id AND rootNodeFeatures._condensed_node_type = hydratedNeighborhoodNodes.root_node.condensed_node_type
        """,
        )
      } else {
        spark.sql(
          f"""
          SELECT
            hydratedNeighborhoodNodes.root_node AS root_node,
            struct (
              hydratedNeighborhoodNodes.neighbor_nodes AS nodes,
              hydratedNeighborhoodEdges.neighbor_edges AS edges
            ) as neighborhood
          FROM ${rnnNodeHydratedVIEW} AS hydratedNeighborhoodNodes
          JOIN ${rnnEdgeHydratedVIEW} AS hydratedNeighborhoodEdges
          ON hydratedNeighborhoodNodes.root_node.node_id = hydratedNeighborhoodEdges.root_node.node_id and hydratedNeighborhoodNodes.root_node.condensed_node_type = hydratedNeighborhoodEdges.root_node.condensed_node_type
        """,
        )
      }
    val hydratedRnnDS: Dataset[RootedNodeNeighborhood] = hydratedRnnDF.as[RootedNodeNeighborhood]

    if (shuoldCacheResult) {
      val cachedHydratedRnnDS = hydratedRnnDS.persist(StorageLevel.MEMORY_ONLY)
      println(
        f"cachedHydratedRnnDS count: ${cachedHydratedRnnDS.count()}",
      ) // force trigger action to cache
      val rnnHydrateEndTime     = System.currentTimeMillis()
      val rnnHydrateTimeSeconds = (rnnHydrateEndTime - rnnHydrateStartTime) / 1000
      println(f"RNNs Hydrate time: ${rnnHydrateTimeSeconds / 60}m ${rnnHydrateTimeSeconds % 60}s")
      cachedHydratedRnnDS
    } else {
      hydratedRnnDS
    }
  }

  // Function checks if a node is isolated by checking if the neighborhood only contains the root node
  def isIsolatedNode(rnn: RootedNodeNeighborhood): Boolean = {
    val rootNode          = rnn.rootNode.get
    val neighborhoodNodes = rnn.neighborhood.get.nodes
    neighborhoodNodes.size == 1 && neighborhoodNodes.contains(rootNode)
  }
}
