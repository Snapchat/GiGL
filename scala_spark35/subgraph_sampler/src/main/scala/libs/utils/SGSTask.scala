package libs.utils

import com.typesafe.scalalogging.LazyLogging
import common.types.GraphTypes.CondensedEdgeType
import common.types.GraphTypes.CondensedNodeType
import common.types.GraphTypes.NodeType
import common.types.Metapath
import common.types.pb_wrappers.GbmlConfigPbWrapper
import common.types.pb_wrappers.GraphMetadataPbWrapper
import common.utils.SparkSessionEntry.getActiveSparkSession
import libs.utils.DataLoading.loadEdgeDataframeIntoSparkSql
import libs.utils.DataLoading.loadNodeDataframeIntoSparkSql
import libs.utils.Spark.applyCachingToDataFrameVIEW
import libs.utils.Spark.uniqueTempViewSuffix
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel
import scalapb.spark.Implicits._
import snapchat.research.gbml.graph_schema.EdgeType
import snapchat.research.gbml.preprocessed_metadata.PreprocessedMetadata
import snapchat.research.gbml.training_samples_schema.RootedNodeNeighborhood

import scala.collection.mutable

object SGSTask extends Serializable with LazyLogging {
  val spark: SparkSession = getActiveSparkSession

  // we do a traversal and get all possible metapaths
  // metapaths return the relation of the EdgTypes
  // TODO (yliu2) can return EdgeType directly instead
  def getMetapaths(
    nodeTypes: Seq[NodeType],
    graphMetadataPbWrapper: GraphMetadataPbWrapper,
    numHops: Int,
  ): Seq[Metapath] = {
    val edgeTypes                = graphMetadataPbWrapper.graphMetadataPb.edgeTypes
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
    numNodeIds: Option[Int] = None,
    numPartitions: Option[Int] = None,
    shouldCacheResult: Boolean = false,
    storageLevel: StorageLevel = StorageLevel.DISK_ONLY,
  ): Dataset[Int] = {
    val nodeIdsDF = spark.sql(
      f"""
        SELECT
          _node_id
        FROM ${hydratedNodeVIEW}
        WHERE _condensed_node_type = ${condensedNodeType}
        """,
    )

    val nodeIdsDS: Dataset[Int] = if (numNodeIds.isDefined && numNodeIds.get > 0) {
      nodeIdsDF.limit(numNodeIds.get).as[Int]
    } else {
      nodeIdsDF.as[Int]
    }

    println("Number of partitions: " + nodeIdsDS.rdd.getNumPartitions)

    val repartitionedNodeIdsDS = numPartitions match {
      case Some(n) => nodeIdsDS.repartition(n)
      case None    => nodeIdsDS
    }
    println(
      "Number of partitions after repartition: " + repartitionedNodeIdsDS.rdd.getNumPartitions,
    )

    if (shouldCacheResult) {
      val cachedRepartitionedNodeIdsDS = repartitionedNodeIdsDS.persist(storageLevel)
      println(
        f"NodeIdsDS count for node type ${condensedNodeType}: ${cachedRepartitionedNodeIdsDS.count()}",
      ) // force trigger action to cache
      cachedRepartitionedNodeIdsDS
    } else {
      repartitionedNodeIdsDS
    }
  }

  def loadHydratedNodeDataFrame(
    condensedNodeTypes: Seq[CondensedNodeType],
    gbmlConfigWrapper: GbmlConfigPbWrapper,
  ): String = {
    val hydratedNodeDFs: Seq[DataFrame] = condensedNodeTypes.map { condensedNodeType =>
      /** For given condensed_node_types, loads the node feature/type dataframe with columns:
       * _node_id: Integer, _node_features: Array(), _condensed_node_type: Integer. Returns VIEW name
       * associated with the node dataframe.
       */
      val nodeMetadataOutputPb: PreprocessedMetadata.NodeMetadataOutput =
        gbmlConfigWrapper.preprocessedMetadataWrapper
          .getPreprocessedMetadataForCondensedNodeType(condensedNodeType = condensedNodeType)

      val featureKeys: List[String] = nodeMetadataOutputPb.featureKeys.toList
      val nodeIdKey: String         = nodeMetadataOutputPb.nodeIdKey
      val nodeDfUri: String         = nodeMetadataOutputPb.tfrecordUriPrefix

      val hydratedNodeDF = loadNodeDataframeIntoSparkSql(
        condensedNodeType = condensedNodeType,
        featureKeys = featureKeys,
        nodeIdKey = nodeIdKey,
        nodeDfUri = nodeDfUri,
        shouldCacheResult = false,
      )
      hydratedNodeDF
    }
    val hydratedNodeDF = hydratedNodeDFs.reduce((df1, df2) => df1.unionByName(df2))
    val hydratedNodeVIEW: String =
      "hydratedNodeDF" + uniqueTempViewSuffix
    hydratedNodeDF.createOrReplaceTempView(
      hydratedNodeVIEW,
    )
    val cachedHydratedNodeVIEW = applyCachingToDataFrameVIEW(
      dfVIEW = hydratedNodeVIEW,
      forceTriggerCaching = true,
      withRepartition = false,
    )
    cachedHydratedNodeVIEW
  }

  def loadHydratedEdgeDataFrame(
    condensedEdgeTypes: Seq[CondensedEdgeType],
    gbmlConfigWrapper: GbmlConfigPbWrapper,
  ): String = {
    val hydratedEdgeDFs: Seq[DataFrame] = condensedEdgeTypes.map { condensedEdgeType =>
      val edgeMetadataOutputPb: PreprocessedMetadata.EdgeMetadataOutput =
        gbmlConfigWrapper.preprocessedMetadataWrapper
          .getPreprocessedMetadataForCondensedEdgeType(condensedEdgeType = condensedEdgeType)

      val edgeInfo = edgeMetadataOutputPb.mainEdgeInfo

      println(
        s"loadEdgeDataframeIntoSparkSql - condensedEdgeType: ${condensedEdgeType}, edgeInfo: ${edgeInfo}, " +
          s"tfrecordUriPrefix: ${edgeInfo.get.tfrecordUriPrefix}",
      )

      val srcNodeIdKey: String = edgeMetadataOutputPb.srcNodeIdKey
      val dstNodeIdKey: String = edgeMetadataOutputPb.dstNodeIdKey
      val edgeDfUri: String    = edgeInfo.get.tfrecordUriPrefix
      val hasEdgeFeatures: Boolean =
        edgeInfo.get.featureKeys.nonEmpty // hasEdgeFeatures can be inferred from preprocessed metadata
      val edgeFeatureKeys: List[String] = edgeInfo.get.featureKeys.toList
      val hydratedEdgeDF = loadEdgeDataframeIntoSparkSql(
        condensedEdgeType = condensedEdgeType,
        srcNodeIdKey = srcNodeIdKey,
        dstNodeIdKey = dstNodeIdKey,
        edgeDfUri = edgeDfUri,
        edgeFeatureKeys = edgeFeatureKeys,
      )
      hydratedEdgeDF
    }
    val hydratedEdgeDF = hydratedEdgeDFs.reduce((df1, df2) => df1.unionByName(df2))
    var hydratedEdgeVIEW: String =
      "hydratedEdgeDF" + uniqueTempViewSuffix
    hydratedEdgeDF.createOrReplaceTempView(
      hydratedEdgeVIEW,
    )

    // @spark: we cache hydratedEdgeDF here to trim DAG and avoid running same stages again and again,
    // saves at least 1hr of runtime for MAU
    val cachedHydratedEdgeVIEW = applyCachingToDataFrameVIEW(
      dfVIEW = hydratedEdgeVIEW,
      forceTriggerCaching = true,
      withRepartition = true,
      repartitionFactor = 1,
      colNameForRepartition = "_from",
    )

    cachedHydratedEdgeVIEW
  }

  def hydrateRnn(
    rnnDS: Dataset[RootedNodeNeighborhood],
    hydratedNodeVIEW: String,
    hydratedEdgeVIEW: String,
    shouldHydrateRootNode: Boolean = true,
    shouldHydrateEdges: Boolean = true,
    shouldCacheResult: Boolean = true,
    storageLevel: StorageLevel = StorageLevel.DISK_ONLY,
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
    @param shouldCacheResult: Cache the result
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
    // NOTE: Isolated nodes will have empty edge list, they will be excluded from hydratedEdgeDF
    //    so we need to LEFT JOIN the hydratedEdgeDF in the final hydratedRNN join
    val rnnEdgeHydratedDF = if (shouldHydrateEdges) {
      spark.sql(
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
    } else {
      println("Skipping edge hydration")
      spark.sql(
        f"""
        SELECT root_node, neighborhood.edges AS neighbor_edges
        FROM ${rnnDFVIEW}
        """,
      )
    }
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
          LEFT JOIN ${rnnEdgeHydratedVIEW} AS hydratedNeighborhoodEdges
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
          LEFT JOIN ${rnnEdgeHydratedVIEW} AS hydratedNeighborhoodEdges
          ON hydratedNeighborhoodNodes.root_node.node_id = hydratedNeighborhoodEdges.root_node.node_id and hydratedNeighborhoodNodes.root_node.condensed_node_type = hydratedNeighborhoodEdges.root_node.condensed_node_type
        """,
        )
      }
    val hydratedRnnDS: Dataset[RootedNodeNeighborhood] = hydratedRnnDF.as[RootedNodeNeighborhood]

    if (shouldCacheResult) {
      val cachedHydratedRnnDS = hydratedRnnDS.persist(storageLevel)
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
