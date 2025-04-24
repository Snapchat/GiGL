package libs.task.graphdb

import common.graphdb.NebulaGraphDBClient
import common.src.main.scala.types.EdgeUsageType
import common.types.GraphTypes.CondensedEdgeType
import common.types.GraphTypes.CondensedNodeType
import common.types.GraphTypes.NodeType
import common.types.Metapath
import common.types.pb_wrappers.GbmlConfigPbWrapper
import common.types.pb_wrappers.GraphMetadataPbWrapper
import common.utils.TFRecordIO.writeDatasetToTfrecord
import common.utils.TrainingSamplesHelper.mergeGraphs
import common.utils.TrainingSamplesHelper.mergeRootedNodeNeighborhoods
import libs.sampler.NebulaHeteroKHopSampler
import libs.task.SubgraphSamplerTask
import libs.task.TaskOutputValidator
import libs.task.graphdb.Utils.getAnchorNodeTypes
import libs.task.graphdb.Utils.getMetapaths
import libs.task.graphdb.Utils.hydrateRnn
import libs.task.graphdb.Utils.loadEdgeDataframeIntoSparkSql
import libs.task.graphdb.Utils.loadNodeDataframeIntoSparkSql
import libs.task.graphdb.Utils.loadNodeIds
import libs.task.graphdb.Utils.uniqueTempViewSuffix
import org.apache.spark.sql.Dataset
import org.apache.spark.storage.StorageLevel
import scalapb.spark.Implicits._
import snapchat.research.gbml.graph_schema.Edge
import snapchat.research.gbml.graph_schema.EdgeType
import snapchat.research.gbml.graph_schema.Graph
import snapchat.research.gbml.graph_schema.Node
import snapchat.research.gbml.training_samples_schema.NodeAnchorBasedLinkPredictionSample
import snapchat.research.gbml.training_samples_schema.RootedNodeNeighborhood

class GraphDBUnsupervisedNodeAnchorBasedLinkPredictionTask(
  gbmlConfigWrapper: GbmlConfigPbWrapper,
  graphMetadataPbWrapper: GraphMetadataPbWrapper)
    extends SubgraphSamplerTask(gbmlConfigWrapper) {

  // TODO (yliu2-sc) can get numCores from sparkContenxt
  val numPartitions: Int    = 6400
  val numNodeIdsToTest: Int = -1
//  val numPartitions: Int    = 1
//  val numNodeIdsToTest: Int = 10

  def run(): Unit = {
    // TODO (yliu2-sc) graphDBClient class should be read as parameter
    val graphMetadataPbWrapper: GraphMetadataPbWrapper = GraphMetadataPbWrapper(
      gbmlConfigWrapper.graphMetadataPb,
    )
    val numNeighborsToSample: Int = gbmlConfigWrapper.subgraphSamplerConfigPb.numNeighborsToSample
    val numHops: Int              = gbmlConfigWrapper.subgraphSamplerConfigPb.numHops
    val shouldIncludeIsolatedNodesInTraining: Boolean = true
    // TODO (yliu2-sc) add proto field
//    val shouldIncludeIsolatedNodesInTraining: Boolean =
//      gbmlConfigWrapper.sharedConfigPb.shouldIncludeIsolatedNodesInTraining.getOrElse(
//        false,
//      )

    val nablpTaskOutputUri =
      gbmlConfigWrapper.flattenedGraphMetadataPb.getNodeAnchorBasedLinkPredictionOutput

    val taskMetadata = gbmlConfigWrapper.taskMetadataPb.getNodeAnchorBasedLinkPredictionTaskMetadata
    val graphdbArgs  = gbmlConfigWrapper.subgraphSamplerConfigPb.graphDbConfig.get.graphDbArgs

    val allNodeTypes = graphMetadataPbWrapper.nodeTypes
    val allEdgeTypes = graphMetadataPbWrapper.edgeTypes

    // In phase 1 we will only support 1 supervision edge type, and 1 anchor node type
    val supervisionEdgeTypes: Seq[EdgeType] = taskMetadata.supervisionEdgeTypes
    val anchorNodeTypes: Seq[NodeType] =
      getAnchorNodeTypes(supervisionEdgeTypes = supervisionEdgeTypes)
    val supervisionSrcDstNodeTypes: Seq[NodeType] = supervisionEdgeTypes
      .flatMap(edgeType => Seq(edgeType.srcNodeType, edgeType.dstNodeType))
      .distinct
    println(s"Supervision edge types: $supervisionEdgeTypes")
    println(s"Anchor node types: $anchorNodeTypes")
    println(s"Supervision unique src-dst node types: $supervisionSrcDstNodeTypes")

    val numPositiveSamples: Int = gbmlConfigWrapper.subgraphSamplerConfigPb.numPositiveSamples
    val numTrainingSamples: Int =
      gbmlConfigWrapper.subgraphSamplerConfigPb.numMaxTrainingSamplesToOutput
    println(s"numPositiveSamples: $numPositiveSamples")
    println(s"numTrainingSamples: $numTrainingSamples")

    val supervisionCondensedEdgeTypes = supervisionEdgeTypes.map(edgeType =>
      graphMetadataPbWrapper.edgeTypeToCondensedEdgeTypeMap.get(edgeType).get,
    )

    val allCondensedNodeTypes: Seq[CondensedNodeType] = graphMetadataPbWrapper.condensedNodeTypes
    val allCondensedEdgeTypes: Seq[CondensedEdgeType] = graphMetadataPbWrapper.condensedEdgeTypes

    val loadDataStartTime = System.currentTimeMillis()
    val hydratedNodeVIEW: String =
      loadNodeDataframeIntoSparkSql(
        condensedNodeTypes = allCondensedNodeTypes,
        gbmlConfigWrapper = gbmlConfigWrapper,
      )
    val hydratedEdgeVIEW: String =
      loadEdgeDataframeIntoSparkSql(
        condensedEdgeTypes = allCondensedEdgeTypes,
        edgeUsageType = EdgeUsageType.MAIN,
        gbmlConfigWrapper = gbmlConfigWrapper,
      )
    val loadDataEndTime     = System.currentTimeMillis()
    val loadDataTimeSeconds = (loadDataEndTime - loadDataStartTime) / 1000
    println(f"Load data time: ${loadDataTimeSeconds / 60}m ${loadDataTimeSeconds % 60}s")

    // Generate RNNs for each node type
    val hydratedRnnMap =
      scala.collection.mutable.Map[NodeType, Dataset[RootedNodeNeighborhood]]()
    for (nodeType <- supervisionSrcDstNodeTypes) {
      val condensedNodeType =
        graphMetadataPbWrapper.nodeTypeToCondensedNodeTypeMap.get(nodeType).get
      // TODO (yliu2-sc) Allow users to provide metapaths in graphdbArgs
      val metapaths: Seq[Metapath] = getMetapaths(
        nodeTypes = Seq(nodeType),
        graphMetadataPbWrapper = graphMetadataPbWrapper,
        numHops = numHops,
      )
      val nodeIdsDS = loadNodeIds(
        hydratedNodeVIEW = hydratedNodeVIEW,
        condensedNodeType = condensedNodeType,
        numPartitions = numPartitions,
        numNodeIds = numNodeIdsToTest,
      )

      // TODO (svij-sc) We should probably add a stopwatch or timer utility
      val rnnStartTime = System.currentTimeMillis()
      println(f"\nGenerate RNNs for node type ${condensedNodeType} [${nodeType}] ")
      val rnnDS: Dataset[RootedNodeNeighborhood] =
        nodeIdsDS.mapPartitions(nodeIds => {
          // NOTES (yliu2)
          // graphDBclient needs to be serializable if defined outside of mapPartitions, nebula client however is has underlying classes that are not serializable
          // ConnectionPool + getSession - java.io.NotSerializableException: com.vesoft.nebula.client.graph.net.RoundRobinLoadBalancer
          // SessionPool - Task not serializable: java.io.NotSerializableException: java.util.concurrent.ScheduledThreadPoolExecutor
          // each partition has it's own graphDBClient instance/connection, since it's on different executors and threads
          // each graphDBClient instance is tied to otne connection, ie. Nebula session
          // Nebula sessions are not thread-safe, each partition has it's own connection.
          // SessionPool implementation however is threadsafe
          // there is always a worry that we're creating and destroying too many objects, the partition size plays a factor here yet to see if it's an issue

          val graphDBClient = new NebulaGraphDBClient(gbmlConfigWrapper = gbmlConfigWrapper)
          graphDBClient.connect()
          val kHopSampler = new NebulaHeteroKHopSampler(
            graphDBClient = graphDBClient,
            graphMetadataPbWrapper = graphMetadataPbWrapper,
          )

          val rnns: Iterator[RootedNodeNeighborhood] = nodeIds.map(rootNodeId => {
            kHopSampler.getKHopSubgraphForRootNodes(
              rootNodeIds = List(rootNodeId),
              metapaths = metapaths,
              numNeighborsToSample = numNeighborsToSample,
            )(0)
          })
          val rnnsList =
            rnns.toList // force trigger map operation before session pool is terminated

          graphDBClient.terminate()
          rnnsList.iterator // return value
        })

      // cache the rnnDS for the hydrate step so we don't repeat querying graphDB for subgraphs
      val cachedRnnDS = rnnDS.persist(StorageLevel.MEMORY_ONLY)
      println(
        f"rnnDS count for node type ${condensedNodeType} [${nodeType}]: ${cachedRnnDS.count()}",
      ) // force trigger action to cache
      val rnnEndTime     = System.currentTimeMillis()
      val rnnTimeSeconds = (rnnEndTime - rnnStartTime) / 1000
      println(
        f"RNNs generation time for node type ${condensedNodeType} [${nodeType}]: ${rnnTimeSeconds / 60}m ${rnnTimeSeconds % 60}s",
      )
      nodeIdsDS.unpersist()

      // hydrate RNNs
      val cachedHydratedRnnDS = hydrateRnn(
        rnnDS = cachedRnnDS,
        hydratedNodeVIEW = hydratedNodeVIEW,
        hydratedEdgeVIEW = hydratedEdgeVIEW,
      )
      cachedRnnDS.unpersist() // unpersist the non-hydrated RNNs, since hydrated RNNs are cached
      hydratedRnnMap(nodeType) = cachedHydratedRnnDS

      val validatedRnnWithSchemaDS = TaskOutputValidator.validateRootedNodeNeighborhoodSamples(
        rootedNodeNeighborhoodSamples = cachedHydratedRnnDS,
        graphMetadataPbWrapper = graphMetadataPbWrapper,
      )
      writeDatasetToTfrecord[RootedNodeNeighborhood](
        inputDS = validatedRnnWithSchemaDS,
        gcsUri = nablpTaskOutputUri.nodeTypeToRandomNegativeTfrecordUriPrefix.getOrElse(
          nodeType,
          throw new Exception(
            "if you are seeing this, it means the node_type_to_random_negative_tfrecord_uri_prefix is missing the dstNodeType of the first supervision edge type, please check the frozen config",
          ),
        ),
      )
    }

    // Stage 1 supports 1 supervision edge type, therefore 1 anchor node type
    // TODO (yliu2-sc) can unpersist the rest
    val cachedHydratedRnnDS     = hydratedRnnMap(anchorNodeTypes(0))
    val cachedHydratedRnnDF     = cachedHydratedRnnDS.toDF()
    val cachedHydratedRnnDFVIEW = "cachedHydratedRnnDF" + uniqueTempViewSuffix
    cachedHydratedRnnDF.createOrReplaceTempView(cachedHydratedRnnDFVIEW)
    println(
      "cachedHydratedRnnDS partitions: " + cachedHydratedRnnDS.rdd.getNumPartitions,
    )
    val metapaths: Seq[Metapath] = getMetapaths(
      nodeTypes = anchorNodeTypes,
      graphMetadataPbWrapper = graphMetadataPbWrapper,
      numHops = numHops,
    )

    // Downsample the training samples
    val rnnTrainingSamplesDS: Dataset[RootedNodeNeighborhood] =
      // proto field default is 0
      if (numTrainingSamples > 0) {
        // limit in spark returns a new dataset, need to repartition otherwise spark will do 1 partition
        cachedHydratedRnnDS
          .limit(
            numTrainingSamples,
          )
          .repartition(
            numPartitions,
          ) // this will be random every time its run since reading from tfrecord is random
      } else {
        cachedHydratedRnnDS
          .map(identity)
          .repartition(numPartitions) // make copy so we can unpersist
      }
    val cachedRnnTrainingSamplesDS: Dataset[RootedNodeNeighborhood] =
      rnnTrainingSamplesDS.persist(StorageLevel.MEMORY_ONLY)
    println(
      f"cachedRnnTrainingSamplesDS count: ${cachedRnnTrainingSamplesDS.count()}",
    ) // force trigger action to cache
    cachedHydratedRnnDS
      .unpersist() // only need the downsampled version for the next steps, unpersist full hydratedDS

    val cachedRnnTrainingSamplesDF     = cachedRnnTrainingSamplesDS.toDF()
    val cachedRnnTrainingSamplesDFVIEW = "cachedRnnTrainingSamplesDF" + uniqueTempViewSuffix
    cachedRnnTrainingSamplesDF.createOrReplaceTempView(cachedRnnTrainingSamplesDFVIEW)

    // Generate NABLPs
    println("\nGenerate NABLPs")
    val startTimeNABLP = System.currentTimeMillis()

    val rootNodesForTrainingSamplesDS: Dataset[Node] =
      cachedRnnTrainingSamplesDS.map(_.rootNode.get)
    println(
      "rootNodesForTrainingSamplesDS partitions: " + rootNodesForTrainingSamplesDS.rdd.getNumPartitions,
    )
    val repartitionedRootNodesForTrainingSamplesDS: Dataset[Node] =
      rootNodesForTrainingSamplesDS.repartition(numPartitions)

    // TODO (yliu2-sc) opportunity to abstract the querying/mapPartition component out
    // Query the positive neighborhoods with graphdb
    val posEdgeNeighborhoodsDS: Dataset[(Node, Seq[Edge], Graph)] =
      repartitionedRootNodesForTrainingSamplesDS
        .mapPartitions(rootNodes => {
          val graphDBClient = new NebulaGraphDBClient(gbmlConfigWrapper = gbmlConfigWrapper)
          graphDBClient.connect()
          val kHopSampler = new NebulaHeteroKHopSampler(
            graphDBClient = graphDBClient,
            graphMetadataPbWrapper = graphMetadataPbWrapper,
          )

          val posEdgeNeighborhoods: Iterator[(Node, Seq[Edge], Graph)] = rootNodes.map(rootNode => {
            val (posEdges, posNeighborhoods): (Seq[Edge], Seq[RootedNodeNeighborhood]) =
              kHopSampler.samplePositiveEdgeNeighborhoods(
                rootNodeId = rootNode.nodeId,
                numPositives = numPositiveSamples,
                condensed_edge_type = supervisionCondensedEdgeTypes(0),
                // reverse the metapath for bipartite graph, for full hetero graph we will need to traverse 1 hop deeper to get metapath
                metapaths = Seq(Metapath(Seq(metapaths(0).path(1), metapaths(0).path(0)))),
                numNeighborsToSample = numNeighborsToSample,
              )
            if (posEdges.nonEmpty) {
              val mergedPosNeighborhood =
                mergeRootedNodeNeighborhoods(posNeighborhoods).neighborhood.get
              (rootNode, posEdges, mergedPosNeighborhood)
            } else {
              (rootNode, Seq.empty, Graph())
            }
          })
          // NOTE: (yliu2) Isolated nodes without positive neighborhs will be filtered out here for NABLP samples
          //  if shouldIncludeIsolatedNodesInTraining is set to True, Isolated nodes will eventually be included
          //  in the last join after positive neighborhoods are hydrated
          val posEdgeNeighborhoodsList: List[(Node, Seq[Edge], Graph)] =
            posEdgeNeighborhoods
              .filter(_._2.nonEmpty)
              .toList // force trigger map operation before session pool is terminated

          graphDBClient.terminate()
          posEdgeNeighborhoodsList.iterator // return value
        })
    val cachedPosEdgeNeighborhoodsDS = posEdgeNeighborhoodsDS.persist(StorageLevel.MEMORY_ONLY)
    println(
      f"cachedPosEdgeNeighborhoodsDS count: ${cachedPosEdgeNeighborhoodsDS.count()}",
    ) // force trigger action to cache
    val cachedPosEdgeNeighborhoodsDF =
      cachedPosEdgeNeighborhoodsDS.toDF("root_node", "pos_edges", "pos_neighborhood")
    val endTimePosNeighborhood     = System.currentTimeMillis()
    val timeSecondsPosNeighborhood = (endTimePosNeighborhood - startTimeNABLP) / 1000
    println(
      f"Positive edge neighborhoods generation time: ${timeSecondsPosNeighborhood / 60}m ${timeSecondsPosNeighborhood % 60}s",
    )

    // contains root node, pos_neighborhood as neighborhood
    val cachedHydratedPosNeighborhoodDS: Dataset[RootedNodeNeighborhood] = hydrateRnn(
      rnnDS = cachedPosEdgeNeighborhoodsDF
        .select("root_node", "pos_neighborhood")
        .withColumnRenamed("pos_neighborhood", "neighborhood")
        .as[RootedNodeNeighborhood],
      hydratedNodeVIEW = hydratedNodeVIEW,
      hydratedEdgeVIEW = hydratedEdgeVIEW,
      shouldHydrateRootNode = false,
    )
    val cachedHydratedPosNeighborhoodDF   = cachedHydratedPosNeighborhoodDS.toDF()
    val cachedHydratedPosNeighborhoodVIEW = "cachedHydratedPosNeighborhoodDF" + uniqueTempViewSuffix
    cachedHydratedPosNeighborhoodDF.createOrReplaceTempView(cachedHydratedPosNeighborhoodVIEW)

    val posEdgesDF   = cachedPosEdgeNeighborhoodsDF.select("root_node", "pos_edges")
    val posEdgesVIEW = "posEdgesDF" + uniqueTempViewSuffix
    posEdgesDF.createOrReplaceTempView(posEdgesVIEW)
    val hydratedPosEdgesDF = spark.sql(
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
          ) AS pos_edges
        FROM (
          SELECT root_node, explode(pos_edges) as edge
          FROM ${posEdgesVIEW}
        )
        LEFT JOIN ${hydratedEdgeVIEW} as edgeFeatures
        ON edge.src_node_id = edgeFeatures._from AND edge.dst_node_id = edgeFeatures._to AND edge.condensed_edge_type = edgeFeatures._condensed_edge_type
        GROUP BY root_node
        """,
    )
    val hydratedPosEdgesVIEW = "hydratedPosEdgesDF" + uniqueTempViewSuffix
    hydratedPosEdgesDF.createOrReplaceTempView(hydratedPosEdgesVIEW)

    // NOTE (yliu2) we use LEFT JOIN here to include isolated nodes
    val nablpPosNeighborhoodsDF = if (shouldIncludeIsolatedNodesInTraining) {
      spark.sql(
        f"""
        SELECT
          rnnSamples.root_node AS root_node,
          posEdges.pos_edges AS pos_edges,
          rnnSamples.neighborhood AS root_node_neigborhood,
          posNeighborhoods.neighborhood AS pos_neighborhood
        FROM ${cachedRnnTrainingSamplesDFVIEW} AS rnnSamples
        LEFT JOIN ${hydratedPosEdgesVIEW} AS posEdges
        ON rnnSamples.root_node.node_id = posEdges.root_node.node_id
          AND rnnSamples.root_node.condensed_node_type = posEdges.root_node.condensed_node_type
        LEFT JOIN ${cachedHydratedPosNeighborhoodVIEW} AS posNeighborhoods
        ON posEdges.root_node.node_id = posNeighborhoods.root_node.node_id
          AND posEdges.root_node.condensed_node_type = posNeighborhoods.root_node.condensed_node_type
      """,
      )
    } else {
      spark.sql(
        f"""
        SELECT
          rnnSamples.root_node AS root_node,
          posEdges.pos_edges AS pos_edges,
          rnnSamples.neighborhood AS root_node_neigborhood,
          posNeighborhoods.neighborhood AS pos_neighborhood
        FROM ${cachedRnnTrainingSamplesDFVIEW} AS rnnSamples
        JOIN ${hydratedPosEdgesVIEW} AS posEdges
        ON rnnSamples.root_node.node_id = posEdges.root_node.node_id
          AND rnnSamples.root_node.condensed_node_type = posEdges.root_node.condensed_node_type
        JOIN ${cachedHydratedPosNeighborhoodVIEW} AS posNeighborhoods
        ON posEdges.root_node.node_id = posNeighborhoods.root_node.node_id
          AND posEdges.root_node.condensed_node_type = posNeighborhoods.root_node.condensed_node_type
      """,
      )
    }

    //    // Avoid using UDF since in spark SQL proto types aren't supported and will make UDF unncesessarily complex
    val nablpPosNeighborhoodsDS = nablpPosNeighborhoodsDF.as[(Node, Seq[Edge], Graph, Graph)]
    val nablpDS: Dataset[NodeAnchorBasedLinkPredictionSample] = nablpPosNeighborhoodsDS.map {
      case (rootNode, posEdges, rootNodeNeighborhood, posNeighborhood) =>
        val mergedNeighborhood = mergeGraphs(
          Seq(rootNodeNeighborhood, Option(posNeighborhood).getOrElse(Graph())),
        )
        NodeAnchorBasedLinkPredictionSample(
          rootNode = Some(rootNode),
          posEdges = Option(posEdges).getOrElse(Seq.empty),
          neighborhood = Some(mergedNeighborhood),
        )
    }
    println(s"Count NABLP: ${nablpDS.count()}") // action will trigger recomputation
    val endTimeNABLP     = System.currentTimeMillis()
    val timeSecondsNABLP = (endTimeNABLP - startTimeNABLP) / 1000
    println(f"NABLPs generation time: ${timeSecondsNABLP / 60}m ${timeSecondsNABLP % 60}s")

    val validatedNablpWithSchemaDS = TaskOutputValidator.validateMainSamples(
      mainSampleDS = nablpDS,
      graphMetadataPbWrapper = graphMetadataPbWrapper,
    )
    writeDatasetToTfrecord[NodeAnchorBasedLinkPredictionSample](
      inputDS = validatedNablpWithSchemaDS,
      gcsUri = nablpTaskOutputUri.tfrecordUriPrefix,
    )
  }
}
