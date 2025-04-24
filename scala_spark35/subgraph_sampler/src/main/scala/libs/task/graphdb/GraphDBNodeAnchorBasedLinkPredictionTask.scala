package libs.task.graphdb

import common.types.GraphTypes.CondensedNodeType
import common.types.GraphTypes.NodeId
import common.types.GraphTypes.NodeType
import common.types.SamplingOpDAG
import common.types.pb_wrappers.GbmlConfigPbWrapper
import common.types.pb_wrappers.GraphMetadataPbWrapper
import common.types.pb_wrappers.GraphPbWrappers
import common.types.pb_wrappers.ResourceConfigPbWrapper
import common.types.pb_wrappers.TaskMetadataPbWrapper
import common.utils.GiGLComponents
import common.utils.NumCores
import common.utils.TFRecordIO.writeDatasetToTfrecord
import libs.sampler.KHopSamplerServiceFactory
import libs.task.SubgraphSamplerTask
import libs.task.TaskOutputValidator
import libs.utils.SGSTask.hydrateRnn
import libs.utils.SGSTask.loadHydratedEdgeDataFrame
import libs.utils.SGSTask.loadHydratedNodeDataFrame
import libs.utils.SGSTask.loadNodeIds
import libs.utils.Spark.uniqueTempViewSuffix
import org.apache.spark.sql.Dataset
import scalapb.spark.Implicits._
import snapchat.research.gbml.graph_schema.Edge
import snapchat.research.gbml.graph_schema.EdgeType
import snapchat.research.gbml.graph_schema.Graph
import snapchat.research.gbml.graph_schema.Node
import snapchat.research.gbml.training_samples_schema.NodeAnchorBasedLinkPredictionSample
import snapchat.research.gbml.training_samples_schema.RootedNodeNeighborhood

class GraphDBNodeAnchorBasedLinkPredictionTask(
  gbmlConfigWrapper: GbmlConfigPbWrapper,
  giglResourceConfigWrapper: ResourceConfigPbWrapper)
    extends SubgraphSamplerTask(gbmlConfigWrapper) {

  val graphMetadataPbWrapper = gbmlConfigWrapper.graphMetadataPbWrapper
  val numVCPUs: Int = NumCores.getNumVCPUs(
    giglResourceConfigWrapper = giglResourceConfigWrapper,
    component = GiGLComponents.SubgraphSampler,
  )
  // Internal experiments have shown that 5 partitions per vCPU achieves a good balance for
  // most use cases experimented with. This is definitely not a one-size-fits-all and neither
  // is it heavily optimized for any particular use case.
  val numPartitions: Int = numVCPUs * 5

  // TODO (svij) these need to be piped in
  // val numNodeIdsToTest: Int = 10

  def getRootedNodeNeighborhood(
    nodeIdsDS: Dataset[NodeId],
    condensedNodeType: CondensedNodeType,
    samplingOpDag: SamplingOpDAG,
  ): Dataset[RootedNodeNeighborhood] = {
    println(
      s"Generating RNNs for node type: ${condensedNodeType} with samplingOpDag: ${samplingOpDag}",
    )
    val samplerService = KHopSamplerServiceFactory.createKHopServiceSampler(
      gbmlConfigWrapper = gbmlConfigWrapper,
      giglResourceConfigWrapper = giglResourceConfigWrapper,
    )
    val rnnDS: Dataset[RootedNodeNeighborhood] = nodeIdsDS.mapPartitions(nodeIds => {
      samplerService.setup()
      val rnns: Iterator[RootedNodeNeighborhood] = nodeIds.map(rootNodeId => {
        samplerService.getKHopSubgraphForRootNode(
          rootNode = Node(
            nodeId = rootNodeId,
            condensedNodeType = Some(condensedNodeType),
          ),
          samplingOpDag = samplingOpDag,
        )
      })
      val rnnsList = rnns.toList // force trigger map operation before session pool is terminated
      samplerService.teardown()
      rnnsList.iterator
    })
    rnnDS
  }

  def getPositiveNeighborNodeNeighborhoods(
    rootNodeDS: Dataset[Node],
    positiveEdgeType: EdgeType,
    numPositives: Int,
    positiveNeighborSamplingOpDag: SamplingOpDAG,
  ): Dataset[(Node, Seq[Edge], Graph)] = {
    val samplerService = KHopSamplerServiceFactory.createKHopServiceSampler(
      gbmlConfigWrapper = gbmlConfigWrapper,
      giglResourceConfigWrapper = giglResourceConfigWrapper,
    )
    val posEdgeNeighborhoodsDS: Dataset[(Node, Seq[Edge], Graph)] = rootNodeDS
      .mapPartitions(rootNodes => {
        samplerService.setup()
        val posEdgeNeighborhoods: Iterator[(Node, Seq[Edge], Graph)] = rootNodes.map(rootNode => {
          val (posEdges, posNeighborhoods): (Seq[Edge], Seq[Graph]) =
            samplerService.samplePositiveEdgeNeighborhoods(
              rootNode = rootNode,
              edgeType = positiveEdgeType,
              numPositives = numPositives,
              samplingOpDag = positiveNeighborSamplingOpDag,
            )
          if (posEdges.nonEmpty) {
            val mergedPosNeighborhood = GraphPbWrappers.mergeGraphs(posNeighborhoods)
            (rootNode, posEdges, mergedPosNeighborhood)
          } else {
            (rootNode, Seq.empty, Graph())
          }
        })
        // NOTE: (yliu2) Isolated nodes without positive neighbors will be filtered out here for NABLP samples
        //  if shouldIncludeIsolatedNodesInTraining is set to True, Isolated nodes will eventually be included
        //  in the last join after positive neighborhoods are hydrated
        val posEdgeNeighborhoodsList: List[(Node, Seq[Edge], Graph)] =
          posEdgeNeighborhoods
            .filter(_._2.nonEmpty)
            .toList // force trigger map operation before session pool is terminated

        samplerService.teardown()
        posEdgeNeighborhoodsList.iterator // return value
      })
    posEdgeNeighborhoodsDS
  }

  def run(): Unit = {
    // TODO (yliu2) graphDBClient class should be read as parameter
    val graphMetadataPbWrapper: GraphMetadataPbWrapper = GraphMetadataPbWrapper(
      gbmlConfigWrapper.graphMetadataPb,
    )
    val subgraphSamplerConfigPb     = gbmlConfigWrapper.subgraphSamplerConfigPb
    val preprocessedMetadataWrapper = gbmlConfigWrapper.preprocessedMetadataWrapper

    println(
      s"shouldIncludeIsolatedNodesInTraining: ${gbmlConfigWrapper.sharedConfigPb.shouldIncludeIsolatedNodesInTraining}",
    )

    val nablpTaskOutputUri =
      gbmlConfigWrapper.flattenedGraphMetadataPb.getNodeAnchorBasedLinkPredictionOutput

    val taskMetadataPbWrapper = TaskMetadataPbWrapper(gbmlConfigWrapper.taskMetadataPb)
    val nablpTaskMetadata =
      gbmlConfigWrapper.taskMetadataPb.getNodeAnchorBasedLinkPredictionTaskMetadata

    // In phase 1 we will only support 1 supervision edge type, and 1 anchor node type
    val distinctAnchorTargetNodeTypes: Seq[NodeType] =
      (taskMetadataPbWrapper.anchorNodeTypes ++ taskMetadataPbWrapper.targetNodeTypes).distinct
    println(s"Supervision edge types: ${nablpTaskMetadata.supervisionEdgeTypes}")
    println(s"Anchor node types: ${taskMetadataPbWrapper.anchorNodeTypes}")
    println(s"Target node types: ${taskMetadataPbWrapper.targetNodeTypes}")
    println(s"Distinct anchor and target node types: ${distinctAnchorTargetNodeTypes}")

    println(s"numPositiveSamples: ${subgraphSamplerConfigPb.numPositiveSamples}")
    println(s"numTrainingSamples: ${subgraphSamplerConfigPb.numMaxTrainingSamplesToOutput}")

    val supervisionCondensedEdgeTypes =
      nablpTaskMetadata.supervisionEdgeTypes.map(edgeType =>
        graphMetadataPbWrapper.edgeTypeToCondensedEdgeTypeMap.get(edgeType).get,
      )

    val loadDataStartTime = System.currentTimeMillis()
    val hydratedNodeVIEW: String =
      loadHydratedNodeDataFrame(
        condensedNodeTypes = graphMetadataPbWrapper.condensedNodeTypes,
        gbmlConfigWrapper = gbmlConfigWrapper,
      )

    val hydratedEdgeVIEW: String =
      loadHydratedEdgeDataFrame(
        condensedEdgeTypes = graphMetadataPbWrapper.condensedEdgeTypes,
        gbmlConfigWrapper = gbmlConfigWrapper,
      )

    KHopSamplerServiceFactory.initializeLocalDbIfNeeded(
      gbmlConfigWrapper = gbmlConfigWrapper,
      hydratedEdgeVIEW = hydratedEdgeVIEW,
    )

    val loadDataEndTime     = System.currentTimeMillis()
    val loadDataTimeSeconds = (loadDataEndTime - loadDataStartTime) / 1000
    println(f"Load data time: ${loadDataTimeSeconds / 60}m ${loadDataTimeSeconds % 60}s")

    // Generate RNNs for each node type
    val cachedHydratedRnnMap =
      scala.collection.mutable.Map[NodeType, Dataset[RootedNodeNeighborhood]]()

    println(
      "Will generate RNNs for DistinctAnchorTargetNodeTypes: " + distinctAnchorTargetNodeTypes,
    )
    val nodeTypeToSamplingOpDagMap: Map[NodeType, SamplingOpDAG] =
      gbmlConfigWrapper.subgraphSamplingStrategyPbWrapper.getNodeTypeToSamplingOpDagMap

    for (nodeType <- distinctAnchorTargetNodeTypes) {
      println("Generate RNNs for node type: " + nodeType)
      val condensedNodeType =
        graphMetadataPbWrapper.nodeTypeToCondensedNodeTypeMap.get(nodeType).get
      val nodeIdsDS: Dataset[NodeId] = loadNodeIds(
        hydratedNodeVIEW = hydratedNodeVIEW,
        condensedNodeType = condensedNodeType,
        numPartitions = Some(numPartitions),
        // numNodeIds = Some(numNodeIdsToTest), // for testing to limit the number of nodes
        shouldCacheResult = true,
        storageLevel = storageLevel,
      )
      val numNodeIds = nodeIdsDS.count() // force trigger action to cache
      println(
        f"nodeIdsDS count for node type ${condensedNodeType} [${nodeType}]: ${numNodeIds}",
      )

      // TODO (svij) We should probably add a stopwatch or timer utility
      val rnnStartTime  = System.currentTimeMillis()
      val samplingOpDag = nodeTypeToSamplingOpDagMap(nodeType)
      val allSamplingOpDagEdgeTypes =
        samplingOpDag.rootSamplingOpNodes.map(_.samplingOp.edgeType).distinct
      // if any of the edge types have edge features, we will hydrate the edges
      val hasAnyEdgeFeatures = allSamplingOpDagEdgeTypes.exists(edgeType =>
        preprocessedMetadataWrapper.hasEdgeFeatures(
          graphMetadataPbWrapper.edgeTypeToCondensedEdgeTypeMap(edgeType.get),
        ),
      )
      val rnnDS: Dataset[RootedNodeNeighborhood] = getRootedNodeNeighborhood(
        nodeIdsDS = nodeIdsDS,
        condensedNodeType = condensedNodeType,
        samplingOpDag = samplingOpDag,
      )
      // cache the rnnDS for the hydrate step so we don't repeat querying graphDB for subgraphs
      val cachedRnnDS = rnnDS.persist(storageLevel)
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
        shouldCacheResult = true,
        storageLevel = storageLevel,
        shouldHydrateEdges = hasAnyEdgeFeatures,
      )
      cachedRnnDS.unpersist() // unpersist the non-hydrated RNNs, since hydrated RNNs are cached
      cachedHydratedRnnMap(nodeType) = cachedHydratedRnnDS

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

    // Stage 1 HGH supports 1 supervision edge type, therefore 1 anchor node type
    val positiveEdgeType = nablpTaskMetadata.supervisionEdgeTypes(0)
    val condensedPositiveEdgeType =
      graphMetadataPbWrapper.edgeTypeToCondensedEdgeTypeMap(positiveEdgeType)
    val trainingSampleRootNodeType     = positiveEdgeType.srcNodeType
    val trainingSamplePositiveNodeType = positiveEdgeType.dstNodeType
    val trainingSamplePositiveCondensedNodeType =
      graphMetadataPbWrapper.nodeTypeToCondensedNodeTypeMap.get(positiveEdgeType.srcNodeType).get
    val positiveNodeMessagePasingSamplingOpDag = nodeTypeToSamplingOpDagMap(
      trainingSamplePositiveNodeType,
    )
    val allPositiveSamplingOpDagEdgeTypes =
      positiveNodeMessagePasingSamplingOpDag.rootSamplingOpNodes.map(_.samplingOp.edgeType).distinct
    // if any of the positive edge types have edge features, we will hydrate the edges
    val positiveNeighborhoodHasAnyEdgeFeatures =
      allPositiveSamplingOpDagEdgeTypes.exists(edgeType =>
        preprocessedMetadataWrapper.hasEdgeFeatures(
          graphMetadataPbWrapper.edgeTypeToCondensedEdgeTypeMap(edgeType.get),
        ),
      )

    // This should be cached already so count should be quick
    // TODO: (svij) Ensure that this is the case ^
    val countRnnForTrainingSampleRootNode = cachedHydratedRnnMap(trainingSampleRootNodeType).count()
    val numTrainingSamplesToGenerate =
      if (subgraphSamplerConfigPb.numMaxTrainingSamplesToOutput > 0) {
        subgraphSamplerConfigPb.numMaxTrainingSamplesToOutput
      } else {
        countRnnForTrainingSampleRootNode.toInt
      }
    println(
      s"Number of RNNs for generated for root node: ${countRnnForTrainingSampleRootNode}; " +
        s"for generating training samples we will downsample to ${numTrainingSamplesToGenerate}",
    )

    // We SAMPLE rather than using limit. Limit should only be used if absolutely needed - its an expensive operation!!!
    // Spark performs limit in two steps, first, it does LocalLimit and then does GlobalLimit.
    // Iteratively tasks are executed on paritions until the limit is satisfied. The # tasks are scheduled in
    // "scale up manner" using spark.sql.limit.scaleUpFactor". Finally, a GlobalLimit is performed where
    // only 1 task is run - inducing a shuffle. We have seen this take hours when doing LIMIT(100Ms of rows).
    // https://towardsdatascience.com/stop-using-the-limit-clause-wrong-with-spark-646e328774f5
    val sampleFraction = numTrainingSamplesToGenerate.toDouble / countRnnForTrainingSampleRootNode
    println("Will sample to following fraction: " + sampleFraction)
    val sampledRnnTrainingSamplesDS: Dataset[RootedNodeNeighborhood] =
      if (numTrainingSamplesToGenerate < countRnnForTrainingSampleRootNode) {
        cachedHydratedRnnMap(trainingSampleRootNodeType)
          .sample(sampleFraction)
      } else {
        cachedHydratedRnnMap(trainingSampleRootNodeType)
      }

    val cachedRnnTrainingSamplesDS: Dataset[RootedNodeNeighborhood] =
      sampledRnnTrainingSamplesDS
        .repartition(numPartitions)
        .persist(storageLevel)

    val countCachedRnnTrainingSamplesDS =
      cachedRnnTrainingSamplesDS.count() // force trigger action to cache
    println(
      f"cachedRnnTrainingSamplesDS count: ${countCachedRnnTrainingSamplesDS}",
    )

    // We clear up memory by unpersist unneeded cached RNNs
    for (cachedRnnDS <- cachedHydratedRnnMap.values) {
      cachedRnnDS.unpersist()
    }

    val cachedRnnTrainingSamplesDFVIEW = "cachedRnnTrainingSamplesDF" + uniqueTempViewSuffix
    cachedRnnTrainingSamplesDS.createOrReplaceTempView(cachedRnnTrainingSamplesDFVIEW)

    // Generate NABLPs
    println("\nGenerate NABLP samples")
    val startTimeNABLP = System.currentTimeMillis()

    val repartitionedRootNodesForTrainingSamplesDS: Dataset[Node] =
      cachedRnnTrainingSamplesDS.map(_.rootNode.get).repartition(numPartitions)
    println(
      "repartitionedRootNodesForTrainingSamplesDS partitions: " + repartitionedRootNodesForTrainingSamplesDS.rdd.getNumPartitions + "expected: " + numPartitions,
    )

    // TODO (yliu2) opportunity to abstract the querying/mapPartition component out
    // Query the positive neighborhoods with graphdb

    println(
      f"Generate NABLP for supervision edge type: ${positiveEdgeType} anchor node type: ${positiveEdgeType}",
    )
    println(
      f"positiveNodeMessagePasingSamplingOpDag for node type ${trainingSamplePositiveNodeType}: ${positiveNodeMessagePasingSamplingOpDag}",
    )

    val posEdgeNeighborhoodsDS: Dataset[(Node, Seq[Edge], Graph)] =
      getPositiveNeighborNodeNeighborhoods(
        rootNodeDS = repartitionedRootNodesForTrainingSamplesDS,
        positiveEdgeType = positiveEdgeType,
        numPositives = subgraphSamplerConfigPb.numPositiveSamples,
        positiveNeighborSamplingOpDag = positiveNodeMessagePasingSamplingOpDag,
      )
    val cachedPosEdgeNeighborhoodsDS = posEdgeNeighborhoodsDS.persist(storageLevel)
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
      shouldCacheResult = true,
      storageLevel = storageLevel,
      shouldHydrateEdges = positiveNeighborhoodHasAnyEdgeFeatures,
    )
    println(
      "Created cachedHydratedPosNeighborhoodDS",
    )

    val cachedHydratedPosNeighborhoodDF   = cachedHydratedPosNeighborhoodDS.toDF()
    val cachedHydratedPosNeighborhoodVIEW = "cachedHydratedPosNeighborhoodDF" + uniqueTempViewSuffix
    cachedHydratedPosNeighborhoodDF.createOrReplaceTempView(cachedHydratedPosNeighborhoodVIEW)

    val posEdgesDF   = cachedPosEdgeNeighborhoodsDF.select("root_node", "pos_edges")
    val posEdgesVIEW = "posEdgesDF" + uniqueTempViewSuffix
    posEdgesDF.createOrReplaceTempView(posEdgesVIEW)

    val posEdgeHasEdgeFeatures =
      preprocessedMetadataWrapper.hasEdgeFeatures(condensedPositiveEdgeType)
    // No need to hydrate edge if there are no edge features
    val hydratedPosEdgesDF = if (posEdgeHasEdgeFeatures) {
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
    } else {
      println("Skipping edge hydration for positive edges")
      posEdgesDF
    }
    val hydratedPosEdgesVIEW = "hydratedPosEdgesDF" + uniqueTempViewSuffix
    hydratedPosEdgesDF.createOrReplaceTempView(hydratedPosEdgesVIEW)

    // NOTE (yliu2) we use LEFT JOIN here to include isolated nodes
    val nablpPosNeighborhoodsDF =
      if (gbmlConfigWrapper.sharedConfigPb.shouldIncludeIsolatedNodesInTraining) {
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
        val mergedNeighborhood = GraphPbWrappers.mergeGraphs(
          Seq(rootNodeNeighborhood, Option(posNeighborhood).getOrElse(Graph())),
        )
        NodeAnchorBasedLinkPredictionSample(
          rootNode = Some(rootNode),
          posEdges = Option(posEdges).getOrElse(Seq.empty),
          neighborhood = Some(mergedNeighborhood),
        )
    }
    // Forces peristing data as a precursor to writing data - prevents recomputation on failures w/ writing
    // output - which we have witnessed some.
    // Also, allows us to print the count for debugging without recomputing nablpDS for validation step.
    nablpDS.persist(storageLevel)
    println(s"Count NABLP: ${nablpDS.count()}") // action will force cache
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
