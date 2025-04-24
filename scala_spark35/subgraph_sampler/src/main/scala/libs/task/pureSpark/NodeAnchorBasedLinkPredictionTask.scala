package libs.task.pureSpark

import common.types.EdgeUsageType
import common.types.pb_wrappers.GbmlConfigPbWrapper
import common.types.pb_wrappers.GraphMetadataPbWrapper
import common.utils.TFRecordIO.writeDatasetToTfrecord
import libs.task.SamplingStrategy.PermutationStrategy
import libs.task.TaskOutputValidator
import org.apache.spark.sql.Dataset
import scalapb.spark.Implicits._
import snapchat.research.gbml.training_samples_schema.NodeAnchorBasedLinkPredictionSample
import snapchat.research.gbml.training_samples_schema.RootedNodeNeighborhood

class NodeAnchorBasedLinkPredictionTask(
  gbmlConfigWrapper: GbmlConfigPbWrapper,
  graphMetadataPbWrapper: GraphMetadataPbWrapper)
    extends NodeAnchorBasedLinkPredictionBaseTask(gbmlConfigWrapper) {
  // constants used for repartitioning DFs, as their size grows to avoid spills and OOM errors
  private val RepartitionFactorForRefSubgraphDf =
    10 // as numNeighborsToSample and feature dim increases, we need to use this (if we're not changing cluster size)
  private val RepartitionFactorForPositiveEdgeDf =
    if (gbmlConfigWrapper.subgraphSamplerConfigPb.numMaxTrainingSamplesToOutput > 0) {
      2
    } else {
      10
    } // used for repartitioning DFs for final join of subgraph and hydrated pos edges

  def run(): Unit = {
    // We get default condensed edge type for now; in the future this might be
    // done in a more iterative fashion to support hetero use cases.
    val defaultCondensedEdgeType: Int =
      gbmlConfigWrapper.preprocessedMetadataWrapper.preprocessedMetadataPb.condensedEdgeTypeToPreprocessedMetadata.keysIterator
        .next()
    val defaultCondensedNodeType: Int =
      gbmlConfigWrapper.preprocessedMetadataWrapper.preprocessedMetadataPb.condensedNodeTypeToPreprocessedMetadata.keysIterator
        .next()
    val experimentalFlags = gbmlConfigWrapper.subgraphSamplerConfigPb.experimentalFlags
    val permutationStrategy =
      experimentalFlags.getOrElse("permutation_strategy", PermutationStrategy.NonDeterministic)
    val numNeighborsToSample = gbmlConfigWrapper.subgraphSamplerConfigPb.numNeighborsToSample
    val includeIsolatedNodesInTrainingSamples =
      false // must be added to subgraphSamplerConfig in gbml config [As of now, SGS v2 with Spark, isolated nodes are NOT included in training samples ]
    // NOTE: we don't have downsampleNumberOfNodes applied to isolated nodes, so if in future users want to
    // include isolated nodes in training samples, all isolated nodes will be included in training samples.
    // If users want to downsample isolated nodes, we need to add a dedicated flag for num of isloated nodes in SGS configs
    val numPositiveSamples = gbmlConfigWrapper.subgraphSamplerConfigPb.numPositiveSamples
    val numTrainingSamples = gbmlConfigWrapper.subgraphSamplerConfigPb.numMaxTrainingSamplesToOutput

    val nablpTaskOutputUri =
      gbmlConfigWrapper.flattenedGraphMetadataPb.getNodeAnchorBasedLinkPredictionOutput

    val hydratedNodeVIEW: String =
      loadNodeDataframeIntoSparkSql(condensedNodeType = defaultCondensedNodeType)
    val hydratedEdgeVIEW: String =
      loadEdgeDataframeIntoSparkSql(condensedEdgeType = defaultCondensedEdgeType)
    val unhydratedEdgeVIEW: String =
      loadUnhydratedEdgeDataframeIntoSparkSql(hydratedEdgeVIEW = hydratedEdgeVIEW)

    val subgraphVIEW = createSubgraph(
      numNeighborsToSample = numNeighborsToSample,
      hydratedNodeVIEW = hydratedNodeVIEW,
      hydratedEdgeVIEW = hydratedEdgeVIEW,
      unhydratedEdgeVIEW = unhydratedEdgeVIEW,
      permutationStrategy = permutationStrategy,
    ) // does not include isolated nodes and we must cache this DF
    // @spark: caching the DF associated with subgraphVIEW is CRITICAL, substantially reduces job time. Cache is
    // triggered once the action (i.e. *write* RootedNodeNeighborhoodSample to GCS) is finished.
    val cachedSubgraphVIEW = applyCachingToDf(
      dfVIEW = subgraphVIEW,
      forceTriggerCaching = false,
      withRepartition = false,
    ) // set to true if we have both high dim edge/node feats, and we don't wanna scale out cluster nodes. Then use RepartitionFactorForRefSubgraphDf, and colNameForRepartition='_root_node'
    val rnnVIEW = createRootedNodeNeighborhoodSubgraph(
      hydratedNodeVIEW = hydratedNodeVIEW,
      unhydratedEdgeVIEW = unhydratedEdgeVIEW,
      subgraphVIEW = cachedSubgraphVIEW,// NOTE to pass **cached** subgraphVIEW
    )                                   // has isolated nodes

    val rnnWithSchemaVIEW = castToRootedNodeNeighborhoodProtoSchema(dfVIEW = rnnVIEW)
    val rnnWithSchemaDF   = spark.table(rnnWithSchemaVIEW)
    // @spark: NOTE 1: for all tasks, we must first write RootedNodeNeighborhood samples.
    // since write is the `action` that triggers caching on subgraphDF. The other sample types, such as
    // NodeAnchorBasedLinkPredictionSample, use cachedSubgraphDF to avoid repeating neighbor sampling/hydration.
    val rnnWithSchemaDS: Dataset[RootedNodeNeighborhood] =
      rnnWithSchemaDF.as[RootedNodeNeighborhood]
    // validate output before writing to storage
    println("start validating rooted node neighborhood samples")
    val validatedRnnWithSchemaDS = TaskOutputValidator.validateRootedNodeNeighborhoodSamples(
      rootedNodeNeighborhoodSamples = rnnWithSchemaDS,
      graphMetadataPbWrapper = graphMetadataPbWrapper,
    )
    // TODO (Tong): The following ~10 line of codes assumes homogeneous graphs, need to update to support heterogeneous graphs in HGS.
    //              They may work for stage 1 of HGS, but needs validation.
    val defaultTargetNodeType =
      gbmlConfigWrapper.taskMetadataPb.getNodeAnchorBasedLinkPredictionTaskMetadata.supervisionEdgeTypes.head.dstNodeType
    writeDatasetToTfrecord[RootedNodeNeighborhood](
      inputDS = validatedRnnWithSchemaDS,
      gcsUri = nablpTaskOutputUri.nodeTypeToRandomNegativeTfrecordUriPrefix.getOrElse(
        defaultTargetNodeType,
        throw new Exception(
          "If you are seeing this, it means the node_type_to_random_negative_tfrecord_uri_prefix is missing the dstNodeType of the first supervision edge type, please check the frozen config",
        ),
      ),
    )
    println("Finished writing RootedNodeNeighborhood samples")

    if (
      gbmlConfigWrapper.sharedConfigPb.shouldSkipTraining && gbmlConfigWrapper.sharedConfigPb.shouldSkipModelEvaluation
    ) {
      println(
        "shouldSkipTraining and shouldSkipModelEvaluation is set to true, thus skipping generating NodeAnchorBasedLinkPrediction samples",
      )
    } else {
      val nablpVIEW = createNodeAnchorBasedLinkPredictionSubgraph(
        numPositiveSamples = numPositiveSamples,
        includeIsolatedNodesInTrainingSamples = includeIsolatedNodesInTrainingSamples,
        unhydratedEdgeVIEW = unhydratedEdgeVIEW,
        hydratedEdgeVIEW = hydratedEdgeVIEW,
        hydratedNodeVIEW = hydratedNodeVIEW,
        subgraphVIEW = cachedSubgraphVIEW, // NOTE to pass **cached** subgraphVIEW
        permutationStrategy = permutationStrategy,
        numTrainingSamples = numTrainingSamples,
      )

      val nablpWithSchemaVIEW = castToTrainingSampleProtoSchema(dfVIEW = nablpVIEW)
      val nablpWithSchemaDF   = spark.table(nablpWithSchemaVIEW)
      val nablpWithSchemaDS: Dataset[NodeAnchorBasedLinkPredictionSample] =
        nablpWithSchemaDF.as[NodeAnchorBasedLinkPredictionSample]
      // validate output before writing to storage
      println("start validating main samples")
      val validatedNablpWithSchemaDS = TaskOutputValidator.validateMainSamples(
        mainSampleDS = nablpWithSchemaDS,
        graphMetadataPbWrapper = graphMetadataPbWrapper,
      )
      writeDatasetToTfrecord[NodeAnchorBasedLinkPredictionSample](
        inputDS = validatedNablpWithSchemaDS,
        gcsUri = nablpTaskOutputUri.tfrecordUriPrefix,
      )
    }
  }

  def createNodeAnchorBasedLinkPredictionSubgraph(
    numPositiveSamples: Int,
    includeIsolatedNodesInTrainingSamples: Boolean,
    unhydratedEdgeVIEW: String,
    hydratedEdgeVIEW: String,
    hydratedNodeVIEW: String,
    subgraphVIEW: String,
    permutationStrategy: String,
    numTrainingSamples: Int,
  ): String = {

    /**
      * 1. samples pos dst nodes
      * 2. creates neighborhood for sampled pos dst nodes, by looking up neighborhood from subgraph DF
      * 3. hydrated pos edges with feature values/types
      * 4. adds pos neighborhoods to subgraph DF [@spark: This is the stage that requires optimization
      * since as we join the size of data significantly increases and lead to spills and even OOMs.]
      */

    val sampledPosVIEW = sampleDstNodesUniformly(
      numDstSamples = numPositiveSamples,
      unhydratedEdgeVIEW = unhydratedEdgeVIEW,
      permutationStrategy = permutationStrategy,
      numTrainingSamples = numTrainingSamples,
      edgeUsageType = EdgeUsageType.POS,
    ) // any time we use a function that samples, we need to use a different seed to maintain sampling diversity
    // here we already have sampled 1-hop with seed*1, 2-hop with seed*2, now we use seed*3
    val posNodeNeighborhoodVIEW = lookupDstNodeNeighborhood(
      sampledDstNodesVIEW = sampledPosVIEW,
      subgraphVIEW = subgraphVIEW,
      edgeUsageType = EdgeUsageType.POS,
    )

    val hydratedPosEdgeVIEW =
      hydrateTaskBasedEdges(
        sampledTaskBasedEdgesVIEW = sampledPosVIEW,
        hydratedEdgeVIEW = hydratedEdgeVIEW,
        edgeUsageType = EdgeUsageType.POS,
      )
    // @spark: If this stage has spills, adjust RepartitionFactorForRefSubgraphDf
    val subgraphWithPosNeighborhoodDF = spark.sql(f"""
        SELECT
          *
        FROM
          ${subgraphVIEW}
        INNER JOIN
          ${posNodeNeighborhoodVIEW}
        ON
          ${subgraphVIEW}._root_node = ${posNodeNeighborhoodVIEW}._src_node
        """)

    val subgraphWithPosNeighborhoodVIEW = "subgraphWithPosNeighborhoodDF" + uniqueTempViewSuffix
    subgraphWithPosNeighborhoodDF.createOrReplaceTempView(subgraphWithPosNeighborhoodVIEW)

    val collectedSubgraphWithPosNeighborhoodDF = spark.sql(f"""
        SELECT
          _root_node,
          array_distinct(CONCAT(_neighbor_edges, _pos_neighbor_edges)) AS _neighbor_edges,
          array_distinct(CONCAT(_neighbor_nodes, _pos_neighbor_nodes)) AS _neighbor_nodes,
          _node_features,
          _condensed_node_type
        FROM
          ${subgraphWithPosNeighborhoodVIEW}
        """)

    val collectedSubgraphWithPosNeighborhoodVIEW =
      "collectedSubgraphWithPosNeighborhoodDF" + uniqueTempViewSuffix
    collectedSubgraphWithPosNeighborhoodDF.createOrReplaceTempView(
      collectedSubgraphWithPosNeighborhoodVIEW,
    )

    val validRootNodesForTrainingSubgraphVIEW =
      if (!gbmlConfigWrapper.sharedConfigPb.isGraphDirected) {
        collectedSubgraphWithPosNeighborhoodVIEW
      } else {
        // for directed graphs, any node that has an out edge is a valid training sample
        // so we add srcOnlyNodes to collectedSubgraphWithPosNeighborhoodVIEW too.
        val srcOnlyNodesNeighborhoodVIEW = formNeighborhoodForSrcOnlyNodes(
          sampledPosVIEW = sampledPosVIEW,
          unhydratedEdgeVIEW = unhydratedEdgeVIEW,
          posNodeNeighborhoodVIEW = posNodeNeighborhoodVIEW,
          hydratedNodeVIEW = hydratedNodeVIEW,
        )
        val allRootNodesWithPosEdgeDF = spark.sql(f"""
            SELECT
              *
            FROM
              ${collectedSubgraphWithPosNeighborhoodVIEW} UNION
            SELECT
              _root_node,
              _neighbor_edges,
              _neighbor_nodes,
              _node_features,
              _condensed_node_type
            FROM ${srcOnlyNodesNeighborhoodVIEW}
          """)
        val allRootNodesWithPosEdgeVIEW = "allRootNodesWithPosEdgeDF" + uniqueTempViewSuffix
        allRootNodesWithPosEdgeDF.createOrReplaceTempView(allRootNodesWithPosEdgeVIEW)
        allRootNodesWithPosEdgeVIEW
      }

    // repartition pos_neighbor and pos_edge dfs
    val repartitionedSubgraphWithPosNeighborhoodDF = applyRepartitionToDataFrame(
      df = spark.table(validRootNodesForTrainingSubgraphVIEW), // convert VIEW to DF
      colName = "_root_node",
      repartitionFactor = RepartitionFactorForPositiveEdgeDf,
    )

    val hydratedPosEdgeDF = spark.table(hydratedPosEdgeVIEW)
    val repartitionedHydratedPosEdgeDF = applyRepartitionToDataFrame(
      df = hydratedPosEdgeDF,
      colName = "_root_node",
      repartitionFactor = RepartitionFactorForPositiveEdgeDf,
    )

    val repartitionedSubgraphWithPosNeighborhoodVIEW =
      "repartitionedSubgraphWithPosNeighborhoodDF" + uniqueTempViewSuffix
    repartitionedSubgraphWithPosNeighborhoodDF.createOrReplaceTempView(
      repartitionedSubgraphWithPosNeighborhoodVIEW,
    )

    val repartitionedHydratedPosEdgeVIEW = "repartitionedHydratedPosEdgeDF" + uniqueTempViewSuffix
    repartitionedHydratedPosEdgeDF.createOrReplaceTempView(repartitionedHydratedPosEdgeVIEW)
    // @spark: This is the stage that needs most optimization efforts. For now we use repartition as the strategy.
    // however, if the data size grows significantly we need to scale out cluster workers.
    val subgraphWithPosNeighborsAndEdgesDF = spark
      .sql(f"""
          SELECT
            ${repartitionedSubgraphWithPosNeighborhoodVIEW}._root_node,
            _neighbor_edges,
            _neighbor_nodes,
            _pos_hydrated_edges,
            _node_features,
            _condensed_node_type
          FROM
            ${repartitionedSubgraphWithPosNeighborhoodVIEW}
          INNER JOIN
            ${repartitionedHydratedPosEdgeVIEW}
          ON
            ${repartitionedSubgraphWithPosNeighborhoodVIEW}._root_node = ${repartitionedHydratedPosEdgeVIEW}._root_node
          """)
      .dropDuplicates(
        "_root_node",
      ) // dropping duplicate if any, as we may reduce the number of root nodes in downsampleNumberOfRootNodes. There should never be a duplicate unless downsampleNumberOfRootNodes is non deterministic
    val nablpSubgraphVIEW = if (!includeIsolatedNodesInTrainingSamples) {
      val subgraphWithPosNeighborsAndEdgesVIEW =
        "subgraphWithPosNeighborsAndEdgesDF" + uniqueTempViewSuffix
      subgraphWithPosNeighborsAndEdgesDF.createOrReplaceTempView(
        subgraphWithPosNeighborsAndEdgesVIEW,
      )
      subgraphWithPosNeighborsAndEdgesVIEW
    } else {
      val subgraphWithPosNeighborsAndEdgesVIEW =
        "subgraphWithPosNeighborsAndEdgesDF" + uniqueTempViewSuffix
      subgraphWithPosNeighborsAndEdgesDF.createOrReplaceTempView(
        subgraphWithPosNeighborsAndEdgesVIEW,
      )

      val subgraphWithPosNeighborsEdgesAndIsolatedNodesVIEW = appendIsolatedNodesToTrainingSamples(
        trainingSubgraphVIEW = subgraphWithPosNeighborsAndEdgesVIEW,
        hydratedNodeVIEW = hydratedNodeVIEW,
        unhydratedEdgeVIEW = unhydratedEdgeVIEW,
      )
      subgraphWithPosNeighborsEdgesAndIsolatedNodesVIEW
    }
    nablpSubgraphVIEW
  }

}
