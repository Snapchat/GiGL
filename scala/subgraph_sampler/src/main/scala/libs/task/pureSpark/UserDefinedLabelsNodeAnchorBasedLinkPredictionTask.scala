package libs.task.pureSpark

import common.src.main.scala.types.EdgeUsageType
import common.types.pb_wrappers.GbmlConfigPbWrapper
import common.types.pb_wrappers.GraphMetadataPbWrapper
import common.utils.TFRecordIO.writeDatasetToTfrecord
import libs.task.SamplingStrategy.PermutationStrategy
import libs.task.TaskOutputValidator
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.{functions => F}
import scalapb.spark.Implicits._
import snapchat.research.gbml.training_samples_schema.NodeAnchorBasedLinkPredictionSample
import snapchat.research.gbml.training_samples_schema.RootedNodeNeighborhood

/**
  * This task is used to generate training samples for node anchor based link prediction when at least
  * either of user defined pos edges or neg edges are provided.
  * 1. We first add src only nodes from **user defined pos/neg** edges to the reference subgraph, for both dir/undir graphs.
  *  NOTE that if
  *     graph is directed: src only nodes from **main** edges are already included in cachedRnnVIEW by applying `createNeighborlessNodesSubgraph()`
  * 2. We then use step 1 to create pos subgraph which includes hydrated pos edges and pos dst node's neighborhood
  * 3. We then use step 1 to create neg subgraph which includes hydrated neg edges and neg dst node's neighborhood
  * 4. We join 2 and 3. (for cases that we do NOT have hard neg from user defined negatives we should use RNN as random negatives)
  * 5. Finally, create UserDefinedLabelsNodeAnchorBasedLinkPrediction Subgraph by joining 4 with 1
  * @spark: the idea for the order of ssteps above is to keep size of data at each stage balanced and avoid addressing several large joins but only focusing on scaling one/two large joins at the end.
  * @param gbmlConfigWrapper
  * @param graphMetadataPbWrapper
  */

// (yliu2) this task can potentially be merged with NodeAnchorBasedLinkPredictionTask in the future,
// given that the optimization strategies are not much different
class UserDefinedLabelsNodeAnchorBasedLinkPredictionTask(
  gbmlConfigWrapper: GbmlConfigPbWrapper,
  graphMetadataPbWrapper: GraphMetadataPbWrapper,
  isPosUserDefined: Boolean,
  isNegUserDefined: Boolean)
    extends NodeAnchorBasedLinkPredictionBaseTask(gbmlConfigWrapper) {
  // constants used for repartitioning DFs, as their size grows to avoid spills and OOM errors
  private val RepartitionFactorForRefSubgraphDf =
    10 // as numNeighborsToSample and feature dim increases, we need to use this (if we're not changing cluster size)
  private val RepartitionFactorForPositiveEdgeDf =
    if (gbmlConfigWrapper.subgraphSamplerConfigPb.numMaxTrainingSamplesToOutput > 0) {
      5
    } else {
      10
    } // used for repartitioning DFs for final join of subgraph and hydrated pos edges
  private val RepartitionFactorForUserDefinedEdgeDf =
    if (gbmlConfigWrapper.subgraphSamplerConfigPb.numMaxTrainingSamplesToOutput > 0) {
      5
    } else {
      10
    } // used for repartitioning DFs for final join of subgraph and hydrated user defined edges

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
    val sampleWithReplacement: Boolean =
      experimentalFlags.getOrElse("sample_with_replacement", "false").toBoolean
    println(f"Sample with replacement: ${sampleWithReplacement}")
    val numNeighborsToSample = gbmlConfigWrapper.subgraphSamplerConfigPb.numNeighborsToSample
    val includeIsolatedNodesInTrainingSamples =
      false // must be added to subgraphSamplerConfig in gbml config [As of now, SGS v2 with Spark, isolated nodes are NOT included in training samples ]
    // NOTE: we don't have downsampleNumberOfNodes applied to isolated nodes, so if in future users want to
    // include isolated nodes in training samples, all isolated nodes will be included in training samples.
    // If users want to downsample isolated nodes, we need to add a dedicated flag for num of isloated nodes in SGS configs
    val numPositiveSamples = gbmlConfigWrapper.subgraphSamplerConfigPb.numPositiveSamples
    val numUserDefinedPositiveSamples =
      gbmlConfigWrapper.subgraphSamplerConfigPb.numUserDefinedPositiveSamples
    val numUserDefinedNegativeSamples =
      gbmlConfigWrapper.subgraphSamplerConfigPb.numUserDefinedNegativeSamples

    println(f"User defined Positive edges are provided: ${isPosUserDefined}")
    println(f"User defined Negative edges are provided: ${isNegUserDefined}")

    if (isPosUserDefined) {
      println(f"numUserDefinedPositiveSamples: ${numUserDefinedPositiveSamples}")
      assert(
        numUserDefinedPositiveSamples > 0,
        "numUserDefinedPositiveSamples must be provided in subgraphSamplerConfig and > 0 if user defined pos edges are provided",
      )
    }
    if (isNegUserDefined) {
      println(f"numUserDefinedNegativeSamples: ${numUserDefinedNegativeSamples}")
      assert(
        numUserDefinedNegativeSamples > 0,
        "numUserDefinedNegativeSamples must be provided in subgraphSamplerConfig and > 0 if user defined neg edges are provided",
      )
    }

    val numTrainingSamples = gbmlConfigWrapper.subgraphSamplerConfigPb.numMaxTrainingSamplesToOutput

    val nablpTaskOutputUri =
      gbmlConfigWrapper.flattenedGraphMetadataPb.getNodeAnchorBasedLinkPredictionOutput

    val hydratedNodeVIEW: String =
      loadNodeDataframeIntoSparkSql(condensedNodeType = defaultCondensedNodeType)
    val hydratedEdgeVIEW: String =
      loadEdgeDataframeIntoSparkSql(
        condensedEdgeType = defaultCondensedEdgeType,
        edgeUsageType = EdgeUsageType.MAIN,
      )
    val unhydratedEdgeVIEW: String =
      loadUnhydratedEdgeDataframeIntoSparkSql(
        hydratedEdgeVIEW = hydratedEdgeVIEW,
        edgeUsageType = EdgeUsageType.MAIN,
      )

    val subgraphVIEW = createSubgraph(
      numNeighborsToSample = numNeighborsToSample,
      hydratedNodeVIEW = hydratedNodeVIEW,
      hydratedEdgeVIEW = hydratedEdgeVIEW,
      unhydratedEdgeVIEW = unhydratedEdgeVIEW,
      permutationStrategy = permutationStrategy,
      sampleWithReplacement = sampleWithReplacement,
    ) // does not include isolated nodes and we must cache this DF
    // @spark: caching the DF associated with subgraphVIEW is CRITICAL, substantially reduces job time. Cache is
    // triggered once the action (i.e. *write* RootedNodeNeighborhoodSample to GCS) is finished.
    //    val cachedSubgraphVIEW = applyCachingToDf(
    //      dfVIEW = subgraphVIEW,
    //      forceTriggerCaching = false,
    //      withRepartition = false,
    //    ) // set to true if we have both high dim edge/node feats, and we don't wanna scale out cluster nodes. Then use RepartitionFactorForRefSubgraphDf, and colNameForRepartition='_root_node'

    // no src only nodes if treated as undirected
    val rnnVIEW = createRootedNodeNeighborhoodSubgraph(
      hydratedNodeVIEW = hydratedNodeVIEW,
      unhydratedEdgeVIEW = unhydratedEdgeVIEW,
      subgraphVIEW = subgraphVIEW,// NOTE to pass **cached** subgraphVIEW
    )                             // has isolated nodes

    val cachedRnnVIEW = applyCachingToDf(
      dfVIEW = rnnVIEW,
      forceTriggerCaching = false,
      withRepartition = false,
    )

    val rnnWithSchemaVIEW = castToRootedNodeNeighborhoodProtoSchema(dfVIEW = cachedRnnVIEW)
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
      val (positiveSubgraphVIEW, referenceSubgraphWithPosSrcOnlyNodesVIEW) = createSampleSubgraph(
        isUserDefined = isPosUserDefined,
        numSamples = if (!isPosUserDefined) numPositiveSamples else numUserDefinedPositiveSamples,
        permutationStrategy = permutationStrategy,
        hydratedNodeVIEW = hydratedNodeVIEW,
        referenceSubgraphVIEW = cachedRnnVIEW,
        hydratedMainEdgeVIEW = hydratedEdgeVIEW,
        unhydratedMainEdgeVIEW = unhydratedEdgeVIEW,
        numTrainingSamples = numTrainingSamples,
        defaultCondensedEdgeType = defaultCondensedEdgeType,
        edgeUsageType = EdgeUsageType.POS,
      )

      val (negativeSubgraphVIEW, referenceSubgraphWithPosNegSrcOnlyNodesVIEW) =
        if (isNegUserDefined) {
          createSampleSubgraph(
            isUserDefined = isNegUserDefined,
            numSamples = numUserDefinedNegativeSamples,
            permutationStrategy = permutationStrategy,
            hydratedNodeVIEW = hydratedNodeVIEW,
            referenceSubgraphVIEW = referenceSubgraphWithPosSrcOnlyNodesVIEW,
            hydratedMainEdgeVIEW = hydratedEdgeVIEW,
            unhydratedMainEdgeVIEW = unhydratedEdgeVIEW,
            numTrainingSamples = 0,
            defaultCondensedEdgeType = defaultCondensedEdgeType,
            edgeUsageType = EdgeUsageType.NEG,
          )
        } else { (null, null) }

      val userDefNablpVIEW = if (negativeSubgraphVIEW != null) {
        createNodeAnchorBasedLinkPredictionUserDefinedLabelsSubgraphWithPosNeg(
          posSubgraphVIEW = positiveSubgraphVIEW,
          referenceSubgraphVIEW =
            referenceSubgraphWithPosNegSrcOnlyNodesVIEW, // we pass this VIEW, as long as for a root/anchor node, we have at least one pos edge, we can use it as a training sample
          negSubgraphVIEW = negativeSubgraphVIEW,
        )
      } else {
        createNodeAnchorBasedLinkPredictionUserDefinedLabelsSubgraphWithPosOnly(
          posSubgraphVIEW = positiveSubgraphVIEW,
          referenceSubgraphVIEW =
            referenceSubgraphWithPosSrcOnlyNodesVIEW, // we pass this VIEW, as long as for a root/anchor node, we have at least one pos edge, we can use it as a training sample
        )
      }

      println("start creating NodeAnchorBasedLinkPrediction samples")

      val nablpWithSchemaVIEW =
        castToTrainingSampleProtoSchema(dfVIEW = userDefNablpVIEW, hasHardNegs = isNegUserDefined)
      val nablpWithSchemaDF = spark.table(nablpWithSchemaVIEW)
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

  def addUserDefSrcOnlyNodesToRNNSubgraph(
    unhydratedMainEdgeVIEW: String,
    unhydratedUserDefEdgeVIEW: String,
    referenceSubgraphVIEW: String,
    hydratedNodeVIEW: String,
  ): String = {

    /**
      * needed for directed case. we will not be able to join in the pos/neg neighborhood
      *  in lookup neighborhood if src only nodes don't exist in RNN view, causing keyError
      *  down the line with the pos/neg node not included in the neighborhood after the joins.
      *  Adding src only nodes to RNN subgraph also ensures they're considered as root nodes
      */
    val allEdgesDF = spark.sql(f"""
       SELECT
         _src_node, _dst_node
       FROM
         ${unhydratedMainEdgeVIEW}
       UNION
       SELECT
         _src_node, _dst_node
       FROM
         ${unhydratedUserDefEdgeVIEW}
     """)
    val srcNodesDF         = allEdgesDF.select("_src_node").distinct()
    val dstNodeDF          = allEdgesDF.select("_dst_node").distinct()
    val srcOnlyNodeIdsDF   = srcNodesDF.exceptAll(dstNodeDF)
    val srcOnlyNodeIdsVIEW = "srcOnlyNodeIdsDF" + uniqueTempViewSuffix
    srcOnlyNodeIdsDF.createOrReplaceTempView(srcOnlyNodeIdsVIEW)
    val hydratedSrcOnlyNodesDF = spark.sql(f"""
       SELECT
         _node_id AS _root_node,
         NULL AS _neighbor_edges,
         ARRAY(STRUCT(_node_id,
               _condensed_node_type,
               _node_features AS _feature_values)) AS _neighbor_nodes,
         _node_features,
         _condensed_node_type
       FROM
         ${hydratedNodeVIEW}
       INNER JOIN
         ${srcOnlyNodeIdsVIEW}
       ON
         ${hydratedNodeVIEW}._node_id = ${srcOnlyNodeIdsVIEW}._src_node
     """)
    val hydratedSrcOnlyNodesVIEW = "hydratedSrcOnlyNodesDF" + uniqueTempViewSuffix
    hydratedSrcOnlyNodesDF.createOrReplaceTempView(hydratedSrcOnlyNodesVIEW)

    val rnnWithSrcOnlyNodesDF = spark.sql(f"""
       SELECT
         *
       FROM
         ${referenceSubgraphVIEW}
       UNION
       SELECT
         *
       FROM
         ${hydratedSrcOnlyNodesVIEW}
     """)

    val rnnWithSrcOnlyNodesVIEW = "rnnWithSrcOnlyNodesDF" + uniqueTempViewSuffix
    rnnWithSrcOnlyNodesDF.createOrReplaceTempView(rnnWithSrcOnlyNodesVIEW)
    rnnWithSrcOnlyNodesVIEW
  }

  def createSampleSubgraph(
    isUserDefined: Boolean,
    numSamples: Int,
    permutationStrategy: String,
    hydratedNodeVIEW: String,
    referenceSubgraphVIEW: String,
    hydratedMainEdgeVIEW: String,
    unhydratedMainEdgeVIEW: String,
    numTrainingSamples: Int,
    defaultCondensedEdgeType: Int,
    edgeUsageType: EdgeUsageType.EdgeUsageType,
  ): (String, String) = {

    /**
      * Creates subgraph (sample, join, hydrate) for positive / negative samples
      */
    val hydratedTaskBasedEdgeVIEW = if (isUserDefined) {
      println(s"Loading user defined ${edgeUsageType} edge")
      loadEdgeDataframeIntoSparkSql(
        condensedEdgeType = defaultCondensedEdgeType,
        edgeUsageType = edgeUsageType,
      )
    } else if (edgeUsageType == EdgeUsageType.POS) {
      println("Using main edges for sampling pos edges")
      hydratedMainEdgeVIEW
    } else {
      throw new RuntimeException(
        "Negative edges must be user defined for NodeAnchorBasedLinkPredictionUserDefinedLabelsTask",
      )
    }

    // TODO (yliu2-sc) maybe we can operate all on hydrated
    val unhydratedTaskBasedEdgeVIEW: String = if (isUserDefined) {
      loadUnhydratedEdgeDataframeIntoSparkSql(
        hydratedEdgeVIEW = hydratedTaskBasedEdgeVIEW,
        edgeUsageType = edgeUsageType,
      )
    } else if (edgeUsageType == EdgeUsageType.POS) {
      unhydratedMainEdgeVIEW
    } else {
      throw new RuntimeException(
        "Negative edges must be user defined for NodeAnchorBasedLinkPredictionUserDefinedLabelsTask",
      )
    }

    val sampledDstNodesVIEW = sampleDstNodesUniformly(
      numDstSamples = numSamples,
      unhydratedEdgeVIEW = unhydratedTaskBasedEdgeVIEW,
      permutationStrategy = permutationStrategy,
      numTrainingSamples = numTrainingSamples,
      edgeUsageType = edgeUsageType,
    )

    val referenceSubgraphWithSrcOnlyNodesVIEW =
      if (gbmlConfigWrapper.sharedConfigPb.isGraphDirected) {
        println(
          f"add src only nodes to isGraphDirected=${gbmlConfigWrapper.sharedConfigPb.isGraphDirected} graph",
        )
        addUserDefSrcOnlyNodesToRNNSubgraph(
          unhydratedMainEdgeVIEW = unhydratedMainEdgeVIEW,
          unhydratedUserDefEdgeVIEW = unhydratedTaskBasedEdgeVIEW,
          referenceSubgraphVIEW = referenceSubgraphVIEW,
          hydratedNodeVIEW = hydratedNodeVIEW,
        )
      } else {
        // NOTE: we don't add src only nodes to undirected graph, since they are already included in referenceSubgraphVIEW=cachedRnnVIEW by applying `createNeighborlessNodesSubgraph()`
        referenceSubgraphVIEW
      }

    val dstNodeNeighborhoodVIEW = lookupDstNodeNeighborhood(
      sampledDstNodesVIEW = sampledDstNodesVIEW,
      subgraphVIEW = referenceSubgraphWithSrcOnlyNodesVIEW,
      edgeUsageType = edgeUsageType,
    )

    val hydratedSampledTaskBasedEdgeVIEW = hydrateTaskBasedEdges(
      sampledTaskBasedEdgesVIEW = sampledDstNodesVIEW,
      hydratedEdgeVIEW = hydratedTaskBasedEdgeVIEW,
      edgeUsageType = edgeUsageType,
    )

    val taskBasedEdgeSubgraphDF = spark.sql(f"""
          SELECT
            _src_node AS _${edgeUsageType}_src_node, _${edgeUsageType}_neighbor_edges, _${edgeUsageType}_neighbor_nodes, _${edgeUsageType}_hydrated_edges
          FROM
            ${dstNodeNeighborhoodVIEW}
          INNER JOIN
            ${hydratedSampledTaskBasedEdgeVIEW}
          ON
            ${dstNodeNeighborhoodVIEW}._src_node = ${hydratedSampledTaskBasedEdgeVIEW}._root_node
          """)
    val taskBasedEdgeSubgraphVIEW =
      s"taskBasedEdge${edgeUsageType}SubgraphDF" + uniqueTempViewSuffix
    taskBasedEdgeSubgraphDF.createOrReplaceTempView(taskBasedEdgeSubgraphVIEW)

    (taskBasedEdgeSubgraphVIEW, referenceSubgraphWithSrcOnlyNodesVIEW)
  }

  def createNodeAnchorBasedLinkPredictionUserDefinedLabelsSubgraphWithPosNeg(
    posSubgraphVIEW: String,
    referenceSubgraphVIEW: String,
    negSubgraphVIEW: String,
  ): String = {

    /**
      * Creates NABLP samples for cases where user defined pos & neg exist
      */
    // join pos and neg subgraphs
    val posNegSubgraphDF = spark.sql(f"""
        SELECT
          _pos_src_node , _pos_neighbor_edges, _neg_neighbor_edges, _pos_hydrated_edges, _pos_neighbor_nodes, _neg_neighbor_nodes , _neg_hydrated_edges
        FROM
          ${posSubgraphVIEW}
        LEFT JOIN
          ${negSubgraphVIEW}
        ON
          ${posSubgraphVIEW}._pos_src_node = ${negSubgraphVIEW}._neg_src_node
      """)

    val collectedPosNegSubgraphDF =
      posNegSubgraphDF.select(
        F.col("_pos_src_node").alias("_root_node"),
        F.col("_pos_hydrated_edges"),
        F.col("_neg_hydrated_edges"),
        F.array_distinct(
          F.concat(
            F.col("_pos_neighbor_edges"),
            F.coalesce(F.col("_neg_neighbor_edges"), F.array()),
          ),
        ).alias("_udl_neighbor_edges"),
        F.array_distinct(
          F.concat(
            F.col("_pos_neighbor_nodes"),
            F.coalesce(F.col("_neg_neighbor_nodes"), F.array()),
          ),
        ).alias("_udl_neighbor_nodes"),
      )

    val collectedPosNegSubgraphVIEW = "collectedPosNegSubgraphDF" + uniqueTempViewSuffix
    collectedPosNegSubgraphDF.createOrReplaceTempView(collectedPosNegSubgraphVIEW)

    // Repartition
    val repartitionedReferenceSubgraphVIEW = applyRepartitionToDataFrameVIEW(
      dfVIEW = referenceSubgraphVIEW,
      colName = "_root_node",
      repartitionFactor = RepartitionFactorForUserDefinedEdgeDf,
    )
    val repartitionedCollectedPosNegSubgraphVIEW = applyRepartitionToDataFrameVIEW(
      dfVIEW = collectedPosNegSubgraphVIEW,
      colName = "_root_node",
      repartitionFactor = RepartitionFactorForUserDefinedEdgeDf,
    )

    // join result with rnn
    val udlSubgraphDF = spark.sql(f"""
        SELECT
          ${repartitionedReferenceSubgraphVIEW}._root_node,
          _neighbor_edges,
          _neighbor_nodes,
          _node_features,
          _condensed_node_type,
          _pos_hydrated_edges,
          _neg_hydrated_edges,
          _udl_neighbor_edges,
          _udl_neighbor_nodes
        FROM
          ${repartitionedReferenceSubgraphVIEW}
        INNER JOIN
          ${repartitionedCollectedPosNegSubgraphVIEW}
        ON ${repartitionedReferenceSubgraphVIEW}._root_node = ${repartitionedCollectedPosNegSubgraphVIEW}._root_node
      """)

    // We use dataframe API here for ease of handling scala and spark data types in terms of Null and empty arrays.
    val collectedUdlSubgraphDF = udlSubgraphDF.select(
      F.col("_root_node"),
      F.array_distinct(
        F.concat(
          F.coalesce(F.col("_neighbor_edges"), F.array()),
          F.coalesce(F.col("_udl_neighbor_edges"), F.array()),
        ),
      ).alias("_neighbor_edges"),
      F.array_distinct(F.concat(F.col("_neighbor_nodes"), F.col("_udl_neighbor_nodes")))
        .alias("_neighbor_nodes"),
      F.col("_node_features"),
      F.col("_condensed_node_type"),
      F.col("_pos_hydrated_edges"),
      F.col("_neg_hydrated_edges"),
    )
    val collectedUdlSubgraphVIEW = "collectedUdlSubgraphDF" + uniqueTempViewSuffix
    collectedUdlSubgraphDF.createOrReplaceTempView(collectedUdlSubgraphVIEW)

    collectedUdlSubgraphVIEW
  }

  def createNodeAnchorBasedLinkPredictionUserDefinedLabelsSubgraphWithPosOnly(
    posSubgraphVIEW: String,
    referenceSubgraphVIEW: String,
  ): String = {

    /**
      * Creates NABLP samples for cases where only user defined pos exist
      */
    // join pos subgraphs
    val posSubgraphDF = spark.sql(f"""
    SELECT
      _pos_src_node , _pos_neighbor_edges, _pos_hydrated_edges, _pos_neighbor_nodes
    FROM
      ${posSubgraphVIEW}
  """)

    val collectedPosSubgraphDF =
      posSubgraphDF.select(
        F.col("_pos_src_node").alias("_root_node"),
        F.col("_pos_hydrated_edges"),
        F.array_distinct(
          F.col("_pos_neighbor_edges"),
        ).alias("_udl_neighbor_edges"),
        F.array_distinct(
          F.col("_pos_neighbor_nodes"),
        ).alias("_udl_neighbor_nodes"),
      )

    val collectedPosSubgraphVIEW = "collectedPosSubgraphDF" + uniqueTempViewSuffix
    collectedPosSubgraphDF.createOrReplaceTempView(collectedPosSubgraphVIEW)

    // Repartition
    val repartitionedReferenceSubgraphVIEW = applyRepartitionToDataFrameVIEW(
      dfVIEW = referenceSubgraphVIEW,
      colName = "_root_node",
      repartitionFactor = RepartitionFactorForUserDefinedEdgeDf,
    )
    val repartitionedCollectedPosSubgraphVIEW = applyRepartitionToDataFrameVIEW(
      dfVIEW = collectedPosSubgraphVIEW,
      colName = "_root_node",
      repartitionFactor = RepartitionFactorForUserDefinedEdgeDf,
    )

    // join result with rnn
    val udlSubgraphDF = spark.sql(f"""
    SELECT
      ${repartitionedReferenceSubgraphVIEW}._root_node,
      _neighbor_edges,
      _neighbor_nodes,
      _node_features,
      _condensed_node_type,
      _pos_hydrated_edges,
      _udl_neighbor_edges,
      _udl_neighbor_nodes
    FROM
      ${repartitionedReferenceSubgraphVIEW}
    INNER JOIN
      ${repartitionedCollectedPosSubgraphVIEW}
    ON ${repartitionedReferenceSubgraphVIEW}._root_node = ${repartitionedCollectedPosSubgraphVIEW}._root_node
  """)

    // We use dataframe API here for ease of handling scala and spark data types in terms of Null and empty arrays.
    val collectedUdlSubgraphDF = udlSubgraphDF.select(
      F.col("_root_node"),
      F.array_distinct(
        F.concat(
          F.coalesce(F.col("_neighbor_edges"), F.array()),
          F.coalesce(F.col("_udl_neighbor_edges"), F.array()),
        ),
      ).alias("_neighbor_edges"),
      F.array_distinct(F.concat(F.col("_neighbor_nodes"), F.col("_udl_neighbor_nodes")))
        .alias("_neighbor_nodes"),
      F.col("_node_features"),
      F.col("_condensed_node_type"),
      F.col("_pos_hydrated_edges"),
    )
    val collectedUdlSubgraphVIEW = "collectedUdlSubgraphDF" + uniqueTempViewSuffix
    collectedUdlSubgraphDF.createOrReplaceTempView(collectedUdlSubgraphVIEW)

    collectedUdlSubgraphVIEW
  }

}
