package libs.task.pureSpark

import common.src.main.scala.types.EdgeUsageType
import common.types.pb_wrappers.GbmlConfigPbWrapper
import libs.task.SamplingStrategy.PermutationStrategy
import libs.task.SamplingStrategy.hashBasedUniformPermutation
import libs.task.SamplingStrategy.shuffleBasedUniformPermutation
import org.apache.spark.sql.{functions => F}

abstract class NodeAnchorBasedLinkPredictionBaseTask(
  gbmlConfigWrapper: GbmlConfigPbWrapper)
    extends SGSPureSparkV1Task(gbmlConfigWrapper) {

  def run(): Unit

  // (yliu2) NOTE: we use dynamic columns (ex. _${edgeUsageType}_dst_node_arr) in SQL queries so that we can easily
  // debug in the spark SQL execution plan. The spark SQL plan does not show table names, and if columns names are
  // not dynamic we can't tell different joins apart
  def sampleDstNodesUniformly(
    numDstSamples: Int,
    unhydratedEdgeVIEW: String,
    permutationStrategy: String,
    numTrainingSamples: Int,
    edgeUsageType: EdgeUsageType.EdgeUsageType,
  ): String = {

    /**
      * Given root node (_src_node) samples _dst_node as pos / neg (label) destination nodes
      * @param unhydratedEdgeVIEW with columns:
      * _src_node: Int
      * _dst_node: Int
      */

    val dstArrayDF = spark.sql(f"""
      SELECT
        _src_node,
        array_sort(collect_list(_dst_node)) AS _${edgeUsageType}_dst_node_arr
      FROM
        ${unhydratedEdgeVIEW}
      GROUP BY
        _src_node
      """)

    val permutedDstArrayDF = if (permutationStrategy == PermutationStrategy.Deterministic) {
      hashBasedUniformPermutation(
        sortedArrayDF = dstArrayDF,
        arrayColName = f"_${edgeUsageType}_dst_node_arr",
        seed = samplingSeed,
      )
    } else {
      shuffleBasedUniformPermutation(
        sortedArrayDF = dstArrayDF,
        arrayColName = f"_${edgeUsageType}_dst_node_arr",
      )
    }

    val permutedDstArrayVIEW = f"permuted${edgeUsageType}ArrayDF" + uniqueTempViewSuffix
    permutedDstArrayDF.createOrReplaceTempView(permutedDstArrayVIEW)

    // @spark: NON deterministic sampling must be addressed here using cache
    val sampledDstDF = spark.sql(f"""
      SELECT
        _src_node,
        slice(_shuffled_${edgeUsageType}_dst_node_arr,
          1,
          ${numDstSamples}) AS _sampled_${edgeUsageType}_arr
      FROM
        ${permutedDstArrayVIEW}
      """)
    // @spark: critical NOTE: cache is necessary here, to break parallelism and enforce random shuffle/sampling happens once, to circumvent NON-determinism in F.shuffle. Otherwise, for all calls to sampledPositiveDF downstream, this stage will run in parallel and mess up pos samples!
    val sampledDstVIEW = f"sampled${edgeUsageType}DF" + uniqueTempViewSuffix
    sampledDstDF.createOrReplaceTempView(sampledDstVIEW)

    val cachedSampledDstVIEW = applyCachingToDf(
      dfVIEW = sampledDstVIEW,
      forceTriggerCaching = true,
      withRepartition = false,
    )
    // downsample the number of anchor nodes (to reduce the number of training samples for larger graphs
    // we only down sample when sampling pos nodes to guarantee the number of training samples
    // if we down sample both pos & neg when labels are user defined we'd end up with less than numTrainingSamples
    val retainedDstSamplesVIEW = if (numTrainingSamples > 0 && edgeUsageType == EdgeUsageType.POS) {
      println(f"Downsampling number of anchor nodes to ${numTrainingSamples}")
      val downsampledSampledDstVIEW = downsampleNumberOfNodes(
        numSamples = numTrainingSamples,
        dfVIEW = cachedSampledDstVIEW,
      )

      downsampledSampledDstVIEW
    } else {
      cachedSampledDstVIEW
    }

    val explodedDstSamplesDF = spark.sql(f"""
      SELECT
        _src_node,
        explode(_sampled_${edgeUsageType}_arr) AS _${edgeUsageType}_dst_node
      FROM
        ${retainedDstSamplesVIEW}
      """)
    val explodedDstSamplesVIEW = f"exploded${edgeUsageType}SamplesDF" + uniqueTempViewSuffix
    explodedDstSamplesDF.createOrReplaceTempView(explodedDstSamplesVIEW)
    explodedDstSamplesVIEW
  }

  def lookupDstNodeNeighborhood(
    sampledDstNodesVIEW: String,
    subgraphVIEW: String,
    edgeUsageType: EdgeUsageType.EdgeUsageType,
  ): String = {

    /**
      * We don't repeat sampling for khop pos neighborhood, instead we look up from **cached** subgraph VIEW
      * @param sampledVIEW with columns:
      * _src_node: Int which is equivalent to root node of each sample
      * _pos_dst_node / _neg_dst_node: Int
      * @param subgraphVIEW with columns:
      * _root_node: Int
      * _neighbor_edges: Array(Struct)
      * _neighbor_nodes: Array(Struct)
      * _node_features: Array
      * _condensed_node_type: Int
      */

    val dstNodeNeighborhoodDF = spark.sql(f"""
      SELECT
        _neighbor_edges AS _${edgeUsageType}_neighbor_edges,
        _neighbor_nodes AS _${edgeUsageType}_neighbor_nodes,
        _src_node,
        _${edgeUsageType}_dst_node
      FROM
        ${subgraphVIEW}
      INNER JOIN
        ${sampledDstNodesVIEW}
      ON
        ${subgraphVIEW}._root_node = ${sampledDstNodesVIEW}._${edgeUsageType}_dst_node
      """)
    val dstNodeNeighborhoodVIEW = f"${edgeUsageType}NodeNeighborhoodDF" + uniqueTempViewSuffix
    dstNodeNeighborhoodDF.createOrReplaceTempView(dstNodeNeighborhoodVIEW)

    // (yliu2) keeping the following here in case we ever want to come back to the force dst node solution
    // if the subgraphVIEW does not contain src only, we would have to do a right join instead of inner join
    // in the query above
    //    // Hydrate dst node with features, inject directly into neighborhood
    //    val hydratedDstNodeNeighborhoodDF = spark.sql(f"""
    //      SELECT
    //        _${edgeUsageType}_neighbor_edges,
    //        _${edgeUsageType}_neighbor_nodes,
    //        _src_node,
    //        STRUCT(
    //          _${edgeUsageType}_dst_node AS _node_id,
    //          _condensed_node_type,
    //          _node_features AS _feature_values
    //         ) AS _${edgeUsageType}_dst_node_hydrated
    //      FROM
    //        ${dstNodeNeighborhoodVIEW}
    //      INNER JOIN
    //        ${hydratedNodeVIEW}
    //      ON
    //        ${dstNodeNeighborhoodVIEW}._${edgeUsageType}_dst_node = ${hydratedNodeVIEW}._node_id
    //      """)
    //    val hydratedDstNodeNeighborhoodVIEW =
    //      f"hydrated${edgeUsageType}NodeNeighborhoodDF" + uniqueTempViewSuffix
    //    hydratedDstNodeNeighborhoodDF.createOrReplaceTempView(hydratedDstNodeNeighborhoodVIEW)

    //    val collectedDstNodeNeighborhoodDF = spark.sql(f"""
    //      SELECT
    //        /*+ REPARTITION(_src_node) */ _src_node,
    //        array_distinct(flatten(collect_list(_${edgeUsageType}_neighbor_edges))) AS _${edgeUsageType}_neighbor_edges,
    //        array_distinct(
    //            CONCAT(
    //                flatten(collect_list(_${edgeUsageType}_neighbor_nodes)),
    //                collect_list(_${edgeUsageType}_dst_node_hydrated)
    //            )
    //        ) AS _${edgeUsageType}_neighbor_nodes
    //      FROM
    //        ${hydratedDstNodeNeighborhoodVIEW}
    //      GROUP BY
    //        _src_node
    //      """)
    val collectedDstNodeNeighborhoodDF = spark.sql(
      f"""
        SELECT
          /*+ REPARTITION(_src_node) */ _src_node,
          array_distinct(flatten(collect_list(_${edgeUsageType}_neighbor_edges))) AS _${edgeUsageType}_neighbor_edges,
          array_distinct(flatten(collect_list(_${edgeUsageType}_neighbor_nodes))) AS _${edgeUsageType}_neighbor_nodes
        FROM
          ${dstNodeNeighborhoodVIEW}
        GROUP BY
          _src_node
        """,
    )
    val collectedDstNodeNeighborhoodVIEW =
      f"collected${edgeUsageType}NodeNeighborhoodDF" + uniqueTempViewSuffix
    collectedDstNodeNeighborhoodDF.createOrReplaceTempView(collectedDstNodeNeighborhoodVIEW)

    collectedDstNodeNeighborhoodVIEW
  }

  def formNeighborhoodForSrcOnlyNodes(
    sampledPosVIEW: String,
    unhydratedEdgeVIEW: String,
    posNodeNeighborhoodVIEW: String,
    hydratedNodeVIEW: String,
  ): String = {

    /**
      * srcOnlyNodes (as root node) are valid training samples for NodeAnchorBasedLinkPrediction and
      * are a particular case of directed graphs. Hence, this function creates a neighborhood for such nodes.
      */
    // srcOnlyNodes are those nodes that are included in _src_node of positive edges, but not in _dst_node of the graph.
    val allDstNodeIdsDF =
      spark.table(unhydratedEdgeVIEW).select(F.col("_dst_node")).dropDuplicates("_dst_node")
    val sampeledPosSrcNodeIds =
      spark.table(sampledPosVIEW).select(F.col("_src_node")).dropDuplicates("_src_node")
    val srcOnlyNodeIdsDF = sampeledPosSrcNodeIds
      .exceptAll(allDstNodeIdsDF)
      .select(F.col("_src_node").alias("_src_only_node"))
    val srcOnlyNodeIdsVIEW = "srcOnlyNodeIdsDF" + uniqueTempViewSuffix
    srcOnlyNodeIdsDF.createOrReplaceTempView(srcOnlyNodeIdsVIEW)

    // form neighborhood for srcOnlyNodes
    val srcOnlyNodesNeighborhoodDF = spark.sql(f"""
        SELECT
          _src_only_node AS _root_node,
          _pos_neighbor_edges AS _neighbor_edges,
          _pos_neighbor_nodes AS _neighbor_nodes
        FROM
          ${posNodeNeighborhoodVIEW}
        INNER JOIN
          ${srcOnlyNodeIdsVIEW}
        ON
          ${posNodeNeighborhoodVIEW}._src_node = ${srcOnlyNodeIdsVIEW}._src_only_node
      """)
    val srcOnlyNodesNeighborhoodVIEW = "srcOnlyNodesNeighborhoodDF" + uniqueTempViewSuffix
    srcOnlyNodesNeighborhoodDF.createOrReplaceTempView(srcOnlyNodesNeighborhoodVIEW)
    // add hydrated root node to neighborhood
    val srcOnlyNodesNeighborhoodWithHydratedRootNodeDF = spark.sql(f"""
          SELECT
            _root_node,
            _neighbor_edges,
            _neighbor_nodes,
          ARRAY(STRUCT(_root_node AS _node_id,
                _condensed_node_type,
                _node_features AS _feature_values)) AS _root_node_hydrated,
          _node_features, _condensed_node_type
          FROM
            ${srcOnlyNodesNeighborhoodVIEW}
          INNER JOIN
            ${hydratedNodeVIEW}
          ON
            ${srcOnlyNodesNeighborhoodVIEW}._root_node = ${hydratedNodeVIEW}._node_id
      """)
    val srcOnlyNodesNeighborhoodWithHydratedRootNodeVIEW =
      " srcOnlyNodesNeighborhoodWithHydratedRootNodeDF" + uniqueTempViewSuffix
    srcOnlyNodesNeighborhoodWithHydratedRootNodeDF.createOrReplaceTempView(
      srcOnlyNodesNeighborhoodWithHydratedRootNodeVIEW,
    )
    // addresses a corner case, where we have a root node that is dstOnly node, with number of edges > numSamples st
    // src nodes are a combination of srcOnly nodes and nodes with both in-edges and out-edges
    // ensures root node is included in neighborhood
    val srcOnlyNodesNeighborhoodWithHydratedRootNodeAddedToNeighborhoodDF = spark.sql(f"""
        SELECT
          _root_node,
          _neighbor_edges,
          array_distinct(CONCAT(_neighbor_nodes, _root_node_hydrated)) AS _neighbor_nodes,
          _node_features,
          _condensed_node_type
        FROM
          ${srcOnlyNodesNeighborhoodWithHydratedRootNodeVIEW}
    """)
    val srcOnlyNodesNeighborhoodWithHydratedRootNodeAddedToNeighborhoodVIEW =
      "srcOnlyNodesNeighborhoodWithHydratedRootNodeAddedToNeighborhoodDF" + uniqueTempViewSuffix
    srcOnlyNodesNeighborhoodWithHydratedRootNodeAddedToNeighborhoodDF.createOrReplaceTempView(
      srcOnlyNodesNeighborhoodWithHydratedRootNodeAddedToNeighborhoodVIEW,
    )
    srcOnlyNodesNeighborhoodWithHydratedRootNodeAddedToNeighborhoodVIEW
  }

  def hydrateTaskBasedEdges(
    sampledTaskBasedEdgesVIEW: String,
    hydratedEdgeVIEW: String,
    edgeUsageType: EdgeUsageType.EdgeUsageType,
  ): String = {

    /**
      * Adds feature values and condensed edge type to pos edges.
      * @param sampledTaskBasedEdgesVIEW with columns:
      * _src_node: Int which is equivalent to root node of each sample
      * _pos_dst_node: Int
      * @param hydratedEdgeVIEW with columns:
      * _from: Int, src node id
      * _to: Int, dst node id
      * _condensed_edge_type: Int
      * _edge_features: Array, including all features
      */

    val hydratedTaskBasedEdgeDF = spark
      .sql(f"""
        SELECT
          *
        FROM
          ${hydratedEdgeVIEW}
        INNER JOIN
          ${sampledTaskBasedEdgesVIEW}
        ON
          (${sampledTaskBasedEdgesVIEW}._src_node = ${hydratedEdgeVIEW}._from
            AND ${sampledTaskBasedEdgesVIEW}._${edgeUsageType}_dst_node = ${hydratedEdgeVIEW}._to)
        """)
      .drop("_from")
      .drop("_to")

    // collect all positive edges for each root node, after hydration
    val hydratedTaskBasedEdgeVIEW =
      f"hydratedTaskBased${edgeUsageType}EdgeDF" + uniqueTempViewSuffix
    hydratedTaskBasedEdgeDF.createOrReplaceTempView(hydratedTaskBasedEdgeVIEW)

    val collectedHydratedTaskBasedEdgeDF = spark.sql(f"""
      SELECT
        /*+ REPARTITION(_root_node) */ _src_node AS _root_node,
        collect_list(STRUCT(_src_node,
            _${edgeUsageType}_dst_node AS _dst_node,
            _condensed_edge_type,
            _edge_features AS _feature_values)) AS _${edgeUsageType}_hydrated_edges
      FROM
        ${hydratedTaskBasedEdgeVIEW}
      GROUP BY
        _root_node
      """)
    val collectedHydratedTaskBasedEdgeVIEW =
      f"collectedHydrated${edgeUsageType}EdgeDF" + uniqueTempViewSuffix
    collectedHydratedTaskBasedEdgeDF.createOrReplaceTempView(collectedHydratedTaskBasedEdgeVIEW)
    collectedHydratedTaskBasedEdgeVIEW
  }

  def appendIsolatedNodesToTrainingSamples(
    trainingSubgraphVIEW: String,
    hydratedNodeVIEW: String,
    unhydratedEdgeVIEW: String,
  ): String = {

    /**
      * For cases that users want to include isolated nodes in task specific samples.
      * @param trainingSubgraphVIEW trainingSubgraph without isolated nodes, with columns:
      * _root_node: Int
      * _neighbor_edges: Array(Struct)
      * _neighbor_nodes: Array(Struct)
      * _node_features: Array
      * _condensed_node_type: Int
      * _label: Int, label value for each node id
      * _label_type: String
      */
    val isolatedNodesSubgraphVIEW = createIsolatedNodesSubgraph(
      hydratedNodeVIEW = hydratedNodeVIEW,
      unhydratedEdgeVIEW = unhydratedEdgeVIEW,
    )
    val trainingSubgraphsWithIsolatedNodesDF = spark.sql(f"""
        SELECT
          _root_node,
          _neighbor_edges,
          _neighbor_nodes,
          _pos_hydrated_edges,
          _node_features,
          _condensed_node_type
        FROM
          ${trainingSubgraphVIEW} UNION
        SELECT
          _root_node,
          NULL AS _neighbor_edges,
          ARRAY(STRUCT(_root_node AS _node_id,
              _condensed_node_type,
              _node_features AS _feature_values)) AS _neighbor_nodes,
          NULL AS _pos_hydrated_edges,
          NULL AS _neg_hydrated_edges,
          _node_features,
          _condensed_node_type
        FROM
          ${isolatedNodesSubgraphVIEW}
      """)
    val trainingSubgraphsWithIsolatedNodesVIEW =
      "trainingSubgraphsWithIsolatedNodesDF" + uniqueTempViewSuffix
    trainingSubgraphsWithIsolatedNodesDF.createOrReplaceTempView(
      trainingSubgraphsWithIsolatedNodesVIEW,
    )
    trainingSubgraphsWithIsolatedNodesVIEW
  }

  def castToTrainingSampleProtoSchema(
    dfVIEW: String,
    hasHardNegs: Boolean = false,
  ): String = {
    // Forms dfVIEW schema into NodeAnchorBasedLinkPredictionSample message in training_samples_schema.proto
    val nablpDF = if (!hasHardNegs) {
      spark.sql(f"""
      SELECT
        STRUCT( _root_node AS node_id,
          _condensed_node_type AS condensed_node_type,
          _node_features AS feature_values ) AS root_node,
        ARRAY() AS hard_neg_edges,
        _pos_hydrated_edges AS pos_edges,
        ARRAY() AS neg_edges,
        STRUCT( _neighbor_nodes AS nodes,
          _neighbor_edges AS edges ) AS neighborhood
      FROM
        ${dfVIEW}
                """)
    } else {
      spark.sql(f"""
      SELECT
        STRUCT( _root_node AS node_id,
          _condensed_node_type AS condensed_node_type,
          _node_features AS feature_values ) AS root_node,
        _neg_hydrated_edges AS hard_neg_edges,
        _pos_hydrated_edges AS pos_edges,
        ARRAY() AS neg_edges,
        STRUCT( _neighbor_nodes AS nodes,
          _neighbor_edges AS edges ) AS neighborhood
      FROM
        ${dfVIEW}
                """)
    }

    val nablpVIEW = "nablpDF" + uniqueTempViewSuffix
    nablpDF.createOrReplaceTempView(nablpVIEW)
    nablpVIEW
  }

}
