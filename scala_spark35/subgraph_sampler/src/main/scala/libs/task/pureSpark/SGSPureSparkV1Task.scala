package libs.task.pureSpark

import common.types.EdgeUsageType
import common.types.pb_wrappers.GbmlConfigPbWrapper
import common.utils.SparkSessionEntry.getActiveSparkSession
import common.utils.SparkSessionEntry.getNumCurrentShufflePartitions
import common.utils.TFRecordIO.RecordTypes
import common.utils.TFRecordIO.readDataframeFromTfrecord
import libs.task.SamplingStrategy.PermutationStrategy
import libs.task.SamplingStrategy.hashBasedUniformPermutation
import libs.task.SamplingStrategy.shuffleBasedUniformPermutation
import libs.task.SubgraphSamplerTask
import org.apache.spark.sql.Column
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{functions => F}
import org.apache.spark.storage.StorageLevel
import snapchat.research.gbml.preprocessed_metadata.PreprocessedMetadata

import java.util.UUID.randomUUID
import scala.collection.mutable.ListBuffer

abstract class SGSPureSparkV1Task(
  gbmlConfigWrapper: GbmlConfigPbWrapper)
    extends SubgraphSamplerTask(gbmlConfigWrapper) {

  // var lockCallToMethod: Boolean = false

  private def generateUniqueSuffix: String = {

    /** create a unique suffix for table view names, upon initialization of this calss
      */
    val uniqueTempViewSuffix = randomUUID.toString.replace("-", "_")
    "_" + uniqueTempViewSuffix
  }

  val uniqueTempViewSuffix: String = generateUniqueSuffix

  def loadNodeDataframeIntoSparkSql(condensedNodeType: Int): String = {

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
    // NOTE1: in graph_schema.proto: node_id is of type uint32, however for the input node table
    // node_id is long. This casting makes datatype compatible with protobufs
    // We can remove this casting step, by changing type in graph_schema.proto
    // from uint32 to uint64 in the future.
    // As of now making such fundamental change to protobuf message may mess up people's usage of mocked data.

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

  def loadEdgeDataframeIntoSparkSql(
    condensedEdgeType: Int,
    edgeUsageType: EdgeUsageType.EdgeUsageType = EdgeUsageType.MAIN,
  ): String = {

    /** For a given condensed_edge_type loads edge dataframe with columns: _from: Int, _to: Int,
      * _condensed_edge_type: Int, _edge_features: Array()(if features are provided) (used for
      * hydration)
      * edgeUsageType: EdgeUsageType.EdgeUsageType = EdgeUsageType.MAIN, EdgeUsageType.POS, EdgeUsageType.NEG
      */
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

    var hydratedEdgeVIEW: String =
      f"hydrated${condensedEdgeType}${edgeUsageType}EdgeDF" + uniqueTempViewSuffix
    hydratedEdgeDF.createOrReplaceTempView(
      hydratedEdgeVIEW,
    )

    def enforceBidirectionalization(hydratedEdgeVIEW: String): String = {

      // I don't think we need to include GROUP BY in below query.
      // check edge features sanity, once we have edge features.
      val undirectedEdgesDF = spark.sql(f"""
          SELECT DISTINCT
            LEAST(_from, _to) AS _from,
            GREATEST(_from, _to) AS _to,
            _condensed_edge_type,
            _edge_features
          FROM
            ${hydratedEdgeVIEW}
          GROUP BY
            LEAST(_from, _to),
            GREATEST(_from, _to),
            _condensed_edge_type,
            _edge_features
          """)
      val uniqueUndirectedEdgesDF = undirectedEdgesDF.dropDuplicates("_from", "_to")
      val uniqueUndirectedEdgesVIEW =
        f"uniqueUndirected${edgeUsageType}EdgesDF" + uniqueTempViewSuffix
      uniqueUndirectedEdgesDF.createOrReplaceTempView(uniqueUndirectedEdgesVIEW)
      val uniqueDirectedEdgesDF = spark.sql(f"""
          SELECT
            _from,
            _to,
            _condensed_edge_type,
            _edge_features
          FROM
            ${uniqueUndirectedEdgesVIEW} UNION
          SELECT
            _to,
            _from,
            _condensed_edge_type,
            _edge_features
          FROM
            ${uniqueUndirectedEdgesVIEW}
          """)
      uniqueDirectedEdgesDF.createOrReplaceTempView(hydratedEdgeVIEW)
      hydratedEdgeVIEW
    }

    // Is graph directed specifies if we want to treat the graph as directed or undirected
    println(f"isGraphDirected: ${gbmlConfigWrapper.sharedConfigPb.isGraphDirected}")

    // We enforce only bidirectionalization for MAIN edges, user defined edges are always kept directed
    hydratedEdgeVIEW =
      if (
        !(gbmlConfigWrapper.sharedConfigPb.isGraphDirected) && (edgeUsageType == EdgeUsageType.MAIN)
      ) {
        println(
          f"Enforce bidirectionalization for condensedEdgeType: ${condensedEdgeType}, edgeUsageType: ${edgeUsageType}",
        )
        enforceBidirectionalization(hydratedEdgeVIEW = hydratedEdgeVIEW)
      } else {
        hydratedEdgeVIEW
      }
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

  def sampleOnehopSrcNodesUniformly(
    numNeighborsToSample: Int,
    unhydratedEdgeVIEW: String,
    permutationStrategy: String,
  ): String = {

    /** 1. for each dst node, take all in-edges as onehop array 2. randomly shuffle onehop array
      *  elements and take first numNeighborsToSample elements (random shuffle is a
      *  NON-deterministic transformation and cannot be assigned with a fixed random seed.)
      *  Creates sampledOnehopDF with columns: _dst_node: Int, _sampled_1_hop_arr: Arr. Returns
      *  associated sampledOnehopVIEW
      * @param numNeighborsToSample taken from config
      * @param unhydratedEdgeVIEW with columns:
      * _src_node: Int, src node id
      * _dst_node: Int, dst node id
      */

    // @spark: collect_list requires shuffle which happens at two levels: at executor level (partially) and then at workers level (fully), by repartitoning dataframe the number of shuffle at the executor level will be reduced
    // See the following for reparition hints:
    // https://spark.apache.org/docs/latest/sql-ref-syntax-qry-select-hints.html#partitioning-hints-types
    val onehopArrayDF: DataFrame = spark.sql(f"""
        SELECT
          /*+ REPARTITION(_dst_node) */ _dst_node,
          array_sort(collect_list(_src_node)) AS _1_hop_arr
        FROM
          ${unhydratedEdgeVIEW}
        GROUP BY
          _dst_node
        """)
    val permutedOnehopArrayDF = if (permutationStrategy == PermutationStrategy.Deterministic) {
      hashBasedUniformPermutation(
        sortedArrayDF = onehopArrayDF,
        arrayColName = "_1_hop_arr",
        seed = samplingSeed,
      )
    } else {
      shuffleBasedUniformPermutation(sortedArrayDF = onehopArrayDF, arrayColName = "_1_hop_arr")
    }
    val permutedOnehopArrayVIEW = "permutedOnehopArrayDF" + uniqueTempViewSuffix
    permutedOnehopArrayDF.createOrReplaceTempView(permutedOnehopArrayVIEW)

    val sampledOnehopDF: DataFrame = spark.sql(f"""
          SELECT
            _dst_node AS _0_hop,
            slice(_shuffled_1_hop_arr, 1, ${numNeighborsToSample}) AS _sampled_1_hop_arr
          FROM
            ${permutedOnehopArrayVIEW}
          """)

    // @spark: critical NOTE: cache is necessary here, to break parallelism and enforce random shuffle/sampling happens once, to circumvent NON-determinism in F.shuffle. Otherwise, for all calls to sampledOnehopDF downstream, this stage will run in parallel and mess up onehop samples!
    // @spark: maybe in future we wanna try caching the exploded verison of below DF.
    val sampledOnehopVIEW = "sampledOnehopDF" + uniqueTempViewSuffix
    sampledOnehopDF.createOrReplaceTempView(sampledOnehopVIEW)
    val cachedSampledOnehopVIEW = applyCachingToDf(
      dfVIEW = sampledOnehopVIEW,
      forceTriggerCaching = true,
      withRepartition = false,
    )
    cachedSampledOnehopVIEW

  }

  def sampleTwohopSrcNodesUniformly(
    numNeighborsToSample: Int,
    unhydratedEdgeVIEW: String,
    sampledOnehopVIEW: String,
    permutationStrategy: String,
  ): String = {

    /**   1. uses onehop nodes in sampledOnehopVIEW as reference to obtain twohop neighbors for each
      *      root node 2. randomly shuffles twohop array elements and takes first num_samples
      *      elements. Creates sampledTwohopDF.
      * @param numNeighborsToSample taken from config
      * @param unhydratedEdgeVIEW with columns:
      * _src_node: Int, src node id
      * _dst_node: Int, dst node id
      * @param sampledOnehopVIEW with columns:
      * _0_hop: Int, root node id
      * _sampled_1_hop_arr: Array, including sampled src nodes for one hop edges
      * TODO: get khop neighbor (in a generic way: [iteratively or recursively? + cache + repartiton?])
      */

    val explodedOnehopDF: DataFrame = spark.sql(f"""
          SELECT
            _0_hop,
            explode(_sampled_1_hop_arr) AS _1_hop
          FROM
            ${sampledOnehopVIEW}
        """)

    val explodedOnehopVIEW = "explodedOnehopDF" + uniqueTempViewSuffix
    explodedOnehopDF.createOrReplaceTempView(explodedOnehopVIEW)

    // @spark: bigger df on the left side of join, to avoid more shuffles
    val allTwohopDstDF: DataFrame = spark.sql(f"""
          SELECT
            _0_hop,
            _1_hop,
            _src_node AS _2_hop
          FROM
            ${unhydratedEdgeVIEW}
          INNER JOIN
            ${explodedOnehopVIEW}
          ON
            ${unhydratedEdgeVIEW}._dst_node = ${explodedOnehopVIEW}._1_hop
        """)
    val allTwohopDstVIEW = "allTwohopDstDF" + uniqueTempViewSuffix
    allTwohopDstDF.createOrReplaceTempView(allTwohopDstVIEW)

    val twohopArrayDstDF = spark.sql(f"""
          SELECT
            /*+ REPARTITION(_0_hop, _1_hop) */ _0_hop,
            _1_hop,
            array_sort(collect_list(_2_hop)) AS _2_hop_arr
          FROM
            ${allTwohopDstVIEW}
          GROUP BY
            _0_hop,
            _1_hop
          """)

    val permutedTwohopArrayDF = if (permutationStrategy == PermutationStrategy.Deterministic) {
      hashBasedUniformPermutation(
        sortedArrayDF = twohopArrayDstDF,
        arrayColName = "_2_hop_arr",
        seed = samplingSeed,
      )
    } else {
      shuffleBasedUniformPermutation(sortedArrayDF = twohopArrayDstDF, arrayColName = "_2_hop_arr")
    }
    val permutedTwohopArrayVIEW = "permutedTwohopArrayDF" + uniqueTempViewSuffix
    permutedTwohopArrayDF.createOrReplaceTempView(permutedTwohopArrayVIEW)
    // @spark: Since twohop df is called only once, no need to take care of NON determinism in the shuffle/sampling (for k hop any k-1 df must be cached)
    val sampledTwohopDF: DataFrame = spark.sql(f"""
        SELECT
          _0_hop,
          _1_hop,
          slice(_shuffled_2_hop_arr ,1 , ${numNeighborsToSample}) AS _sampled_2_hop_arr
        FROM
          ${permutedTwohopArrayVIEW}
        """)
    val sampledTwohopVIEW = "sampledTwohopDF" + uniqueTempViewSuffix
    sampledTwohopDF.createOrReplaceTempView(sampledTwohopVIEW)
    //
    val cachedSampledTwohopVIEW = applyCachingToDf(
      dfVIEW = sampledTwohopVIEW,
      forceTriggerCaching = true,
      withRepartition = false,
    )
    cachedSampledTwohopVIEW
  }

  def hydrateNodes(
    k: Int,
    hydratedNodeVIEW: String,
    sampledKhopVIEW: String,
  ): String = {

    /** Hydrates (only) khop neighbor nodes (not root_node) with node features/types. Returns hydratedKhopNodesVIEW with columns:
      * _node_id, _node_features, _condensed_node_type, _0_hop, _1_hop, ..., _k_hop
      * @param k number of current hop
      * @param hydratedNodeVIEW with columns:
      * _node_id: Int, root node id
      * _condensed_node_type: Int
      * _node_features: Array
      * @param sampledKhopVIEW with columns:
      * _0_hop: Int, root node id
      * _1_hop, ..., _k-1_hop: Int, kth hop node id
      * _sampled_k_hop_arr, Array
      */

    val allColsExceptSampled =
      spark
        .table(sampledKhopVIEW)
        .columns
        .filter(col => col != f"_sampled_${k}_hop_arr")
        .mkString(", ")
    val khopDF = spark.sql(f"""
          SELECT
            ${allColsExceptSampled},
            explode(_sampled_${k}_hop_arr) AS _${k}_hop
          FROM
            ${sampledKhopVIEW}
          """)

    val khopVIEW =
      f"${k}hopDF" + uniqueTempViewSuffix // NOTE: Name of this view MUST be unique, otherwise the name will be overwitten and the associated table will be lost. Spark starts repeating all stages to retrieve lost table and redo khop sampling: sampling is done again and we will have new frontiers which lead to an invalid neighborhood!
    khopDF.createOrReplaceTempView(khopVIEW)

    val hydratedKhopNodesDF = spark.sql(f"""
        SELECT
          *
        FROM
          ${hydratedNodeVIEW}
        INNER JOIN
          ${khopVIEW}
        ON
          ${khopVIEW}._${k}_hop = ${hydratedNodeVIEW}._node_id
      """)

    val hydratedKhopNodesVIEW = f"hydrated${k}hopNodesDF" + uniqueTempViewSuffix
    hydratedKhopNodesDF.createOrReplaceTempView(hydratedKhopNodesVIEW)
    hydratedKhopNodesVIEW
  }

  def hydrateEdges(
    k: Int,
    hydratedEdgeVIEW: String,
    hydratedKhopNodesVIEW: String,
  ): String = {

    /** Hydrates (only) khop neighbor edges with edge features/types. Note that for the case of no
      * edge feature, we still should do this step, to hydrate edges with edge types and empty edge
      * features. Returns hydratedKhopNodesEdgesVIEW with
      * columns: _condensed_edge_type, _edge_features, _node_id, _node_features,
      * _condensed_node_type, _0_hop, _1_hop, ..., _k_hop
      * @param hydratedEdgeVIEW with columns:
      * _src_node: Int
      * _dst_node: Int
      * _condensed_edge_type: Int
      * _edge_features: Array
      * @param hydratedKhopNodesVIEW with columns:
      * _node_id: Int, root node id
      * _node_features: Array
      * _condensed_node_type: Int
      * _0_hop: Int, root node id
      * _1_hop: Int, src node id for 1 hop edge
      * ...
      * _k_hop: Int, src node id for k hop edge
      */

    val hydratedKhopNodesEdgesDF = spark
      .sql(f"""
        SELECT
          *
        FROM
          ${hydratedKhopNodesVIEW}
        INNER JOIN
          ${hydratedEdgeVIEW}
        ON
          (${hydratedKhopNodesVIEW}._${k}_hop = ${hydratedEdgeVIEW}._from
            AND ${hydratedKhopNodesVIEW}._${k - 1}_hop = ${hydratedEdgeVIEW}._to)
        """)
      .drop("_to")
      .drop("_from")

    val hydratedKhopNodesEdgesVIEW = f"hydrated${k}hopNodesEdgesDF" + uniqueTempViewSuffix
    hydratedKhopNodesEdgesDF.createOrReplaceTempView(hydratedKhopNodesEdgesVIEW)
    hydratedKhopNodesEdgesVIEW
  }

  private def collectHydratedKthHopNeighborhood(
    k: Int,
    hydratedKhopNodesEdgesVIEW: String,
  ): String = {

    /** Collect all kth hop neighbors for each root node, after hydration. Form neighbors schema st
      * it conforms with graph_schema.proto
      * @param hydratedKhopNodesEdgesVIEW with columns:
      * _root_node: Int
      * _neighbor_edges: Array(Struct), where struct includes {src, dst, edge_type, edge_feature}
      * _neighbor_nodes: Array(Struct), where struct includes {node id, node_type, node_feature}
      * _node_feature: Array, nodes features of root node
      * _condensed_node_type: Int, node_type of root node
      * @param k number of current hop
      */
    // @spark: ensure hydratedKhopNodesEdgesVIEW names are unique for each hop, otherwise
    // the cached df in `sampleOnehopDstNodesUniformly()` will break/be evicted.
    // reason: when we use SQL API, each `createOrReplaceTempView` is a SQL query. If we register
    // different dfs with the same name, the `createOrReplaceTempView` will repeat any previous stage
    // that is required to create that df and will ignore cached stages
    val hydratedKthHopDF = spark.sql(f"""
        SELECT
          _0_hop,
          collect_list(STRUCT(_${k}_hop AS _src_node,
              _${k - 1}_hop AS _dst_node,
              _condensed_edge_type,
              _edge_features AS _feature_values)) AS _${k}_hop_hydrated_edges,
          array_distinct(collect_list(STRUCT(_${k}_hop AS _node_id,
                _condensed_node_type,
                _node_features AS _feature_values))) AS _${k}_hop_hydrated_nodes
        FROM
          ${hydratedKhopNodesEdgesVIEW}
        GROUP BY
          _0_hop
        """)
    val hydratedKthHopVIEW = f"hydrated${k}thHopDF" + uniqueTempViewSuffix
    hydratedKthHopDF.createOrReplaceTempView(hydratedKthHopVIEW)
    hydratedKthHopVIEW
  }

  def createKthHydratedNeighborhood(
    k: Int,
    sampledKhopVIEW: String,
    hydratedNodeVIEW: String,
    hydratedEdgeVIEW: String,
  ): String = {
    val hydratedKthHopNodesVIEW = hydrateNodes(
      k = k,
      hydratedNodeVIEW = hydratedNodeVIEW,
      sampledKhopVIEW = sampledKhopVIEW,
    )
    val hydratedKthHopNodesEdgesVIEW = hydrateEdges(
      k = k,
      hydratedEdgeVIEW = hydratedEdgeVIEW,
      hydratedKhopNodesVIEW = hydratedKthHopNodesVIEW,
    )
    val collectedHydratedKthHopNeighborsVIEW = collectHydratedKthHopNeighborhood(
      k = k,
      hydratedKhopNodesEdgesVIEW = hydratedKthHopNodesEdgesVIEW,
    )
    val hydratedKthHopNeighborsVIEW =
      if (
        gbmlConfigWrapper.sharedConfigPb.isGraphDirected && k < gbmlConfigWrapper.subgraphSamplerConfigPb.numHops
      ) { // cache hydrated df for k < numHops
        val cachedHydratedKthHopNeighborsVIEW = applyCachingToDf(
          dfVIEW = collectedHydratedKthHopNeighborsVIEW,
          forceTriggerCaching = true,
          withRepartition = false,
        )
        cachedHydratedKthHopNeighborsVIEW
      } else {
        collectedHydratedKthHopNeighborsVIEW
      }
    hydratedKthHopNeighborsVIEW
  }

  def createSubgraph(
    hydratedNodeVIEW: String,
    hydratedEdgeVIEW: String,
    unhydratedEdgeVIEW: String,
    numNeighborsToSample: Int,
    permutationStrategy: String,
  ): String = {

    /** Adds root node features to hydrated neighborhood and creates final subgraphDF for each root
      * node. subgraphDF will be cached and used to construct
      * different task-based samples. (eg for NodeAnchorBasedLinkPredictionTask positive
      * node neighborhoods are looked up from subgraphDF without repeating khop sampling.)
      * @return subgraphsVIEW with columns:
      * _root_node: Int,
      * _neighbor_edges: Array(Struct)
      * _neighbor_nodes: Array(Struct)
      * _node_features: Array, root node features [please note that root node features are included in _neighbor_nodes,
      * we keep this column because of proto structure]
      * _condensed_node_type : Int [same reason as above]
      */

    val sampledOnehopVIEW = sampleOnehopSrcNodesUniformly(
      numNeighborsToSample = numNeighborsToSample,
      unhydratedEdgeVIEW = unhydratedEdgeVIEW,
      permutationStrategy = permutationStrategy,
    )
    val sampledTwohopVIEW = sampleTwohopSrcNodesUniformly(
      numNeighborsToSample = numNeighborsToSample,
      unhydratedEdgeVIEW = unhydratedEdgeVIEW,
      sampledOnehopVIEW = sampledOnehopVIEW,
      permutationStrategy = permutationStrategy,
    )
    // hydrate onehop neighbors
    val hydratedOnehopNeighborsVIEW = createKthHydratedNeighborhood(
      k = 1,
      sampledKhopVIEW = sampledOnehopVIEW,
      hydratedNodeVIEW = hydratedNodeVIEW,
      hydratedEdgeVIEW = hydratedEdgeVIEW,
    )
    // hydrate twohop neighbor
    val hydratedTwohopNeighborsVIEW = createKthHydratedNeighborhood(
      k = 2,
      sampledKhopVIEW = sampledTwohopVIEW,
      hydratedNodeVIEW = hydratedNodeVIEW,
      hydratedEdgeVIEW = hydratedEdgeVIEW,
    )

    val hydratedNeighborhoodDF = spark.sql(f"""
        SELECT /*+ REPARTITION(_root_node) */
          ${hydratedOnehopNeighborsVIEW}._0_hop AS _root_node,
          CONCAT(_1_hop_hydrated_edges, _2_hop_hydrated_edges ) AS _neighbor_edges,
          array_distinct(CONCAT(_1_hop_hydrated_nodes, _2_hop_hydrated_nodes)) AS _neighbor_nodes
        FROM
          ${hydratedTwohopNeighborsVIEW}
        INNER JOIN
          ${hydratedOnehopNeighborsVIEW}
        ON
          ${hydratedTwohopNeighborsVIEW}._0_hop = ${hydratedOnehopNeighborsVIEW}._0_hop
      """)

    val hydratedNeighborhoodVIEW = "hydratedNeighborhoodDF" + uniqueTempViewSuffix
    hydratedNeighborhoodDF.createOrReplaceTempView(hydratedNeighborhoodVIEW)

    val completeHydratedNeighborhoodVIEW =
      if (!(gbmlConfigWrapper.sharedConfigPb.isGraphDirected)) {
        hydratedNeighborhoodVIEW
      } else {
        // if graph is directed, append all subgraphs with less than k hop neighbors to hydratedNeighborhoodVIEW
        // NOTE: isolated nodes will not be appended in this function. They'll be appended separately.
        val lessThanKHopNeighborsNodeIdsVIEW = findLessThanKHopNeighborsNodeIds(
          khopSamplesVIEW = sampledTwohopVIEW,
          lessThanKHopSamplesVIEW = sampledOnehopVIEW,
        )

        val lessThanKHopNeighborhoodDF = spark.sql(f"""
            SELECT /*+ REPARTITION(_root_node) */
              ${lessThanKHopNeighborsNodeIdsVIEW}._0_hop AS _root_node,
              _1_hop_hydrated_edges,
              _1_hop_hydrated_nodes
            FROM
              ${hydratedOnehopNeighborsVIEW}
            INNER JOIN
              ${lessThanKHopNeighborsNodeIdsVIEW}
            ON
              ${hydratedOnehopNeighborsVIEW}._0_hop = ${lessThanKHopNeighborsNodeIdsVIEW}._0_hop
          """)

        val lessThanKHopNeighborhoodVIEW = "lessThanKHopNeighborhoodDF" + uniqueTempViewSuffix
        lessThanKHopNeighborhoodDF.createOrReplaceTempView(lessThanKHopNeighborhoodVIEW)

        val allLessThanOrEqualToKHydratedNeighborhoodDF = spark.sql(f"""
            SELECT *
            FROM
              ${hydratedNeighborhoodVIEW}
            UNION
            SELECT *
            FROM
              ${lessThanKHopNeighborhoodVIEW}
          """)

        val allLessThanOrEqualToKHydratedNeighborhoodVIEW =
          "allLessThanOrEqualToKHydratedNeighborhoodDF" + uniqueTempViewSuffix
        allLessThanOrEqualToKHydratedNeighborhoodDF.createOrReplaceTempView(
          allLessThanOrEqualToKHydratedNeighborhoodVIEW,
        )
        allLessThanOrEqualToKHydratedNeighborhoodVIEW
      }
    // add features of root node to completeHydratedNeighborhoodVIEW
    val hydratedNeighborhoodWithRootNodeDF = spark
      .sql(f"""
          SELECT /*+ REPARTITION(_root_node) */
            _root_node,
            _neighbor_edges,
            _neighbor_nodes,
            ARRAY(STRUCT(_root_node AS _node_id,
                _condensed_node_type,
                _node_features AS _feature_values)) AS _root_node_hydrated,
            _node_features,
            _condensed_node_type
          FROM
            ${completeHydratedNeighborhoodVIEW}
          INNER JOIN
            ${hydratedNodeVIEW}
          ON
            ${completeHydratedNeighborhoodVIEW}._root_node = ${hydratedNodeVIEW}._node_id
          """)
    val hydratedNeighborhoodWithRootNodeVIEW =
      "hydratedNeighborhoodWithRootNodeDF" + uniqueTempViewSuffix
    hydratedNeighborhoodWithRootNodeDF.createOrReplaceTempView(hydratedNeighborhoodWithRootNodeVIEW)

    // TODO: once write is added, try assigning empty arr, and null to _node_features and _condensed_node_type
    // to have node_ids with empty features/type in proto, to save some space
    val subgraphsDF = spark.sql(f"""
          SELECT /*+ REPARTITION(_root_node) */
            _root_node,
            _neighbor_edges,
            array_distinct(CONCAT(_neighbor_nodes, _root_node_hydrated)) AS _neighbor_nodes,
            _node_features,
            _condensed_node_type
          FROM
            ${hydratedNeighborhoodWithRootNodeVIEW}
    """)

    val subgraphsVIEW = "subgraphsDF" + uniqueTempViewSuffix
    subgraphsDF.createOrReplaceTempView(subgraphsVIEW)
    subgraphsVIEW
  }

  def findLessThanKHopNeighborsNodeIds(
    khopSamplesVIEW: String,
    lessThanKHopSamplesVIEW: String,
  ): String = {

    /**
      * For directed graphs, to have all valid neighborhoods included in samples, we'll need to
      * include all 1, ..., k-1 neighborhood of root nodes.
      * For k=2, we find nodes that have ONLY k=1 hop, by finding the nodes ids that are in 1hop's root_node (_0_hop)
      * but not in 2hop's root_node (_0_hop)
      * @param khopSamplesVIEW with columns
      * _0_hop, _1_hop, ..., _k_hop
      * @param lessThanKHopSamplesVIEW with columns
      * _0_hop, _1_hop, ..., _k-1_hop
      * NOTE, currently k is hardcoded to 2. For k>2, we can use this functions within a for loop to find
      * less than khop sample node ids.
      */
    val lessThanKHopSamplesDF       = spark.table(lessThanKHopSamplesVIEW).select("_0_hop")
    val khopSamplesDF               = spark.table(khopSamplesVIEW).select("_0_hop")
    val lessThanKNeighborsNodesDF   = lessThanKHopSamplesDF.exceptAll(khopSamplesDF)
    val lessThanKNeighborsNodesVIEW = "lessThanKNeighborsNodesDF" + uniqueTempViewSuffix
    lessThanKNeighborsNodesDF.createOrReplaceTempView(lessThanKNeighborsNodesVIEW)
    lessThanKNeighborsNodesVIEW
  }

  def createIsolatedNodesSubgraph(
    hydratedNodeVIEW: String,
    unhydratedEdgeVIEW: String,
  ): String = {

    /**
      * find all node ids from node df and and all node ids that have at least one edge from edge df
      * the difference will be isolated nodes
      */
    val allNodeIdsDF = spark.sql(f"""
        SELECT
          _node_id
        FROM
          ${hydratedNodeVIEW}
        """)
    val allNodeIdsVIEW = "allNodeIdsDF" + uniqueTempViewSuffix
    allNodeIdsDF.createOrReplaceTempView(allNodeIdsVIEW)

    val nodeIDsWithAtLeastOneEdgeDF = spark
      .sql(f"""
        SELECT
            _dst_node AS _nonisolated_node_id
        FROM
            ${unhydratedEdgeVIEW} UNION
        SELECT
            _src_node AS _nonisolated_node_id
        FROM
            ${unhydratedEdgeVIEW}
        """)
      .dropDuplicates("_nonisolated_node_id")
    val nodeIDsWithAtLeastOneEdgeVIEW = "nodeIDsWithAtLeastOneEdgeDF" + uniqueTempViewSuffix
    nodeIDsWithAtLeastOneEdgeDF.createOrReplaceTempView(nodeIDsWithAtLeastOneEdgeVIEW)
    val isolatedNodeIdsDF = allNodeIdsDF
      .exceptAll(nodeIDsWithAtLeastOneEdgeDF)
      .select(F.col("_node_id").alias("_isolated_node"))

    val isolatedNodeIdsVIEW = "isolatedNodeIdsDF" + uniqueTempViewSuffix
    isolatedNodeIdsDF.createOrReplaceTempView(isolatedNodeIdsVIEW)
    val hydratedIsolatedNodesDF = spark.sql(f"""
        SELECT
          _node_id AS _root_node,
          _node_features,
          _condensed_node_type
        FROM
          ${hydratedNodeVIEW}
        INNER JOIN
          ${isolatedNodeIdsVIEW}
        ON
          ${hydratedNodeVIEW}._node_id = ${isolatedNodeIdsVIEW}._isolated_node
      """)
    val hydratedIsolatedNodesVIEW = "hydratedIsolatedNodesDF" + uniqueTempViewSuffix
    hydratedIsolatedNodesDF.createOrReplaceTempView(hydratedIsolatedNodesVIEW)
    hydratedIsolatedNodesVIEW
  }

  def createNeighborlessNodesSubgraph(
    hydratedNodeVIEW: String,
    unhydratedEdgeVIEW: String,
  ): String = {

    /** If a graph (such as MAU) has neighborless nodes (1. isolated nodes; i.e. # of node_ids > # of unique
      * src_ids or dst_ids OR 2. src only nodes in case of directed graphs) we need to have neighborless nodes and their features included in
      * RootedNodeNeighborhood (required for inference) neighborless nodes can be included in
      * NodeAnchorBasedLinkPredictionSamples, if users want so. Returns
      * subgraphsWithneighborlessNodesVIEW with columns: _root_node, _neighbor_edges, _neighbor_nodes,
      * _node_features,_condensed_node_type st for neighborless nodes _neighbor_edges is null.
      * @param hydratedNodeVIEW with columns
      * _node_id: Int
      * _node_features: Array
      * _condensed_node_type: Int
      * @param unhydratedEdgeVIEW with columns
      * _src_node: Int
      * _dst_node: Int
      */

    val isolatedNodesSubgraphVIEW = createIsolatedNodesSubgraph(
      hydratedNodeVIEW = hydratedNodeVIEW,
      unhydratedEdgeVIEW = unhydratedEdgeVIEW,
    )

    // neighborlessNodeIDs: for undirected graphs will be only isolated nodes
    // and for directed graphs will be isolated nodes + nodes with no in-edges (i.e. srcOnlyNodes, any node that does not form a neighborhood)
    val neighborlessNodesSubgraphVIEW = if (!(gbmlConfigWrapper.sharedConfigPb.isGraphDirected)) {
      isolatedNodesSubgraphVIEW
    } else {
      val srcNodeDF          = spark.table(unhydratedEdgeVIEW).select("_src_node")
      val dstNodeDF          = spark.table(unhydratedEdgeVIEW).select("_dst_node")
      val srcOnlyNodeIdsDF   = srcNodeDF.exceptAll(dstNodeDF).dropDuplicates("_src_node")
      val srcOnlyNodeIdsVIEW = "srcOnlyNodeIdsDF" + uniqueTempViewSuffix
      srcOnlyNodeIdsDF.createOrReplaceTempView(srcOnlyNodeIdsVIEW)
      val srcOnlyNodesSubgraphDF = spark.sql(f"""
          SELECT
            _node_id AS _root_node,
            _node_features,
            _condensed_node_type
          FROM
            ${hydratedNodeVIEW}
          INNER JOIN
            ${srcOnlyNodeIdsVIEW}
          ON
            ${hydratedNodeVIEW}._node_id = ${srcOnlyNodeIdsVIEW}._src_node
        """)
      val srcOnlyNodesSubgraphVIEW = "srcOnlyNodesSubgraphDF" + uniqueTempViewSuffix
      srcOnlyNodesSubgraphDF.createOrReplaceTempView(srcOnlyNodesSubgraphVIEW)

      val isolatedAndSrcOnlySubgraphDF = spark.sql(f"""
          SELECT
            _root_node,
            _node_features,
            _condensed_node_type
          FROM
            ${isolatedNodesSubgraphVIEW} UNION
          SELECT
            _root_node,
            _node_features,
            _condensed_node_type
          FROM
            ${srcOnlyNodesSubgraphVIEW}
        """)
      val isolatedAndSrcOnlySubgraphVIEW = "isolatedAndSrcOnlySubgraphDF" + uniqueTempViewSuffix
      isolatedAndSrcOnlySubgraphDF.createOrReplaceTempView(isolatedAndSrcOnlySubgraphVIEW)
      isolatedAndSrcOnlySubgraphVIEW
    }
    neighborlessNodesSubgraphVIEW
  }

  def createRootedNodeNeighborhoodSubgraph(
    hydratedNodeVIEW: String,
    unhydratedEdgeVIEW: String,
    subgraphVIEW: String,
  ): String = {

    /**
      * Note that solated nodes are always included in RNN samples.
      * @param subgraphVIEW with columns
      * _root_node: Int
      * _neighbor_edges: Array(Struct)
      * _neighbor_nodes: Array(Struct)
      * _node_features: Array
      * _condensed_node_type: Int
      */

    val neighborlessNodesSubgraphVIEW = createNeighborlessNodesSubgraph(
      hydratedNodeVIEW = hydratedNodeVIEW,
      unhydratedEdgeVIEW = unhydratedEdgeVIEW,
    )
    val subgraphsWithNeighborlessNodesDF = spark.sql(f"""
        SELECT
          _root_node,
          _neighbor_edges,
          _neighbor_nodes,
          _node_features,
          _condensed_node_type
        FROM
          ${subgraphVIEW} UNION
        SELECT
          _root_node,
          NULL AS _neighbor_edges,
          ARRAY(STRUCT(_root_node AS _node_id,
              _condensed_node_type,
              _node_features AS _feature_values)) AS _neighbor_nodes,
          _node_features,
          _condensed_node_type
        FROM
          ${neighborlessNodesSubgraphVIEW}
      """)
    val subgraphsWithNeighborlessNodesVIEW =
      "subgraphsWithNeighborlessNodesDF" + uniqueTempViewSuffix
    subgraphsWithNeighborlessNodesDF.createOrReplaceTempView(subgraphsWithNeighborlessNodesVIEW)
    subgraphsWithNeighborlessNodesVIEW
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
      * @param dfVIEW associated with the DataFrame we want to cache
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

  def run():
  /**
    * All sgs task will have these two outputs, which will be written to gcs inside run()
    * 1. inference samples (i.e. RootedNodeNeighborhood) [which can be used as Random negative samples and unlabeled samples too]
    * 2. training samples (e.g. NodeAnchorBasedLinkPrediction, SupervisedNodeClassification)
    */
  Unit
}
