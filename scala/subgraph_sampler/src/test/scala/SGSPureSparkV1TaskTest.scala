import common.src.main.scala.types.EdgeUsageType
import common.test.SharedSparkSession
import common.types.pb_wrappers.GbmlConfigPbWrapper
import common.utils.ProtoLoader.populateProtoFromYaml
import libs.task.pureSpark.SGSPureSparkV1Task
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.apache.spark.sql.{functions => F}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers._
import snapchat.research.gbml.gbml_config.GbmlConfig

import java.util.UUID.randomUUID

class SGSPureSparkV1TaskTest extends AnyFunSuite with BeforeAndAfterAll with SharedSparkSession {

  import sqlImplicits._
  // commmon/reused vars among tests which must be assigned in beforeAll():
  var sgsTask: SGSPureSparkV1Task            = _
  var gbmlConfigWrapper: GbmlConfigPbWrapper = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    val frozenGbmlConfigUriTest =
      "common/src/test/assets/subgraph_sampler/node_anchor_based_link_prediction/frozen_gbml_config.yaml"
    val gbmlConfigProto =
      populateProtoFromYaml[GbmlConfig](uri = frozenGbmlConfigUriTest)
    gbmlConfigWrapper = GbmlConfigPbWrapper(gbmlConfigPb = gbmlConfigProto)
    class MockSGSPureSparkV1Task(
      gbmlConfigWrapper: GbmlConfigPbWrapper)
        extends SGSPureSparkV1Task(gbmlConfigWrapper) {
      def applyCachingToSubgraphDf(
        dfVIEW: String,
        withRepartition: Boolean,
      ): String = ???
      def run() = ???
    }
    sgsTask = new MockSGSPureSparkV1Task(gbmlConfigWrapper = gbmlConfigWrapper)

  }

  def mockUnhydratedEdgeForCurrentTest: DataFrame = {
    val edgeData = Seq(
      (0, 1),
      (0, 2),
      (0, 3),
      (0, 4),
      (0, 5),
      (0, 6),
      (0, 7),
      (0, 8),
      (1, 2),
      (1, 3),
      (1, 0),
      (2, 0),
      (3, 0),
      (4, 0),
      (5, 0),
      (6, 0),
      (7, 0),
      (8, 0),
      (2, 1),
      (3, 1),
    )

    val unhydratedEdgeDF = edgeData.toDF("_src_node", "_dst_node")
    unhydratedEdgeDF
  }

  def mockHydratedEdgeForCurrentTest: DataFrame = {
    // must have same edges as edgeData in mockUnhydratedEdgeForCurrentTest
    val hydratedEdgeData = Seq(
      (0, 1, Seq(0.5), 0),
      (0, 2, Seq(1.0), 0),
      (0, 3, Seq(1.5), 0),
      (0, 4, Seq(2.0), 0),
      (0, 5, Seq(2.5), 0),
      (0, 6, Seq(3.0), 0),
      (0, 7, Seq(3.5), 0),
      (0, 8, Seq(4.0), 0),
      (1, 2, Seq(1.5), 0),
      (1, 3, Seq(2.0), 0),
      (1, 0, Seq(0.5), 0),
      (2, 0, Seq(1.0), 0),
      (3, 0, Seq(1.5), 0),
      (4, 0, Seq(2.0), 0),
      (5, 0, Seq(2.5), 0),
      (6, 0, Seq(3.0), 0),
      (7, 0, Seq(3.5), 0),
      (8, 0, Seq(4.0), 0),
      (2, 1, Seq(1.5), 0),
      (3, 1, Seq(2.0), 0),
    )

    val hydratedEdgeDF = hydratedEdgeData
      .toDF("_from", "_to", "_edge_features", "_condensed_edge_type")
    hydratedEdgeDF
  }

  def mockHydratedNodeForCurrentTest: DataFrame = {
    // must include isolated node (ie 9,10,11)
    val nodeData = Seq(
      (0, 0.0, 0),
      (1, 0.1, 0),
      (2, 0.2, 0),
      (3, 0.3, 0),
      (4, 0.4, 0),
      (5, 0.5, 0),
      (6, 0.6, 0),
      (7, 0.7, 0),
      (8, 0.8, 0),
      (9, 0.9, 0),
      (10, 0.01, 0),
      (11, 0.11, 0),
    )
    val hydratedNodeDF =
      nodeData.toDF("_node_id", "_node_features", "_condensed_node_type")
    hydratedNodeDF
  }

  test("Node Dataframe with features and node type loads with correct cloumns.") {
    val hydratedNodeVIEW = sgsTask.loadNodeDataframeIntoSparkSql(condensedNodeType = 0)
    val hydratedNodeDF   = sparkTest.table(hydratedNodeVIEW)
    val expectedColSize  = 3
    assert(hydratedNodeDF.columns.size == expectedColSize)
    val loadedNodeDfColumns: Seq[String] =
      hydratedNodeDF.columns.toSeq
    assert(loadedNodeDfColumns.contains("_node_id"))
    assert(loadedNodeDfColumns.contains("_node_features"))
    assert(loadedNodeDfColumns.contains("_condensed_node_type"))
    // need to check dtypes too? how to find field's dtype from proto easily?
    // print(graph_schema.Node.scalaDescriptor.fields(0))
  }

  def is_edge_dataframe_loaded_with_correct_columns(
    edgeUsageType: EdgeUsageType.EdgeUsageType,
    check_self_loops: Boolean = false,
  ): Unit = {
    val hydratedEdgeVIEW =
      sgsTask.loadEdgeDataframeIntoSparkSql(condensedEdgeType = 0, edgeUsageType = edgeUsageType)
    val hydratedEdgeDF  = sparkTest.table(hydratedEdgeVIEW)
    val expectedColSize = 4
    assert(hydratedEdgeDF.columns.size == expectedColSize)
    val hydratedEdgeDfCols: Seq[String] = hydratedEdgeDF.columns.toSeq
    assert(hydratedEdgeDfCols.contains("_from"))
    assert(hydratedEdgeDfCols.contains("_to"))
    assert(hydratedEdgeDfCols.contains("_condensed_edge_type"))
    assert(hydratedEdgeDfCols.contains("_edge_features"))

    val unhydratedEdgeVIEW =
      sgsTask.loadUnhydratedEdgeDataframeIntoSparkSql(hydratedEdgeVIEW = hydratedEdgeVIEW)
    val unhydratedEdgeDF            = sparkTest.table(unhydratedEdgeVIEW)
    val unhydratedDfExpectedColSize = 2
    assert(unhydratedEdgeDF.columns.size == unhydratedDfExpectedColSize)
    val unhydratedEdgeDfCols = unhydratedEdgeDF.columns.toSeq
    assert(unhydratedEdgeDfCols.contains("_src_node"))
    assert(unhydratedEdgeDfCols.contains("_dst_node"))

    assert(gbmlConfigWrapper.sharedConfigPb.isGraphDirected == false)

    if (check_self_loops) {
      var srcIds: List[Any] = hydratedEdgeDF.sort("_from").select("_from").collect.map(_(0)).toList
      var dstIds: List[Any] = hydratedEdgeDF.sort("_to").select("_to").collect.map(_(0)).toList
      srcIds shouldBe dstIds

      srcIds = unhydratedEdgeDF.sort("_src_node").select("_src_node").collect.map(_(0)).toList
      dstIds = unhydratedEdgeDF.sort("_dst_node").select("_dst_node").collect.map(_(0)).toList
      srcIds shouldBe dstIds
      // graph should not include any self loops
      val selfLoopIds = hydratedEdgeDF.filter(F.col("_from") === F.col("_to"))
      assert(selfLoopIds.count() == 0)
    }

  }

  test("Edge Dataframes are loaded with correct columns and bidirectionalized if required.") {
    is_edge_dataframe_loaded_with_correct_columns(
      edgeUsageType = EdgeUsageType.MAIN,
      check_self_loops = true,
    )
    is_edge_dataframe_loaded_with_correct_columns(edgeUsageType = EdgeUsageType.POS)
    is_edge_dataframe_loaded_with_correct_columns(edgeUsageType = EdgeUsageType.NEG)
    // TODO perhaps add asserts on the content itself
    // TODO test if 3 edge dataframes are loaded with correct directions in scenarios for directed vs undirected graphs.
    //    val hydratedEdgeVIEW = sgsTask.loadEdgeDataframeIntoSparkSql(condensedEdgeType = 0)
  }

  test("Onehop samples are valid.") {
    val uniqueTestViewSuffix = "_" + randomUUID.toString.replace("-", "_")
    val unhydratedEdgeDF     = mockUnhydratedEdgeForCurrentTest
    val unhydratedEdgeVIEW   = "unhydratedEdgeDF" + uniqueTestViewSuffix
    unhydratedEdgeDF.createOrReplaceTempView(unhydratedEdgeVIEW)
    var numNeighborsToSample = 3
    val sampledOnehopVIEW =
      sgsTask.sampleOnehopSrcNodesUniformly(
        numNeighborsToSample = numNeighborsToSample,
        unhydratedEdgeVIEW = unhydratedEdgeVIEW,
        permutationStrategy = "non-deterministic",
      )
    val sampledOnehopDF = sparkTest.table(sampledOnehopVIEW)
    // must choose a nodeId st number of its in-edges is > numNeighborsToSample [to test randomness]
    var nodeId   = 0
    val nodeList = Seq(1, 2, 3, 4, 5, 6, 7, 8)
    val randomSamplesList = sampledOnehopDF
      .filter(F.col("_0_hop") === nodeId)
      .select("_sampled_1_hop_arr")
      .first
      .getSeq[Integer](0)
    nodeList should contain allElementsOf randomSamplesList
    assert(randomSamplesList.length == numNeighborsToSample)
  }

  test("Onehop samples with replacement are valid.") {
    val uniqueTestViewSuffix = "_" + randomUUID.toString.replace("-", "_")
    val unhydratedEdgeDF     = mockUnhydratedEdgeForCurrentTest
    val unhydratedEdgeVIEW   = "unhydratedEdgeDF" + uniqueTestViewSuffix
    unhydratedEdgeDF.createOrReplaceTempView(unhydratedEdgeVIEW)
    var numNeighborsToSample = 10
    val sampledOnehopVIEW =
      sgsTask.sampleOnehopSrcNodesUniformly(
        numNeighborsToSample = numNeighborsToSample,
        unhydratedEdgeVIEW = unhydratedEdgeVIEW,
        permutationStrategy = "non-deterministic",
        sampleWithReplacement = true,
      )
    val sampledOnehopDF = sparkTest.table(sampledOnehopVIEW)
    // must choose a nodeId st number of its in-edges is > numNeighborsToSample [to test randomness]
    var nodeId   = 0
    val nodeList = Seq(1, 2, 3, 4, 5, 6, 7, 8)
    val randomSamplesList = sampledOnehopDF
      .filter(F.col("_0_hop") === nodeId)
      .select("_sampled_1_hop_arr")
      .first
      .getSeq[Integer](0)
    nodeList should contain allElementsOf randomSamplesList
    assert(randomSamplesList.length == numNeighborsToSample)
  }

  test("Twohop samples are valid.") {
    val uniqueTestViewSuffix = "_" + randomUUID.toString.replace("-", "_")
    val unhydratedEdgeDF     = mockUnhydratedEdgeForCurrentTest
    val unhydratedEdgeVIEW   = "unhydratedEdgeDF" + "_" + uniqueTestViewSuffix
    unhydratedEdgeDF.createOrReplaceTempView(unhydratedEdgeVIEW)
    val onehopData      = Seq((1, Seq(3, 0, 2)), (3, Seq(0, 1)), (2, Seq(0, 1)), (0, Seq(1, 3, 2)))
    val sampledOnehopDF = onehopData.toDF("_0_hop", "_sampled_1_hop_arr")
    val sampledOnehopVIEW = "sampledOnehopDF" + uniqueTestViewSuffix
    sampledOnehopDF.createOrReplaceTempView(sampledOnehopVIEW)
    var numNeighborsToSample = 3

    val sampledTwohopVIEW = sgsTask.sampleTwohopSrcNodesUniformly(
      numNeighborsToSample = numNeighborsToSample,
      unhydratedEdgeVIEW = unhydratedEdgeVIEW,
      sampledOnehopVIEW = sampledOnehopVIEW,
      permutationStrategy = "non-deterministic",
    )
    val sampledTwohopDF = sparkTest.table(sampledTwohopVIEW)
    // must choose a nodeId st number of its in-edges is <= numNeighborsToSample [no randomness]
    var zerohopId = 1
    // must choose a nodeId st number of its in-edges is > numNeighborsToSample [to test randomness]
    var onehopId   = 0
    var twohopList = Seq(1, 2, 3, 4, 5, 6, 7, 8)
    val randomTwohopSamplesList = sampledTwohopDF
      .filter(F.col("_0_hop") === zerohopId && F.col("_1_hop") === onehopId)
      .select("_sampled_2_hop_arr")
      .first
      .getSeq[Integer](0)
    twohopList should contain allElementsOf randomTwohopSamplesList
    assert(randomTwohopSamplesList.length == numNeighborsToSample)
  }

  test("Twohop samples with replacement are valid.") {
    val uniqueTestViewSuffix = "_" + randomUUID.toString.replace("-", "_")
    val unhydratedEdgeDF     = mockUnhydratedEdgeForCurrentTest
    val unhydratedEdgeVIEW   = "unhydratedEdgeDF" + uniqueTestViewSuffix
    unhydratedEdgeDF.createOrReplaceTempView(unhydratedEdgeVIEW)
    val onehopData = Seq((1, Seq(3, 0, 2)), (3, Seq(0, 1, 1)), (2, Seq(0, 1, 0)), (0, Seq(1, 3, 2)))
    val sampledOnehopDF   = onehopData.toDF("_0_hop", "_sampled_1_hop_arr")
    val sampledOnehopVIEW = "sampledOnehopDF" + uniqueTestViewSuffix
    sampledOnehopDF.createOrReplaceTempView(sampledOnehopVIEW)
    var numNeighborsToSample = 10

    val sampledTwohopVIEW = sgsTask.sampleTwohopSrcNodesUniformly(
      numNeighborsToSample = numNeighborsToSample,
      unhydratedEdgeVIEW = unhydratedEdgeVIEW,
      sampledOnehopVIEW = sampledOnehopVIEW,
      permutationStrategy = "non-deterministic",
      sampleWithReplacement = true,
    )
    val sampledTwohopDF = sparkTest.table(sampledTwohopVIEW)
    // must choose a nodeId st number of its in-edges is <= numNeighborsToSample [no randomness]
    var zerohopId = 1
    // must choose a nodeId st number of its in-edges is > numNeighborsToSample [to test randomness]
    var onehopId   = 0
    var twohopList = Seq(1, 2, 3, 4, 5, 6, 7, 8)
    val randomTwohopSamplesList = sampledTwohopDF
      .filter(F.col("_0_hop") === zerohopId && F.col("_1_hop") === onehopId)
      .select("_sampled_2_hop_arr")
      .first
      .getSeq[Integer](0)
    twohopList should contain allElementsOf randomTwohopSamplesList
    assert(randomTwohopSamplesList.length == numNeighborsToSample)
  }

  test("Hydrated kth hop nodes have right col and node ids") {
    val uniqueTestViewSuffix = "_" + randomUUID.toString.replace("-", "_")
    val edgeData = Seq(
      (0, 1),
      (0, 2),
      (0, 3),
      (1, 2),
      (1, 3),
      (1, 0),
      (2, 0),
      (3, 0),
      (2, 1),
      (3, 1),
    )
    val unhydratedEdgeDF   = edgeData.toDF("_src_node", "_dst_node")
    val unhydratedEdgeVIEW = "unhydratedEdgeDF" + uniqueTestViewSuffix
    unhydratedEdgeDF.createOrReplaceTempView(unhydratedEdgeVIEW)

    val nodeData = Seq(
      (0, 0.0, 0),
      (1, 0.1, 0),
      (2, 0.2, 0),
      (3, 0.3, 0),
    )
    val hydratedNodeDF =
      nodeData.toDF("_node_id", "_node_features", "_condensed_node_type")
    val hydratedNodeVIEW = "hydratedNodeDF" + uniqueTestViewSuffix
    hydratedNodeDF.createOrReplaceTempView(hydratedNodeVIEW)

    val onehopData      = Seq((1, Seq(3, 0, 2)), (3, Seq(0, 1)), (2, Seq(0, 1)), (0, Seq(1, 3, 2)))
    val sampledOnehopDF = onehopData.toDF("_0_hop", "_sampled_1_hop_arr")
    val sampledOnehopVIEW = "sampledOnehopDF" + uniqueTestViewSuffix
    sampledOnehopDF.createOrReplaceTempView(sampledOnehopVIEW)

    val twohopData = Seq(
      (1, 0, Seq(1, 2, 3)),
      (3, 1, Seq(0, 3, 2)),
      (1, 2, Seq(0, 1)),
      (1, 3, Seq(0, 1)),
      (2, 1, Seq(2, 0, 3)),
      (0, 1, Seq(2, 3, 0)),
      (2, 0, Seq(3, 2, 1)),
      (0, 2, Seq(0, 1)),
      (0, 3, Seq(1, 0)),
      (3, 0, Seq(2, 3, 1)),
    )
    val sampledTwohopDF =
      twohopData.toDF("_0_hop", "_1_hop", "_sampled_2_hop_arr")
    val sampledTwohopVIEW = "sampledTwohopDF" + uniqueTestViewSuffix
    sampledTwohopDF.createOrReplaceTempView(sampledTwohopVIEW)

    // must be <= max num out edges of unhydratedEdgeDF [we don't want to test randomness here]
    var numNeighborsToSample = 3
    // check hydrated onehop sanity
    val hydratedOnehopNodesVIEW = sgsTask.hydrateNodes(
      k = 1,
      hydratedNodeVIEW = hydratedNodeVIEW,
      sampledKhopVIEW = sampledOnehopVIEW,
    )
    val hydratedOnehopNodesDF = sparkTest.table(hydratedOnehopNodesVIEW)
    val expectedOnehopColNames: Seq[String] =
      Seq("_node_id", "_node_features", "_condensed_node_type", "_0_hop", "_1_hop")
    val loadedOnehopColNames: Seq[String] = hydratedOnehopNodesDF.columns.toSeq
    expectedOnehopColNames should contain theSameElementsAs loadedOnehopColNames

    var diffDF = hydratedOnehopNodesDF.filter(!(F.col("_node_id").contains(F.col("_1_hop"))))
    assert(diffDF.count() == 0)

    // check hydrated twohop sanity
    val hydratedTwohopNodesVIEW = sgsTask.hydrateNodes(
      k = 2,
      hydratedNodeVIEW = hydratedNodeVIEW,
      sampledKhopVIEW = sampledTwohopVIEW,
    )
    val hydratedTwohopNodesDF = sparkTest.table(hydratedTwohopNodesVIEW)
    val expectedTwohopColNames: Seq[String] =
      Seq("_node_id", "_node_features", "_condensed_node_type", "_0_hop", "_1_hop", "_2_hop")
    val loadedTwohopColNames: Seq[String] = hydratedTwohopNodesDF.columns.toSeq
    expectedTwohopColNames should contain theSameElementsAs loadedTwohopColNames

    diffDF = hydratedTwohopNodesDF.filter(!(F.col("_node_id").contains(F.col("_2_hop"))))
    assert(diffDF.count() == 0)

  }

  test("Rooted Node Neighborhood is valid.") {
    // delete all prev view names from current sparkTest session
    sparkTest.sqlContext.tableNames().foreach(sparkTest.catalog.dropTempView(_))
    val uniqueTestViewSuffix = "_" + randomUUID.toString.replace("-", "_")

    val unhydratedEdgeDF   = mockUnhydratedEdgeForCurrentTest
    val unhydratedEdgeVIEW = "unhydratedEdgeDF" + uniqueTestViewSuffix
    unhydratedEdgeDF.createOrReplaceTempView(unhydratedEdgeVIEW)

    val hydratedEdgeDF   = mockHydratedEdgeForCurrentTest
    val hydratedEdgeVIEW = "hydratedEdgeDF" + uniqueTestViewSuffix
    hydratedEdgeDF.createOrReplaceTempView(hydratedEdgeVIEW)

    val hydratedNodeDF   = mockHydratedNodeForCurrentTest
    val hydratedNodeVIEW = "hydratedNodeDF" + uniqueTestViewSuffix
    hydratedNodeDF.createOrReplaceTempView(hydratedNodeVIEW)

    var numNeighborsToSample = 3
    val rnnVIEW = sgsTask.createSubgraph(
      numNeighborsToSample = numNeighborsToSample,
      hydratedNodeVIEW = hydratedNodeVIEW,
      hydratedEdgeVIEW = hydratedEdgeVIEW,
      unhydratedEdgeVIEW = unhydratedEdgeVIEW,
      permutationStrategy = "non-detrministic",
    )
    val rnnDF = sparkTest.table(rnnVIEW)
    // sparkTest.table(rnnVIEW).show(50, truncate = false)
    // sparkTest.table(rnnVIEW).printSchema()

    // view names parametrized by hop number are valid
    val parametrizedViews = Array(
      "1hopdf",
      "2hopdf",
      "hydrated1hopnodesdf",
      "hydrated2hopnodesdf",
      "hydrated1thhopdf",
      "hydrated2thhopdf",
      "hydrated1hopnodesedgesdf",
      "hydrated2hopnodesedgesdf",
    )
    val expectedParametrizedViews: Array[String] =
      parametrizedViews.map(_ + sgsTask.uniqueTempViewSuffix)
    val curViews = sparkTest.sqlContext.tableNames()
    curViews should contain allElementsOf expectedParametrizedViews
    // edges are valid, i.e neighborhood is a valid subgraph. [ensures caching sampledDF is done correctly]
    // 1. MUST choose a root node with in edges > NumNeighborsToSample
    val rootNodeId = 0
    // 2. for rootNodeId, take 1hop src_node ids
    val onehopSrcIds: Seq[Int] = rnnDF
      .filter(F.col("_root_node") === rootNodeId)
      .select(
        F.filter(F.col("_neighbor_edges"), c => c.apply("_src_node") !== rootNodeId)
          .getItem("_src_node"),
      )
      .first
      .getSeq[Int](0)
      .take(numNeighborsToSample)
    // 3. for each of onehopSrcIds verify that twohopDstId exists
    val firstSampledSrcId = onehopSrcIds.apply(0)
    var twohopDst: DataFrame = rnnDF
      .filter(F.col("_root_node") === rootNodeId)
      .select(F.filter(F.col("_neighbor_edges"), c => c.apply("_dst_node") === firstSampledSrcId))
    assert(twohopDst.count() == 1)

    val secondSampledSrcId = onehopSrcIds.apply(1)
    twohopDst = rnnDF
      .filter(F.col("_root_node") === rootNodeId)
      .select(F.filter(F.col("_neighbor_edges"), c => c.apply("_dst_node") === secondSampledSrcId))
    assert(twohopDst.count() == 1)

    val thirdSampledSrcId = onehopSrcIds.apply(2)
    twohopDst = rnnDF
      .filter(F.col("_root_node") === rootNodeId)
      .select(F.filter(F.col("_neighbor_edges"), c => c.apply("_dst_node") === thirdSampledSrcId))
    assert(twohopDst.count() == 1)

    // verify node ids in _neighbor_nodes match with node ids in _neighbor_edges
    val srcIdList = rnnDF
      .filter(F.col("_root_node") === rootNodeId)
      .select(F.col("_neighbor_edges").getItem("_src_node"))
      .first
      .getSeq[Integer](0)
    val dstIdList = rnnDF
      .filter(F.col("_root_node") === rootNodeId)
      .select(F.col("_neighbor_edges").getItem("_dst_node"))
      .first
      .getSeq[Integer](0)
    var allIdsFromEdges: Seq[Integer] = srcIdList ++ dstIdList
    allIdsFromEdges = allIdsFromEdges.distinct
    val allIdsFromNodes: Seq[Integer] = rnnDF
      .filter(F.col("_root_node") === rootNodeId)
      .select(F.col("_neighbor_nodes").getItem("_node_id"))
      .first
      .getSeq[Integer](0)
    allIdsFromEdges.sorted shouldBe allIdsFromNodes.sorted

    // root nodes are included in neighborhood
    val rootNodeIdDF = rnnDF
      .filter(F.col("_root_node") === rootNodeId)
      .select(
        F.filter(F.col("_neighbor_nodes"), c => c.apply("_node_id") === rootNodeId)
          .alias("_curr_root_node"),
      )
    val hydratedRootNodeId: Integer = rootNodeIdDF
      .select(F.col("_curr_root_node").getItem("_node_id"))
      .first
      .getSeq[Integer](0)
      .apply(0)
    val hydratedRootNodeFeature: Double = rootNodeIdDF
      .select(F.col("_curr_root_node").getItem("_feature_values"))
      .first
      .getSeq[Double](0)
      .apply(0)
    assert(hydratedRootNodeId == rootNodeId)
    assert(hydratedRootNodeFeature == 0.0)

  }
  // TODO: add unittest here for directed graphs to check if neighborless nodes are all included
  test("Isolated nodes are included in inference/rooted node neighbor subgraph DF.") {
    val uniqueTestViewSuffix = "_" + randomUUID.toString.replace("-", "_")
    val edgeData             = Seq((0, 1), (0, 2), (0, 3), (1, 3), (1, 0), (2, 0), (3, 0), (3, 1))
    val edgeRDD              = sparkTest.sparkContext.parallelize(edgeData)
    val unhydratedEdgeDF     = sparkTest.createDataFrame(edgeRDD).toDF("_src_node", "_dst_node")
    val unhydratedEdgeVIEW   = "unhydratedEdgeDF" + uniqueTestViewSuffix
    unhydratedEdgeDF.createOrReplaceTempView(unhydratedEdgeVIEW)
    val nodeData = Seq(
      (0, Seq(0.0f), 0),
      (1, Seq(0.1f), 0),
      (2, Seq(0.2f), 0),
      (3, Seq(0.3f), 0),
      (4, Seq(0.4f), 0),
      (5, Seq(0.5f), 0),
    )
    val nodeRDD = sparkTest.sparkContext.parallelize(nodeData)
    val hydratedNodeDF =
      sparkTest.createDataFrame(nodeRDD).toDF("_node_id", "_node_features", "_condensed_node_type")
    val hydratedNodeVIEW = "hydratedNodeDF" + uniqueTestViewSuffix
    hydratedNodeDF.createOrReplaceTempView(hydratedNodeVIEW)

    // note that subgraph data does not form an actual graph, only root_id and dtypes matter for this test
    val emptyFeats = Seq.empty[Float]
    val subgraphData = Seq(
      Row(0, List(Row(1, 0, 0, emptyFeats)), List(Row(1, 0, Seq(0.1f))), Seq(0.0f), 0),
      Row(1, List(Row(1, 0, 0, emptyFeats)), List(Row(1, 0, Seq(0.1f))), Seq(0.0f), 0),
      Row(2, List(Row(1, 0, 0, emptyFeats)), List(Row(1, 0, Seq(0.1f))), Seq(0.0f), 0),
      Row(3, List(Row(1, 0, 0, emptyFeats)), List(Row(1, 0, Seq(0.1f))), Seq(0.0f), 0),
    )
    val arrayStruct = new StructType()
      .add("_root_node", IntegerType)
      .add(
        "_neighbor_edges",
        ArrayType(
          new StructType()
            .add("_src_node", IntegerType)
            .add("_dst_node", IntegerType)
            .add("_condensed_edge_type", IntegerType)
            .add("_feature_values", ArrayType(FloatType)),
        ),
      )
      .add(
        "_neighbor_nodes",
        ArrayType(
          new StructType()
            .add("_node_id", IntegerType)
            .add("_condensed_node_type", IntegerType)
            .add("_feature_values", ArrayType(FloatType)),
        ),
      )
      .add("_node_features", ArrayType(FloatType))
      .add("_condensed_node_type", IntegerType)
    val subgraphRDD      = sparkTest.sparkContext.parallelize(subgraphData)
    val mockSubgraphDF   = sparkTest.createDataFrame(subgraphRDD, arrayStruct)
    val mockSubgraphVIEW = "mockSubgraphDF" + uniqueTestViewSuffix
    mockSubgraphDF.createOrReplaceTempView(mockSubgraphVIEW)
    val subgraphWithIsolatedNodesVIEW = sgsTask.createRootedNodeNeighborhoodSubgraph(
      hydratedNodeVIEW = hydratedNodeVIEW,
      unhydratedEdgeVIEW = unhydratedEdgeVIEW,
      subgraphVIEW = mockSubgraphVIEW,
    )
    val subgraphWithIsolatedNodesDF = sparkTest.table(subgraphWithIsolatedNodesVIEW)
    // subgraphWithIsolatedNodesDF.show(truncate = false)
    val expectedIsolatedNodeList = Seq(4, 5)
    val curIsolatedNodesList = subgraphWithIsolatedNodesDF
      .filter(F.col("_neighbor_edges").isNull)
      .select("_root_node")
      .collect
      .map(_(0))
      .toList
    expectedIsolatedNodeList should contain allElementsOf curIsolatedNodesList
  }

  test("sampleWithReplacementUDF returns correct number of samples") {
    // Use the already registered UDF from SGSPureSparkV1Task
    val sampleWithReplacementUDF = sgsTask.sampleWithReplacementUDF

    // Create a DataFrame with sample data
    val data = Seq(
      (Seq(1, 2, 3, 4, 5), 10),
      (Seq(6, 7, 8, 9, 10), 2),
      (Seq.empty[Int], 3),
      (null, 3),
    ).toDF("array", "numSamples")

    // Apply the UDF
    val resultDF =
      data.withColumn("samples", sampleWithReplacementUDF(F.col("array"), F.col("numSamples")))

    // Collect the results
    val results = resultDF.collect()

    // Write assertions
    assert(results(0).getAs[Seq[Int]]("samples").length == 10)
    assert(results(1).getAs[Seq[Int]]("samples").length == 2)
    assert(results(2).getAs[Seq[Int]]("samples").isEmpty)
    assert(results(3).getAs[Seq[Int]]("samples").isEmpty)
  }

}
