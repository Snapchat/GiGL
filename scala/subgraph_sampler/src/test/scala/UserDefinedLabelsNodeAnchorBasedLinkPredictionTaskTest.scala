import common.src.main.scala.types.EdgeUsageType
import common.test.SharedSparkSession
import common.types.pb_wrappers.GbmlConfigPbWrapper
import common.types.pb_wrappers.GraphMetadataPbWrapper
import common.utils.ProtoLoader.populateProtoFromYaml
import libs.task.pureSpark.UserDefinedLabelsNodeAnchorBasedLinkPredictionTask
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.apache.spark.sql.{functions => F}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers._
import snapchat.research.gbml.gbml_config.GbmlConfig

import java.util.UUID.randomUUID

class UserDefinedLabelsNodeAnchorBasedLinkPredictionTaskTest
    extends AnyFunSuite
    with BeforeAndAfterAll
    with SharedSparkSession {

  // commmon/reused vars among tests which must be assigned in beforeAll():
  var nablpTask: UserDefinedLabelsNodeAnchorBasedLinkPredictionTask = _
  var gbmlConfigWrapper: GbmlConfigPbWrapper                        = _
  var graphMetadataPbWrapper: GraphMetadataPbWrapper                = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    val frozenGbmlConfigUriTest =
      "common/src/test/assets/subgraph_sampler/node_anchor_based_link_prediction/frozen_gbml_config.yaml"
    val gbmlConfigProto =
      populateProtoFromYaml[GbmlConfig](uri = frozenGbmlConfigUriTest)
    gbmlConfigWrapper = GbmlConfigPbWrapper(gbmlConfigPb = gbmlConfigProto)

    val graphMetadataPbWrapper: GraphMetadataPbWrapper = GraphMetadataPbWrapper(
      gbmlConfigWrapper.graphMetadataPb,
    )

    nablpTask = new UserDefinedLabelsNodeAnchorBasedLinkPredictionTask(
      gbmlConfigWrapper = gbmlConfigWrapper,
      graphMetadataPbWrapper = graphMetadataPbWrapper,
      isPosUserDefined = true,
      isNegUserDefined = true,
    )

  }

  def mockNumberOfNodesForCurrentTest: Int = {
    val numNodes: Int = 9
    numNodes
  }

  def mockUnhydratedEdgeForCurrentTest: DataFrame = {
    // Making scope local to avoid clash with scalapb.spark.Implicits._
    import sqlImplicits._
    // Note that below graph is bidirected
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

  def mockUnhydratedUserDefinedPosEdgesForCurrentTest: DataFrame = {
    // Making scope local to avoid clash with scalapb.spark.Implicits._
    import sqlImplicits._
    val userDefinedPosEdgeData = Seq(
      (0, 1),
      (0, 3),
      (0, 2),
      (1, 2),
      (8, 0), // src only
      (3, 1),
    )
    val userDefinePosEdgeDF = userDefinedPosEdgeData.toDF("_src_node", "_dst_node")
    userDefinePosEdgeDF
  }

  def mockUnhydratedUserDefinedNegEdgesForCurrentTest: DataFrame = {
    // Making scope local to avoid clash with scalapb.spark.Implicits._
    import sqlImplicits._
    val userDefinedNegEdgeData = Seq(
      (2, 4),
      (3, 2),
      (6, 7),
      (2, 7),
      (8, 4),
      (2, 6),
    )
    val userDefineNegEdgeDF = userDefinedNegEdgeData.toDF("_src_node", "_dst_node")
    userDefineNegEdgeDF
  }

  def mockHydratedNodeForCurrentTest: DataFrame = {
    // Making scope local to avoid clash with scalapb.spark.Implicits._
    import sqlImplicits._
    // must include isolated node (ie 9,10,11)
    val nodeData = Seq(
      (0, Seq(0.0f), 0),
      (1, Seq(0.1f), 0),
      (2, Seq(0.2f), 0),
      (3, Seq(0.3f), 0),
      (4, Seq(0.4f), 0),
      (5, Seq(0.5f), 0),
      (6, Seq(0.6f), 0),
      (7, Seq(0.7f), 0),
      (8, Seq(0.8f), 0),
      (9, Seq(0.9f), 0),
      (10, Seq(1.0f), 0),
      (11, Seq(1.1f), 0),
    )
    val hydratedNodeDF =
      nodeData.toDF("_node_id", "_node_features", "_condensed_node_type")
    hydratedNodeDF
  }

  def mockSubgraphForCurrentTest: DataFrame = {
    val emptyFeats = Seq.empty[Float]
    val subgraphData = Seq(
      Row(0, List(Row(1, 0, 0, emptyFeats)), List(Row(1, 0, Seq(0.1f))), Seq(0.0f), 0),
      Row(1, List(Row(2, 1, 0, emptyFeats)), List(Row(2, 0, Seq(0.2f))), Seq(0.1f), 0),
      Row(2, List(Row(1, 2, 0, emptyFeats)), List(Row(1, 0, Seq(0.1f))), Seq(0.2f), 0),
      Row(3, List(Row(0, 3, 0, emptyFeats)), List(Row(0, 0, Seq(0.0f))), Seq(0.3f), 0),
      Row(4, List(Row(0, 4, 0, emptyFeats)), List(Row(0, 0, Seq(0.0f))), Seq(0.4f), 0),
      Row(5, List(Row(0, 5, 0, emptyFeats)), List(Row(0, 0, Seq(0.0f))), Seq(0.5f), 0),
      Row(6, List(Row(0, 6, 0, emptyFeats)), List(Row(0, 0, Seq(0.0f))), Seq(0.6f), 0),
      Row(7, List(Row(0, 7, 0, emptyFeats)), List(Row(0, 0, Seq(0.0f))), Seq(0.7f), 0),
      Row(8, List(Row(8, 0, 0, emptyFeats)), List(Row(0, 0, Seq(0.0f))), Seq(0.8f), 0),
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
    val subgraphRDD    = sparkTest.sparkContext.parallelize(subgraphData)
    val mockSubgraphDF = sparkTest.createDataFrame(subgraphRDD, arrayStruct)

    mockSubgraphDF
  }

  def are_user_defined_samples_valid(edgeUsageType: EdgeUsageType.EdgeUsageType): Unit = {
    val uniqueTestViewSuffix = "_" + randomUUID.toString.replace("-", "_")
    val unhydratedEdgeDF =
      if (edgeUsageType == EdgeUsageType.POS) mockUnhydratedUserDefinedPosEdgesForCurrentTest
      else mockUnhydratedUserDefinedNegEdgesForCurrentTest
    val unhydratedEdgeVIEW = s"unhydrated${edgeUsageType}EdgeDF" + uniqueTestViewSuffix
    unhydratedEdgeDF.createOrReplaceTempView(unhydratedEdgeVIEW)

    val numTrainingSamples = mockNumberOfNodesForCurrentTest // number of all nodes in graph
    val numSamples         = 2
    val sampledDstVIEW = nablpTask.sampleDstNodesUniformly(
      numDstSamples = numSamples,
      unhydratedEdgeVIEW = unhydratedEdgeVIEW,
      permutationStrategy = "non-deterministic",
      numTrainingSamples = numTrainingSamples,
      edgeUsageType = edgeUsageType,
    )
    val sampledDstDF = sparkTest.table(sampledDstVIEW)
    // must choose a nodeId st number of its out-edges is > numSamples [to test randomness]
    val nodeId         = if (edgeUsageType == EdgeUsageType.POS) 0 else 2
    val sampleNodeList = if (edgeUsageType == EdgeUsageType.POS) Seq(1, 2, 3) else Seq(4, 6, 7)
    val randomSamplesList = sampledDstDF
      .filter(F.col("_src_node") === nodeId)
      .select(f"_${edgeUsageType}_dst_node")
      .collect
      .map(_(0))
      .toList
    sampleNodeList should contain allElementsOf randomSamplesList
  }

  // test correctness of sampleDstNodesUniformly function for UDL
  test("User Defined Positive & Negative samples are valid") {
    are_user_defined_samples_valid(EdgeUsageType.POS)
    are_user_defined_samples_valid(EdgeUsageType.NEG)
  }

  // test correctness of lookupDstNodeNeighborhood function for UDL
  test("Negative neighbors are valid with correct columns") {
    // Making scope local to avoid clash with scalapb.spark.Implicits._
    val uniqueTestViewSuffix = "_" + randomUUID.toString.replace("-", "_")
    val mockSubgraphDF       = mockSubgraphForCurrentTest
    val mockSubgraphVIEW     = "mockSubgraphDF" + uniqueTestViewSuffix
    mockSubgraphDF.createOrReplaceTempView(mockSubgraphVIEW)
    val negEdgesDF = mockUnhydratedUserDefinedNegEdgesForCurrentTest
    val sampledNegDF =
      negEdgesDF.select(F.col("_src_node"), F.col("_dst_node").alias("_neg_dst_node"))
    val sampledNegVIEW = "sampledNegDF" + uniqueTestViewSuffix
    sampledNegDF.createOrReplaceTempView(sampledNegVIEW)
    val negNeighborhoodVIEW = nablpTask.lookupDstNodeNeighborhood(
      sampledDstNodesVIEW = sampledNegVIEW,
      subgraphVIEW = mockSubgraphVIEW,
      edgeUsageType = EdgeUsageType.NEG,
    )
    val negNeighborhoodDF = sparkTest.table(negNeighborhoodVIEW)
    // ensures the direction is preserved i.e. from _src_node to _pos_node not the other way
    val expectedSrcNodeList = Seq(2, 3, 6, 8)
    val currentSrcNodeList  = negNeighborhoodDF.select("_src_node").collect.map(_(0)).toList
    expectedSrcNodeList should contain allElementsOf currentSrcNodeList
    // ensures neighbors are valid
    // for src node 1, pos dst node is 2
    val negNodeDstId = 7
    val srcNodeId    = 6

    val expectedNeighborEdges = mockSubgraphDF
      .filter(F.col("_root_node") === negNodeDstId)
      .select("_neighbor_edges")
      .first
      .toSeq
    val currentNeighborEdges = negNeighborhoodDF
      .filter(F.col("_src_node") === srcNodeId)
      .select("_neg_neighbor_edges")
      .first
      .toSeq
    expectedNeighborEdges should contain allElementsOf currentNeighborEdges

  }

  test("Src Only nodes added to reference subgraph") {
    // Making scope local to avoid clash with scalapb.spark.Implicits._
    val uniqueTestViewSuffix = "_" + randomUUID.toString.replace("-", "_")

    // initialize assets
    val unhydratedEdgeDF   = mockUnhydratedEdgeForCurrentTest
    val unhydratedEdgeVIEW = "unhydratedEdgeDF" + uniqueTestViewSuffix
    unhydratedEdgeDF.createOrReplaceTempView(unhydratedEdgeVIEW)

    val unhydratedUserDefinedPosEdgesDF   = mockUnhydratedUserDefinedPosEdgesForCurrentTest
    val unhydratedUserDefinedPosEdgesVIEW = "unhydratedUserDefinedPosEdgesDF" + uniqueTestViewSuffix
    unhydratedUserDefinedPosEdgesDF.createOrReplaceTempView(unhydratedUserDefinedPosEdgesVIEW)

    val mockSubgraphDF   = mockSubgraphForCurrentTest
    val mockSubgraphVIEW = "mockSubgraphDF" + uniqueTestViewSuffix
    mockSubgraphDF.createOrReplaceTempView(mockSubgraphVIEW)

    val mockHydratedNodeDF   = mockHydratedNodeForCurrentTest
    val mockHydratedNodeVIEW = "mockHydratedNodeDF" + uniqueTestViewSuffix
    mockHydratedNodeDF.createOrReplaceTempView(mockHydratedNodeVIEW)

    // run function
    val subgraphWithUdlPosSrcOnlyView = nablpTask.addUserDefSrcOnlyNodesToRNNSubgraph(
      unhydratedMainEdgeVIEW = unhydratedEdgeVIEW,
      unhydratedUserDefEdgeVIEW = unhydratedUserDefinedPosEdgesVIEW,
      referenceSubgraphVIEW = mockSubgraphVIEW,
      hydratedNodeVIEW = mockHydratedNodeVIEW,
    )

    // check outputs
    val userDefinedPosSrcNode =
      unhydratedUserDefinedPosEdgesDF.select("_src_node").collect.map(_(0)).toList
    val subgraphWithUdlPosSrcOnlyRootNodes =
      sparkTest.table(subgraphWithUdlPosSrcOnlyView).select("_root_node").collect.map(_(0)).toList

    subgraphWithUdlPosSrcOnlyRootNodes should contain allElementsOf userDefinedPosSrcNode
  }

  // TODO
  // ensure caching sanity, by checking pos samples and hydrated nei, in integration test

  // TODO(yliu2) maybe can create test to check each sample subgraph output
  // check resulting subgraph pos is in nodes, hard neg is in nodes
}
