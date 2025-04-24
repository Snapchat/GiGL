import common.src.main.scala.types.EdgeUsageType
import common.test.SharedSparkSession
import common.types.pb_wrappers.GbmlConfigPbWrapper
import common.types.pb_wrappers.GraphMetadataPbWrapper
import common.utils.ProtoLoader.populateProtoFromYaml
import libs.task.TaskOutputValidator
import libs.task.pureSpark.NodeAnchorBasedLinkPredictionTask
import org.apache.spark.SparkException
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.apache.spark.sql.{functions => F}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers._
import scalapb.spark.Implicits._
import scalapb.spark.Implicits._
import snapchat.research.gbml.gbml_config.GbmlConfig
import snapchat.research.gbml.graph_schema.Edge
import snapchat.research.gbml.graph_schema.Graph
import snapchat.research.gbml.graph_schema.Node
import snapchat.research.gbml.training_samples_schema.NodeAnchorBasedLinkPredictionSample

import java.util.UUID.randomUUID

class NodeAnchorBasedLinkPredictionTaskTest
    extends AnyFunSuite
    with BeforeAndAfterAll
    with SharedSparkSession {

  // commmon/reused vars among tests which must be assigned in beforeAll():
  var nablpTask: NodeAnchorBasedLinkPredictionTask   = _
  var gbmlConfigWrapper: GbmlConfigPbWrapper         = _
  var graphMetadataPbWrapper: GraphMetadataPbWrapper = _

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

    nablpTask = new NodeAnchorBasedLinkPredictionTask(
      gbmlConfigWrapper = gbmlConfigWrapper,
      graphMetadataPbWrapper = graphMetadataPbWrapper,
    )

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
      Row(8, List(Row(0, 8, 0, emptyFeats)), List(Row(0, 0, Seq(0.0f))), Seq(0.8f), 0),
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

  test("Positive samples are valid.") {
    val uniqueTestViewSuffix = "_" + randomUUID.toString.replace("-", "_")
    val unhydratedEdgeDF     = mockUnhydratedEdgeForCurrentTest
    val unhydratedEdgeVIEW   = "unhydratedEdgeDF" + uniqueTestViewSuffix
    unhydratedEdgeDF.createOrReplaceTempView(unhydratedEdgeVIEW)

    val numTrainingSamples =
      unhydratedEdgeDF.distinct().count().toInt // number of all nodes in graph
    val numPositiveSamples = 2
    val sampledPosVIEW = nablpTask.sampleDstNodesUniformly(
      numDstSamples = numPositiveSamples,
      unhydratedEdgeVIEW = unhydratedEdgeVIEW,
      permutationStrategy = "non-deterministic",
      numTrainingSamples = numTrainingSamples,
      edgeUsageType = EdgeUsageType.POS,
    )
    val sampledPosDF = sparkTest.table(sampledPosVIEW)
    // must choose a nodeId st number of its out-edges is > numPositiveSamples [to test randomness]
    val nodeId      = 0
    val posNodeList = Seq(1, 2, 3, 4, 5, 6, 7, 8)
    val randomPosSamplesList = sampledPosDF
      .filter(F.col("_src_node") === nodeId)
      .select("_pos_dst_node")
      .collect
      .map(_(0))
      .toList
    posNodeList should contain allElementsOf randomPosSamplesList

  }

  test("Positive node neighbors are valid with correct columns.") {
    // Making scope local to avoid clash with scalapb.spark.Implicits._
    import sqlImplicits._
    val uniqueTestViewSuffix = "_" + randomUUID.toString.replace("-", "_")
    val mockSubgraphDF       = mockSubgraphForCurrentTest
    val mockSubgraphVIEW     = "mockSubgraphDF" + uniqueTestViewSuffix
    mockSubgraphDF.createOrReplaceTempView(mockSubgraphVIEW)
    val sampledPosData = Seq((0, 1), (0, 3), (0, 2), (1, 2), (4, 0))
    val sampledPosDF   = sampledPosData.toDF("_src_node", "_pos_dst_node")
    val sampledPosVIEW = "sampledPosDF" + uniqueTestViewSuffix
    sampledPosDF.createOrReplaceTempView(sampledPosVIEW)
    val posNeighborhoodVIEW = nablpTask.lookupDstNodeNeighborhood(
      sampledDstNodesVIEW = sampledPosVIEW,
      subgraphVIEW = mockSubgraphVIEW,
      edgeUsageType = EdgeUsageType.POS,
    )
    val posNeighborhoodDF = sparkTest.table(posNeighborhoodVIEW)
    // sparkTest.table(posNeighborhoodVIEW).show()
    // ensures the direction is preserved i.e. from _src_node to _pos_node not the other way
    val expectedSrcNodeList = Seq(1, 4, 0)
    val currentSrcNodeList  = posNeighborhoodDF.select("_src_node").collect.map(_(0)).toList
    expectedSrcNodeList should contain allElementsOf currentSrcNodeList
    // ensures neighbors are valid
    // for src node 1, pos dst node is 2
    val posNodeDstId = 2
    val srcNodeId    = 1
    val expectedNeighborEdges = mockSubgraphDF
      .filter(F.col("_root_node") === posNodeDstId)
      .select("_neighbor_edges")
      .first
      .toSeq
    val currentNeighborEdges = posNeighborhoodDF
      .filter(F.col("_src_node") === srcNodeId)
      .select("_pos_neighbor_edges")
      .first
      .toSeq
    expectedNeighborEdges should contain allElementsOf currentNeighborEdges
  }

  test("validation fails if nodes of supervision edges not present in neighborhood nodes") {
    val rootNode = Node(nodeId = 0)

    // Create the nodes for neighborhood
    val neighborhoodNodes = Seq(
      Node(nodeId = 0),
      Node(nodeId = 1),
      Node(nodeId = 2),
    )

    // Create the edges for neighborhood
    val neighborhoodEdges = Seq(
      Edge(srcNodeId = 0, dstNodeId = 1),
      Edge(srcNodeId = 1, dstNodeId = 2),
    )

    // Create the pos_edges (with one node of the edge not in neighborhood nodes)
    val posEdges = Seq(
      Edge(srcNodeId = 0, dstNodeId = 3),
    )

    // Create the NodeAnchorBasedLinkPredictionSample dataset
    val sample = Seq(
      NodeAnchorBasedLinkPredictionSample(
        rootNode = Some(rootNode),
        neighborhood = Some(Graph(nodes = neighborhoodNodes, edges = neighborhoodEdges)),
        posEdges = posEdges,
      ),
    )

    val sampleDS = sparkTest.createDataset(sample)

    // validateMainSamples method is not an action, performing collect for triggering computation
    assertThrows[SparkException](
      TaskOutputValidator.validateMainSamples(sampleDS, graphMetadataPbWrapper).collect(),
    )
  }

  // TODO
  // ensure caching sanity, by checking pos samples and hydrated nei, in integration test

  // TODO test create node anchor sample subgraph
  // check resulting subgraph pos is in nodes, hard neg is in nodes
}
