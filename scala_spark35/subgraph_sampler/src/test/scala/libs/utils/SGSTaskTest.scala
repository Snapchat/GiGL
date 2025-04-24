import common.test.testLibs.SharedSparkSession
import common.types.pb_wrappers.GbmlConfigPbWrapper
import common.types.pb_wrappers.GraphMetadataPbWrapper
import common.utils.ProtoLoader.populateProtoFromYaml
import libs.utils.SGSTask.hydrateRnn
import libs.utils.SGSTask.isIsolatedNode
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Dataset
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
import snapchat.research.gbml.training_samples_schema.RootedNodeNeighborhood

class SGSTaskTest extends AnyFunSuite with BeforeAndAfterAll with SharedSparkSession {
  var gbmlConfigWrapper: GbmlConfigPbWrapper         = _
  var graphMetadataPbWrapper: GraphMetadataPbWrapper = _

  // Nodes
  val node_0_0: Node = Node(nodeId = 0, condensedNodeType = Some(0), featureValues = Seq(0.0f))
  val node_1_0: Node = Node(nodeId = 1, condensedNodeType = Some(0), featureValues = Seq(0.1f))
  val node_2_0: Node = Node(nodeId = 2, condensedNodeType = Some(0), featureValues = Seq(0.2f))
  val node_3_1: Node = Node(nodeId = 3, condensedNodeType = Some(1), featureValues = Seq(0.3f))
  val node_4_1: Node = Node(nodeId = 4, condensedNodeType = Some(1), featureValues = Seq(0.4f))
  val node_5_0: Node =
    Node(nodeId = 5, condensedNodeType = Some(2), featureValues = Seq(0.5f)) // isolated node
  val nodes: Seq[Node] = Seq(node_0_0, node_1_0, node_2_0, node_3_1, node_4_1, node_5_0)

  // Edges
  val edge_0_1_0: Edge =
    Edge(srcNodeId = 0, dstNodeId = 1, condensedEdgeType = Some(0), featureValues = Seq(0.5f))
  val edge_2_0_0: Edge =
    Edge(srcNodeId = 2, dstNodeId = 0, condensedEdgeType = Some(0), featureValues = Seq(1.0f))
  val edge_0_3_1: Edge =
    Edge(srcNodeId = 0, dstNodeId = 3, condensedEdgeType = Some(1), featureValues = Seq(1.5f))
  val edge_1_4_1: Edge =
    Edge(srcNodeId = 1, dstNodeId = 4, condensedEdgeType = Some(1), featureValues = Seq(1.5f))
  val edge_3_1_2: Edge =
    Edge(srcNodeId = 3, dstNodeId = 1, condensedEdgeType = Some(2), featureValues = Seq(2.0f))
  val edge_4_0_2: Edge =
    Edge(srcNodeId = 4, dstNodeId = 0, condensedEdgeType = Some(2), featureValues = Seq(2.5f))
  val edges: Seq[Edge] = Seq(edge_0_1_0, edge_2_0_0, edge_0_3_1, edge_1_4_1, edge_3_1_2, edge_4_0_2)

  override def beforeAll(): Unit = {
    super.beforeAll()
    val frozenGbmlConfigUriTest =
      "common/src/test/assets/subgraph_sampler/heterogeneous/node_anchor_based_link_prediction/frozen_gbml_config_graphdb_dblp_local.yaml"
    val gbmlConfigProto =
      populateProtoFromYaml[GbmlConfig](uri = frozenGbmlConfigUriTest)
    gbmlConfigWrapper = GbmlConfigPbWrapper(gbmlConfigPb = gbmlConfigProto)

    val graphMetadataPbWrapper: GraphMetadataPbWrapper = GraphMetadataPbWrapper(
      gbmlConfigWrapper.graphMetadataPb,
    )
  }

  def mockHydratedNode: DataFrame = {
    import sqlImplicits._
    val nodeData = nodes.map(node => (node.nodeId, node.featureValues, node.condensedNodeType))
    val hydratedNodeDF =
      nodeData.toDF("_node_id", "_node_features", "_condensed_node_type")
    hydratedNodeDF
  }
  def mockHydratedEdge: DataFrame = {
    import sqlImplicits._
    val hydratedEdgeData = edges.map(edge =>
      (edge.srcNodeId, edge.dstNodeId, edge.featureValues, edge.condensedEdgeType),
    )
    val hydratedEdgeDF = hydratedEdgeData
      .toDF("_from", "_to", "_edge_features", "_condensed_edge_type")
    hydratedEdgeDF
  }

  def mockUnhydratedRnnDS: Dataset[RootedNodeNeighborhood] = {
    val rnnData = Seq(
      RootedNodeNeighborhood(
        rootNode = Some(node_0_0.copy(featureValues = Seq.empty)),
        neighborhood = Some(
          Graph(
            nodes =
              Seq(node_0_0, node_1_0, node_2_0, node_4_1).map(_.copy(featureValues = Seq.empty)),
            edges = Seq(edge_4_0_2, edge_1_4_1, edge_2_0_0).map(_.copy(featureValues = Seq.empty)),
          ),
        ),
      ),
      RootedNodeNeighborhood(
        rootNode = Some(node_1_0.copy(featureValues = Seq.empty)),
        neighborhood = Some(
          Graph(
            nodes =
              Seq(node_0_0, node_1_0, node_3_1, node_4_1).map(_.copy(featureValues = Seq.empty)),
            edges = Seq(edge_4_0_2, edge_0_1_0, edge_0_3_1, edge_3_1_2).map(
              _.copy(featureValues = Seq.empty),
            ),
          ),
        ),
      ),
    )

    val rnnDS = sparkTest.createDataset(rnnData).as[RootedNodeNeighborhood]
    rnnDS
  }

  def mockHydratedRnnDS: Dataset[RootedNodeNeighborhood] = {
    val rnnData = Seq(
      RootedNodeNeighborhood(
        rootNode = Some(node_0_0),
        neighborhood = Some(
          Graph(
            nodes = Seq(node_0_0, node_1_0, node_2_0, node_4_1),
            edges = Seq(edge_4_0_2, edge_1_4_1, edge_2_0_0),
          ),
        ),
      ),
      RootedNodeNeighborhood(
        rootNode = Some(node_1_0),
        neighborhood = Some(
          Graph(
            nodes = Seq(node_0_0, node_1_0, node_3_1, node_4_1),
            edges = Seq(edge_4_0_2, edge_0_1_0, edge_0_3_1, edge_3_1_2),
          ),
        ),
      ),
    )

    val rnnDS = sparkTest.createDataset(rnnData).as[RootedNodeNeighborhood]
    rnnDS
  }

  test(
    "hydrateRnn - test if rnn is hydrated correctly",
  ) {

    // Create mock views for node and edge features
    val mockNodeFeaturesDF   = mockHydratedNode
    val mockEdgeFeaturesDF   = mockHydratedEdge
    val mockNodeFeaturesView = "mockNodeFeaturesVIEW"
    val mockEdgeFeaturesView = "mockEdgeFeaturesVIEW"
    mockNodeFeaturesDF.createOrReplaceTempView(mockNodeFeaturesView)
    mockEdgeFeaturesDF.createOrReplaceTempView(mockEdgeFeaturesView)

    val result =
      hydrateRnn(
        rnnDS = mockUnhydratedRnnDS,
        hydratedNodeVIEW = "mockNodeFeaturesVIEW",
        hydratedEdgeVIEW = "mockEdgeFeaturesVIEW",
      )

    val expectedResult       = mockHydratedRnnDS
    val sortedResult         = result.orderBy(F.col("root_node.node_id"))
    val sortedExpectedResult = expectedResult.orderBy(F.col("root_node.node_id"))

    sortedResult.collect() shouldEqual sortedExpectedResult.collect()
  }

  test(
    "isIsolatedNode - test if isolated nodes are detected correctly",
  ) {
    // Create a mock RootedNodeNeighborhood dataset
    val mockIsolatedRnn = RootedNodeNeighborhood(
      rootNode = Some(node_5_0),
      neighborhood = Some(
        Graph(
          nodes = Seq(node_5_0),
          edges = Seq.empty,
        ),
      ),
    )

    val mockNonIsolatedRnn = RootedNodeNeighborhood(
      rootNode = Some(node_0_0),
      neighborhood = Some(
        Graph(
          nodes = Seq(node_0_0, node_1_0, node_2_0),
          edges = Seq(edge_0_1_0, edge_2_0_0),
        ),
      ),
    )

    // Call the function with the mock data
    val resultIsolated    = isIsolatedNode(rnn = mockIsolatedRnn)
    val resultNonIsolated = isIsolatedNode(rnn = mockNonIsolatedRnn)

    resultIsolated shouldEqual true
    resultNonIsolated shouldEqual false
  }
}
