import common.types.pb_wrappers.NodeAnchorBasedLinkPredictionSamplePbWrapper
import org.scalatest.funsuite.AnyFunSuite
import snapchat.research.gbml.graph_schema.Graph
import snapchat.research.gbml.training_samples_schema.NodeAnchorBasedLinkPredictionSample
import snapchat.research.gbml.training_samples_schema.RootedNodeNeighborhood

class NodeAnchorBasedLinkpredictionSamplePbWrapperTest
    extends AnyFunSuite
    with PbWrappersTestGraphSetup {
  test(
    "createNodeAnchorBasedLinkPredictionSample - test NodeAnchorBasedLinkPredictionSample created as expected",
  ) {
    val rnn = RootedNodeNeighborhood(
      rootNode = Some(node_1_0),
      neighborhood = Some(
        Graph(
          nodes = Seq(node_1_0, node_2_1),
          edges = Seq(edge_1_2_1),
        ),
      ),
    )
    val positiveEdges = Seq(edge_1_2_1)
    val positiveNeighborhoods = Seq(
      Graph(
        nodes = Seq(node_3_0, node_4_1),
        edges = Seq(edge_2_3_2, edge_2_4_0),
      ),
    )
    val hardNegativeEdges = Seq(edge_2_3_2)
    val hardNegativeNeighborhoods = Seq(
      Graph(
        nodes = Seq(node_4_1, node_3_0),
        edges = Seq(edge_3_4_1),
      ),
    )
    val sample =
      NodeAnchorBasedLinkPredictionSamplePbWrapper.NodeAnchorBasedLinkPredictionSampleFrom(
        rnn = rnn,
        positiveEdges = positiveEdges,
        positiveNeighborhoods = positiveNeighborhoods,
        hardNegativeEdges = Some(hardNegativeEdges),
        hardNegativeNeighborhoods = Some(hardNegativeNeighborhoods),
      )
    val expectedSample = NodeAnchorBasedLinkPredictionSample(
      rootNode = Some(node_1_0),
      posEdges = Seq(edge_1_2_1),
      hardNegEdges = Seq(edge_2_3_2),
      neighborhood = Some(
        Graph(
          Seq(node_1_0, node_2_1, node_3_0, node_4_1),
          Seq(edge_1_2_1, edge_2_3_2, edge_2_4_0, edge_3_4_1),
        ),
      ),
    )
    assert(sample == expectedSample)
  }
}
