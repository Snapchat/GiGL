import common.types.pb_wrappers.RootedNodeNeighborhoodPbWrapper
import org.scalatest.funsuite.AnyFunSuite
import snapchat.research.gbml.graph_schema.Graph
import snapchat.research.gbml.training_samples_schema.RootedNodeNeighborhood

class RootedNodeNeighborhoodPbWrapperTest extends AnyFunSuite with PbWrappersTestGraphSetup {
  test(
    "mergeRootedNodeNeighborhoods - test RootedNodeNeighborhood protos merge as expected with mergeRootedNodeNeighborhoods",
  ) {
    val rnn1 = RootedNodeNeighborhood(
      rootNode = Some(node_1_0),
      neighborhood = Some(
        Graph(
          nodes = Seq(node_1_0, node_2_1),
          edges = Seq(edge_1_2_1),
        ),
      ),
    )
    val rnn2 = RootedNodeNeighborhood(
      rootNode = Some(node_1_0),
      neighborhood = Some(
        Graph(
          nodes = Seq(node_2_1, node_3_0, node_4_1),
          edges = Seq(edge_2_3_2, edge_3_4_1, edge_2_4_0),
        ),
      ),
    )
    val mergedRnn =
      RootedNodeNeighborhoodPbWrapper.mergeRootedNodeNeighborhoods(rnns = Seq(rnn1, rnn2))
    val mergedRnnExpected = RootedNodeNeighborhood(
      rootNode = Some(node_1_0),
      neighborhood = Some(
        Graph(
          nodes = Seq(node_1_0, node_2_1, node_3_0, node_4_1),
          edges = Seq(edge_1_2_1, edge_2_3_2, edge_3_4_1, edge_2_4_0),
        ),
      ),
    )
    assert(mergedRnn == mergedRnnExpected)
  }

  test(
    "mergeRootedNodeNeighborhoods - test assert thrown when mergeRootedNodeNeighborhoods merges RootNodeNeighborhoods with different root nodes",
  ) {
    val rnn1 = RootedNodeNeighborhood(
      rootNode = Some(node_1_0),
      neighborhood = Some(
        Graph(
          nodes = Seq(node_1_0, node_2_1),
          edges = Seq(edge_1_2_1),
        ),
      ),
    )
    val rnn2 = RootedNodeNeighborhood(
      rootNode = Some(node_2_1),
      neighborhood = Some(
        Graph(
          nodes = Seq(node_2_1, node_3_0, node_4_1),
          edges = Seq(edge_2_3_2, edge_3_4_1, edge_2_4_0),
        ),
      ),
    )
    assertThrows[AssertionError] {
      RootedNodeNeighborhoodPbWrapper.mergeRootedNodeNeighborhoods(rnns = Seq(rnn1, rnn2))
    }
  }

  test("checkAllRootNodesAreSame - test checkAllRootNodesAreSame is true") {
    val rnn1 =
      RootedNodeNeighborhood(
        rootNode = Some(node_1_0),
        neighborhood = Some(Graph()),
      )
    val rnn2 =
      RootedNodeNeighborhood(
        rootNode = Some(node_1_0),
        neighborhood = Some(Graph()),
      )
    val rnn3 =
      RootedNodeNeighborhood(
        rootNode = Some(node_1_0),
        neighborhood = Some(Graph()),
      )
    assert(RootedNodeNeighborhoodPbWrapper.checkAllRootNodesAreSame(rnns = Seq(rnn1, rnn2, rnn3)))
  }

  test(
    "checkAllRootNodesAreSame - test checkAllRootNodesAreSame root nodes are different",
  ) {
    val rnn1 =
      RootedNodeNeighborhood(
        rootNode = Some(node_1_0),
        neighborhood = Some(Graph()),
      )
    val rnn2 =
      RootedNodeNeighborhood(
        rootNode = Some(node_1_0),
        neighborhood = Some(Graph()),
      )
    val rnn3 =
      RootedNodeNeighborhood(
        rootNode = Some(node_2_1),
        neighborhood = Some(Graph()),
      )
    assert(!RootedNodeNeighborhoodPbWrapper.checkAllRootNodesAreSame(rnns = Seq(rnn1, rnn2, rnn3)))
  }
}
