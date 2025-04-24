import common.types.pb_wrappers.GraphPbWrappers
import org.scalatest.funsuite.AnyFunSuite
import snapchat.research.gbml.graph_schema.Graph

class GraphPbWrappersTest extends AnyFunSuite with PbWrappersTestGraphSetup {
  test(
    "mergeGraphs - test Graph protos merged as expected with mergeGraphs",
  ) {
    val graph1 = Graph(
      nodes = Seq(node_1_0, node_2_1, node_3_0),
      edges = Seq(edge_1_2_1, edge_2_3_2),
    )

    val graph2 = Graph(
      nodes = Seq(node_2_1, node_3_0, node_4_1),
      edges = Seq(edge_2_3_2, edge_3_4_1, edge_2_4_0),
    )

    val mergedGraph = GraphPbWrappers.mergeGraphs(Seq(graph1, graph2))
    val mergedGraphExpected = Graph(
      nodes = Seq(node_1_0, node_2_1, node_3_0, node_4_1),
      edges = Seq(edge_1_2_1, edge_2_3_2, edge_3_4_1, edge_2_4_0),
    )

    val sortedMergedGraphNodes =
      mergedGraph.nodes.sortBy(node => (node.nodeId, node.condensedNodeType))
    val sortedMergedGraphExpectedNodes =
      mergedGraphExpected.nodes.sortBy(node => (node.nodeId, node.condensedNodeType))
    assert(sortedMergedGraphNodes.sameElements(sortedMergedGraphExpectedNodes))

    val sortedMergedGraphEdges =
      mergedGraph.edges.sortBy(edge => (edge.srcNodeId, edge.dstNodeId, edge.condensedEdgeType))
    val sortedMergedGraphExpectedEdges = mergedGraphExpected.edges.sortBy(edge =>
      (edge.srcNodeId, edge.dstNodeId, edge.condensedEdgeType),
    )

    assert(sortedMergedGraphEdges.sameElements(sortedMergedGraphExpectedEdges))
  }

}
