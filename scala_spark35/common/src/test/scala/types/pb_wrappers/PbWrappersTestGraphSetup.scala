import snapchat.research.gbml.graph_schema.Edge
import snapchat.research.gbml.graph_schema.Node

trait PbWrappersTestGraphSetup {
  val node_1_0: Node = Node(nodeId = 1, condensedNodeType = Some(0))
  val node_2_1: Node = Node(nodeId = 2, condensedNodeType = Some(1))
  val node_3_0: Node = Node(nodeId = 3, condensedNodeType = Some(0))
  val node_4_1: Node = Node(nodeId = 4, condensedNodeType = Some(1))

  val edge_1_2_1: Edge = Edge(srcNodeId = 1, dstNodeId = 2, condensedEdgeType = Some(1))
  val edge_2_3_2: Edge = Edge(srcNodeId = 2, dstNodeId = 3, condensedEdgeType = Some(2))
  val edge_2_4_0: Edge = Edge(srcNodeId = 2, dstNodeId = 4, condensedEdgeType = Some(0))
  val edge_3_4_1: Edge = Edge(srcNodeId = 3, dstNodeId = 4, condensedEdgeType = Some(1))
}
