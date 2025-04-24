package common.types.pb_wrappers

import common.types.GraphTypes.CondensedEdgeType
import common.types.GraphTypes.CondensedNodeType
import common.types.GraphTypes.NodeId
import snapchat.research.gbml.graph_schema.Edge
import snapchat.research.gbml.graph_schema.Node

object GraphPbWrappers {

  val DefaultCondensedNodeType: CondensedNodeType = 0
  val DefaultCondensedEdgeType: CondensedEdgeType = 0

  case class EdgePbWrapper(edge: Edge) {
    def srcNodeId: NodeId = edge.srcNodeId
    def dstNodeId: NodeId = edge.dstNodeId
    def condensedEdgeType: CondensedEdgeType =
      edge.condensedEdgeType.getOrElse(DefaultCondensedEdgeType)
    def uniqueId: Array[Byte] = {
      s"$srcNodeId-$condensedEdgeType-$dstNodeId".getBytes()
    }
    def reverseEdgePb: EdgePbWrapper = {
      EdgePbWrapper(
        Edge(
          srcNodeId = dstNodeId,
          dstNodeId = srcNodeId,
          condensedEdgeType = Some(condensedEdgeType),
        ),
      )
    }
  }

  case class NodePbWrapper(node: Node) {
    def nodeId: NodeId = node.nodeId
    def condensedNodeType: CondensedNodeType =
      node.condensedNodeType.getOrElse(DefaultCondensedNodeType)
    def uniqueId: Array[Byte] = {
      s"$nodeId-$condensedNodeType".getBytes()
    }
  }
}
