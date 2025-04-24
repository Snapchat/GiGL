package common.types.pb_wrappers

import common.types.GraphTypes.CondensedEdgeType
import common.types.GraphTypes.CondensedNodeType
import common.types.GraphTypes.NodeId
import snapchat.research.gbml.graph_schema.Edge
import snapchat.research.gbml.graph_schema.Graph
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

  def mergeGraphs(graphs: Seq[Graph]): Graph = {

    /**
     * Merges multiple graphs into a single graph, by comparing node ids and condensed node/edge types, without
     * comparing the features for efficiency. We make the assumption that there are no discrepancies of features
     * between the same nodes and same edges.
     * @param graphs: Sequence of graphs to merge
     * @return Merged graph
     */
    val nodeMap = graphs
      .flatMap(graph => graph.nodes)
      .map(node => (node.nodeId, node.condensedNodeType) -> node)
      .toMap
    val edgeMap = graphs
      .flatMap(graph => graph.edges)
      .map(edge => (edge.srcNodeId, edge.dstNodeId, edge.condensedEdgeType) -> edge)
      .toMap

    val uniqueNodes =
      nodeMap.values.groupBy(node => (node.nodeId, node.condensedNodeType)).map(_._2.head).toSeq
    val uniqueEdges = edgeMap.values
      .groupBy(edge => (edge.srcNodeId, edge.dstNodeId, edge.condensedEdgeType))
      .map(_._2.head)
      .toSeq

    Graph(nodes = uniqueNodes, edges = uniqueEdges)
  }
}
