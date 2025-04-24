package common.graphdb.nebula

import common.graphdb.nebula.types.NebulaEdgeType
import common.graphdb.nebula.types.NebulaVID
import common.types.GraphTypes.CondensedNodeType
import common.types.GraphTypes.NodeId
import snapchat.research.gbml.graph_schema.EdgeType
import snapchat.research.gbml.graph_schema.Node

object GraphEntityTranslator {

  private val MAX_UINT32         = 0xffffffff
  private val NUM_BITS_IN_UINT32 = 32

  def nebulaVIDFromNodeComponents(
    nodeId: NodeId,
    condensedNodeType: CondensedNodeType,
  ): NebulaVID = {
    (condensedNodeType.toLong << NUM_BITS_IN_UINT32) | nodeId
  }

  def nebulaVIDToNodeComponents(
    nebulaVID: NebulaVID,
  ): Tuple2[NodeId, CondensedNodeType] = {
    val condensedNodeType = nebulaVID >> NUM_BITS_IN_UINT32
    val nodeId            = nebulaVID & MAX_UINT32
    (nodeId.toInt, condensedNodeType.toInt)
  }

  def nebulaVIDToNode(
    nebulaVID: NebulaVID,
  ): Node = {
    val (nodeId, condensedNodeType) = nebulaVIDToNodeComponents(nebulaVID)
    Node(nodeId, Some(condensedNodeType))
  }

  def nebulaEdgeType(
    edgeType: EdgeType,
  ): NebulaEdgeType = {
    s"${edgeType.srcNodeType}_${edgeType.relation}_${edgeType.dstNodeType}"
  }

  def edgeTypeFromNebulaEdgeType(
    graphServiceEdgeType: NebulaEdgeType,
  ): EdgeType = {
    val Array(srcNodeType, relation, dstNodeType) = graphServiceEdgeType.split("_")
    EdgeType(
      srcNodeType = srcNodeType,
      relation = relation,
      dstNodeType = dstNodeType,
    )
  }

}
