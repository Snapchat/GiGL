package common.types.pb_wrappers

import common.types.GraphTypes.CondensedEdgeType
import common.types.GraphTypes.CondensedNodeType
import common.types.GraphTypes.NodeType
import snapchat.research.gbml.graph_schema.EdgeType
import snapchat.research.gbml.graph_schema.GraphMetadata
import snapchat.research.gbml.graph_schema.Node

import GraphPbWrappers.EdgePbWrapper

case class GraphMetadataPbWrapper(graphMetadataPb: GraphMetadata) {
  val nodeTypes: Seq[NodeType] = graphMetadataPb.nodeTypes
  val edgeTypes: Seq[EdgeType] = graphMetadataPb.edgeTypes

  val condensedNodeTypeMap: Map[CondensedNodeType, NodeType] = graphMetadataPb.condensedNodeTypeMap
  val condensedEdgeTypeMap: Map[CondensedEdgeType, EdgeType] = graphMetadataPb.condensedEdgeTypeMap

  val condensedNodeTypes: Seq[CondensedNodeType] =
    condensedNodeTypeMap.seq.keys.toSeq
  val condensedEdgeTypes: Seq[CondensedEdgeType] =
    condensedEdgeTypeMap.seq.keys.toSeq

  val dstNodeTypeToEdgeTypeMap: Map[NodeType, Seq[EdgeType]] = {
    graphMetadataPb.edgeTypes.groupBy(_.dstNodeType)
  }

  val edgeTypeToSrcDstNodeTypesMap: Map[EdgeType, (NodeType, NodeType)] = {
    graphMetadataPb.edgeTypes
      .map(edgeType => edgeType -> (edgeType.srcNodeType, edgeType.dstNodeType))
      .toMap
  }

  val nodeTypeToCondensedNodeTypeMap: Map[NodeType, CondensedNodeType] = {
    graphMetadataPb.condensedNodeTypeMap.seq.map { case (condensedNodeType, nodeType) =>
      nodeType -> condensedNodeType
    }
  }

  val edgeTypeToCondensedEdgeTypeMap: Map[EdgeType, CondensedEdgeType] = {
    graphMetadataPb.condensedEdgeTypeMap.seq.map { case (condensedEdgeType, edgeType) =>
      edgeType -> condensedEdgeType
    }
  }

  val relationToEdgeTypeMap: Map[String, EdgeType] = {
    graphMetadataPb.edgeTypes.map(edgeType => edgeType.relation -> edgeType).toMap
  }

  val edgeTypeToRelationMap: Map[EdgeType, String] = {
    graphMetadataPb.edgeTypes.map(edgeType => edgeType -> edgeType.relation).toMap
  }

  val getEdgeTypeFromSrcDstNodeTypes: ((NodeType, NodeType)) => Option[EdgeType] = {
    case (srcNodeType, dstNodeType) =>
      edgeTypes.find(edgeType =>
        edgeType.srcNodeType == srcNodeType && edgeType.dstNodeType == dstNodeType,
      )
  }

  val condensedEdgeTypeToCondensedNodeTypes
    : Map[CondensedEdgeType, (CondensedNodeType, CondensedNodeType)] = {
    val nodeTypeToCondensedNodeTypeMap: Map[NodeType, CondensedNodeType] =
      graphMetadataPb.condensedNodeTypeMap.seq.map { case (condensedNodeType, nodeType) =>
        nodeType -> condensedNodeType
      }
    graphMetadataPb.condensedEdgeTypeMap.seq.map { case (condensedEdgeType, edgeType) =>
      (condensedEdgeType -> (
        nodeTypeToCondensedNodeTypeMap.get(edgeType.srcNodeType).get,
        nodeTypeToCondensedNodeTypeMap.get(edgeType.dstNodeType).get
      ))
    }
  }

  def getFeaturelessNodePbsFromEdge(edgeWrapper: EdgePbWrapper): Seq[Node] = {
    val condensedNodeTypes =
      condensedEdgeTypeToCondensedNodeTypes.get(edgeWrapper.condensedEdgeType).get
    val srcNode =
      Node(nodeId = edgeWrapper.srcNodeId, condensedNodeType = Some(condensedNodeTypes._1))
    val dstNode =
      Node(nodeId = edgeWrapper.dstNodeId, condensedNodeType = Some(condensedNodeTypes._2))
    Seq(srcNode, dstNode)
  }
}
