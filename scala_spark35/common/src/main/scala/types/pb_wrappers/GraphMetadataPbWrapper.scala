package common.types.pb_wrappers

import common.types.GraphTypes.CondensedEdgeType
import common.types.GraphTypes.CondensedNodeType
import common.types.GraphTypes.NodeType
import common.types.GraphTypes.Relation
import snapchat.research.gbml.graph_schema.EdgeType
import snapchat.research.gbml.graph_schema.GraphMetadata
import snapchat.research.gbml.graph_schema.Node

import GraphPbWrappers.EdgePbWrapper

case class GraphMetadataPbWrapper(graphMetadataPb: GraphMetadata) {
  val condensedNodeTypes: Seq[CondensedNodeType] =
    graphMetadataPb.condensedNodeTypeMap.seq.keys.toSeq
  val condensedEdgeTypes: Seq[CondensedEdgeType] =
    graphMetadataPb.condensedEdgeTypeMap.seq.keys.toSeq

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

  // It is possible for multiple edge types to have the same relation
  // Ex. ("user", "to", "story"), ("story", "to", "user")
  // Depending on relation description, the mapping between relation and edge type can also be unique
  // Ex. ("user", "user_to_story", "story"), ("story", "story_to_user", "user")
  val relationToEdgeTypesMap: Map[Relation, Seq[EdgeType]] = {
    graphMetadataPb.edgeTypes.groupBy(_.relation)
  }

  val edgeTypeToRelationMap: Map[EdgeType, Relation] = {
    graphMetadataPb.edgeTypes.map(edgeType => edgeType -> edgeType.relation).toMap
  }

  val getEdgeTypesFromSrcDstNodeTypes: ((NodeType, NodeType)) => Seq[EdgeType] = {
    case (srcNodeType, dstNodeType) =>
      graphMetadataPb.edgeTypes.filter(edgeType =>
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
