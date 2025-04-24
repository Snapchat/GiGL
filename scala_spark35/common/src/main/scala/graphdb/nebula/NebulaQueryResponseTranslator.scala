package common.graphdb.nebula

import common.graphdb.DBResult
import common.graphdb.QueryResponseTranslator
import common.types.pb_wrappers.GraphMetadataPbWrapper
import snapchat.research.gbml.graph_schema.Edge
import snapchat.research.gbml.graph_schema.Node
import snapchat.research.gbml.subgraph_sampling_strategy.SamplingDirection
import snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp

import scala.collection.mutable.HashSet

object NebulaQueryResponseTranslator {
  val RESULT_SRC_NODE_ID_COL_NAME = "src"
  val RESULT_DST_NODE_ID_COL_NAME = "dst"
}
class NebulaQueryResponseTranslator(
  graphMetadataPbWrapper: GraphMetadataPbWrapper)
    extends QueryResponseTranslator
    with Serializable {

  def translate(
    samplingOp: SamplingOp,
    rootNodes: Seq[
      Node,
    ],
  ): types.nGQLQuery = {
    if (rootNodes.isEmpty) {
      throw new IllegalArgumentException("Root nodes cannot be empty; atleast one must be provided")
    }
    val query = rootNodes
      .map(rootNode => {
        translate(samplingOp, rootNode)
      })
      .mkString("\nUNION\n")
    query
  }

  def translate(
    samplingOp: SamplingOp,
    rootNode: Node,
  ): types.nGQLQuery = {

    val nebulaVID = GraphEntityTranslator.nebulaVIDFromNodeComponents(
      nodeId = rootNode.nodeId,
      condensedNodeType = rootNode.condensedNodeType.get,
    )
    val nebulaEdgeType: types.NebulaEdgeType = GraphEntityTranslator.nebulaEdgeType(
      samplingOp.edgeType.get,
    )
    // Adding `REVERSELY` identifier to the query just means we want to get incoming edges instead of default
    // outgoing edges.
    val useOutgoingEdges = samplingOp.samplingDirection match {
      case SamplingDirection.OUTGOING => true
      case _                          => false
    }
    val outgoingEdgesModifier = if (useOutgoingEdges) "" else "REVERSELY"

    val query = samplingOp.samplingMethod match {
      case SamplingOp.SamplingMethod.RandomUniform(value) => {
        val numNodesToSample = value.numNodesToSample
        s"""GO 1 STEP
        FROM ${nebulaVID} 
        OVER ${nebulaEdgeType} ${outgoingEdgesModifier}
        YIELD
          src(edge) as ${NebulaQueryResponseTranslator.RESULT_SRC_NODE_ID_COL_NAME},
          dst(edge) as ${NebulaQueryResponseTranslator.RESULT_DST_NODE_ID_COL_NAME}
        LIMIT [${numNodesToSample}]
        """
      }
      // TODO: (svij) This is not TRUE RandomWeighted sampling.
      // Consider this a placeholder that tries to do a "similar computation"
      // Work needs to be done to implement true RandomWeighted sampling.
      case SamplingOp.SamplingMethod.RandomWeighted(value) => {
        val numNodesToSample = value.numNodesToSample
        val edgeFeatName     = value.edgeFeatName
        s"""GO 1 STEP 
        FROM ${nebulaVID} 
        OVER ${nebulaEdgeType} ${outgoingEdgesModifier}
        YIELD 
          src(edge) as ${NebulaQueryResponseTranslator.RESULT_SRC_NODE_ID_COL_NAME},
          dst(edge) as ${NebulaQueryResponseTranslator.RESULT_DST_NODE_ID_COL_NAME},
          ${nebulaEdgeType}.${edgeFeatName} * rand() as ${edgeFeatName} |
        ORDER BY $$-.${edgeFeatName} DESC |
        LIMIT ${numNodesToSample} |
        YIELD 
          $$-.${NebulaQueryResponseTranslator.RESULT_SRC_NODE_ID_COL_NAME} AS ${NebulaQueryResponseTranslator.RESULT_SRC_NODE_ID_COL_NAME}, 
          $$-.${NebulaQueryResponseTranslator.RESULT_DST_NODE_ID_COL_NAME} AS ${NebulaQueryResponseTranslator.RESULT_DST_NODE_ID_COL_NAME}"""
      }
      case SamplingOp.SamplingMethod.TopK(value) => {
        val numNodesToSample = value.numNodesToSample
        val edgeFeatName     = value.edgeFeatName
        s"""GO 1 STEP 
        FROM ${nebulaVID} 
        OVER ${nebulaEdgeType} ${outgoingEdgesModifier}
        YIELD 
          src(edge) as ${NebulaQueryResponseTranslator.RESULT_SRC_NODE_ID_COL_NAME},
          dst(edge) as ${NebulaQueryResponseTranslator.RESULT_DST_NODE_ID_COL_NAME},
          ${nebulaEdgeType}.${edgeFeatName} as ${edgeFeatName} |
        ORDER BY $$-.${edgeFeatName} DESC |
        LIMIT ${numNodesToSample} |
        YIELD 
          $$-.${NebulaQueryResponseTranslator.RESULT_SRC_NODE_ID_COL_NAME} AS ${NebulaQueryResponseTranslator.RESULT_SRC_NODE_ID_COL_NAME}, 
          $$-.${NebulaQueryResponseTranslator.RESULT_DST_NODE_ID_COL_NAME} AS ${NebulaQueryResponseTranslator.RESULT_DST_NODE_ID_COL_NAME}"""
      }
      case SamplingOp.SamplingMethod.UserDefined(value) => {
        throw new NotImplementedError("UserDefined sampling strategy is not supported yet")
      }
      case _ => { // Empty / default case
        throw new NotImplementedError("Sampling method is not supported yet")
      }
    }

    // Take all new lines (\n), multiple whitespaces (\\s+) and replace them with a single whitespace
    // to prevent error prone query formation.
    query.replaceAll("\n", " ").replaceAll("\\s+", " ").trim()
  }

  def parseEdgesAndNodesFromResult(
    result: DBResult,
    samplingOp: SamplingOp,
  ): Tuple2[HashSet[Edge], HashSet[Node]] = {

    val neighborhoodEdgeSet: HashSet[Edge] = HashSet.empty[Edge]
    val resultingNodeSet: HashSet[Node]    = HashSet.empty[Node]
    if (!result.isEmpty()) {
      val dstNodes: Seq[Node] = result
        .colValues(NebulaQueryResponseTranslator.RESULT_DST_NODE_ID_COL_NAME)
        .asInstanceOf[List[types.NebulaVID]]
        .toSeq
        .map(vid => GraphEntityTranslator.nebulaVIDToNode(vid))
      val srcNodes: Seq[Node] = result
        .colValues(NebulaQueryResponseTranslator.RESULT_SRC_NODE_ID_COL_NAME)
        .asInstanceOf[List[types.NebulaVID]]
        .toSeq
        .map(vid => GraphEntityTranslator.nebulaVIDToNode(vid))

      // If we are using outgoing edges, then we want to add dstNodes to the resultingNodeSet
      // For incoming edges, we want to add srcNodes to the resultingNodeSet
      // The resultingNodeSet defines the "next hop nodes"; inclding the other node here breaks
      // the query contract - it may still be included in the case of self loops, which is expected.
      val useOutgoingEdges = samplingOp.samplingDirection match {
        case SamplingDirection.OUTGOING => true
        case _                          => false
      }
      if (useOutgoingEdges) {
        resultingNodeSet ++= dstNodes
      } else {
        resultingNodeSet ++= srcNodes
      }

      val edges: Seq[Edge] = srcNodes.zip(dstNodes).map { case (srcNode, dstNode) =>
        val edgeType          = samplingOp.edgeType.get
        val condensedEdgeType = graphMetadataPbWrapper.edgeTypeToCondensedEdgeTypeMap(edgeType)
        Edge(
          srcNodeId = srcNode.nodeId,
          dstNodeId = dstNode.nodeId,
          condensedEdgeType = Some(condensedEdgeType),
        )
      }
      neighborhoodEdgeSet ++= edges
    }
    Tuple2(neighborhoodEdgeSet, resultingNodeSet)
  }

}
