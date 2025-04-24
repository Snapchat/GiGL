import common.graphdb.nebula.GraphEntityTranslator
import org.scalatest.funsuite.AnyFunSuite
import snapchat.research.gbml.graph_schema.EdgeType

class GraphEntityTranslatorTest extends AnyFunSuite {

  test(
    "test_can_encode_and_decode a nebulaVID",
  ) {
    val nodeId            = 1
    val condensedNodeType = 2
    val nebulaVID =
      GraphEntityTranslator.nebulaVIDFromNodeComponents(nodeId, condensedNodeType)
    val (decodedNodeId, decodedCondensedNodeType) =
      GraphEntityTranslator.nebulaVIDToNodeComponents(nebulaVID)
    assert(decodedNodeId == nodeId)
    assert(decodedCondensedNodeType == condensedNodeType)
  }

  test(
    "test_can_encode_and_decode a graphServiceEdgeType",
  ) {
    val edgeType = EdgeType(
      srcNodeType = "src",
      relation = "rel",
      dstNodeType = "dst",
    )
    val graphServiceEdgeType = GraphEntityTranslator.nebulaEdgeType(edgeType)
    val decodedEdgeType = GraphEntityTranslator.edgeTypeFromNebulaEdgeType(graphServiceEdgeType)
    assert(decodedEdgeType == edgeType)
  }

}
