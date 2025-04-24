import common.types.pb_wrappers.GraphMetadataPbWrapper
import org.scalatest.funsuite.AnyFunSuite
import snapchat.research.gbml.graph_schema.EdgeType
import snapchat.research.gbml.graph_schema.GraphMetadata

class GraphMetadataPbWrapperTest extends AnyFunSuite {

  /**
   * Test the GraphMetadataWrapper class.
   * Making sure that all fields in the wrapper yields desired results.
   */
  test("GraphMetadataWrapperTest") {
    val edgeType0 = EdgeType(srcNodeType = "User", relation = "UserToStory", dstNodeType = "Story")
    val edgeType1 = EdgeType(srcNodeType = "Story", relation = "StoryToUser", dstNodeType = "User")
    val nodeType0 = "User"
    val nodeType1 = "Story"
    val condensedNodeType_0 = 0
    val condensedNodeType_1 = 1
    val condensedEdgeType_0 = 0
    val condensedEdgeType_1 = 1
    val relationType_0      = "UserToStory"
    val relationType_1      = "StoryToUser"
    val condensedEdgeTypeMap =
      Map(condensedEdgeType_0 -> edgeType0, condensedEdgeType_1 -> edgeType1)
    val condensedNodeTypeMap =
      Map(condensedNodeType_0 -> nodeType0, condensedNodeType_1 -> nodeType1)

    val mockGraphMetadata = GraphMetadata(
      edgeTypes = Seq(edgeType0, edgeType1),
      nodeTypes = Seq(nodeType0, nodeType1),
      condensedEdgeTypeMap = condensedEdgeTypeMap,
      condensedNodeTypeMap = condensedNodeTypeMap,
    )
    val mockGraphMetadataWrapper = GraphMetadataPbWrapper(mockGraphMetadata)

    assert(
      mockGraphMetadataWrapper.condensedNodeTypes == Seq(condensedNodeType_0, condensedNodeType_1),
    )
    assert(
      mockGraphMetadataWrapper.condensedEdgeTypes == Seq(condensedEdgeType_0, condensedEdgeType_1),
    )
    assert(
      mockGraphMetadataWrapper.dstNodeTypeToEdgeTypeMap == Map(
        nodeType0 -> Seq(edgeType1),
        "Story"   -> Seq(edgeType0),
      ),
    )
    assert(
      mockGraphMetadataWrapper.edgeTypeToSrcDstNodeTypesMap == Map(
        edgeType0 -> (nodeType0, nodeType1),
        edgeType1 -> (nodeType1, nodeType0),
      ),
    )
    assert(
      mockGraphMetadataWrapper.nodeTypeToCondensedNodeTypeMap == Map(
        nodeType0 -> condensedNodeType_0,
        nodeType1 -> condensedNodeType_1,
      ),
    )
    assert(
      mockGraphMetadataWrapper.edgeTypeToCondensedEdgeTypeMap == Map(
        edgeType0 -> condensedEdgeType_0,
        edgeType1 -> condensedEdgeType_1,
      ),
    )
    assert(
      mockGraphMetadataWrapper.relationToEdgeTypesMap == Map(
        relationType_0 -> Seq(edgeType0),
        relationType_1 -> Seq(edgeType1),
      ),
    )
    assert(
      mockGraphMetadataWrapper.edgeTypeToRelationMap == Map(
        edgeType0 -> relationType_0,
        edgeType1 -> relationType_1,
      ),
    )
    assert(
      mockGraphMetadataWrapper.condensedEdgeTypeToCondensedNodeTypes == Map(
        condensedEdgeType_0 -> (condensedNodeType_0, condensedNodeType_1),
        condensedEdgeType_1 -> (condensedNodeType_1, condensedNodeType_0),
      ),
    )
    assert(
      mockGraphMetadataWrapper.getEdgeTypesFromSrcDstNodeTypes((nodeType0, nodeType1)) == Seq(
        edgeType0,
      ),
    )
    assert(
      mockGraphMetadataWrapper.getEdgeTypesFromSrcDstNodeTypes((nodeType1, nodeType0)) == Seq(
        edgeType1,
      ),
    )
    assert(
      mockGraphMetadataWrapper.getEdgeTypesFromSrcDstNodeTypes(("Nonexistent", nodeType0)) == Seq
        .empty[EdgeType],
    )
  }
}
