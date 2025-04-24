package splitgenerator.test

import common.types.pb_wrappers.GraphMetadataPbWrapper
import common.types.pb_wrappers.GraphPbWrappers.EdgePbWrapper
import snapchat.research.gbml.graph_schema.Edge
import snapchat.research.gbml.graph_schema.EdgeType
import snapchat.research.gbml.graph_schema.Graph
import snapchat.research.gbml.graph_schema.GraphMetadata
import snapchat.research.gbml.graph_schema.Node
import snapchat.research.gbml.training_samples_schema.NodeAnchorBasedLinkPredictionSample
import snapchat.research.gbml.training_samples_schema.RootedNodeNeighborhood

import scala.collection.mutable.ArrayBuffer

object SplitGeneratorTestUtils {
  def getGraphMetadataWrapper(): GraphMetadataPbWrapper = {
    val defaultCondensedEdgeType = 0
    val defaultCondensedNodeType = 0
    val defaultNodeType          = defaultCondensedNodeType.toString()
    val defaultEdgeType = EdgeType(
      srcNodeType = defaultNodeType,
      dstNodeType = defaultCondensedNodeType.toString(),
      relation = defaultNodeType,
    )

    val condensedNodeTypeMap =
      Map[Int, String](defaultCondensedNodeType -> defaultCondensedNodeType.toString())
    val condensedEdgeTypeMap = Map[Int, EdgeType](
      defaultCondensedEdgeType -> EdgeType(
        srcNodeType = defaultCondensedNodeType.toString(),
        dstNodeType = defaultCondensedNodeType.toString(),
        relation = defaultCondensedEdgeType.toString(),
      ),
    )

    GraphMetadataPbWrapper(
      GraphMetadata(
        Seq(defaultNodeType),
        Seq(defaultEdgeType),
        condensedEdgeTypeMap,
        condensedNodeTypeMap,
      ),
    )
  }

  /**
      * Create edgePbWrapper mocks. 100 bi-directional edges
      * We create 100 edges so that there is a very chance that each bucket has atleast 1 edge assigned.
      */
  def getMockEdgePbWrappers(): Seq[EdgePbWrapper] = {
    val NumberOfMockObjects = 100
    var edgeWrappers        = ArrayBuffer[EdgePbWrapper]()
    for (i <- 1 to NumberOfMockObjects) {
      edgeWrappers += EdgePbWrapper(
        Edge(srcNodeId = 2 * i - 1, dstNodeId = 2 * i, condensedEdgeType = Some(0)),
      )
      edgeWrappers += EdgePbWrapper(
        Edge(srcNodeId = 2 * i, dstNodeId = 2 * i - 1, condensedEdgeType = Some(0)),
      )
    }
    edgeWrappers
  }

  /**
      * Generated from Toy Graph data
      */
  def getMockMainInputSample: NodeAnchorBasedLinkPredictionSample = {
    // Create the Node for root_node
    val rootNode = Node(
      nodeId = 5,
      condensedNodeType = Some(0),
      featureValues = Seq(1.0f),
    )

    // Create the nodes for neighborhood
    val neighborhoodNodes = Seq(
      Node(nodeId = 1, condensedNodeType = Some(0), featureValues = Seq(1.0f)),
      Node(nodeId = 3, condensedNodeType = Some(0), featureValues = Seq(1.0f)),
      Node(nodeId = 0, condensedNodeType = Some(0), featureValues = Seq(1.0f)),
      Node(nodeId = 8, condensedNodeType = Some(0), featureValues = Seq(1.0f)),
      Node(nodeId = 7, condensedNodeType = Some(0), featureValues = Seq(1.0f)),
      Node(nodeId = 5, condensedNodeType = Some(0), featureValues = Seq(1.0f)),
      Node(nodeId = 6, condensedNodeType = Some(0), featureValues = Seq(1.0f)),
    )

    // Create the edges for neighborhood
    val neighborhoodEdges = Seq(
      Edge(srcNodeId = 1, dstNodeId = 5, condensedEdgeType = Some(0)),
      Edge(srcNodeId = 3, dstNodeId = 5, condensedEdgeType = Some(0)),
      Edge(srcNodeId = 0, dstNodeId = 1, condensedEdgeType = Some(0)),
      Edge(srcNodeId = 8, dstNodeId = 1, condensedEdgeType = Some(0)),
      Edge(srcNodeId = 7, dstNodeId = 3, condensedEdgeType = Some(0)),
      Edge(srcNodeId = 0, dstNodeId = 3, condensedEdgeType = Some(0)),
      Edge(srcNodeId = 5, dstNodeId = 1, condensedEdgeType = Some(0)),
      Edge(srcNodeId = 6, dstNodeId = 8, condensedEdgeType = Some(0)),
      Edge(srcNodeId = 1, dstNodeId = 8, condensedEdgeType = Some(0)),
    )

    // Create the pos_edges
    val posEdges = Seq(
      Edge(srcNodeId = 5, dstNodeId = 1, condensedEdgeType = Some(0)),
    )

    // Create the NodeAnchorBasedLinkPredictionSample
    NodeAnchorBasedLinkPredictionSample(
      rootNode = Some(rootNode),
      neighborhood = Some(Graph(nodes = neighborhoodNodes, edges = neighborhoodEdges)),
      posEdges = posEdges,
    )
  }

  def getMockRootedNodeNeighborhoodSample: RootedNodeNeighborhood = {
    // Create the Node for root_node
    val rootNode = Node(
      nodeId = 5,
      condensedNodeType = Some(0),
      featureValues = Seq(1.0f),
    )

    // Create the nodes for neighborhood
    val neighborhoodNodes = Seq(
      Node(nodeId = 1, condensedNodeType = Some(0), featureValues = Seq(1.0f)),
      Node(nodeId = 3, condensedNodeType = Some(0), featureValues = Seq(1.0f)),
      Node(nodeId = 0, condensedNodeType = Some(0), featureValues = Seq(1.0f)),
      Node(nodeId = 8, condensedNodeType = Some(0), featureValues = Seq(1.0f)),
      Node(nodeId = 6, condensedNodeType = Some(0), featureValues = Seq(1.0f)),
      Node(nodeId = 5, condensedNodeType = Some(0), featureValues = Seq(1.0f)),
    )

    // Create the edges for neighborhood
    val neighborhoodEdges = Seq(
      Edge(srcNodeId = 1, dstNodeId = 5, condensedEdgeType = Some(0)),
      Edge(srcNodeId = 5, dstNodeId = 3, condensedEdgeType = Some(0)),
      Edge(srcNodeId = 0, dstNodeId = 1, condensedEdgeType = Some(0)),
      Edge(srcNodeId = 8, dstNodeId = 1, condensedEdgeType = Some(0)),
      Edge(srcNodeId = 6, dstNodeId = 3, condensedEdgeType = Some(0)),
      Edge(srcNodeId = 0, dstNodeId = 3, condensedEdgeType = Some(0)),
    )

    RootedNodeNeighborhood(
      rootNode = Some(rootNode),
      neighborhood = Some(Graph(neighborhoodNodes, neighborhoodEdges)),
    )
  }

}
