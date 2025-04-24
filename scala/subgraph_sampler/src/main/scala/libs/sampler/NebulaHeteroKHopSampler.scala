package libs.sampler

import com.vesoft.nebula.client.graph.data.ResultSet
import common.graphdb.NebulaGraphDBClient
import common.types.GraphTypes.NodeType
import common.types.Metapath
import common.types.pb_wrappers.GraphMetadataPbWrapper
import common.utils.TrainingSamplesHelper.mergeRootedNodeNeighborhoods
import snapchat.research.gbml.graph_schema.Edge
import snapchat.research.gbml.graph_schema.EdgeType
import snapchat.research.gbml.graph_schema.Graph
import snapchat.research.gbml.graph_schema.Node
import snapchat.research.gbml.training_samples_schema.RootedNodeNeighborhood

import scala.collection.JavaConverters._

class NebulaHeteroKHopSampler(
  graphDBClient: NebulaGraphDBClient,
  graphMetadataPbWrapper: GraphMetadataPbWrapper)
    extends KHopSampler {

  // specific to the bipartite use case, where we assume only the first edge type is ingested
  val ingestedEdgeType: String = graphMetadataPbWrapper.edgeTypeToRelationMap
    .get(graphMetadataPbWrapper.condensedEdgeTypeMap.get(0).get)
    .get
  println(f"ingestedEdgeType: ${ingestedEdgeType}")

  def getKHopSubgraphForRootNodes(
    rootNodeIds: Seq[Int],
    metapaths: Seq[Metapath],
    numNeighborsToSample: Int,
  ): Seq[RootedNodeNeighborhood] = {
    // TODO (yliu2-sc) this implementation could potentially be optimized by batching queries
    rootNodeIds.toList.map(rootNodeId =>
      getKHopSubgraphForRootNode(rootNodeId, metapaths, numNeighborsToSample),
    )
  }

  def getKHopSubgraphForRootNode(
    rootNodeId: Int,
    metapaths: Seq[Metapath],
    numNeighborsToSample: Int,
  ): RootedNodeNeighborhood = {
    // TODO (yliu2-sc) Can possibly combine all metapaths into one query in the future
    val kHopSubgraphForMetapaths: Seq[RootedNodeNeighborhood] =
      metapaths
        .map(metapath => getSubgraphOnMetapath(rootNodeId, metapath, numNeighborsToSample))
    mergeRootedNodeNeighborhoods(kHopSubgraphForMetapaths)
  }

  // Currently only support Bipartite graph with one edge type and the reverse,
  // assuming features in both directions and the same
  def getSubgraphOnMetapath(
    rootNodeId: Int,
    metapath: Metapath,
    numNeighborsToSample: Int,
  ): RootedNodeNeighborhood = {
    val metapathEdgeTypes =
      metapath.path.map(relation => graphMetadataPbWrapper.relationToEdgeTypeMap.get(relation).get)

    // TODO (yliu2-sc) Only for bipartite graph with one edge type ingested, will need to update logic for full HGS
    val queryStringFirstHop = if (metapath.path(1) == ingestedEdgeType) {
      getFirstHopGoQuery(rootNodeId, metapath.path(1), numNeighborsToSample)
    } else {
      getSecondHopGoQuery(rootNodeId, metapath.path(0), numNeighborsToSample)
    }

    // GO //
    // Run graphDB queries
    val resultSetOneHop = graphDBClient.executeQuery(queryStringFirstHop)
    val oneHopNodeIds = if (resultSetOneHop.rowsSize() > 0) {
      resultSetOneHop.colValues("src").asScala.map(_.asLong().toInt).toSeq
    } else {
      Seq.empty[Int]
    }

    // if node is isolated, return the empty result set
    if (oneHopNodeIds.isEmpty) {
      return resultSetsGoQueryToRNN(
        rootNodeId = rootNodeId,
        resultSets = Seq(resultSetOneHop),
        metapath = metapath,
      )
    }
    val queryStringSecondHop = if (metapath.path(1) == ingestedEdgeType) {
      oneHopNodeIds
        .map(nodeId =>
          getSecondHopGoQuery(
            nodeId,
            metapath.path(1),
            numNeighborsToSample,
          ),
        )
        .mkString("\nUNION\n")
    } else {
      oneHopNodeIds
        .map(nodeId =>
          getFirstHopGoQuery(
            nodeId,
            metapath.path(0),
            numNeighborsToSample,
          ),
        )
        .mkString("\nUNION\n")
    }
    val resultSetTwoHop = graphDBClient.executeQuery(queryStringSecondHop)

    if (metapath.path(1) == ingestedEdgeType) {
      mergeRootedNodeNeighborhoods(
        Seq(
          resultSetsGoQueryToRNN(
            rootNodeId = rootNodeId,
            resultSets = Seq(resultSetOneHop),
            metapath = metapath,
            reverse = true,
          ),
          resultSetsGoQueryToRNN(
            rootNodeId = rootNodeId,
            resultSets = Seq(resultSetTwoHop),
            metapath = metapath,
          ),
        ),
      )
    } else {
      mergeRootedNodeNeighborhoods(
        Seq(
          resultSetsGoQueryToRNN(
            rootNodeId = rootNodeId,
            resultSets = Seq(resultSetOneHop),
            metapath = metapath,
          ),
          resultSetsGoQueryToRNN(
            rootNodeId = rootNodeId,
            resultSets = Seq(resultSetTwoHop),
            metapath = metapath,
            reverse = true,
          ),
        ),
      )
    }
  }

  // return positive edges and positive neighborhood based on 1 hop as positive
  // if isolated node, return empty result set, as we don't want to include in training samples
  def samplePositiveEdgeNeighborhoods(
    rootNodeId: Int,
    numPositives: Int,
    condensed_edge_type: Int,
    metapaths: Seq[Metapath],
    numNeighborsToSample: Int,
  ): (Seq[Edge], Seq[RootedNodeNeighborhood]) = {
    val edgeType = graphMetadataPbWrapper.condensedEdgeTypeMap.get(condensed_edge_type).get
    val relation = graphMetadataPbWrapper.edgeTypeToRelationMap.get(edgeType).get
    // supervision story_to_user, ingested user_to_story. For bipartite. Need to update for full HGS.
    val positiveNeighborQuery = if (relation != ingestedEdgeType) {
      getFirstHopGoQuery(
        nodeId = rootNodeId,
        edgeType = ingestedEdgeType,
        numNeighborsToSample = numPositives,
      )
    } else {
      getSecondHopGoQuery(
        nodeId = rootNodeId,
        edgeType = ingestedEdgeType,
        numNeighborsToSample = numPositives,
      )
    }
    val resultSetOneHop = graphDBClient.executeQuery(positiveNeighborQuery)
    val oneHopNodeIds = if (resultSetOneHop.rowsSize() > 0) {
      resultSetOneHop.colValues("src").asScala.map(_.asLong().toInt).toSeq
    } else {
      return (Seq.empty[Edge], Seq.empty[RootedNodeNeighborhood])
    }

    val positiveEdges = oneHopNodeIds.map(nodeId => {
      Edge(
        srcNodeId = nodeId,
        dstNodeId = rootNodeId,
        featureValues = Array.empty[Float],
        condensedEdgeType = Some(condensed_edge_type),
      )
    })

    val positiveNeighborhoods: Seq[RootedNodeNeighborhood] = oneHopNodeIds.map(nodeId => {
      getKHopSubgraphForRootNodes(
        rootNodeIds = Seq(nodeId),
        metapaths = metapaths,
        numNeighborsToSample = numNeighborsToSample,
      )(0)
    })

    (positiveEdges, positiveNeighborhoods)
  }

  def getFirstHopGoQuery(
    nodeId: Int,
    edgeType: String,
    numNeighborsToSample: Int,
    reversely: Boolean = false,
  ): String = {
    // user_to_story edge is used as edgeType
    val query_string =
      f"""GO 1 STEP FROM ${nodeId} OVER ${edgeType}
                           YIELD ${edgeType}._src AS dst, ${edgeType}._dst AS src,
                           TYPE(EDGE) AS edge_type
                           LIMIT [${numNeighborsToSample}]"""
    query_string
  }

  def getSecondHopGoQuery(
    nodeId: Int,
    edgeType: String,
    numNeighborsToSample: Int,
    reversely: Boolean = false,
  ): String = {
    val query_string =
      f"""GO 1 STEP FROM ${nodeId} OVER ${edgeType} REVERSELY
                           YIELD ${edgeType}._src AS dst, ${edgeType}._dst AS src,
                           TYPE(EDGE) AS edge_type
                           LIMIT [${numNeighborsToSample}]"""
    query_string
  }

  // NOTE (yliu2) For isolated nodes, we return the neighborhood with only the node itself
  //  we could handle the isolated inference case by adding a self loop to the node in trainer,
  //  and the GNN models will be updated to handle isolated cases for inference.
  //  Isolated root nodes are not included in training or training samples.
  //  In the long term, for Stage 2,3 we will have self loop edges in GraphMetadata.
  def resultSetsGoQueryToRNN(
    rootNodeId: Int,
    resultSets: Seq[ResultSet],
    metapath: Metapath,
    reverse: Boolean = false,
  ): RootedNodeNeighborhood = {
    val nodes = scala.collection.mutable.ListBuffer[Node]()
    val edges = scala.collection.mutable.ListBuffer[Edge]()

    for (resultSet <- resultSets) {
      for (i <- 0 until resultSet.rowsSize()) {
        val record = resultSet.rowValues(i)

        val src                = record.get("src").asLong().toInt
        val dst                = record.get("dst").asLong().toInt
        val edge_relation_type = record.get("edge_type").asString()

        // we only query graph structure, setting feature to empty to hydrate later on
        val edge_features     = Array.empty[Float]
        val src_node_features = Array.empty[Float]
        val dst_node_features = Array.empty[Float]

        val edge_type: EdgeType =
          graphMetadataPbWrapper.relationToEdgeTypeMap.get(edge_relation_type).get
        val src_node_type: NodeType =
          graphMetadataPbWrapper.edgeTypeToSrcDstNodeTypesMap.get(edge_type).get._1
        val dst_node_type: NodeType =
          graphMetadataPbWrapper.edgeTypeToSrcDstNodeTypesMap.get(edge_type).get._2

        val (
          condensed_edge_type,
          condensed_src_node_type,
          condensed_dst_node_type,
        ) = if (reverse) {
          // Reverse it
          val reverse_edge_type: EdgeType =
            graphMetadataPbWrapper.getEdgeTypeFromSrcDstNodeTypes(dst_node_type, src_node_type).get
          (
            graphMetadataPbWrapper.edgeTypeToCondensedEdgeTypeMap.get(reverse_edge_type).get,
            graphMetadataPbWrapper.nodeTypeToCondensedNodeTypeMap.get(dst_node_type).get,
            graphMetadataPbWrapper.nodeTypeToCondensedNodeTypeMap.get(src_node_type).get,
          )
        } else {
          (
            graphMetadataPbWrapper.edgeTypeToCondensedEdgeTypeMap.get(edge_type).get,
            graphMetadataPbWrapper.nodeTypeToCondensedNodeTypeMap.get(src_node_type).get,
            graphMetadataPbWrapper.nodeTypeToCondensedNodeTypeMap.get(dst_node_type).get,
          )
        }

        nodes ++= Seq(
          Node(
            nodeId = src,
            featureValues = src_node_features,
            condensedNodeType = Some(condensed_src_node_type),
          ),
          Node(
            nodeId = dst,
            featureValues = dst_node_features,
            condensedNodeType = Some(condensed_dst_node_type),
          ),
        )
        edges += Edge(
          srcNodeId = src,
          dstNodeId = dst,
          featureValues = edge_features,
          condensedEdgeType = Some(condensed_edge_type),
        )
      }
    }
    val rootNodeType =
      graphMetadataPbWrapper.relationToEdgeTypeMap.get(metapath.path(0)).get.dstNodeType
    val condensedRootNodeType =
      graphMetadataPbWrapper.nodeTypeToCondensedNodeTypeMap.get(rootNodeType).get

    var rootNode: Option[Node] =
      nodes.find((node: Node) =>
        node.nodeId == rootNodeId && node.condensedNodeType == Some(condensedRootNodeType),
      ) // returns first match of Node proto

    if (rootNode.isEmpty) {
      logger.debug(
        "Root node not found in nodes set in the Neighborhood Graph from query result. " +
          f"Root node is likely an isolated node. RootNodeId: ${rootNodeId}. nodes: ${nodes}. edges ${edges}.",
      )
      val root_node_features = Array.empty[Float]
      rootNode = Some(
        Node(
          nodeId = rootNodeId,
          featureValues = root_node_features,
          condensedNodeType = Some(condensedRootNodeType),
        ),
      )
      nodes += rootNode.get
    }
    assert(rootNode.nonEmpty, "Root node is empty.")

//    println(f"goToRNN rootNodeId: ${rootNodeId} nodes: ${nodes.map(_.nodeId)}")
//    println(f"goToRNN rootNodeId: ${rootNodeId} nodesSeq: ${nodes.toSeq.map(_.nodeId)}")
//    println(f"goToRNN rootNodeId: ${rootNodeId} rnnNodes ${RootedNodeNeighborhood(rootNode = rootNode, neighborhood = Some(Graph(nodes = nodes.toSeq, edges = edges.toSeq))).neighborhood.get.nodes.map(_.nodeId)}")
    RootedNodeNeighborhood(
      rootNode = rootNode,
      neighborhood = Some(
        Graph(nodes = nodes.distinct.toSeq, edges = edges.distinct.toSeq),
      ), // requires Option[Graph], so add Some()
    )
  }
}
