package libs.sampler

import com.typesafe.scalalogging.LazyLogging
import com.vesoft.nebula.client.graph.data.ResultSet
import common.graphdb.KHopSamplerService
import common.graphdb.nebula.NebulaDBClient
import common.types.GraphTypes.NodeType
import common.types.Metapath
import common.types.SamplingOpDAG
import common.types.pb_wrappers.GbmlConfigPbWrapper
import common.types.pb_wrappers.GraphMetadataPbWrapper
import common.types.pb_wrappers.RootedNodeNeighborhoodPbWrapper
import snapchat.research.gbml.graph_schema.Edge
import snapchat.research.gbml.graph_schema.EdgeType
import snapchat.research.gbml.graph_schema.Graph
import snapchat.research.gbml.graph_schema.Node
import snapchat.research.gbml.training_samples_schema.RootedNodeNeighborhood

import scala.collection.JavaConverters._

object NebulaSamplerService extends LazyLogging {
  def apply(
    gbmlConfigWrapper: GbmlConfigPbWrapper,
  ): NebulaSamplerService = {
    val graphMetadataPbWrapper  = gbmlConfigWrapper.graphMetadataPbWrapper
    val subgraphSamplerConfigPb = gbmlConfigWrapper.subgraphSamplerConfigPb
    // Nebula graphDB parameters
    val graphdbArgs = subgraphSamplerConfigPb.graphDbConfig.get.graphDbArgs
    val graphSpace = graphdbArgs.getOrElse(
      "space",
      throw new Exception("NebulaClient requires 'space' field in graphdbArgs."),
    )
    val hosts = graphdbArgs
      .getOrElse(
        "hosts",
        throw new Exception("NebulaClient requires 'hosts' field in graphdbArgs."),
      )
      .split(";")
    // all hosts should be using the same port
    val port = graphdbArgs
      .getOrElse(
        "port",
        throw new Exception("NebulaClient requires 'port' field in graphdbArgs."),
      )
      .toInt
    val graphDBClient =
      new NebulaDBClient(graphSpace = graphSpace, hosts = hosts, port = port)
    graphDBClient.connect()
    new NebulaSamplerService(
      graphDBClient = graphDBClient,
      graphMetadataPbWrapper = graphMetadataPbWrapper,
    )
  }
}
class NebulaSamplerService(
  graphDBClient: NebulaDBClient,
  graphMetadataPbWrapper: GraphMetadataPbWrapper)
    extends KHopSamplerService
    with LazyLogging {

  // specific to the bipartite use case, where we assume only the first edge type is ingested
  val ingestedEdgeType: String = graphMetadataPbWrapper.edgeTypeToRelationMap
    .get(graphMetadataPbWrapper.graphMetadataPb.condensedEdgeTypeMap.get(0).get)
    .get
  println(f"ingestedEdgeType: ${ingestedEdgeType}")

  def setup(): Unit = {
    graphDBClient.connect()
  }
  def teardown(): Unit = {
    graphDBClient.terminate()
  }

  def getKHopSubgraphForRootNode(
    rootNode: Node,
    samplingOpDag: SamplingOpDAG,
  ): RootedNodeNeighborhood = {
    val rootNodeId = rootNode.nodeId
    throw new NotImplementedError(
      "Not implemented; metapaths need to be deprecated in favor of SamplingOpDAG",
    )
    // getKHopSubgraphForRootNode(rootNodeId, metapaths, numNeighborsToSample)
  }

  def getKHopSubgraphForRootNodes(
    rootNodes: Seq[Node],
    samplingOpDag: SamplingOpDAG,
  ): Seq[RootedNodeNeighborhood] = {
    val rootNodeIds = rootNodes.map(_.nodeId)
    throw new NotImplementedError(
      "Not implemented; metapaths need to be deprecated in favor of SamplingOpDAG",
    )
    // rootNodeIds.toList.map(rootNodeId =>
    //   getKHopSubgraphForRootNode(rootNodeId, metapaths, numNeighborsToSample),
    // )
  }

  def getKHopSubgraphForRootNode(
    rootNodeId: Int,
    metapaths: Seq[Metapath],
    numNeighborsToSample: Int,
  ): RootedNodeNeighborhood = {
    // TODO (yliu2) Can possibly combine all metapaths into one query in the future
    val kHopSubgraphForMetapaths: Seq[RootedNodeNeighborhood] =
      metapaths
        .map(metapath => getSubgraphOnMetapath(rootNodeId, metapath, numNeighborsToSample))
    RootedNodeNeighborhoodPbWrapper.mergeRootedNodeNeighborhoods(kHopSubgraphForMetapaths)
  }

  // Currently only support Bipartite graph with one edge type and the reverse,
  // assuming features in both directions and the same
  def getSubgraphOnMetapath(
    rootNodeId: Int,
    metapath: Metapath,
    numNeighborsToSample: Int,
  ): RootedNodeNeighborhood = {
    val metapathEdgeTypes =
      metapath.path.map(relation =>
        graphMetadataPbWrapper.relationToEdgeTypesMap.get(relation).get(0),
      )

    // TODO (yliu2) Only for bipartite graph with one edge type ingested, will need to update logic for full HGS
    val queryStringFirstHop = if (metapath.path(1) == ingestedEdgeType) {
      getOneHopGoQuery(rootNodeId, metapath.path(1), numNeighborsToSample)
    } else {
      getOneHopGoQuery(rootNodeId, metapath.path(0), numNeighborsToSample, reversely = true)
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
          getOneHopGoQuery(
            nodeId,
            metapath.path(1),
            numNeighborsToSample,
            reversely = true,
          ),
        )
        .mkString("\nUNION\n")
    } else {
      oneHopNodeIds
        .map(nodeId =>
          getOneHopGoQuery(
            nodeId,
            metapath.path(0),
            numNeighborsToSample,
          ),
        )
        .mkString("\nUNION\n")
    }
    val resultSetTwoHop = graphDBClient.executeQuery(queryStringSecondHop)

    if (metapath.path(1) == ingestedEdgeType) {
      RootedNodeNeighborhoodPbWrapper.mergeRootedNodeNeighborhoods(
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
      RootedNodeNeighborhoodPbWrapper.mergeRootedNodeNeighborhoods(
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
    rootNode: Node,
    edgeType: EdgeType,
    numPositives: Int,
    samplingOpDag: SamplingOpDAG,
  ): Tuple2[Seq[Edge], Seq[Graph]] = {
    val rootNodeId        = rootNode.nodeId
    val relation          = graphMetadataPbWrapper.edgeTypeToRelationMap.get(edgeType).get
    val condensedEdgeType = graphMetadataPbWrapper.edgeTypeToCondensedEdgeTypeMap.get(edgeType).get
    // supervision story_to_user, ingested user_to_story. For bipartite. Need to update for full HGS.
    // for positives we take out edges along the direction of supervision edge type
    val positiveNeighborQuery = if (relation != ingestedEdgeType) {
      getOneHopGoQuery(
        nodeId = rootNodeId,
        edgeType = ingestedEdgeType,
        numNeighborsToSample = numPositives,
        reversely = true,
        edgeDirection = "OUT",
      )
    } else {
      getOneHopGoQuery(
        nodeId = rootNodeId,
        edgeType = ingestedEdgeType,
        numNeighborsToSample = numPositives,
        edgeDirection = "OUT",
      )
    }
    val resultSetOneHop = graphDBClient.executeQuery(positiveNeighborQuery)
    val oneHopNodeIds = if (resultSetOneHop.rowsSize() > 0) {
      resultSetOneHop.colValues("src").asScala.map(_.asLong().toInt).toSeq
    } else {
      return (Seq.empty[Edge], Seq.empty[Graph])
    }

    val positiveEdges = oneHopNodeIds.map(nodeId => {
      Edge(
        srcNodeId = nodeId,
        dstNodeId = rootNodeId,
        featureValues = Array.empty[Float],
        condensedEdgeType = Some(condensedEdgeType),
      )
    })

    // TODO: (svij) Tech debt Fix this. Currently below does not work (intentionally), as prior
    // sampling strategy focused solely aourund bipartite graph has been deprecated.
    // val positiveNeighborhoods: Seq[RootedNodeNeighborhood] = oneHopNodeIds.map(nodeId => {
    //   getKHopSubgraphForRootNodes(
    //     rootNodeIds = Seq(nodeId),
    //     metapaths = metapaths,
    //     numNeighborsToSample = numNeighborsToSample,
    //   )(0)
    // })
    val positiveNeighborhoods: Seq[Graph] = positiveEdges.map(edge =>
      Graph(
        nodes = Seq(Node(edge.dstNodeId)),
      ),
    )

    (positiveEdges, positiveNeighborhoods)
  }

  def getOneHopGoQuery(
    nodeId: Int,
    edgeType: String,
    numNeighborsToSample: Int,
    reversely: Boolean = false,
    edgeDirection: String = "IN",
  ): String = {
    val query_string = if (edgeDirection == "IN") {
      if (reversely) {
        f"""GO 1 STEP FROM ${nodeId} OVER ${edgeType} REVERSELY
                           YIELD ${edgeType}._src AS dst, ${edgeType}._dst AS src,
                           TYPE(EDGE) AS edge_type
                           LIMIT [${numNeighborsToSample}]"""
      } else {
        f"""GO 1 STEP FROM ${nodeId} OVER ${edgeType}
                           YIELD ${edgeType}._src AS dst, ${edgeType}._dst AS src,
                           TYPE(EDGE) AS edge_type
                           LIMIT [${numNeighborsToSample}]"""
      }
    } else {
      if (reversely) {
        f"""GO 1 STEP FROM ${nodeId} OVER ${edgeType} REVERSELY
                           YIELD ${edgeType}._src AS src, ${edgeType}._dst AS dst,
                           TYPE(EDGE) AS edge_type
                           LIMIT [${numNeighborsToSample}]"""
      } else {
        f"""GO 1 STEP FROM ${nodeId} OVER ${edgeType}
                           YIELD ${edgeType}._src AS src, ${edgeType}._dst AS dst,
                           TYPE(EDGE) AS edge_type
                           LIMIT [${numNeighborsToSample}]"""
      }
    }
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
          graphMetadataPbWrapper.relationToEdgeTypesMap.get(edge_relation_type).get(0)
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
            graphMetadataPbWrapper.getEdgeTypesFromSrcDstNodeTypes(dst_node_type, src_node_type)(0)
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
      graphMetadataPbWrapper.relationToEdgeTypesMap.get(metapath.path(0)).get(0).dstNodeType
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
      ), // requires Option[Graph], add Some()
    )
  }
}
