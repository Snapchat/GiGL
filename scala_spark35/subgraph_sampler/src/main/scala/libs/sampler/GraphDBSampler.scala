package libs.sampler

import common.graphdb.DBClient
import common.graphdb.DBResult
import common.graphdb.KHopSamplerService
import common.graphdb.QueryResponseTranslator
import common.graphdb.nebula.types.nGQLQuery
import common.types.GraphTypes.CondensedNodeType
import common.types.GraphTypes.NodeId
import common.types.SamplingOpDAG
import common.types.SamplingOpNode
import common.types.pb_wrappers.GraphMetadataPbWrapper
import snapchat.research.gbml.graph_schema.Edge
import snapchat.research.gbml.graph_schema.EdgeType
import snapchat.research.gbml.graph_schema.Graph
import snapchat.research.gbml.graph_schema.Node
import snapchat.research.gbml.subgraph_sampling_strategy.RandomUniform
import snapchat.research.gbml.subgraph_sampling_strategy.SamplingDirection
import snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp
import snapchat.research.gbml.training_samples_schema.RootedNodeNeighborhood

import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet
import scala.collection.mutable.Queue

import util.control.Breaks.break
import util.control.Breaks.breakable

case class SamplerOpRecord(
  val samplingOpNode: SamplingOpNode,
  val result: Seq[Tuple2[NodeId, CondensedNodeType]])

class GraphDBSampler(
  graphMetadataPbWrapper: GraphMetadataPbWrapper,
  dbClient: DBClient[DBResult],
  queryResponseTranslator: QueryResponseTranslator)
    extends KHopSamplerService
    with Serializable {

  def getKHopSubgraphForRootNode(
    rootNode: Node,
    samplingOpDag: SamplingOpDAG,
  ): RootedNodeNeighborhood = {

    // Traverse through the dag to formulate the RootedNodeNeighborhood
    val samplingOpNodeQueue = Queue[SamplingOpNode]()

    // Enqueue the root samplingOpNodes as they will be processed first
    samplingOpDag.rootSamplingOpNodes.foreach(samplingOpNode =>
      samplingOpNodeQueue.enqueue(samplingOpNode),
    )

    val mapSamplingOpNameToParsedResult: HashMap[String, Tuple2[HashSet[Edge], HashSet[Node]]] =
      HashMap.empty[String, Tuple2[HashSet[Edge], HashSet[Node]]]
    val mapSamplingOpNameToOp: HashMap[String, SamplingOpNode] =
      HashMap.empty[String, SamplingOpNode]

    // We queue root nodes first (above).
    // We dequeue a sampling op, process it, and then we queue its children. If an op is found
    // not having their parents completed it is just skipped. This is because a child with
    // multiple parents will get queued once for each parent. Meaning after the last parent
    // is executed the child will finally execute.
    while (samplingOpNodeQueue.nonEmpty) {
      breakable {
        val samplingOpNode = samplingOpNodeQueue.dequeue()
        val nextDstNodes: Seq[Node] = if (samplingOpNode.parentOpNames.size == 0) { // root ops
          Seq(rootNode)
        } else {
          val areAllParentsProcessed: Boolean = samplingOpNode.parentOpNames.forall(parentOpName =>
            mapSamplingOpNameToParsedResult.contains(parentOpName),
          )
          if (!areAllParentsProcessed) {
            // Skip this samplingOpNode for now. It will get requeued if not already by other parents
            // If the graph is incomplete it will never get requeued and path will be skipped
            // TODO: (svij) We probably need correctness checks instead of just skipping
            break
          }
          samplingOpNode.parentOpNames
            .map(parentOpName => mapSamplingOpNameToParsedResult(parentOpName))
            .flatMap(parsedResult => parsedResult._2)
            .toSeq
        }
        if (nextDstNodes.isEmpty) {
          // There were no parent nodes to process. Our traversal ends here.
          break
        }
        val query: nGQLQuery = queryResponseTranslator.translate(
          samplingOp = samplingOpNode.samplingOp,
          rootNodes = nextDstNodes,
        )
        var result: DBResult                                   = null
        var parsedResult: Tuple2[HashSet[Edge], HashSet[Node]] = null
        try {
          result = dbClient.executeQuery(query)
        } catch {
          case e: Exception => {
            throw new RuntimeException(
              s"ERROR: Failed to query: ${query}, samplingOp: ${samplingOpNode.samplingOp}; nextDstNodes: ${nextDstNodes} result: ${result}: ${e}\n${e.fillInStackTrace()}",
            )
          }
        }
        try {
          parsedResult = queryResponseTranslator.parseEdgesAndNodesFromResult(
            result = result,
            samplingOp = samplingOpNode.samplingOp,
          )
        } catch {
          case e: Exception => {
            throw new RuntimeException(
              s"ERROR: Failed to parse result: ${result} from query: ${query}, result: ${result}: ${e}\n${e.fillInStackTrace()}",
            )
          }
        }

        mapSamplingOpNameToParsedResult(samplingOpNode.opName) = parsedResult
        mapSamplingOpNameToOp(samplingOpNode.opName) = samplingOpNode

        // Now we can queue the child samplingOpNodes
        samplingOpNode.childSamplingOpNodes.foreach(childSamplingOpNode => {
          samplingOpNodeQueue.enqueue(childSamplingOpNode)
        })
      }
    }

    val neighborhoodEdgeSet = HashSet.empty[Edge]
    val neighborhoodNodeSet = HashSet.empty[Node]
    mapSamplingOpNameToParsedResult.foreach {
      case (opName, parsedResult) => {
        val samplingOpNode = mapSamplingOpNameToOp(opName)
        val (edges, nodes) = parsedResult
        neighborhoodEdgeSet ++= edges
        neighborhoodNodeSet ++= nodes
      }
    }
    // We need to add the rootNode to the neighborhood as queries only return next hop nodes
    // So it will never get added to the neighborhood if we don't add it here.
    neighborhoodNodeSet += rootNode
    val rootedNodeNeighborhood = RootedNodeNeighborhood(
      rootNode = Some(rootNode),
      neighborhood = Some(
        Graph(
          edges = neighborhoodEdgeSet.toSeq,
          nodes = neighborhoodNodeSet.toSeq,
        ),
      ),
    )
    rootedNodeNeighborhood
  }

  def getKHopSubgraphForRootNodes(
    rootNodes: Seq[Node],
    samplingOpDag: SamplingOpDAG,
  ): Seq[RootedNodeNeighborhood] = {
    // Batching is not supported yet
    rootNodes.map(rootNode => getKHopSubgraphForRootNode(rootNode, samplingOpDag))
  }

  def _assertMessagePassingOpDagOperatesOnPositiveNodeType(
    positiveNodeMessagePasingSamplingOpDag: SamplingOpDAG,
    positiveEdgeType: EdgeType,
  ): Unit = {
    val positiveNodeType = positiveEdgeType.dstNodeType
    positiveNodeMessagePasingSamplingOpDag.rootSamplingOpNodes.foreach(rootSamplingOpNode => {
      if (rootSamplingOpNode.samplingOp.edgeType.get.dstNodeType != positiveNodeType) {
        // TODO: (svij) Will need to be updated for Stage 2
        throw new IllegalArgumentException(
          s"SamplingOpDAG should have edgeType ${positiveEdgeType} matching for all sampling ops. Mismatch for ${rootSamplingOpNode}",
        )
      }
    })
  }

  // TODO: (svij) Tech debt: Sampler should not have knowledge of "positive edges" - this is task specific design
  def samplePositiveEdgeNeighborhoods(
    rootNode: Node,
    edgeType: EdgeType,
    numPositives: Int,
    samplingOpDag: SamplingOpDAG,
  ): Tuple2[Seq[Edge], Seq[Graph]] = {
    _assertMessagePassingOpDagOperatesOnPositiveNodeType(
      positiveNodeMessagePasingSamplingOpDag = samplingOpDag,
      positiveEdgeType = edgeType,
    )
    val positiveEdgesSamplingOp = SamplingOp(
      opName = "samplePositiveEdges",
      edgeType = Some(edgeType),
      // TODO: (svij) This can be configured from yaml in the future
      samplingMethod = SamplingOp.SamplingMethod.RandomUniform(
        RandomUniform(numPositives),
      ),
      inputOpNames = Seq(),
      samplingDirection = SamplingDirection.OUTGOING,
    )
    val queryPositiveEdges: nGQLQuery = queryResponseTranslator.translate(
      samplingOp = positiveEdgesSamplingOp,
      rootNode = rootNode,
    )
    val resultPositives: DBResult = dbClient.executeQuery(queryPositiveEdges)
    val (positiveEdges, positiveNodes) = queryResponseTranslator.parseEdgesAndNodesFromResult(
      result = resultPositives,
      samplingOp = positiveEdgesSamplingOp,
    )
    val positiveNeighborhoods: Seq[Graph] = positiveNodes
      .map(positiveNode =>
        getKHopSubgraphForRootNode(
          rootNode = positiveNode,
          samplingOpDag = samplingOpDag,
        ).neighborhood.get,
      )
      .toSeq

    (
      positiveEdges.toSeq,
      positiveNeighborhoods,
    )
  }

  def setup(): Unit = {
    dbClient.connect()
  }

  def teardown(): Unit = {
    dbClient.terminate()
  }

}
