package common.utils

import snapchat.research.gbml.graph_schema.Edge
import snapchat.research.gbml.graph_schema.Graph
import snapchat.research.gbml.training_samples_schema.NodeAnchorBasedLinkPredictionSample
import snapchat.research.gbml.training_samples_schema.RootedNodeNeighborhood

object TrainingSamplesHelper {

  def mergeGraphs(graphs: Seq[Graph]): Graph = {
    val nodes = graphs.flatMap(graph => graph.nodes).distinct
    val edges = graphs.flatMap(graph => graph.edges).distinct
    Graph(nodes = nodes, edges = edges)
  }
  def mergeRootedNodeNeighborhoods(rnns: Seq[RootedNodeNeighborhood]): RootedNodeNeighborhood = {
    assert(rnns.nonEmpty, "Sequence of RootedNodeNeighborhood is empty")
    assert(
      checkAllRootNodesAreSame(rnns),
      "All root nodes should be the same in able to merge",
//      f"All root nodes should be the same in able to merge ${rnns.map(_.rootNode.get.nodeId)} ${rnns
//          .map(_.rootNode.get.condensedNodeType)} ${firstFeatureValues.indices
//          .map(i => featureValuesList(i) == firstFeatureValues(i))}",
    )

    val mergedGraph: Graph = mergeGraphs(rnns.flatMap(_.neighborhood))

    RootedNodeNeighborhood(rootNode = rnns.head.rootNode, neighborhood = Some(mergedGraph))
  }

  def checkAllRootNodesAreSame(rnns: Seq[RootedNodeNeighborhood]): Boolean = {
    val firstRootNode = rnns.head.rootNode
    rnns.forall(rnn =>
      rnn.rootNode.get.nodeId == firstRootNode.get.nodeId && rnn.rootNode.get.condensedNodeType == firstRootNode.get.condensedNodeType,
    )
  }

  def createNodeAnchorBasedLinkPredictionSample(
    rnn: RootedNodeNeighborhood,
    positiveEdges: Seq[Edge],
    positiveNeighborhoods: Seq[RootedNodeNeighborhood],
    hardNegativeEdges: Option[Seq[Edge]],
    hardNegativeNeighborhoods: Option[Seq[RootedNodeNeighborhood]],
  ): NodeAnchorBasedLinkPredictionSample = {

    val neighborhoodsToMerge = hardNegativeNeighborhoods match {
      case Some(negativeNeighborhoods) =>
        (Seq(rnn) ++ positiveNeighborhoods ++ negativeNeighborhoods).flatMap(_.neighborhood)
      case None =>
        (Seq(rnn) ++ positiveNeighborhoods).flatMap(_.neighborhood)
    }
    val mergedGraph: Graph = mergeGraphs(graphs = neighborhoodsToMerge)

    NodeAnchorBasedLinkPredictionSample(
      rootNode = rnn.rootNode,
      posEdges = positiveEdges,
      hardNegEdges = hardNegativeEdges.getOrElse(Seq.empty),
      neighborhood = Some(mergedGraph),
    )
  }

}
