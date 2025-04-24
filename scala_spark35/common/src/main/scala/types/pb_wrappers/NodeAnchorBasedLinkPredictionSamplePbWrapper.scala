package common.types.pb_wrappers

import common.types.pb_wrappers.GraphPbWrappers.mergeGraphs
import snapchat.research.gbml.graph_schema.Edge
import snapchat.research.gbml.graph_schema.Graph
import snapchat.research.gbml.training_samples_schema.NodeAnchorBasedLinkPredictionSample
import snapchat.research.gbml.training_samples_schema.RootedNodeNeighborhood

object NodeAnchorBasedLinkPredictionSamplePbWrapper {

  /**
   * Construct a NodeAnchorBasedLinkPredictionSample instance,
   *  given positive and optional negative edges and neighborhoods.
   *
   * @param rnn The RootedNodeNeighborhood instance.
   * @param positiveEdges The sequence of positive edges.
   * @param positiveNeighborhoods The sequence of positive neighborhoods.
   * @param hardNegativeEdges The optional sequence of hard negative edges.
   * @param hardNegativeNeighborhoods The optional sequence of hard negative neighborhoods.
   * @return A NodeAnchorBasedLinkPredictionSample instance.
   */
  def NodeAnchorBasedLinkPredictionSampleFrom(
    rnn: RootedNodeNeighborhood,
    positiveEdges: Seq[Edge],
    positiveNeighborhoods: Seq[Graph],
    hardNegativeEdges: Option[Seq[Edge]],
    hardNegativeNeighborhoods: Option[Seq[Graph]],
  ): NodeAnchorBasedLinkPredictionSample = {

    val neighborhoodsToMerge = scala.collection.mutable.ListBuffer[Graph]()
    neighborhoodsToMerge += rnn.neighborhood.get
    neighborhoodsToMerge ++= positiveNeighborhoods
    if (hardNegativeNeighborhoods.isDefined) {
      neighborhoodsToMerge ++= hardNegativeNeighborhoods.get
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
