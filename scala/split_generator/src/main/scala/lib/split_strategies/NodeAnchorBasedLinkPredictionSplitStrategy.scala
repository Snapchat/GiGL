package splitgenerator.lib.split_strategies

import common.types.pb_wrappers.GraphPbWrappers.EdgePbWrapper
import snapchat.research.gbml.graph_schema.Edge
import snapchat.research.gbml.graph_schema.Node
import snapchat.research.gbml.training_samples_schema.NodeAnchorBasedLinkPredictionSample
import snapchat.research.gbml.training_samples_schema.RootedNodeNeighborhood
import splitgenerator.lib.Types.DatasetSplits.DatasetSplit
import splitgenerator.lib.Types.LinkSplitType
import splitgenerator.lib.Types.SplitSubsamplingRatio
import splitgenerator.lib.assigners.AbstractAssigners.Assigner

abstract class NodeAnchorBasedLinkPredictionSplitStrategy(
  splitStrategyArgs: Map[String, String],
  val edgeToLinkSplitAssigner: Assigner[EdgePbWrapper, LinkSplitType])
    extends SplitStrategy[NodeAnchorBasedLinkPredictionSample](splitStrategyArgs) {

  protected val rootedNodeNeighborhoodSubsamplingRatio: SplitSubsamplingRatio =
    SplitSubsamplingRatio(
      // ratio of random neg train samples to keep
      trainSubsamplingRatio = splitStrategyArgs
        .getOrElse("random_negative_train_subsampling_ratio", "1.0")
        .toFloat,
      // ratio of random neg val samples to keep
      valSubsamplingRatio = splitStrategyArgs
        .getOrElse("random_negative_val_subsampling_ratio", "1.0")
        .toFloat,
      // ratio of random neg test samples to keep
      testSubsamplingRatio = splitStrategyArgs
        .getOrElse("random_negative_test_subsampling_ratio", "1.0")
        .toFloat,
    )

  override val graphMetadataPbWrapper = edgeToLinkSplitAssigner.graphMetadataPbWrapper

  override def splitTrainingSample(
    sample: NodeAnchorBasedLinkPredictionSample,
    datasetSplit: DatasetSplit,
  ): Seq[NodeAnchorBasedLinkPredictionSample]

  def splitRootedNodeNeighborhoodTrainingSample(
    sample: RootedNodeNeighborhood,
    datasetSplit: DatasetSplit,
  ): Seq[RootedNodeNeighborhood]

  /**
    * Method to filter out the neighborhood graph nodes for a particular split.
    * This is required because neighborhood graph has Node features which is needed
    *
    * @param node
    * @param datasetSplit
    * @param messagePassingNodes
    * @return
    */

  def shouldAddMessagePassingNodeFromNeighborhoodGraph(
    node: Node,
    datasetSplit: DatasetSplit,
    messagePassingNodes: Set[
      Node,
    ],
  ): Boolean = {
    // filter out those where node present in either rootnode, supervision edges or messagepassing edges
    val featurelessNode = Node(nodeId = node.nodeId, condensedNodeType = node.condensedNodeType)
    messagePassingNodes.contains(featurelessNode)
  }

  /**
      * Gets the src and dst featureless (node_id and condensed_node_type) Node proto from the Edge proto.
      *
      * @param edge
      * @return
      */
  def getFeaturelessNodePbsFromEdgePb(
    edge: Edge,
  ): Seq[Node] = {
    graphMetadataPbWrapper
      .getFeaturelessNodePbsFromEdge(EdgePbWrapper(edge))
  }

}
