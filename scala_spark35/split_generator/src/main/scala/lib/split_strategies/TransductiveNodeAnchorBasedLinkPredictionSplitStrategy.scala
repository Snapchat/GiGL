package splitgenerator.lib.split_strategies

import common.types.pb_wrappers.GraphPbWrappers.EdgePbWrapper
import snapchat.research.gbml.graph_schema.Edge
import snapchat.research.gbml.graph_schema.Graph
import snapchat.research.gbml.graph_schema.Node
import snapchat.research.gbml.training_samples_schema.NodeAnchorBasedLinkPredictionSample
import snapchat.research.gbml.training_samples_schema.RootedNodeNeighborhood
import splitgenerator.lib.Types.DatasetSplits
import splitgenerator.lib.Types.DatasetSplits.DatasetSplit
import splitgenerator.lib.Types.LinkSplitType
import splitgenerator.lib.Types.LinkUsageTypes
import splitgenerator.lib.assigners.AbstractAssigners.Assigner

class TransductiveNodeAnchorBasedLinkPredictionSplitStrategy(
  splitStrategyArgs: Map[String, String],
  edgeToLinkSplitAssigner: Assigner[EdgePbWrapper, LinkSplitType])
    extends NodeAnchorBasedLinkPredictionSplitStrategy(
      splitStrategyArgs,
      edgeToLinkSplitAssigner,
    ) {

  val isDisjointMode: Boolean = splitStrategyArgs.getOrElse("is_disjoint_mode", "false").toBoolean

  /**
      * An input sample is a single NodeAnchorBasedLinkPredictionSample pbs output by SubgraphSampler.
      * An input sample's neighborhood contains structure used for message passing (in neighborhood field),
      * and positive and negative supervision links (in pos_edges and hard_neg_edges fields, respectively).
      * This splitting procedure aims to create up to 3 total output samples (up to 1 for each DatasetSplit:
      * train, val, test) which are NodeAnchorBasedLinkPredictionSample pbs that only contain
      *     (a) only relevant pos_edges and hard_neg_edges belonging to the split.
      *     (b) only message passing structure which should be visible in the split.
      * An output train-split sample needs to have >0 pos_edges in this setting for loss computation.
      * Output val/test-split samples may have 0 pos_edges (and even 0 hard_neg_edges), since these
      * samples can technically be used to evaluate random negative link likelihoods.
      *
      * @param sample
      * @param datasetSplit
      * @return
      */
  override def splitTrainingSample(
    sample: NodeAnchorBasedLinkPredictionSample,
    datasetSplit: DatasetSplit,
  ): Seq[NodeAnchorBasedLinkPredictionSample] = {

    // Add all supervision edges in input sample to relevant train/val/test samples.
    val supervisionPosEdges: Seq[Edge] = sample.posEdges.filter(edge =>
      shouldAddSupervisionEdgeToSplit(supervisionEdge = edge, datasetSplit = datasetSplit),
    )

    val supervisionNegEdges: Seq[Edge] = sample.negEdges.filter(edge =>
      shouldAddSupervisionEdgeToSplit(supervisionEdge = edge, datasetSplit = datasetSplit),
    )

    val supervisionHardNegEdges: Seq[Edge] = sample.hardNegEdges.filter(edge =>
      shouldAddSupervisionEdgeToSplit(supervisionEdge = edge, datasetSplit = datasetSplit),
    )

    // The train split cannot have 0 pos_edges as it used for Loss computation. Drop sample if that is the case
    // We could optionally also filter val and test samples that don't have pos_edges.
    // However, we currently leave them be, since we may want to compute metrics over only
    // negatives in val/test (potentially even just random negatives).
    // e.g. "what is the average similarity between pairs of negative samples?"
    // As such, output val/test samples may in the extreme have 0 pos_edges and 0 hard_neg_edges,
    // if all the corresponding input sample edges fall in other splits.
    if (supervisionPosEdges.isEmpty && datasetSplit == DatasetSplits.TRAIN) {
      return Seq.empty[NodeAnchorBasedLinkPredictionSample]
    }

    // Add all message passing edges in input sample to relevant train/val/test samples.
    val messagePassingEdges: Seq[Edge] = sample.neighborhood
      .getOrElse(throw new RuntimeException("Neighborhood does not exist in the sample"))
      .edges
      .filter(edge =>
        shouldAddMessagePassingEdgeToDatasetSplit(
          messagePassingEdge = edge,
          datasetSplit = datasetSplit,
        ),
      )

    // featurless nodePB to be included in the message passing graph
    // Remember that we also need features for the root node, which may not already appear in the neighborhood.
    // Hence, populate those features also.
    val messagePassingNodesFeatureless: Set[Node] =
      (messagePassingEdges ++ supervisionPosEdges ++ supervisionNegEdges ++ supervisionHardNegEdges)
        .flatMap(getFeaturelessNodePbsFromEdgePb(_))
        .toSet +
        Node(
          nodeId = sample.rootNode
            .getOrElse(throw new RuntimeException("Root Node does not exist for the sample"))
            .nodeId,
          condensedNodeType = sample.rootNode.get.condensedNodeType,
        ) // feautureless root node pb

    // Populate the the graph with node features from the input negihborhood graph
    val messagePassingNodesWithFeatures: Seq[Node] = sample.neighborhood
      .getOrElse(throw new RuntimeException("Neighborhood does not exist in the sample"))
      .nodes
      .filter(node =>
        shouldAddMessagePassingNodeFromNeighborhoodGraph(
          node = node,
          datasetSplit = datasetSplit,
          messagePassingNodes = messagePassingNodesFeatureless,
        ),
      )

    val splitSample = Seq(
      NodeAnchorBasedLinkPredictionSample(
        rootNode = sample.rootNode,
        hardNegEdges = supervisionHardNegEdges,
        posEdges = supervisionPosEdges,
        negEdges = supervisionNegEdges,
        neighborhood = Some(Graph(messagePassingNodesWithFeatures, messagePassingEdges)),
      ),
    )

    // drop output samples according to subsampling ratio
    if (mainSamplesSubsamplingRatio.shouldSubsample) {
      subsample(
        splitSample = splitSample,
        datasetSplit = datasetSplit,
        subsamplingRatio = mainSamplesSubsamplingRatio,
      )
    } else {
      splitSample
    }
  }

  /**
      * We use the same logic for filtering down only train/val/test-usable message passing graphs
      * as for the main samples here.  We emit 1 train, 1 val, and 1 test sample each random negative.
      */
  override def splitRootedNodeNeighborhoodTrainingSample(
    sample: RootedNodeNeighborhood,
    datasetSplit: DatasetSplit,
  ): Seq[RootedNodeNeighborhood] = {

    // Add all message passing edges in input sample to relevant train/val/test samples.
    val messagePassingEdges: Seq[Edge] = sample.neighborhood
      .getOrElse(throw new RuntimeException("Neighborhood does not exist in the sample"))
      .edges
      .filter(edge =>
        shouldAddMessagePassingEdgeToDatasetSplit(
          messagePassingEdge = edge,
          datasetSplit = datasetSplit,
        ),
      )

    // featurless nodePB to be included in the message passing graph
    // Remember that we also need features for the root node, which may not already appear in the neighborhood.
    // Hence, populate those features also.
    val messagePassingNodesFeatureless =
      messagePassingEdges
        .flatMap(getFeaturelessNodePbsFromEdgePb(_))
        .toSet +
        Node(
          nodeId = sample.rootNode
            .getOrElse(throw new RuntimeException("Root Node does not exist for the sample"))
            .nodeId,
          condensedNodeType = sample.rootNode.get.condensedNodeType,
        ) // feautureless root node pb

    // Populate the the graph with node features from the input negihborhood graph
    val messagePassingNodesWithFeatures: Seq[Node] = sample.neighborhood
      .getOrElse(throw new RuntimeException("Neighborhood does not exist in the sample"))
      .nodes
      .filter(node =>
        shouldAddMessagePassingNodeFromNeighborhoodGraph(
          node = node,
          datasetSplit = datasetSplit,
          messagePassingNodes = messagePassingNodesFeatureless,
        ),
      )

    val splitSample = Seq(
      RootedNodeNeighborhood(
        rootNode = sample.rootNode,
        neighborhood = Some(Graph(messagePassingNodesWithFeatures, messagePassingEdges)),
      ),
    )

    // drop output samples according to subsampling ratio
    if (rootedNodeNeighborhoodSubsamplingRatio.shouldSubsample) {
      subsample(
        splitSample = splitSample,
        datasetSplit = datasetSplit,
        subsamplingRatio = rootedNodeNeighborhoodSubsamplingRatio,
      )
    } else {
      splitSample
    }
  }

  /**
      * Here, we filter out edges which should not be used for message passing in each of the train/val/test settings.
      * There are two different transductive data scenarios typically considered in this setting, with differences
      * in splitting logic: we call them disjoint-training mode and non-disjoint training mode.
      * In a the non-disjoint training mode the proposed splitting logic is:
      *     Split all edges into train/val/test edge sets.
      *     - At train time: Use train edges both for message passing and supervision (the sets are equivalent).
      *     - At val time: Use train edges to predict val edges.
      *     - At test time: Use train & val edges to predict test edges.
      * There is some debate as to whether training supervision edges should be allowed for use in message passing.
      * The rationale is that this is exposing the model to information it should be trying to predict.
      * In such a disjoint-training setting (disjoint_train_ratio > 0.0), the proposed splitting logic is:
      *     Split all edges into train-message/train-supervision/val/test edge sets.
      *     - At train time: Use train message edges to predict train supervision edges (these are disjoint sets).
      *     - At val time: Use train message edges & train supervision edges to predict val edges.
      *     - At test time: Use train message edges & train supervision edges & val edges to predict test edges.
      * References:
      *     - https://zqfang.github.io/2021-08-12-graph-linkpredict/
      *     - http://snap.stanford.edu/class/cs224w-2020/slides/09-theory.pdf
      *     - https://www.youtube.com/watch?v=ewEW_EMzRuo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=25
      *     - DeepSnap implementation (https://snap.stanford.edu/deepsnap/modules/dataset.html, ref edge_train_mode)
      *     - PyG implementation (torch_geometric.transforms.RandomLinkSplit, ref disjoint_train_ratio)
      */
  private def shouldAddMessagePassingEdgeToDatasetSplit(
    messagePassingEdge: Edge,
    datasetSplit: DatasetSplit,
  ): Boolean = {
    val linkSplitType: LinkSplitType =
      edgeToLinkSplitAssigner.assign(EdgePbWrapper(messagePassingEdge))
    datasetSplit match {
      case DatasetSplits.TRAIN => {
        if (!isDisjointMode) {
          // In non-disjoint training, training message and supervision edges are the same.
          // Hence, all edges are MESSAGE_AND_SUPERVISION and can be used in message passing in training.
          linkSplitType.datasetSplit == DatasetSplits.TRAIN
        } else {
          // In disjoint mode, training MESSAGE and SUPERVISION edges are disjoint (do not overlap).
          // Only training MESSAGE edges can be used in message passing during training.
          linkSplitType.datasetSplit == DatasetSplits.TRAIN && linkSplitType.edgeUsage == LinkUsageTypes.MESSAGE
        }
      }
      case DatasetSplits.VAL =>
        // val phase to only contain Train edge splits
        linkSplitType.datasetSplit == DatasetSplits.TRAIN
      case DatasetSplits.TEST =>
        // test phase to only contain Train and Val edge splits
        linkSplitType.datasetSplit == DatasetSplits.TRAIN || linkSplitType.datasetSplit == DatasetSplits.VAL
    }

    // All test edges should not be used for any message passing in train/val/test phases.
  }

  /**
    * Helper function to filter out the supervision edges for a particular split
    *
    * @param supervisionEdge
    * @param datasetSplit
    * @return
    */
  private def shouldAddSupervisionEdgeToSplit(
    supervisionEdge: Edge,
    datasetSplit: DatasetSplit,
  ): Boolean = {
    val linkSplitType: LinkSplitType =
      edgeToLinkSplitAssigner.assign(EdgePbWrapper(supervisionEdge))

    // Training MESSAGE edges must not be used for supervision, hence we do not
    // retain them in pos_edges.
    val isMessageEdge: Boolean =
      linkSplitType.edgeUsage == LinkUsageTypes.MESSAGE && datasetSplit == DatasetSplits.TRAIN

    (!isMessageEdge && linkSplitType.datasetSplit == datasetSplit)
  }

}
