package splitgenerator.lib.split_strategies

import common.types.pb_wrappers.GraphPbWrappers.EdgePbWrapper
import snapchat.research.gbml.graph_schema.Edge
import snapchat.research.gbml.training_samples_schema.NodeAnchorBasedLinkPredictionSample
import snapchat.research.gbml.training_samples_schema.RootedNodeNeighborhood
import splitgenerator.lib.Types.DatasetSplits
import splitgenerator.lib.Types.DatasetSplits.DatasetSplit
import splitgenerator.lib.Types.LinkSplitType
import splitgenerator.lib.Types.LinkUsageTypes
import splitgenerator.lib.assigners.AbstractAssigners.Assigner

/**
  * This class is used for splitting NodeAnchorBasedLinkPredictionSample pbs with user-defined labels provided
  * in the pb.pos_edges and pb.hard_neg_edges fields.  The splitting procedure is as follows:
  * for all train/val/test splits,
  * 1. we retain all pb.neighborhood.edges as message passing edges
  * 2. we mask out pb.pos_edges and pb.hard_neg_edges using the Assigner logic
  * @param splitStrategyArgs
  * @param edgeToLinkSplitAssigner
  */
class UserDefinedLabelsNodeAnchorBasedLinkPredictionSplitStrategy(
  splitStrategyArgs: Map[String, String],
  edgeToLinkSplitAssigner: Assigner[EdgePbWrapper, LinkSplitType])
    extends NodeAnchorBasedLinkPredictionSplitStrategy(
      splitStrategyArgs,
      edgeToLinkSplitAssigner,
    ) {

  /**
      * An input sample is a single NodeAnchorBasedLinkPrediction pbs output by SubgraphSampler.
      * An input sample's neighborhood contains structure used for message passing (in neighborhood field),
      * and positive and negative supervision links (in pos_edges and hard_neg_edges fields, respectively).
      * This splitting procedure aims to create up to 3 total output samples (up to 1 for each DatasetSplit:
      * train, val, test) which are NodeAnchorBasedLinkPrediction pbs that only contain
      *     (a) only relevant pos_edges and hard_neg_edges belonging to the split.
      *     (b) message passing structure which should be pb.neighborhood and therefore the same across all splits
      *     (i.e. no masking).
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
    val splitSample = Seq(
      NodeAnchorBasedLinkPredictionSample(
        rootNode = sample.rootNode,
        hardNegEdges = supervisionHardNegEdges,
        posEdges = supervisionPosEdges,
        negEdges = supervisionNegEdges,
        neighborhood = sample.neighborhood,
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
    val splitSample = Seq(
      RootedNodeNeighborhood(
        rootNode = sample.rootNode,
        neighborhood = sample.neighborhood,
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
    // In this setting, we only have linkSplitType.edgeUsage as SUPERVISION.
    // use condition below if in future we have other edgeUsage types.
    // val isMessageEdge: Boolean =
    //   linkSplitType.edgeUsage == LinkUsageTypes.MESSAGE && datasetSplit == DatasetSplits.TRAIN
    // (!isMessageEdge && linkSplitType.datasetSplit == datasetSplit)
    if (linkSplitType.edgeUsage != LinkUsageTypes.SUPERVISION) {
      throw new RuntimeException(
        f"Unexpected edgeUsage type ${linkSplitType.edgeUsage} for supervision edge ${supervisionEdge}",
      )
    }
    linkSplitType.datasetSplit == datasetSplit
  }

}
