package splitgenerator.lib.split_strategies

import snapchat.research.gbml.graph_schema.Edge
import snapchat.research.gbml.graph_schema.Graph
import snapchat.research.gbml.graph_schema.Node
import snapchat.research.gbml.training_samples_schema.NodeAnchorBasedLinkPredictionSample
import snapchat.research.gbml.training_samples_schema.RootedNodeNeighborhood
import splitgenerator.lib.Types.DatasetSplits
import splitgenerator.lib.Types.DatasetSplits.DatasetSplit
import splitgenerator.lib.Types.SplitSubsamplingRatio
import splitgenerator.lib.assigners.AbstractAssigners.Assigner

/**
  * This class is used for splitting NodeAnchorBasedLinkPredictionSample pbs with user-defined labels provided
  * in the pb.pos_edges and pb.hard_neg_edges fields.  The splitting procedure is as follows:
  * for all train/val/test splits,
  * 1. we retain all pb.neighborhood.edges as message passing edges
  * 2. using assigner logic, we determine to which split anchor nodes belongs
  * 3. we retain all pb.pos_edges and pb.hard_neg_edges that belong to the split determined by anchor node split. So all supervision edges are assigned to one of train/val/test splits.
  * 4. we filter out overlap between user provided supervision edges and message passing edges from pb.neighborhood
  * edges. If specified in splitStrtegyArgs we filter out the reverse supervision edges as well.
  * 5. based on what users specifies in splitStrategyArgs, we determine for which split we want to remove overlaps
  * @param splitStrategyArgs
  * @param edgeToLinkSplitAssigner
  */
class UDLAnchorBasedSupervisionEdgeSplitStrategy(
  splitStrategyArgs: Map[String, String],
  nodeToDatasetSplitAssigner: Assigner[Node, DatasetSplit])
    extends SplitStrategy[NodeAnchorBasedLinkPredictionSample](splitStrategyArgs) {

  /**
      * An input sample is a single NodeAnchorBasedLinkPredictionSample pbs output by SubgraphSampler.
      * An input sample's neighborhood contains structure used for message passing (in neighborhood field),
      * and positive and negative supervision links (in pos_edges and hard_neg_edges fields, respectively).
      * This splitting procedure aims to create up to 3 total output samples (up to 1 for each DatasetSplit:
      * train, val, test) which are NodeAnchorBasedLinkPrediction pbs that only contain
      *     (a) All pos_edges and hard_neg_edges belonging to the split.
      *     (b) message passing structure which should be pb.neighborhood and therefore the same across all splits
      *     (i.e. no masking).
      *     (c) The message passing structure may be filtered down to only include edges that are not in the pos_edges 
      *         and hard_neg_edges.
      * An output train-split sample needs to have >0 pos_edges in this setting for loss computation.
      * Output val/test-split samples may have 0 pos_edges (and even 0 hard_neg_edges), since these
      * samples can technically be used to evaluate random negative link likelihoods.
      *
      * @param sample
      * @param datasetSplit
      * @return
      */

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

  override val graphMetadataPbWrapper = nodeToDatasetSplitAssigner.graphMetadataPbWrapper

  private val __shouldFilterReverseSupervisionEdge: Boolean =
    splitStrategyArgs.getOrElse("should_filter_reverse_supervision_edge", "true").toBoolean

  private val __shouldFilterTrain: Boolean =
    splitStrategyArgs.getOrElse("should_filter_train", "true").toBoolean
  private val __shouldFilterVal: Boolean =
    splitStrategyArgs.getOrElse("should_filter_val", "true").toBoolean
  private val __shouldFilterTest: Boolean =
    splitStrategyArgs.getOrElse("should_filter_test", "true").toBoolean

  /**
   * Usable edge ids are those that are not in the supervision edges and are in the neighborhood edges.
   * @param sample
   * @return
     */
  private def getUsableMessagePassingEdgeIds(
    sample: NodeAnchorBasedLinkPredictionSample,
  ): Seq[(Int, Int)] = {
    val posEdgeIds     = sample.posEdges.map(edge => (edge.srcNodeId, edge.dstNodeId))
    val negEdgeIds     = sample.negEdges.map(edge => (edge.srcNodeId, edge.dstNodeId))
    val hardNegEdgeIds = sample.hardNegEdges.map(edge => (edge.srcNodeId, edge.dstNodeId))

    val unUsableEdgeIds = if (!__shouldFilterReverseSupervisionEdge) {
      posEdgeIds ++ negEdgeIds ++ hardNegEdgeIds
    } else {
      val reversedPosEdgeIds     = sample.posEdges.map(edge => (edge.dstNodeId, edge.srcNodeId))
      val reversedNegEdgeIds     = sample.negEdges.map(edge => (edge.dstNodeId, edge.srcNodeId))
      val reversedHardNegEdgeIds = sample.hardNegEdges.map(edge => (edge.dstNodeId, edge.srcNodeId))
      posEdgeIds ++ reversedPosEdgeIds ++ negEdgeIds ++ reversedNegEdgeIds ++ hardNegEdgeIds ++ reversedHardNegEdgeIds
    }
    val neighborhoodEdgesIds =
      sample.neighborhood.get.edges.map(edge => (edge.srcNodeId, edge.dstNodeId))
    val usableNeighborhoodEdgeIds =
      neighborhoodEdgesIds.filterNot(edge => unUsableEdgeIds.contains(edge))
    usableNeighborhoodEdgeIds

  }

  private def removeSupervisionEdgeOverlapFromNeighborhood(
    sample: NodeAnchorBasedLinkPredictionSample,
  ): NodeAnchorBasedLinkPredictionSample = {

    val usableNeighborhoodEdgeIds = getUsableMessagePassingEdgeIds(sample)
    val filteredNeighborhoodEdges = sample.neighborhood.get.edges.filter(edge =>
      usableNeighborhoodEdgeIds.contains((edge.srcNodeId, edge.dstNodeId)),
    )
    val filteredNodeIds =
      filteredNeighborhoodEdges.flatMap(edge => Seq(edge.srcNodeId, edge.dstNodeId)).distinct
    val filteredNodes =
      sample.neighborhood.get.nodes.filter(node => filteredNodeIds.contains(node.nodeId))

    val filteredSample = NodeAnchorBasedLinkPredictionSample(
      rootNode = sample.rootNode,
      hardNegEdges = sample.hardNegEdges,
      posEdges = sample.posEdges,
      negEdges = sample.negEdges,
      neighborhood = Some(
        Graph(nodes = filteredNodes, edges = filteredNeighborhoodEdges),
      ), // we may have empty neighborhood after filtering, so we use Some for optional field
    )

    filteredSample
  }

  override def splitTrainingSample(
    sample: NodeAnchorBasedLinkPredictionSample,
    datasetSplit: DatasetSplit,
  ): Seq[NodeAnchorBasedLinkPredictionSample] = {
    // val newSample = removeSupervisionEdgeOverlapFromNeighborhood(sample)
    val anchorNode = sample.rootNode.getOrElse(
      throw new RuntimeException("Root node does not exist for sample."),
    )

    val anchorNodeDatasetSplit = nodeToDatasetSplitAssigner.assign(anchorNode)
    if (anchorNodeDatasetSplit != datasetSplit) {
      return Seq.empty[NodeAnchorBasedLinkPredictionSample]
    }

    // filter out supervision edges based on dataset split
    val usableSample =
      if (
        (__shouldFilterTrain && datasetSplit == DatasetSplits.TRAIN) ||
        (__shouldFilterVal && datasetSplit == DatasetSplits.VAL) ||
        (__shouldFilterTest && datasetSplit == DatasetSplits.TEST)
      ) {
        removeSupervisionEdgeOverlapFromNeighborhood(sample)
      } else {
        sample
      }

    // If after filtering overalpping edges the neighborhood is empty, drop that sample as it caanot be used for msg passing
    // Similarly, if there are no positive edges
    // The train split cannot have 0 pos_edges as it used for Loss computation. Drop sample if that is the case
    // We could optionally also filter val and test samples that don't have pos_edges.
    // However, we currently leave them be, since we may want to compute metrics over only
    // negatives in val/test (potentially even just random negatives).
    // e.g. "what is the average similarity between pairs of negative samples?"
    // As such, output val/test samples may in the extreme have 0 pos_edges and 0 hard_neg_edges,
    // if all the corresponding input sample edges fall in other splits.
    // if (supervisionPosEdges.isEmpty && datasetSplit == DatasetSplits.TRAIN) {
    //   return Seq.empty[NodeAnchorBasedLinkPredictionSample]
    // }
    if (
      usableSample.neighborhood.get.edges.isEmpty ||
      (usableSample.posEdges.isEmpty && datasetSplit == DatasetSplits.TRAIN)
    ) {
      return Seq.empty[NodeAnchorBasedLinkPredictionSample]
    }

    // Add all message passing edges in input sample to relevant train/val/test samples.
    val splitSample = Seq(usableSample)

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
      * We use no masking (i.e. no split) for rooted node neighborhood samples.
      */
  def splitRootedNodeNeighborhoodTrainingSample(
    sample: RootedNodeNeighborhood,
    datasetSplit: DatasetSplit,
  ): Seq[RootedNodeNeighborhood] = {

    // Add all message passing edges in input sample to relevant train/val/test samples.
    val splitSample = Seq(sample)

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
    * Helper function to filter out the supervision edges for a particular split based on the anchor node assignment.
    *
    * @param supervisionEdge
    * @param datasetSplit
    * @return
    */
  private def shouldAddSupervisionEdgeToSplit(
    supervisionEdge: Edge,
    datasetSplit: DatasetSplit,
    anchorNode: Node,
  ): Boolean = {

    val anchorNodeDatasetSplit = nodeToDatasetSplitAssigner.assign(anchorNode)
    anchorNodeDatasetSplit == datasetSplit
  }

}
