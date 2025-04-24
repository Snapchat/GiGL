package splitgenerator.lib.split_strategies

import snapchat.research.gbml.graph_schema.Node
import snapchat.research.gbml.training_samples_schema.SupervisedNodeClassificationSample
import splitgenerator.lib.Types.DatasetSplits.DatasetSplit
import splitgenerator.lib.assigners.AbstractAssigners.Assigner

class TransductiveSupervisedNodeClassificationSplitStrategy(
  splitStrategyArgs: Map[String, String],
  nodeToDatasetSplitAssigner: Assigner[Node, DatasetSplit])
    extends SupervisedNodeClassificationSplitStrategy(
      splitStrategyArgs,
      nodeToDatasetSplitAssigner,
    ) {

  /**
    * For transductive node classification splitting, we only split on the node labels.
    * Train/val/test samples all see use the same (entire) graph to compute embeddings.
    * Ref: http://snap.stanford.edu/class/cs224w-2020/slides/09-theory.pdf slides 8-9
    *
    * @param sample
    * @param datasetSplit
    * @return
    */
  override def splitTrainingSample(
    sample: SupervisedNodeClassificationSample,
    datasetSplit: DatasetSplit,
  ): Seq[SupervisedNodeClassificationSample] = {

    // root node will always exist. This will fail the pipeline if rootNode is None.
    // Reason for rootNode to be an Option in proto: https://scalapb.github.io/docs/faq/#why-message-fields-are-wrapped-in-an-option-in-proto3
    val rootNodePb: Node =
      sample.rootNode.getOrElse(throw new RuntimeException("Root node does not exist for sample."))

    // split root node and check if it belongs to this particular split
    val splitSample: Seq[SupervisedNodeClassificationSample] =
      if (nodeToDatasetSplitAssigner.assign(rootNodePb) == datasetSplit) {
        Seq(sample)
      } else {
        Seq.empty[SupervisedNodeClassificationSample]
      }

    subsample(
      splitSample = splitSample,
      datasetSplit = datasetSplit,
      subsamplingRatio = mainSamplesSubsamplingRatio,
    )
  }
}
