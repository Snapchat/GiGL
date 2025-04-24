package splitgenerator.lib.split_strategies

import snapchat.research.gbml.graph_schema.Node
import snapchat.research.gbml.training_samples_schema.SupervisedNodeClassificationSample
import splitgenerator.lib.Types.DatasetSplits.DatasetSplit
import splitgenerator.lib.assigners.AbstractAssigners.Assigner

abstract class SupervisedNodeClassificationSplitStrategy(
  splitStrategyArgs: Map[String, String],
  val nodeToDatasetSplitAssigner: Assigner[Node, DatasetSplit])
    extends SplitStrategy[SupervisedNodeClassificationSample](splitStrategyArgs) {

  override val graphMetadataPbWrapper = nodeToDatasetSplitAssigner.graphMetadataPbWrapper

  override def splitTrainingSample(
    sample: SupervisedNodeClassificationSample,
    datasetSplit: DatasetSplit,
  ): Seq[SupervisedNodeClassificationSample]
}
