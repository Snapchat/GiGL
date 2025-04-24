package splitgenerator.lib.split_strategies

import common.types.pb_wrappers.GraphMetadataPbWrapper
import splitgenerator.lib.Types.DatasetSplits
import splitgenerator.lib.Types.DatasetSplits.DatasetSplit
import splitgenerator.lib.Types.SplitSubsamplingRatio

import scala.util.Random

abstract class SplitStrategy[A](splitStrategyArgs: Map[String, String]) extends Serializable {

  protected val mainSamplesSubsamplingRatio: SplitSubsamplingRatio = SplitSubsamplingRatio(
    // ratio of train samples to keep
    trainSubsamplingRatio = splitStrategyArgs
      .getOrElse("train_subsampling_ratio", "1.0")
      .toFloat,
    // ratio of val samples to keep
    valSubsamplingRatio = splitStrategyArgs
      .getOrElse("val_subsampling_ratio", "1.0")
      .toFloat,
    // ratio of test samples to keep
    testSubsamplingRatio = splitStrategyArgs
      .getOrElse("test_subsampling_ratio", "1.0")
      .toFloat,
  )

  /**
    * Split strategies will want to access GraphMetadata to handle splitting logic.
    * This is because determining how a certain node or edge should be split may depend on
    * certain (readable) node or edge types, field values (e.g. timestamps), etc.
    */
  val graphMetadataPbWrapper: GraphMetadataPbWrapper

  /**
    * Takes in a single "un-split" training sample instance output by SubgraphSampler, 
    * and a DatasetSplit(TRAIN, TEST, VAL) and outputs the the "split" samples for that dataset split
    *
    * @param sample : Input Sample from SGS
    * @param datasetSplit : TRAIN/TEST/VAL
    * @return Seq of "split" samples for particular DatasetSplit
    */
  def splitTrainingSample(
    sample: A,
    datasetSplit: DatasetSplit,
  ): Seq[A]

  def subsample[B](
    splitSample: Seq[B],
    datasetSplit: DatasetSplit,
    subsamplingRatio: SplitSubsamplingRatio,
  ): Seq[B] = {
    val random = Random.nextFloat()
    val datasetSplitSubsampleRatio: Float = datasetSplit match {
      case DatasetSplits.TRAIN => subsamplingRatio.trainSubsamplingRatio
      case DatasetSplits.VAL   => subsamplingRatio.valSubsamplingRatio
      case DatasetSplits.TEST  => subsamplingRatio.testSubsamplingRatio
    }

    if (datasetSplitSubsampleRatio < 1.0f && random < datasetSplitSubsampleRatio) {
      Seq.empty[B]
    } else {
      splitSample
    }
  }
}
