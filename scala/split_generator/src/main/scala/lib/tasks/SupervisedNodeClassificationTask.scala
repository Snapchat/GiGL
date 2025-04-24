package splitgenerator.lib.tasks

import common.types.pb_wrappers.GbmlConfigPbWrapper
import common.utils.TFRecordIO.dataframeToTypedDataset
import scalapb.spark.Implicits._
import snapchat.research.gbml.training_samples_schema.SupervisedNodeClassificationSample
import splitgenerator.lib.Types.SplitOutputPaths
import splitgenerator.lib.split_strategies.SupervisedNodeClassificationSplitStrategy

class SupervisedNodeClassificationTask(
  gbmlConfigWrapper: GbmlConfigPbWrapper,
  splitStrategy: SupervisedNodeClassificationSplitStrategy)
    extends SplitGeneratorTask(gbmlConfigWrapper, splitStrategy) {

  private val __supervisedNodeClassificationDataset =
    gbmlConfigWrapper.sharedConfigPb.datasetMetadata.get.getSupervisedNodeClassificationDataset

  override def run(): Unit = {
    val inputPathForTrainingSamples =
      gbmlConfigWrapper.flattenedGraphMetadataPb.getSupervisedNodeClassificationOutput.labeledTfrecordUriPrefix

    // load, coalesce and cache dataframe from GCS
    val cachedInputDF =
      loadCoalesceCacheDataframe(
        inputPath = inputPathForTrainingSamples,
        coalesceFactor = mainSamplesCoalesceFactor,
      )
    // Convert raw Dataframe to Dataset[SupervisedNodeClassificationSample]
    val inputSamplesDS =
      dataframeToTypedDataset[SupervisedNodeClassificationSample](
        df = cachedInputDF,
      )

    // for each train/test/val, get the split data from the input dataset and write that data back to GCS
    val outputPaths = SplitOutputPaths(
      trainPath = __supervisedNodeClassificationDataset.trainDataUri,
      valPath = __supervisedNodeClassificationDataset.valDataUri,
      testPath = __supervisedNodeClassificationDataset.testDataUri,
    )

    splitSamplesAndWriteToOutputPath(
      outputPaths = outputPaths,
      inputDS = inputSamplesDS,
      splitFn = splitStrategy.splitTrainingSample,
    )

    // Unpersist the cached df
    cachedInputDF.unpersist()
  }
}
