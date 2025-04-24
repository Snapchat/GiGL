package splitgenerator.lib.tasks

import common.types.pb_wrappers.GbmlConfigPbWrapper
import common.utils.SparkSessionEntry
import common.utils.TFRecordIO.RecordTypes
import common.utils.TFRecordIO.readDataframeFromTfrecord
import common.utils.TFRecordIO.writeDatasetToTfrecord
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel
import scalapb.GeneratedMessage
import scalapb.GeneratedMessageCompanion
import splitgenerator.lib.Types.DatasetSplits
import splitgenerator.lib.Types.DatasetSplits.DatasetSplit
import splitgenerator.lib.Types.SplitOutputPaths
import splitgenerator.lib.split_strategies.SplitStrategy

abstract class SplitGeneratorTask(
  gbmlConfigWrapper: GbmlConfigPbWrapper,
  splitStrategy: SplitStrategy[_])
    extends Serializable {
  // We need to extend the Serializable trait because Spark driver serializes this class to send it over the network to the workers. Ommitting causes a Task Not Serializable exception
  // https://stackoverflow.com/questions/22592811/task-not-serializable-java-io-notserializableexception-when-calling-function-ou

  val spark: SparkSession       = SparkSessionEntry.getActiveSparkSession
  val mainSamplesCoalesceFactor = 12

  def run(): Unit

  def loadCoalesceCacheDataframe(
    inputPath: String,
    coalesceFactor: Int,
  ): DataFrame = {
    // read raw dataframe from GCS
    val rawDF: DataFrame = readDataframeFromTfrecord(
      uri = inputPath,
      recordType = RecordTypes.ByteArrayRecordType,
    )

    /**
      * @spark: coalesce partitions (does not incur a shuffle)
      * Without coalesce the CPU utilization of the workers are very less (~25%)
      * Coalescing the partitions leads to better CPU utilization and reduces job time.
      */
    val coalescedDF: DataFrame =
      coalesceInputDataframe(inputDF = rawDF, coalesceFactor = coalesceFactor)

    // no caching is necessary if shouldSkipTrainingflag is set as we only want to write the test split
    if (gbmlConfigWrapper.sharedConfigPb.shouldSkipTraining) {
      coalescedDF
    } else {

      /**
        * @spark: cache the coalesced partition
        * We need to cache the partitions to avoid reading the input from GCS multiple times
        * Without this spark tries to read from source foreach Spark action (in this case we have 3: writing train/ test/ val)
        */
      val cachedDF = coalescedDF.persist(StorageLevel.MEMORY_AND_DISK)
      cachedDF
    }
  }

  def splitSamplesAndWriteToOutputPath[T <: GeneratedMessage: GeneratedMessageCompanion](
    outputPaths: SplitOutputPaths,
    inputDS: Dataset[T],
    splitFn: (T, DatasetSplit) => Seq[T],
  )(implicit encoder: Encoder[T],
  ): Unit = {
    getDatasetSplitsToProcess.map((datasetSplit) => {
      val splitDS: Dataset[T] =
        inputDS
          .flatMap(row => splitFn(row, datasetSplit))

      // write out the samples
      writeDatasetToTfrecord[T](inputDS = splitDS, gcsUri = outputPaths.getPath(datasetSplit))
    })
  }

  private def coalesceInputDataframe(
    inputDF: DataFrame,
    coalesceFactor: Int,
  ): DataFrame = {
    // TODO: Coalesce factor right now set to what performed best in the POC. Need to adjust for different input sizes.
    val numOfInputPartitions = inputDF.rdd.getNumPartitions
    if (numOfInputPartitions < coalesceFactor) {
      // This case to run very small graphs as number of partitions after coalesce must be > 0
      inputDF
    } else {
      inputDF.coalesce(numOfInputPartitions / coalesceFactor)
    }
  }

  private def getDatasetSplitsToProcess: Seq[DatasetSplit] = {
    if (gbmlConfigWrapper.sharedConfigPb.shouldSkipTraining) {
      // Only write out test split if shouldSkipTraining flag is set
      Seq(DatasetSplits.TEST)
    } else {
      Seq(DatasetSplits.TRAIN, DatasetSplits.VAL, DatasetSplits.TEST)
    }
  }
}
