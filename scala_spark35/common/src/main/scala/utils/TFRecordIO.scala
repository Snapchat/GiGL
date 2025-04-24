package common.utils

import common.utils.SparkSessionEntry.getActiveSparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Encoder
import scalapb.GeneratedMessage
import scalapb.GeneratedMessageCompanion
import scalapb.spark.Implicits._

object TFRecordIO {

  object RecordTypes extends Enumeration {
    type RecordType = Value
    val ExampleRecordType: Value   = Value("Example")
    val ByteArrayRecordType: Value = Value("ByteArray")
  }

  import RecordTypes.RecordType

  def readDataframeFromTfrecord(
    uri: String,
    recordType: RecordType,
  ): DataFrame = {
    val spark = getActiveSparkSession
    val df = spark.read
      .format("tfrecord")
      .option("recordType", recordType.toString)
      .option("pathGlobFilter", "*.tfrecord")
      .load(uri)
    df
  }

  def dataframeToTypedDataset[T <: GeneratedMessage: GeneratedMessageCompanion](
    df: DataFrame,
  )(implicit companion: GeneratedMessageCompanion[T],
    encoder: Encoder[T],
  ): Dataset[T] = {
    val byteDS: Dataset[Array[Byte]] = df.as[Array[Byte]]
    val typedDS: Dataset[T]          = byteDS.map(companion.parseFrom(_))
    typedDS
  }

  def writeDataframeToTfrecord[T <: GeneratedMessage: GeneratedMessageCompanion](
    df: DataFrame,
    gcsUri: String,
  )(implicit encoder: Encoder[T],
  ): Unit = {
    val protoDS: Dataset[T] = df.as[(T)] // convert to protoDS
    writeDatasetToTfrecord(inputDS = protoDS, gcsUri = gcsUri)
  }

  def writeDatasetToTfrecord[T <: GeneratedMessage: GeneratedMessageCompanion](
    inputDS: Dataset[T],
    gcsUri: String,
  )(implicit encoder: Encoder[T],
  ): Unit = {
    // convert to byte Array
    val byteArrayDS: Dataset[Array[Byte]] = inputDS.map(_.toByteArray)

    byteArrayDS.write
      .mode("overwrite")
      .format("tfrecord")
      .option(
        "recordType",
        "ByteArray",
      )
      .save(gcsUri)
  }

}
