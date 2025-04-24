import common.types.pb_wrappers.ResourceConfigPbWrapper
import common.utils.GiGLComponents
import common.utils.NumCores
import common.utils.ProtoLoader
import common.utils.SparkSessionEntry.constructSparkSession
import common.utils.SparkSessionEntry.showCurrentSessionConfigs
import snapchat.research.gbml.gigl_resource_config.GiglResourceConfig
import splitgenerator.lib.SplitGeneratorTaskRunner

object Main {

  def main(args: Array[String]): Unit = {

    val sparkAppName               = args(0)
    val frozenGbmlConfigYamlGcsUri = args(1)
    val resourceConfigYamlGcsUri   = args(2)

    val resourceConfigProto: GiglResourceConfig =
      ProtoLoader.populateProtoFromYaml[GiglResourceConfig](
        uri = resourceConfigYamlGcsUri,
        printYamlContents = true,
      )
    val giglResourceConfigWrapper = ResourceConfigPbWrapper(resourceConfigPb = resourceConfigProto)

    val numVCPUs = NumCores.getNumVCPUs(
      giglResourceConfigWrapper = giglResourceConfigWrapper,
      component = GiGLComponents.SplitGenerator,
    )

    val spark = constructSparkSession(
      appName = sparkAppName,
      numVCPUs = numVCPUs,
    ) // Spark session is created ONLY once and passed to downstream classes/functions
    showCurrentSessionConfigs(spark = spark)
    SplitGeneratorTaskRunner.runTask(frozenGbmlConfigUri = frozenGbmlConfigYamlGcsUri)
    spark.stop()
  }
}
