import common.types.pb_wrappers.GbmlConfigPbWrapper
import common.types.pb_wrappers.ResourceConfigPbWrapper
import common.utils.GiGLComponents
import common.utils.NumCores
import common.utils.ProtoLoader
import common.utils.SparkSessionEntry.constructSparkSession
import common.utils.SparkSessionEntry.showCurrentSessionConfigs
import libs.TaskRunner
import snapchat.research.gbml.gbml_config.GbmlConfig
import snapchat.research.gbml.gigl_resource_config.GiglResourceConfig

object Main {

  def main(args: Array[String]): Unit = {

    println(
      s"Starting Spark3.5 implementation of Subgraph Sampler with args: ${args.mkString(",")}",
    )

    val frozenGbmlConfigYamlGcsUri = args(0)
    val sparkAppName               = args(1)
    val resourceConfigYamlGcsUri   = args(2)

    val resourceConfigProto: GiglResourceConfig =
      ProtoLoader.populateProtoFromYaml[GiglResourceConfig](
        uri = resourceConfigYamlGcsUri,
        printYamlContents = true,
      )
    val giglResourceConfigWrapper = ResourceConfigPbWrapper(resourceConfigPb = resourceConfigProto)

    val numVCPUs = NumCores.getNumVCPUs(
      giglResourceConfigWrapper = giglResourceConfigWrapper,
      component = GiGLComponents.SubgraphSampler,
    )

    val gbmlConfigProto: GbmlConfig =
      ProtoLoader.populateProtoFromYaml[GbmlConfig](
        uri = frozenGbmlConfigYamlGcsUri,
        printYamlContents = true,
      )
    val gbmlConfigWrapper = GbmlConfigPbWrapper(gbmlConfigPb = gbmlConfigProto)

    val spark = constructSparkSession(
      appName = sparkAppName,
      numVCPUs = numVCPUs,
    ) // Spark session is created ONLY once and passed to downstream classes/functions
    showCurrentSessionConfigs(spark = spark)
    println("Starting the Spark 3.5 task runner for SGS")

    val taskRunner = TaskRunner.runTask(
      jobName = sparkAppName,
      gbmlConfigWrapper = gbmlConfigWrapper,
      giglResourceConfigWrapper = giglResourceConfigWrapper,
    )
    spark.stop()

  }
}
