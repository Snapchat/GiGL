package common.utils

import common.userDefinedAggregators.RnnUDAF
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import snapchat.research.gbml.training_samples_schema.RootedNodeNeighborhood

object SparkSessionEntry {
  private def setSparkConf(numVCPUs: Int): SparkConf = {
    val conf = new SparkConf()
    conf.set(
      "spark.sql.shuffle.partitions",
      (numVCPUs * 5).toString,
    )
    // Spark performs limit incrementally.
    // It tries to retrieve the given number of rows using one partition. If the number of rows is not satisfied,
    // Spark then queries the next 100 partitions(determined by spark.sql.limit.scaleUpFactor), then 1*100*100
    // and so on until the limit is satisfied or the data is exhausted. Since we are operating on a large
    // amount of data, and partitions we set a larger default value
    conf.set("spark.sql.limit.scaleUpFactor", "100")
    conf.set("spark.sql.adaptive.enabled", "false")
    conf.set("spark.driver.maxResultSize", "64g")
    // disable auto BroadcastJoins (SGS queries are opitmized for SortMerge joins)
    // We have also seen some weird issues where a broadcast join when it shouldn't
    // that is, a dataframe is detected to be less than spark.driver.maxResultSize but it
    // actually is more than that. Which then initiates a broadcast join and then immediately
    // fails w/ Total size of serialized results ... is bigger than spark.driver.maxResultSize.
    // Be careful when changing this value.
    conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")

    conf.registerKryoClasses(
      Array(
        classOf[RootedNodeNeighborhood],
        classOf[RnnUDAF],
        classOf[RnnUDAF.TmpNodeMetadata],
        classOf[RnnUDAF.TmpEdgeMetadata],
        classOf[RnnUDAF.BufferRNN],
        classOf[RnnUDAF.InTwoHopData],
      ),
    )

    // Reset bahaviour to spark 3.1
    conf.set("spark.storage.replication.proactive", "false")
    conf.set("spark.hadoopRDD.ignoreEmptySplits", "false")
    conf.set("spark.dynamicAllocation.shuffleTracking.enabled", "false")
    conf.set("spark.storage.decommission.rddBlocks.enabled", "false")
    conf.set("spark.storage.decommission.shuffleBlocks.enabled", "false")

    conf.set(
      "spark.driver.extraJavaOptions",
      "-XX:+UseCompressedOops",
    )

    // enable graceful node decommission
    // https://spark.apache.org/docs/latest/configuration.html#:~:text=3.1.0-,spark.storage.decommission.rddBlocks.enabled,-true
    // https://aws.github.io/aws-emr-containers-best-practices/cost-optimization/docs/node-decommission/
    conf.set("spark.decommission.enabled", "true")
    conf.set("spark.storage.decommission.enabled", "true")
    // NOTE: "spark.storage.decommission.rddBlocks.enabled" and "spark.storage.decommission.shuffleBlocks.enabled" are
    // by default set to true, but we add them again to be extra sure that they're indeed set to true
    conf.set("spark.storage.decommission.rddBlocks.enabled", "true")
    conf.set("spark.storage.decommission.shuffleBlocks.enabled", "true")

    conf
  }

  def constructSparkSession(
    appName: String,
    numVCPUs: Int,
  ): SparkSession = {
    println(f"Creating SparkSession with name $appName and $numVCPUs vCPUs")
    val session = SparkSession.builder.appName(appName).config(setSparkConf(numVCPUs)).getOrCreate()

    // The DataprocFileOutputCommitter feature is an enhanced version of the open source FileOutputCommitter.
    // It enables concurrent writes by Apache Spark jobs to an output location. We have seen that introducing
    // this committer can lead to a significant decrease in GCS write issues.
    // More info: https://cloud.google.com/dataproc/docs/guides/dataproc-fileoutput-committer
    session.sparkContext.hadoopConfiguration.set(
      "spark.hadoop.mapreduce.outputcommitter.factory.class",
      "org.apache.hadoop.mapreduce.lib.output.DataprocFileOutputCommitterFactory",
    )
    session.sparkContext.hadoopConfiguration.set(
      "spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs",
      "false",
    )
    session
  }

  def getActiveSparkSession(): SparkSession = SparkSession.builder.getOrCreate()

  def showCurrentSessionConfigs(spark: SparkSession): Unit = {
    val arrayConfig = spark.conf.getAll
    println("Current SparkSession configurations:")
    for (conf <- arrayConfig)
      println(conf._1 + ", " + conf._2)
  }

  def getNumCurrentShufflePartitions(spark: SparkSession): Int =
    spark.conf.get("spark.sql.shuffle.partitions").toInt
}
