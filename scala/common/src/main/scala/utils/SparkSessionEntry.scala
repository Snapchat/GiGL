package common.utils

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

object SparkSessionEntry {
  private def setSparkConf(numVCPUs: Int): SparkConf = {
    val conf = new SparkConf()
    // https://spark.apache.org/docs/3.1.3/configuration.html
    // set network communications params (to safeguard againts timeout errors for long running jobs)
    conf.set("spark.reducer.maxReqsInFlight", "1")
    conf.set("spark.shuffle.io.retryWait", "60s")
    conf.set("spark.shuffle.io.maxRetries", "10")
    conf.set("spark.network.timeout", "10000000")
    conf.set("spark.dynamicAllocation.executorIdleTimeout", "600s")
    // shuffle configs
    conf.set(
      "spark.sql.shuffle.partitions",
      Dataproc.computeOptimalNumPartitions(numVCPUs).toString,
    )
    conf.set("spark.shuffle.compress", "true")
    conf.set("spark.shuffle.spill.compress", "true")
    conf.set("spark.rdd.compress", "true")
    // enable adaptive query exec. (Spark execution plan can adapt as datasize grows)
    conf.set("spark.sql.adaptive.enabled", "true")
    // set serializer (for shuffle, cache and pre registered classes, i.e. all built-in Spark.sql functions)
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    // disable auto BroadcastJoins (SGS queries are optimzed for SortMerge joins)
    conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
    // set no limit for driver max size
    conf.set("spark.driver.maxResultSize", "0")
    // make pointers be four bytes instead of eight.
    // https://spark.apache.org/docs/2.1.1/tuning.html
    conf.set(
      "spark.executor.extraJavaOptions",
      "-XX:+UseCompressedOops",
    )
    conf.set(
      "spark.driver.extraJavaOptions",
      "-XX:+UseCompressedOops",
    )
    // enable node decommission
    // https://spark.apache.org/docs/latest/configuration.html#:~:text=3.1.0-,spark.storage.decommission.rddBlocks.enabled,-true
    // https://aws.github.io/aws-emr-containers-best-practices/cost-optimization/docs/node-decommission/
    conf.set("spark.decommission.enabled", "true")
    conf.set("spark.storage.decommission.enabled", "true")
    // NOTE: "spark.storage.decommission.rddBlocks.enabled" and "spark.storage.decommission.shuffleBlocks.enabled" are
    // by default set to true, but we add them again to be extra sure that they're indeed set to true
    conf.set("spark.storage.decommission.rddBlocks.enabled", "true")
    conf.set("spark.storage.decommission.shuffleBlocks.enabled", "true")
  }

  def constructSparkSession(
    appName: String,
    numVCPUs: Int,
  ): SparkSession =
    SparkSession.builder.appName(appName).config(setSparkConf(numVCPUs)).getOrCreate()

  def getActiveSparkSession(): SparkSession = SparkSession.builder.getOrCreate()

  def showCurrentSessionConfigs(spark: SparkSession): Unit = {
    val arrayConfig = spark.conf.getAll
    for (conf <- arrayConfig)
      println(conf._1 + ", " + conf._2)
  }

  def getNumCurrentShufflePartitions(spark: SparkSession): Int =
    spark.conf.get("spark.sql.shuffle.partitions").toInt
}
