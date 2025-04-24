package common.utils

import common.utils.SparkSessionEntry.getActiveSparkSession

object YamlLoader {
  def readYamlAsString(
    uri: String,
  ): String = {
    val spark                  = getActiveSparkSession
    val dataYamlDF             = spark.read.option("wholetext", true).text(uri)
    val dataYamlString: String = dataYamlDF.first().getString(0)
    dataYamlString
  }
}
