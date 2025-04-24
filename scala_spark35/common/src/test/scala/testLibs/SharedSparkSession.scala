package common.test.testLibs

import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.SparkConf
import org.apache.spark.sql.SQLImplicits
import org.apache.spark.sql.SparkSession
import org.scalatest.BeforeAndAfterAll
import org.scalatest.Suite

trait SharedSparkSession extends BeforeAndAfterAll { self: Suite =>

  Logger
    .getLogger("org.apache.spark")
    .setLevel(Level.WARN) //  silence verbose spark logs during tests
  Logger
    .getLogger("org.apache.hadoop.mapreduce.lib.output.FileOutputCommitterFactory")
    .setLevel(Level.WARN)
  Logger
    .getLogger("org.apache.hadoop.mapreduce.lib.output.FileOutputCommitter")
    .setLevel(Level.WARN)

  def sparkTestConf: SparkConf =
    new SparkConf().set("spark.master", "local")

  def createSparkSession: SparkSession =
    SparkSession.builder
      .appName("unittest")
      .config(sparkTestConf)
      .getOrCreate()

  var _spark: SparkSession = _

  protected implicit def sparkTest: SparkSession = _spark

  protected def initializeSession(): Unit = {
    if (_spark == null) {
      _spark = createSparkSession
    }
  }
  // `implicits` object gives implicit conversions for converting Scala objects into Spark Datatypes such as DataFrame,
  // Dataset, Columns, Struct. We need this to create DataFrame from Scala Seq and/or create a DataFrame with nested columns to mock unittest data
  // implicts are bound to SparkSession lifetime, and here can be imported as sqlImplicits._ once we have
  // a sparkSession initialized
  protected lazy val sqlImplicits: SQLImplicits = self.sparkTest.implicits

  protected override def beforeAll(): Unit = {
    initializeSession()
    super.beforeAll()
  }

  // protected override def afterAll(): Unit = {
  //   super.afterAll()
  //   if (_spark != null) {
  //     _spark.sessionState.catalog.reset()
  //     _spark.stop()
  //     _spark = null
  //   }
  // }

}
