package libs.task

import com.typesafe.scalalogging.LazyLogging
import common.types.pb_wrappers.GbmlConfigPbWrapper
import common.utils.SparkSessionEntry.getActiveSparkSession
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel

abstract class SubgraphSamplerTask(
  gbmlConfigWrapper: GbmlConfigPbWrapper)
    extends Serializable
    with LazyLogging {

  val spark: SparkSession = getActiveSparkSession

  val samplingSeed: Int = 42

  val storageLevel: StorageLevel = StorageLevel.DISK_ONLY

  def run():
  /**
    * All subgraph sampler task will have these two outputs, which will be written to gcs inside run()
    * 1. inference samples (i.e. RootedNodeNeighborhood) [which can be used as Random negative samples and unlabeled samples too]
    * 2. training samples (e.g. NodeAnchorBasedLinkPrediction, SupervisedNodeClassification)
    */
  Unit
}
