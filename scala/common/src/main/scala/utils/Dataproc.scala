package common.utils

import scala.sys.process._
import scala.util.Failure
import scala.util.Success
import scala.util.Try

object Dataproc {
  val defaultTotalCores: Int = 20
  private def getClusterName(): Try[String] = {
    // https://github.com/GoogleCloudDataproc/initialization-actions/blob/master/README.md
    Try {
      val clusterName =
        "/usr/share/google/get_metadata_value attributes/dataproc-cluster-name".!!.trim
      println(f"Got cluster name: ${clusterName}")
      clusterName
    }
  }

  def getTotalNumVcores(numVCPUs: Int): Int = {
    val clusterName: String = getClusterName() match {
      case Success(clusterName) => clusterName
      case Failure(error)       => ""
    }

    if (clusterName.nonEmpty) {
      def getNumVcoresFromRuntimeArg(): Int = {
        println(
          s" Running job on Dataproc cluster: $clusterName with total num of cores: $numVCPUs",
        )
        numVCPUs
      }
      getNumVcoresFromRuntimeArg()
    } else {
      println(
        s"Dataproc cluster not found. Running Spark job with default value of total_cores= $defaultTotalCores",
      )
      defaultTotalCores
    }
  }

  def computeOptimalNumPartitions(numVCPUs: Int): Int = {
    val numPartitions =
      getTotalNumVcores(numVCPUs) * 5 // sets parallelization. For SGS job and any graph, x5 works
    println(s"Computed Partitions: $numPartitions")
    numPartitions
  }
}
