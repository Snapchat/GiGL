package common.utils

import common.types.pb_wrappers.ResourceConfigPbWrapper

import scala.util.matching.Regex

object NumCores {

  // Regex pattern to extract the trailing number from strings like "n2-standard-4".
  // `(\d+)$` matches one or more digits at the end of the string.
  private val pattern: Regex = """(\d+)$""".r

  def getNumVCPUs(
    giglResourceConfigWrapper: ResourceConfigPbWrapper,
    component: GiGLComponents.Value,
  ): Int = {
    val (machineType, numReplicas) = component match {
      case GiGLComponents.SubgraphSampler =>
        (
          giglResourceConfigWrapper.subgraphSamplerConfig.machineType,
          giglResourceConfigWrapper.subgraphSamplerConfig.numReplicas,
        )

      case GiGLComponents.SplitGenerator =>
        (
          giglResourceConfigWrapper.splitGeneratorConfig.machineType,
          giglResourceConfigWrapper.splitGeneratorConfig.numReplicas,
        )

      case _ => throw new RuntimeException(s"Unsupported component: $component")
    }

    val coresFromMachineType = pattern.findFirstMatchIn(machineType) match {
      case Some(matched) => matched.group(1).toInt
      case None =>
        throw new RuntimeException(s"No match found for cores in machine type: $machineType")
    }

    coresFromMachineType * numReplicas
  }
}
