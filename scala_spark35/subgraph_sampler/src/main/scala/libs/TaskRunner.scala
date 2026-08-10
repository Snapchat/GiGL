package libs

import common.types.pb_wrappers.GbmlConfigPbWrapper
import common.types.pb_wrappers.ResourceConfigPbWrapper
import common.types.pb_wrappers.TaskMetadataPbWrapper
import common.types.pb_wrappers.TaskMetadataType
import libs.task.SubgraphSamplerTask
import libs.task.graphdb.GraphDBNodeAnchorBasedLinkPredictionTask
import libs.task.pureSpark.NodeAnchorBasedLinkPredictionTask
import libs.task.pureSpark.SupervisedNodeClassificationTask
import libs.task.pureSpark.UserDefinedLabelsNodeAnchorBasedLinkPredictionTask
import libs.task.pureSparkV2.EgoNetGeneration

object TaskRunner {
  def runTask(
    jobName: String,
    gbmlConfigWrapper: GbmlConfigPbWrapper,
    giglResourceConfigWrapper: ResourceConfigPbWrapper,
  ): Unit = {

    val experimentalFlags =
      gbmlConfigWrapper.gbmlConfigPb.datasetConfig.get.subgraphSamplerConfig.get.experimentalFlags
    if (experimentalFlags.getOrElse("compute_ego_net", "False").equals("True")) {
      runEgoNetFlow(
        gbmlConfigWrapper = gbmlConfigWrapper,
        giglResourceConfigWrapper = giglResourceConfigWrapper,
        jobName = jobName,
      )
    } else {
      runSGSFlow(
        gbmlConfigWrapper = gbmlConfigWrapper,
        giglResourceConfigWrapper = giglResourceConfigWrapper,
      )
    }

  }

  def runEgoNetFlow(
    gbmlConfigWrapper: GbmlConfigPbWrapper,
    giglResourceConfigWrapper: ResourceConfigPbWrapper,
    jobName: String,
  ): Unit = {
    val task = new EgoNetGeneration(
      gbmlConfigWrapper = gbmlConfigWrapper,
      giglResourceConfigWrapper = giglResourceConfigWrapper,
      jobName = jobName,
    )
    task.run()
  }

  def runSGSFlow(
    gbmlConfigWrapper: GbmlConfigPbWrapper,
    giglResourceConfigWrapper: ResourceConfigPbWrapper,
  ): Unit = {
    val taskMetadataType = TaskMetadataPbWrapper(gbmlConfigWrapper.taskMetadataPb).taskMetadataType
    val subgraphSamplerConfigPb = gbmlConfigWrapper.subgraphSamplerConfigPb
    val shouldRunGraphdb = if (subgraphSamplerConfigPb.graphDbConfig.isDefined) {
      subgraphSamplerConfigPb.graphDbConfig.get.graphDbArgs.nonEmpty
    } else {
      false
    }
    println(f"shouldRunGraphdb = ${shouldRunGraphdb}")

    val sgsTask: SubgraphSamplerTask =
      if (taskMetadataType.equals(TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK)) {
        if (shouldRunGraphdb) {
          new GraphDBNodeAnchorBasedLinkPredictionTask(
            gbmlConfigWrapper = gbmlConfigWrapper,
            giglResourceConfigWrapper = giglResourceConfigWrapper,
          )
        } else {
          // TODO for heterogeneous will have to update this to be more flexible to detect UDL for different edge types
          val defaultCondensedEdgeType: Int =
            gbmlConfigWrapper.preprocessedMetadataWrapper.preprocessedMetadataPb.condensedEdgeTypeToPreprocessedMetadata.keysIterator
              .next()
          val isPosUserDefined: Boolean = !gbmlConfigWrapper.preprocessedMetadataWrapper
            .isPosUserDefinedForCondensedEdgeType(condensedEdgeType = defaultCondensedEdgeType)
          val isNegUserDefined: Boolean = !gbmlConfigWrapper.preprocessedMetadataWrapper
            .isNegUserDefinedForCondensedEdgeType(condensedEdgeType = defaultCondensedEdgeType)
          if (isPosUserDefined || isNegUserDefined) {
            new UserDefinedLabelsNodeAnchorBasedLinkPredictionTask(
              gbmlConfigWrapper = gbmlConfigWrapper,
              graphMetadataPbWrapper = gbmlConfigWrapper.graphMetadataPbWrapper,
              isPosUserDefined = isPosUserDefined,
              isNegUserDefined = isNegUserDefined,
            )
          } else {
            new NodeAnchorBasedLinkPredictionTask(
              gbmlConfigWrapper = gbmlConfigWrapper,
              graphMetadataPbWrapper = gbmlConfigWrapper.graphMetadataPbWrapper,
            )
          }
        }
      } else {
        // TODO: (svij) Note this implementation is copy from spark 3.2 implementation but untested
        new SupervisedNodeClassificationTask(
          gbmlConfigWrapper = gbmlConfigWrapper,
        )
      }
    println(
      f"\nStarting SGSTask ${sgsTask.getClass.getName}",
    )
    sgsTask.run()
  }
}
