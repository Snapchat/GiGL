package libs

import common.types.pb_wrappers.GbmlConfigPbWrapper
import common.types.pb_wrappers.GraphMetadataPbWrapper
import common.types.pb_wrappers.TaskMetadataPbWrapper
import common.types.pb_wrappers.TaskMetadataType
import common.utils.ProtoLoader.populateProtoFromYaml
import libs.task.SubgraphSamplerTask
import libs.task.graphdb.GraphDBUnsupervisedNodeAnchorBasedLinkPredictionTask
import libs.task.pureSpark.NodeAnchorBasedLinkPredictionTask
import libs.task.pureSpark.SupervisedNodeClassificationTask
import libs.task.pureSpark.UserDefinedLabelsNodeAnchorBasedLinkPredictionTask
import snapchat.research.gbml.gbml_config.GbmlConfig

object TaskRunner {
  def runTask(frozenGbmlConfigUri: String): Unit = {
    println("\nfrozen config info:")
    val gbmlConfigProto =
      populateProtoFromYaml[GbmlConfig](uri = frozenGbmlConfigUri, printYamlContents = true)
    val gbmlConfigWrapper = GbmlConfigPbWrapper(gbmlConfigPb = gbmlConfigProto)
    val taskMetadataType  = TaskMetadataPbWrapper(gbmlConfigProto.taskMetadata.get).taskMetadataType
    val graphMetadataPbWrapper: GraphMetadataPbWrapper = GraphMetadataPbWrapper(
      gbmlConfigWrapper.graphMetadataPb,
    )
    val subgraphSamplerConfigPb = gbmlConfigWrapper.subgraphSamplerConfigPb
    val shouldRunGraphdb = if (subgraphSamplerConfigPb.graphDbConfig.isDefined) {
      subgraphSamplerConfigPb.graphDbConfig.get.graphDbArgs.nonEmpty
    } else {
      false
    }
    println(f"shouldRunGraphdb = ${shouldRunGraphdb}")

    val sgsTask: SubgraphSamplerTask =
      if (taskMetadataType.equals(TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK)) {
        // TODO for heterogeneous will have to update this to be more flexible to detect UDL for different edge types
        val defaultCondensedEdgeType: Int =
          gbmlConfigWrapper.preprocessedMetadataWrapper.preprocessedMetadataPb.condensedEdgeTypeToPreprocessedMetadata.keysIterator
            .next()
        val isPosUserDefined: Boolean = !gbmlConfigWrapper.preprocessedMetadataWrapper
          .isPosUserDefinedForCondensedEdgeType(condensedEdgeType = defaultCondensedEdgeType)
        val isNegUserDefined: Boolean = !gbmlConfigWrapper.preprocessedMetadataWrapper
          .isNegUserDefinedForCondensedEdgeType(condensedEdgeType = defaultCondensedEdgeType)

        if (isPosUserDefined || isNegUserDefined) {
          if (shouldRunGraphdb) {
            new GraphDBUnsupervisedNodeAnchorBasedLinkPredictionTask(
              gbmlConfigWrapper = gbmlConfigWrapper,
              graphMetadataPbWrapper = graphMetadataPbWrapper,
            )
          } else {
            new UserDefinedLabelsNodeAnchorBasedLinkPredictionTask(
              gbmlConfigWrapper = gbmlConfigWrapper,
              graphMetadataPbWrapper = graphMetadataPbWrapper,
              isPosUserDefined = isPosUserDefined,
              isNegUserDefined = isNegUserDefined,
            )
          }
        } else {
          if (shouldRunGraphdb) {
            new GraphDBUnsupervisedNodeAnchorBasedLinkPredictionTask(
              gbmlConfigWrapper = gbmlConfigWrapper,
              graphMetadataPbWrapper = graphMetadataPbWrapper,
            )
          } else {
            new NodeAnchorBasedLinkPredictionTask(
              gbmlConfigWrapper = gbmlConfigWrapper,
              graphMetadataPbWrapper = graphMetadataPbWrapper,
            )
          }
        }
      } else {
        new SupervisedNodeClassificationTask(
          gbmlConfigWrapper = gbmlConfigWrapper,
          graphMetadataPbWrapper = graphMetadataPbWrapper,
        )
      }
    println(
      f"\nStarting SGSTask ${sgsTask.getClass.getName} with frozenGbmlConfigUri = ${frozenGbmlConfigUri}",
    )
    sgsTask.run()
  }
}
