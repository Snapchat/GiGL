package splitgenerator.lib

import common.types.pb_wrappers.GbmlConfigPbWrapper
import common.types.pb_wrappers.GraphMetadataPbWrapper
import common.types.pb_wrappers.TaskMetadataPbWrapper
import common.types.pb_wrappers.TaskMetadataType
import common.utils.ProtoLoader.populateProtoFromYaml
import snapchat.research.gbml.gbml_config.GbmlConfig
import splitgenerator.lib.assigners.AbstractAssigners.Assigner
import splitgenerator.lib.split_strategies.NodeAnchorBasedLinkPredictionSplitStrategy
import splitgenerator.lib.split_strategies.SplitStrategy
import splitgenerator.lib.split_strategies.SupervisedNodeClassificationSplitStrategy
import splitgenerator.lib.split_strategies.UDLAnchorBasedSupervisionEdgeSplitStrategy
import splitgenerator.lib.tasks.NodeAnchorBasedLinkPredictionTask
import splitgenerator.lib.tasks.SupervisedNodeClassificationTask
import splitgenerator.lib.tasks.UDLNodeAnchorBasedLinkPredictionTask

object SplitGeneratorTaskRunner {

  def runTask(frozenGbmlConfigUri: String): Unit = {
    println("\nfrozen config info:") // TODO: Add Logging functionality
    val gbmlConfigProto =
      populateProtoFromYaml[GbmlConfig](uri = frozenGbmlConfigUri, printYamlContents = true)
    val gbmlConfigWrapper = GbmlConfigPbWrapper(gbmlConfigPb = gbmlConfigProto)

    val taskMetadataType = TaskMetadataPbWrapper(gbmlConfigProto.taskMetadata.get).taskMetadataType
    val splitStrategy: SplitStrategy[_] = getSplitStrategyInstance(gbmlConfigWrapper)

    // TODO: (svij-sc) Ideally we dont want to hardcode the mapping with conditionals
    //             dictionary would be preferred.
    // val taskMap: Dict[Class[SplitStrategy], Class[SplitGeneratorTask]] = Dict(
    //   UDLAnchorBasedSupervisionEdgeSplitStrategy -> UDLNodeAnchorBasedLinkPredictionTask,
    //   NodeAnchorBasedLinkPredictionSplitStrategy -> NodeAnchorBasedLinkPredictionTask,
    //   SupervisedNodeClassificationSplitStrategy -> SupervisedNodeClassificationTask,
    // )

    if (taskMetadataType.equals(TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK)) {
      // since we have a v1 UDL strategy that uses NodeAnchorBasedLinkPredictionSplitStrategy and the
      // newly added UDLAnchorBasedSupervisionEdgeSplitStrategy, we have to differentiate between the two based on what user determines
      if (
        splitStrategy.getClass
          .getName() == "splitgenerator.lib.split_strategies.UDLAnchorBasedSupervisionEdgeSplitStrategy"
      ) {
        new UDLNodeAnchorBasedLinkPredictionTask(
          gbmlConfigWrapper = gbmlConfigWrapper,
          splitStrategy = splitStrategy.asInstanceOf[UDLAnchorBasedSupervisionEdgeSplitStrategy],
        ).run()
      } else {
        new NodeAnchorBasedLinkPredictionTask(
          gbmlConfigWrapper = gbmlConfigWrapper,
          splitStrategy = splitStrategy.asInstanceOf[NodeAnchorBasedLinkPredictionSplitStrategy],
        ).run()
      }
    } else if (taskMetadataType.equals(TaskMetadataType.NODE_BASED_TASK)) {
      new SupervisedNodeClassificationTask(
        gbmlConfigWrapper = gbmlConfigWrapper,
        splitStrategy = splitStrategy.asInstanceOf[SupervisedNodeClassificationSplitStrategy],
      ).run()
    } else {
      throw new NotImplementedError(
        s"TaskMetadataType ${taskMetadataType} has not been implemented yet",
      )
    }
  }

  def getSplitStrategyInstance(gbmlConfigWrapper: GbmlConfigPbWrapper): SplitStrategy[_] = {
    val splitGeneratorConfig = gbmlConfigWrapper.datasetConfigPb.splitGeneratorConfig.get

    // Get assigner instance
    val assignerArgs = splitGeneratorConfig.assignerArgs
    val graphMetadataPbWrapper = new GraphMetadataPbWrapper(
      gbmlConfigWrapper.graphMetadataPb,
    )
    val assignerClass: Class[_] = Class.forName(splitGeneratorConfig.assignerClsPath)
    val assignerConstructor = assignerClass.getDeclaredConstructor(
      classOf[Map[String, String]],
      classOf[GraphMetadataPbWrapper],
    )
    val assignerInstance = assignerConstructor
      .newInstance(assignerArgs, graphMetadataPbWrapper)
      .asInstanceOf[Assigner[_, _]]

    // Get Split Strategy Instance
    val splitstrategyArgs            = splitGeneratorConfig.splitStrategyArgs
    val splitstrategyClass: Class[_] = Class.forName(splitGeneratorConfig.splitStrategyClsPath)
    val splitstrategyConstructor =
      splitstrategyClass.getConstructor(classOf[Map[String, String]], classOf[Assigner[_, _]])
    val splitstrategyInstance = splitstrategyConstructor
      .newInstance(splitstrategyArgs, assignerInstance)
      .asInstanceOf[SplitStrategy[_]]

    splitstrategyInstance
  }
}
