import common.types.pb_wrappers.TaskMetadataPbWrapper
import common.types.pb_wrappers.TaskMetadataType
import common.utils.ProtoLoader
import org.scalatest.funsuite.AnyFunSuite
import snapchat.research.gbml.gbml_config.GbmlConfig
import snapchat.research.gbml.preprocessed_metadata.PreprocessedMetadata

import scala.io.Source

class ProtoLoaderTest extends AnyFunSuite {
  test("ProtoLoader.loadYamlToProto - test local yaml file can be loaded") {
    val pathToLinkPredictionTestFile = (
      "common/src/test/assets/subgraph_sampler/node_anchor_based_link_prediction/frozen_gbml_config.yaml"
    )
    val gbmlConfig =
      ProtoLoader.loadYamlStrToProto[GbmlConfig](pathToLinkPredictionTestFile)
    val taskMetadataType = TaskMetadataPbWrapper(gbmlConfig.taskMetadata.get).taskMetadataType
    assert(taskMetadataType.equals(TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK))
    assert(
      gbmlConfig.datasetConfig.get.dataPreprocessorConfig.get.dataPreprocessorConfigClsPath == "this.is.non.existent.test.path.1",
    )

    val pathToPreprocessedMetadataTestFile = (
      "common/src/test/assets/subgraph_sampler/node_anchor_based_link_prediction/preprocessed_metadata.yaml"
    )
    val preprocessed_metadata =
      ProtoLoader.loadYamlStrToProto[PreprocessedMetadata](
        pathToPreprocessedMetadataTestFile,
      )

    assert(preprocessed_metadata.condensedNodeTypeToPreprocessedMetadata.contains(0))
    assert(
      preprocessed_metadata.condensedNodeTypeToPreprocessedMetadata
        .get(0)
        .get
        .featureKeys
        .equals(
          Vector(
            "f0",
            "f1",
          ),
        ),
    )
  }

  test("ProtoLoader.load_yaml_to_proto - test in memory yaml string can be loaded") {
    val pathToLinkPredictionTestFile = (
      "common/src/test/assets/subgraph_sampler/node_anchor_based_link_prediction/frozen_gbml_config.yaml"
    )
    val yamlString = Source.fromFile(pathToLinkPredictionTestFile).mkString

    val gbmlConfig       = ProtoLoader.loadYamlStrToProto[GbmlConfig](yamlString)
    val taskMetadataType = TaskMetadataPbWrapper(gbmlConfig.taskMetadata.get).taskMetadataType
    assert(taskMetadataType.equals(TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK))
    assert(
      gbmlConfig.datasetConfig.get.dataPreprocessorConfig.get.dataPreprocessorConfigClsPath == "this.is.non.existent.test.path.1",
    )
  }
}
