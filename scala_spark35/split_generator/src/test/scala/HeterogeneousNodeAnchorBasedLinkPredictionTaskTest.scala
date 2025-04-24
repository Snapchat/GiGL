package splitgenerator.test

import common.test.testLibs.SharedSparkSession
import common.types.pb_wrappers.GbmlConfigPbWrapper
import common.utils.ProtoLoader
import common.utils.TFRecordIO
import org.apache.spark.sql.Dataset
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers._
import scalapb.spark.Implicits._
import snapchat.research.gbml.gbml_config.GbmlConfig
import snapchat.research.gbml.training_samples_schema.RootedNodeNeighborhood
import splitgenerator.lib.SplitGeneratorTaskRunner
import splitgenerator.lib.split_strategies.TransductiveNodeAnchorBasedLinkPredictionSplitStrategy
import splitgenerator.lib.tasks.NodeAnchorBasedLinkPredictionTask
import splitgenerator.test.SplitGeneratorTestUtils.getSplits

class HeterogeneousNodeAnchorBasedLinkPredictionTaskTest
    extends AnyFunSuite
    with BeforeAndAfterAll
    with SharedSparkSession
    with Serializable {

  var gbmlConfigWrapper: GbmlConfigPbWrapper = _

  var splitStrategy: TransductiveNodeAnchorBasedLinkPredictionSplitStrategy = _

  var splitGeneratorTask: NodeAnchorBasedLinkPredictionTask = _

  override protected def beforeAll(): Unit = {
    super.beforeAll()

    val frozenGbmlConfigUriTest =
      "common/src/test/assets/split_generator/hetero_node_anchor_based_link_prediction/frozen_gbml_config.yaml"
    val gbmlConfigProto =
      ProtoLoader.populateProtoFromYaml[GbmlConfig](uri = frozenGbmlConfigUriTest)
    gbmlConfigWrapper = GbmlConfigPbWrapper(gbmlConfigPb = gbmlConfigProto)

    splitStrategy = SplitGeneratorTaskRunner
      .getSplitStrategyInstance(gbmlConfigWrapper)
      .asInstanceOf[TransductiveNodeAnchorBasedLinkPredictionSplitStrategy]

    splitGeneratorTask = new NodeAnchorBasedLinkPredictionTask(
      gbmlConfigWrapper = gbmlConfigWrapper,
      splitStrategy = splitStrategy,
    )

  }

  test("anchor node type is correct after splitting for RNNs") {

    val anchorNodeType =
      gbmlConfigWrapper.taskMetadataPb.getNodeAnchorBasedLinkPredictionTaskMetadata.supervisionEdgeTypes.head.dstNodeType
    // load rnn samples
    val rnnSamplesDF = splitGeneratorTask.loadCoalesceCacheDataframe(
      inputPath =
        gbmlConfigWrapper.flattenedGraphMetadataPb.getNodeAnchorBasedLinkPredictionOutput.nodeTypeToRandomNegativeTfrecordUriPrefix
          .get(anchorNodeType)
          .get,
      coalesceFactor = 1,
    )

    val rnnSamplesDS: Dataset[RootedNodeNeighborhood] = TFRecordIO.dataframeToTypedDataset(
      df = rnnSamplesDF,
    )

    val (trainSplit, valSplit, testSplit) = getSplits(
      samples = rnnSamplesDS,
      splitFn = splitStrategy.splitRootedNodeNeighborhoodTrainingSample,
    )

    checkAnchorNodeTypeMatchTargetNodeType(
      dataSplit = trainSplit,
      anchorNodeType = anchorNodeType,
      gbmlConfigWrapper = gbmlConfigWrapper,
    )

    checkAnchorNodeTypeMatchTargetNodeType(
      dataSplit = valSplit,
      anchorNodeType = anchorNodeType,
      gbmlConfigWrapper = gbmlConfigWrapper,
    )

    checkAnchorNodeTypeMatchTargetNodeType(
      dataSplit = testSplit,
      anchorNodeType = anchorNodeType,
      gbmlConfigWrapper = gbmlConfigWrapper,
    )

  }

  private def checkAnchorNodeTypeMatchTargetNodeType(
    dataSplit: Dataset[RootedNodeNeighborhood],
    anchorNodeType: String,
    gbmlConfigWrapper: GbmlConfigPbWrapper,
  ): Unit = {
    val splitCondensedAnchorNodeType = dataSplit
      .map(sample => sample.rootNode.get.condensedNodeType.get)
      .collect()

    val allAnchorNodeTypesMatchTargetNodeType = splitCondensedAnchorNodeType
      .map(condensedNodeType =>
        gbmlConfigWrapper.graphMetadataPb.condensedNodeTypeMap
          .get(condensedNodeType)
          .get == anchorNodeType,
      )
      .reduce(_ && _)

    allAnchorNodeTypesMatchTargetNodeType shouldBe true
  }

}
