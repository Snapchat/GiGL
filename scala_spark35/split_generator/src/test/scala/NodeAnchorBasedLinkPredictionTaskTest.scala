package splitgenerator.test

import common.test.testLibs.SharedSparkSession
import common.types.pb_wrappers.GbmlConfigPbWrapper
import common.utils.ProtoLoader
import common.utils.TFRecordIO
import org.apache.spark.sql.Dataset
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import scalapb.spark.Implicits._
import snapchat.research.gbml.gbml_config.GbmlConfig
import snapchat.research.gbml.graph_schema.Edge
import snapchat.research.gbml.training_samples_schema.NodeAnchorBasedLinkPredictionSample
import snapchat.research.gbml.training_samples_schema.RootedNodeNeighborhood
import splitgenerator.lib.SplitGeneratorTaskRunner
import splitgenerator.lib.split_strategies.TransductiveNodeAnchorBasedLinkPredictionSplitStrategy
import splitgenerator.lib.tasks.NodeAnchorBasedLinkPredictionTask
import splitgenerator.test.SplitGeneratorTestUtils.getSplits

class NodeAnchorBasedLinkPredictionTaskTest
    extends AnyFunSuite
    with BeforeAndAfterAll
    with SharedSparkSession {

  var gbmlConfigWrapper: GbmlConfigPbWrapper = _

  var splitStrategy: TransductiveNodeAnchorBasedLinkPredictionSplitStrategy =
    _

  var splitGeneratorTask: NodeAnchorBasedLinkPredictionTask = _

  override protected def beforeAll(): Unit = {
    super.beforeAll()

    val frozenGbmlConfigUriTest =
      "common/src/test/assets/split_generator/node_anchor_based_link_prediction/frozen_gbml_config.yaml"
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

  test("can load and split main samples (Toy graph) using Spark") {
    // load main samples
    val mainSamplesDf = splitGeneratorTask.loadCoalesceCacheDataframe(
      inputPath =
        gbmlConfigWrapper.flattenedGraphMetadataPb.getNodeAnchorBasedLinkPredictionOutput.tfrecordUriPrefix,
      coalesceFactor = 1,
    )

    // try deserialize to proto
    val mainSamplesDS: Dataset[NodeAnchorBasedLinkPredictionSample] =
      TFRecordIO.dataframeToTypedDataset(
        df = mainSamplesDf,
      )

    assert(mainSamplesDS.collect().size > 0)

    val (trainSplit, valSplit, testSplit) =
      getSplits(samples = mainSamplesDS, splitFn = splitStrategy.splitTrainingSample)

    val trainPosEdges: Array[Edge] = trainSplit.flatMap(sample => sample.posEdges).collect()
    val valPosEdges: Array[Edge]   = valSplit.flatMap(sample => sample.posEdges).collect()
    val testPosEdges: Array[Edge]  = testSplit.flatMap(sample => sample.posEdges).collect()

    // All supervision edges are preserved
    val expectedPosEdges = mainSamplesDS.flatMap(sample => sample.posEdges).collect()
    assert(expectedPosEdges.toSet == (trainPosEdges ++ valPosEdges ++ testPosEdges).toSet)

    // train split has lesser samples than main sample (sampls lost due to no positive supervsion edges)
    assert(trainSplit.collect().size < mainSamplesDS.collect().size)

    // val/test has same number of samples
    assert(valSplit.collect().size == mainSamplesDS.collect().size)
    assert(testSplit.collect().size == mainSamplesDS.collect().size)
  }

  test("can load and split rooted node neighborhood samples (Toy graph) using Spark") {
    // load main samples
    val defaultTargetNodeType =
      gbmlConfigWrapper.taskMetadataPb.getNodeAnchorBasedLinkPredictionTaskMetadata.supervisionEdgeTypes.head.dstNodeType
    val rootedNodeNeighborhoodDf = splitGeneratorTask.loadCoalesceCacheDataframe(
      inputPath =
        gbmlConfigWrapper.flattenedGraphMetadataPb.getNodeAnchorBasedLinkPredictionOutput.nodeTypeToRandomNegativeTfrecordUriPrefix
          .getOrElse(
            defaultTargetNodeType,
            throw new Exception(
              "If you are seeing this, it means the node_type_to_random_negative_tfrecord_uri_prefix is missing the dstNodeType of the first supervision edge type, please check the frozen config",
            ),
          ),
      coalesceFactor = 1,
    )

    // try deserialize to proto
    val rootedNodeNeighborhoodDS: Dataset[RootedNodeNeighborhood] =
      TFRecordIO.dataframeToTypedDataset(
        df = rootedNodeNeighborhoodDf,
      )

    assert(rootedNodeNeighborhoodDS.collect().size > 0)

    val (trainSplit, valSplit, testSplit) =
      getSplits(
        samples = rootedNodeNeighborhoodDS,
        splitFn = splitStrategy.splitRootedNodeNeighborhoodTrainingSample,
      )

    val trainNeighborhoodEdges: Array[Edge] =
      trainSplit.flatMap(sample => sample.neighborhood.get.edges).collect()
    val valNeighborhoodEdges: Array[Edge] =
      valSplit.flatMap(sample => sample.neighborhood.get.edges).collect()
    val testNeighborhoodEdges: Array[Edge] =
      testSplit.flatMap(sample => sample.neighborhood.get.edges).collect()

    // train and val neighborhood graph has same number of edges
    assert(trainNeighborhoodEdges.size == valNeighborhoodEdges.size)

    // Test graph should have more edges than val/train graphs.
    assert(testNeighborhoodEdges.size > valNeighborhoodEdges.size)
  }

}
