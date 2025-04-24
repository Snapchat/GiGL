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
import snapchat.research.gbml.training_samples_schema.SupervisedNodeClassificationSample
import splitgenerator.lib.SplitGeneratorTaskRunner
import splitgenerator.lib.split_strategies.TransductiveSupervisedNodeClassificationSplitStrategy
import splitgenerator.lib.tasks.SupervisedNodeClassificationTask

class SupervisedNodeClassificationTaskTest
    extends AnyFunSuite
    with BeforeAndAfterAll
    with SharedSparkSession {

  var gbmlConfigWrapper: GbmlConfigPbWrapper = _

  var splitStrategy: TransductiveSupervisedNodeClassificationSplitStrategy =
    _

  var splitGeneratorTask: SupervisedNodeClassificationTask = _

  override protected def beforeAll(): Unit = {
    super.beforeAll()

    val frozenGbmlConfigUriTest =
      "common/src/test/assets/split_generator/supervised_node_classification/frozen_gbml_config.yaml"
    val gbmlConfigProto =
      ProtoLoader.populateProtoFromYaml[GbmlConfig](uri = frozenGbmlConfigUriTest)
    gbmlConfigWrapper = GbmlConfigPbWrapper(gbmlConfigPb = gbmlConfigProto)

    splitStrategy = SplitGeneratorTaskRunner
      .getSplitStrategyInstance(gbmlConfigWrapper)
      .asInstanceOf[TransductiveSupervisedNodeClassificationSplitStrategy]

    splitGeneratorTask = new SupervisedNodeClassificationTask(
      gbmlConfigWrapper = gbmlConfigWrapper,
      splitStrategy = splitStrategy,
    )
  }

  test("can load supervised node classification samples using Spark") {
    // load main samples
    val splitSamples = splitGeneratorTask.loadCoalesceCacheDataframe(
      inputPath =
        gbmlConfigWrapper.flattenedGraphMetadataPb.getSupervisedNodeClassificationOutput.labeledTfrecordUriPrefix,
      coalesceFactor = 1,
    )

    // try deserialize to proto
    val splitSamplesDS: Dataset[SupervisedNodeClassificationSample] =
      TFRecordIO.dataframeToTypedDataset(
        df = splitSamples,
      )

    assert(splitSamplesDS.collect().size > 0)
  }
}
