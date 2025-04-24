import common.test.SharedSparkSession
import common.types.pb_wrappers.GbmlConfigPbWrapper
import common.types.pb_wrappers.GraphMetadataPbWrapper
import common.utils.ProtoLoader.populateProtoFromYaml
import libs.task.pureSpark.SupervisedNodeClassificationTask
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import snapchat.research.gbml.gbml_config.GbmlConfig

class SupervisedNodeClassificationTaskTest
    extends AnyFunSuite
    with BeforeAndAfterAll
    with SharedSparkSession {

  // commmon/reused vars among tests which must be assigned in beforeAll():
  var sncTask: SupervisedNodeClassificationTask = _
  var gbmlConfigWrapper: GbmlConfigPbWrapper    = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    val frozenGbmlConfigUriTest =
      "common/src/test/assets/subgraph_sampler/supervised_node_classification/frozen_gbml_config.yaml"
    val gbmlConfigProto =
      populateProtoFromYaml[GbmlConfig](uri = frozenGbmlConfigUriTest)
    gbmlConfigWrapper = GbmlConfigPbWrapper(gbmlConfigPb = gbmlConfigProto)

    val graphMetadataPbWrapper: GraphMetadataPbWrapper = GraphMetadataPbWrapper(
      gbmlConfigWrapper.graphMetadataPb,
    )

    sncTask = new SupervisedNodeClassificationTask(
      gbmlConfigWrapper = gbmlConfigWrapper,
      graphMetadataPbWrapper = graphMetadataPbWrapper,
    )

  }

  test("Node label dataframe loads with correct columns.") {
    val nodeLabelVIEW   = sncTask.loadNodeLabelDataframeIntoSparkSql(condensedNodeType = 0)
    val nodeLabelDF     = sparkTest.table(nodeLabelVIEW)
    val expectedColSize = 3
    assert(nodeLabelDF.columns.size == expectedColSize)
    val loadedNodeLabelDfColumns: Seq[String] = nodeLabelDF.columns.toSeq
    assert(loadedNodeLabelDfColumns.contains("_node_id"))
    assert(loadedNodeLabelDfColumns.contains("_label_key"))
    assert(loadedNodeLabelDfColumns.contains("_label_type"))

  }
}
