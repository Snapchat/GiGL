import common.test.testLibs.SharedSparkSession
import common.types.pb_wrappers.RootedNodeNeighborhoodPbWrapper
import common.utils.TFRecordIO.RecordTypes
import common.utils.TFRecordIO.dataframeToTypedDataset
import common.utils.TFRecordIO.readDataframeFromTfrecord
import libs.task.pureSparkV2.EgoNetGeneration
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Dataset
import org.scalatest.BeforeAndAfterAll
import org.scalatest.BeforeAndAfterEach
import org.scalatest.funsuite.AnyFunSuite
import scalapb.spark.Implicits._
import snapchat.research.gbml.graph_schema.Edge
import snapchat.research.gbml.graph_schema.Graph
import snapchat.research.gbml.graph_schema.Node
import snapchat.research.gbml.training_samples_schema.RootedNodeNeighborhood

import java.io.File
import java.nio.file.Paths
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import scala.collection.mutable.HashMap
import scala.reflect.io.Directory

class EgoNetGenerationTest
    extends AnyFunSuite
    with BeforeAndAfterAll
    with BeforeAndAfterEach
    with SharedSparkSession {

  var testTmpDir: Directory = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    val now          = LocalDateTime.now()
    val formatter    = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss")
    val formattedNow = now.format(formatter)
    testTmpDir = new Directory(new File(s"/tmp/ego_net_generation_test/${formattedNow}"))
    println("Created test tmp dir: " + testTmpDir)
  }

  override def beforeEach(): Unit = {
    super.beforeEach()
    // We remove spark warehouse i.e. tmp files to not interfere with prior tests
    val currentDir        = Paths.get("").toAbsolutePath.toString
    val sparkWarehouseDir = new Directory(new File(s"${currentDir}/spark-warehouse"))
    sparkWarehouseDir.deleteRecursively()
  }

  override protected def afterAll(): Unit = {
    println(s"Removing test tmp dir: ${testTmpDir}")
    testTmpDir.deleteRecursively()
    super.afterAll()
  }

  test("Check if we can bidirectionalize edges adequately") {
    import sqlImplicits._

    val rawEdgeData = Seq(
      (1, 0, List(1.0f, 0.0f)),
      (0, 1, List(1.0f, 0.0f)),
      (2, 1, List(2.0f, 2.0f)),
    )
    val rawEdgeDF = rawEdgeData.toDF(
      "src_node_id",
      "dst_node_id",
      "edge_features",
    )
    val expectedBidirectionalData = Seq(
      (1, 0, 0, List(1.0f, 0.0f)),
      (0, 1, 0, List(1.0f, 0.0f)),
      (2, 1, 0, List(2.0f, 2.0f)),
      (1, 2, 0, List(2.0f, 2.0f)),
    )
    val expectedBidirectionalDF = expectedBidirectionalData.toDF(
      "src_node_id",
      "dst_node_id",
      "edge_type",
      "edge_features",
    )

    val rawEdgeTable = "raw_edge_table_test"
    rawEdgeDF.createOrReplaceTempView(rawEdgeTable)

    val df = EgoNetGeneration.biDirectionalizeTable(rawEdgeTable)
    df.show()
    df.printSchema()
    assert(df.collect().toSet == expectedBidirectionalDF.collect().toSet)
  }

  def _mockExpectedRnnFromEdges(
    rootNodeId: Int,
    edges: Seq[(Int, Int)],
  ): RootedNodeNeighborhood = {

    /**
        * Mocks the expected RootedNodeNeighborhood object from the given edges.
        *
        * @param rootNodeId: Int
        * @param edges: Seq[(Int, Int, List[Float])]: List of edges where each edge is a tuple
        *  of (dstNodeId, srcNodeId, edgeFeatures)
        * @return RootedNodeNeighborhood
        */
    val _defaultNodeType = 0
    val _nodes = edges.map { case (dstNodeId, srcNodeId) =>
      Node(
        nodeId = srcNodeId,
        condensedNodeType = Some(_defaultNodeType),
        featureValues = Array[Float](),
      )
    }
    // EgoNetGeneration code creates bidirectional edges; where direction is defined by
    // the edge type. So, we need to create two edges for each edge in the input data.
    val _edges = edges.flatMap { case (dstNodeId, srcNodeId) =>
      Seq(
        Edge(
          srcNodeId = srcNodeId,
          dstNodeId = dstNodeId,
          condensedEdgeType = Some(EgoNetGeneration.DEFAULT_EDGE_TYPE),
          featureValues = Array[Float](),
        ),
      )
    }
    RootedNodeNeighborhood(
      rootNode = Some(
        Node(
          nodeId = rootNodeId,
          condensedNodeType = Some(_defaultNodeType),
          featureValues = Array[Float](),
        ),
      ),
      neighborhood = Some(
        Graph(
          nodes = _nodes,
          edges = _edges,
        ),
      ),
    )
  }

  def generateInputDfAndExpectedRNNs(): Tuple2[DataFrame, HashMap[Int, RootedNodeNeighborhood]] = {
    import sqlImplicits._

    // Edge features are not carried right now for third hop, and also get filtered out in postprocessor.
    // TODO: (svij-sc) This work just needs to be implemented, so for now we test with empty edge features.
    val rawEdgeData = Seq(
      (1, 0, List[Float]()),
      (0, 1, List[Float]()),
      (2, 1, List[Float]()),
      (3, 2, List[Float]()),
      (3, 4, List[Float]()),
    )
    val rawEdgeDF = rawEdgeData.toDF(
      "dst_node_id",
      "src_node_id",
      "edge_features",
    )
    val expectedRnnForNode_0 = _mockExpectedRnnFromEdges(
      rootNodeId = 0,
      // Note we bi-directionalize edges so we expect inward edges only.
      // This both this saves space and reduces complexity for compute.
      // If needed we can always generate the other direction edge on the fly.
      edges = Seq(
        (0, 1), // first hop
        (1, 2),
        (1, 0), // second hop
        (2, 3),
        (2, 1), // third hop
      ),
    )
    val expectedRnnForNode_4 = _mockExpectedRnnFromEdges(
      rootNodeId = 4,
      edges = Seq(
        (4, 3), // first hop
        (3, 2),
        (3, 4), // second hop
        (2, 1),
        (2, 3), // third hop
      ),
    )

    val rootNodeIdToExpectedRnnMap = new HashMap[Int, RootedNodeNeighborhood]()
    rootNodeIdToExpectedRnnMap.put(0, expectedRnnForNode_0)
    rootNodeIdToExpectedRnnMap.put(4, expectedRnnForNode_4)

    (rawEdgeDF, rootNodeIdToExpectedRnnMap)
  }

  test("Test if we can generate ego nets adequately") {
    val (inputDF, rootNodeIdToExpectedRnnMap) = generateInputDfAndExpectedRNNs()
    val outputDir                             = testTmpDir + "/ego_net_generation_test_output/"
    EgoNetGeneration.generateEgoNodes(
      spark = sparkTest,
      rawEdgeDF = inputDF,
      numOptimalPartitions = 10,
      numSlots = 1,
      sampleN = 2,
      numHops = 3,
      outputDir = outputDir,
    )

    // TODO: Read generated tf records and check if they are correct
    val rawEdgeDF: DataFrame = readDataframeFromTfrecord(
      uri = outputDir,
      recordType = RecordTypes.ByteArrayRecordType,
    )
    val rnnDS: Dataset[RootedNodeNeighborhood] =
      dataframeToTypedDataset[RootedNodeNeighborhood](rawEdgeDF)
    val protos = rnnDS.collect()
    assert(protos.length == 5) // We have 4 root nodes: [0, 1, 2, 3, 4] in the input data
    val rootNodeIdToProtoMap = new HashMap[Int, RootedNodeNeighborhood]()
    for (proto <- protos) {
      val rootNodeId = proto.rootNode.get.nodeId
      rootNodeIdToProtoMap.put(rootNodeId, proto)
    }
    for (i <- List(0, 4)) {
      val protoWrapperForNode = new RootedNodeNeighborhoodPbWrapper(rootNodeIdToProtoMap(i))
      val expectedWrapperForNode =
        new RootedNodeNeighborhoodPbWrapper(rootNodeIdToExpectedRnnMap(i))
      assert(protoWrapperForNode.equals(expectedWrapperForNode))
    }
  }

}
