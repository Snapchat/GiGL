import common.test.testLibs.SharedSparkSession
import common.types.pb_wrappers.RootedNodeNeighborhoodPbWrapper
import common.userDefinedAggregators.RnnUDAF
import org.apache.spark.sql.{functions => F}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import scalapb.spark.Implicits._
import snapchat.research.gbml.graph_schema.Edge
import snapchat.research.gbml.graph_schema.Graph
import snapchat.research.gbml.graph_schema.Node
import snapchat.research.gbml.training_samples_schema.RootedNodeNeighborhood

class RnnUDAFTest extends AnyFunSuite with BeforeAndAfterAll with SharedSparkSession {

  // override def beforeAll(): Unit = {
  //   super.beforeAll()
  // }

  test("test RnnUDAF") {

    /**
     * Firstly, create a data frame with the columns that RnnUDAF accepts.
     * @see common.userDefinedAggregators.RnnUDAF
     * Schema:
     * _root_node_id: NodeId,
     * _root_node_type: NodeType,
     * _root_node_features: Seq[Float],
     * _1_hop_node_id: NodeId,
     * _1_hop_node_type: NodeType,
     * _1_hop_node_features: Seq[Float],
     * _1_hop_edge_features: Seq[Float],
     * _1_hop_edge_type: CondensedEdgeType,
     * _2_hop_node_id: NodeId,
     * _2_hop_node_type: NodeType,
     * _2_hop_node_features: Seq[Float],
     * _2_hop_edge_features: Seq[Float],
     * _2_hop_edge_type: CondensedEdgeType
    */
    val node_0_feat     = Seq(999.0f)
    val node_1_feat     = Seq(999.1f)
    val node_2_feat     = Seq(999.2f)
    val node_3_feat     = Seq(999.3f)
    val node_11_feat    = Seq(999.11f)
    val node_22_feat    = Seq(999.22f)
    val edge_1_0_feat   = Seq(1.0f)
    val edge_2_1_feat   = Seq(2.1f)
    val edge_22_11_feat = Seq(22.11f)
    val edge_11_0_feat  = Seq(11.0f)
    val edge_3_2_feat   = Seq(3.2f)
    val edge_3_0_feat   = Seq(3.0f)
    val edge_2_3_feat   = Seq(2.3f)

    val edge_type_1 = 1
    val edge_type_2 = 2
    val node_type_1 = 1
    val node_type_2 = 2

    val dataFrameRow1 = RnnUDAF.InTwoHopData(
      _root_node_id = 0,
      _root_node_type = node_type_1,
      _root_node_features = node_0_feat,
      _1_hop_node_id = 1,
      _1_hop_node_type = node_type_1,
      _1_hop_node_features = node_1_feat,
      _1_hop_edge_features = edge_1_0_feat,
      _1_hop_edge_type = edge_type_1,
      _2_hop_node_id = 2,
      _2_hop_node_type = node_type_1,
      _2_hop_node_features = node_2_feat,
      _2_hop_edge_features = edge_2_1_feat,
      _2_hop_edge_type = edge_type_1,
    )
    val dataFrameRow2 = RnnUDAF.InTwoHopData(
      _root_node_id = 0,
      _root_node_type = node_type_1,
      _root_node_features = node_0_feat,
      _1_hop_node_id = 3,
      _1_hop_node_type = node_type_1,
      _1_hop_node_features = node_3_feat,
      _1_hop_edge_features = edge_3_0_feat,
      _1_hop_edge_type = edge_type_2,
      _2_hop_node_id = 2,
      _2_hop_node_type = node_type_1,
      _2_hop_node_features = node_2_feat,
      _2_hop_edge_features = edge_2_3_feat,
      _2_hop_edge_type = edge_type_1,
    )
    val dataFrameRow3 = RnnUDAF.InTwoHopData(
      _root_node_id = 0,
      _root_node_type = node_type_2,
      _root_node_features = node_0_feat,
      _1_hop_node_id = 11,
      _1_hop_node_type = node_type_1,
      _1_hop_node_features = node_11_feat,
      _1_hop_edge_features = edge_11_0_feat,
      _1_hop_edge_type = edge_type_1,
      _2_hop_node_id = 22,
      _2_hop_node_type = node_type_1,
      _2_hop_node_features = node_22_feat,
      _2_hop_edge_features = edge_22_11_feat,
      _2_hop_edge_type = edge_type_1,
    )

    val dataFrameRow4 = RnnUDAF.InTwoHopData(
      _root_node_id = 1,
      _root_node_type = node_type_1,
      _root_node_features = node_1_feat,
      _1_hop_node_id = 2,
      _1_hop_node_type = node_type_1,
      _1_hop_node_features = node_2_feat,
      _1_hop_edge_features = edge_2_1_feat,
      _1_hop_edge_type = edge_type_1,
      _2_hop_node_id = 3,
      _2_hop_node_type = node_type_1,
      _2_hop_node_features = node_3_feat,
      _2_hop_edge_features = edge_3_2_feat,
      _2_hop_edge_type = edge_type_1,
    )
    val rows = Seq(dataFrameRow1, dataFrameRow2, dataFrameRow3, dataFrameRow4).map(
      RnnUDAF.InTwoHopData.unapply(_).get,
    )

    val expectedRNNForRootNode_0_type_1 = RootedNodeNeighborhood(
      rootNode = Some(
        Node(
          nodeId = 0,
          condensedNodeType = Some(node_type_1),
          featureValues = node_0_feat,
        ),
      ),
      neighborhood = Some(
        Graph(
          nodes = Seq(
            Node(
              nodeId = 0,
              condensedNodeType = Some(node_type_1),
              featureValues = node_0_feat,
            ),
            Node(
              nodeId = 1,
              condensedNodeType = Some(node_type_1),
              featureValues = node_1_feat,
            ),
            Node(
              nodeId = 3,
              condensedNodeType = Some(node_type_1),
              featureValues = node_3_feat,
            ),
            Node(
              nodeId = 2,
              condensedNodeType = Some(node_type_1),
              featureValues = node_2_feat,
            ),
          ),
          edges = Seq(
            Edge(
              srcNodeId = 1,
              dstNodeId = 0,
              condensedEdgeType = Some(edge_type_1),
              featureValues = edge_1_0_feat,
            ),
            Edge(
              srcNodeId = 3,
              dstNodeId = 0,
              condensedEdgeType = Some(edge_type_2),
              featureValues = edge_3_0_feat,
            ),
            Edge(
              srcNodeId = 2,
              dstNodeId = 3,
              condensedEdgeType = Some(edge_type_1),
              featureValues = edge_2_3_feat,
            ),
            Edge(
              srcNodeId = 2,
              dstNodeId = 1,
              condensedEdgeType = Some(edge_type_1),
              featureValues = edge_2_1_feat,
            ),
          ),
        ),
      ),
    )

    val expectedRNNForRootNode_0_type_2 = RootedNodeNeighborhood(
      rootNode = Some(
        Node(
          nodeId = 0,
          condensedNodeType = Some(node_type_2),
          featureValues = node_0_feat,
        ),
      ),
      neighborhood = Some(
        Graph(
          nodes = Seq(
            Node(
              nodeId = 0,
              condensedNodeType = Some(node_type_2),
              featureValues = node_0_feat,
            ),
            Node(
              nodeId = 11,
              condensedNodeType = Some(node_type_1),
              featureValues = node_11_feat,
            ),
            Node(
              nodeId = 22,
              condensedNodeType = Some(node_type_1),
              featureValues = node_22_feat,
            ),
          ),
          edges = Seq(
            Edge(
              srcNodeId = 11,
              dstNodeId = 0,
              condensedEdgeType = Some(edge_type_1),
              featureValues = edge_11_0_feat,
            ),
            Edge(
              srcNodeId = 22,
              dstNodeId = 11,
              condensedEdgeType = Some(edge_type_1),
              featureValues = edge_22_11_feat,
            ),
          ),
        ),
      ),
    )

    val df = sparkTest
      .createDataFrame(rows)
      .toDF(
        "_root_node_id",
        "_root_node_type",
        "_root_node_features",
        "_1_hop_node_id",
        "_1_hop_node_type",
        "_1_hop_node_features",
        "_1_hop_edge_features",
        "_1_hop_edge_type",
        "_2_hop_node_id",
        "_2_hop_node_type",
        "_2_hop_node_features",
        "_2_hop_edge_features",
        "_2_hop_edge_type",
      )

    sparkTest.udf.register("rnnUDAF", F.udaf(new RnnUDAF(sampleN = Some(2))))
    df.createOrReplaceTempView("test_view")
    val result = sparkTest.sql("""
        SELECT
            rnnUDAF(
                _root_node_id,
                _root_node_type,
                _root_node_features,
                _1_hop_node_id,
                _1_hop_node_type,
                _1_hop_node_features,
                _1_hop_edge_features,
                _1_hop_edge_type,
                _2_hop_node_id,
                _2_hop_node_type,
                _2_hop_node_features,
                _2_hop_edge_features,
                _2_hop_edge_type
            ) as result 
        FROM 
            test_view 
        GROUP BY 
            _root_node_id, _root_node_type
    """)

    val protosDS = result.as[Array[Byte]].map(RootedNodeNeighborhood.parseFrom(_))

    val protos = protosDS.collect()
    assert(protos.length == 3) // root node 0; type 1, root node 0; type 2, root node 1; type 1
    val protosMap: Map[
      Tuple2[RnnUDAF.NodeId, RnnUDAF.CondensedNodeType],
      RootedNodeNeighborhood,
    ] = protos.map(p => ((p.rootNode.get.nodeId, p.rootNode.get.condensedNodeType.get), p)).toMap

    assert(
      new RootedNodeNeighborhoodPbWrapper(rootedNodeNeighborhood = protosMap((0, node_type_1))) ==
        new RootedNodeNeighborhoodPbWrapper(
          rootedNodeNeighborhood = expectedRNNForRootNode_0_type_1,
        ),
    )
    assert(
      new RootedNodeNeighborhoodPbWrapper(protosMap((0, node_type_2))) ==
        new RootedNodeNeighborhoodPbWrapper(expectedRNNForRootNode_0_type_2),
    )
  }

}
