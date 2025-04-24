package splitgenerator.test

import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import snapchat.research.gbml.graph_schema.Edge
import snapchat.research.gbml.graph_schema.Graph
import snapchat.research.gbml.graph_schema.Node
import snapchat.research.gbml.training_samples_schema.NodeAnchorBasedLinkPredictionSample
import snapchat.research.gbml.training_samples_schema.RootedNodeNeighborhood
import splitgenerator.lib.Types.DatasetSplits
import splitgenerator.lib.assigners.NodeToDatasetSplitHashingAssigner
import splitgenerator.lib.split_strategies.UDLAnchorBasedSupervisionEdgeSplitStrategy

import SplitGeneratorTestUtils.{getMockMainInputSample, getMockRootedNodeNeighborhoodSample}

class UDLAnchorBasedSupervisionEdgeSplitStrategyTest extends AnyFunSuite with BeforeAndAfterAll {

  val mainSample: NodeAnchorBasedLinkPredictionSample      = getMockMainInputSample
  val rootedNodeNeighborhoodSample: RootedNodeNeighborhood = getMockRootedNodeNeighborhoodSample
  val splitStrategy =
    new UDLAnchorBasedSupervisionEdgeSplitStrategy(
      splitStrategyArgs = Map[String, String](),
      nodeToDatasetSplitAssigner = new NodeToDatasetSplitHashingAssigner(
        assignerArgs = Map[String, String](
          "train_split" -> "1.0", // we use these ratios to enforce pos edge in train split
          "val_split"   -> "0.0",
          "test_split"  -> "0.0",
        ),
        graphMetadataPbWrapper = SplitGeneratorTestUtils.getGraphMetadataWrapper(),
      ),
    )

  // TODO (svij-sc): createRNN, and createNALPSample can probably be abstracted out
  def createRNN(
    rootNodeId: Int,
    edgeList: Seq[Tuple2[Int, Int]],
  ): RootedNodeNeighborhood = {
    val nodes = edgeList.flatMap { case (src, dst) => Seq(src, dst) }.distinct.map { nodeId =>
      Node(nodeId = nodeId, featureValues = Seq(1.0f))
    }
    val edges = edgeList.map { case (src, dst) =>
      Edge(srcNodeId = src, dstNodeId = dst, featureValues = Seq(1.0f))
    }
    RootedNodeNeighborhood(
      rootNode = Some(Node(nodeId = rootNodeId, featureValues = Seq(1.0f))),
      neighborhood = Some(Graph(nodes = nodes, edges = edges)),
    )
  }

  def createNALPSample(
    rootNodeId: Int,
    edgeList: Seq[Tuple2[Int, Int]],
    posEdgeList: Seq[Tuple2[Int, Int]],
    negEdgeList: Seq[Tuple2[Int, Int]],
  ): NodeAnchorBasedLinkPredictionSample = {
    val _tmpRnn = createRNN(rootNodeId, edgeList)
    val posEdges = posEdgeList.map { case (src, dst) =>
      Edge(srcNodeId = src, dstNodeId = dst, featureValues = Seq(1.0f))
    }
    val negEdges = negEdgeList.map { case (src, dst) =>
      Edge(srcNodeId = src, dstNodeId = dst, featureValues = Seq(1.0f))
    }
    NodeAnchorBasedLinkPredictionSample(
      rootNode = _tmpRnn.rootNode,
      neighborhood = _tmpRnn.neighborhood,
      posEdges = posEdges,
      negEdges = negEdges,
    )
  }

  test("main samples split is valid for user defined labels") {

    val sampleNalpFiltersEntireEdgeList = createNALPSample(
      rootNodeId = 1,
      edgeList = Seq((1, 2), (2, 1), (3, 1)),
      posEdgeList = Seq((1, 2)),
      negEdgeList = Seq((1, 3)),
    )
    val splitSeq = splitStrategy.splitTrainingSample(
      sample = sampleNalpFiltersEntireEdgeList,
      datasetSplit = DatasetSplits.TRAIN,
    )
    // We have no neighborhood edges due to filtering all of them out because of pos/neg edges.
    // Thus we should have no training samples.
    assert(splitSeq.length == 0)

    val sampleNalpNoPosEdge = createNALPSample(
      rootNodeId = 1,
      edgeList = Seq((1, 2), (1, 5)),
      posEdgeList = Seq(),
      negEdgeList = Seq((1, 3)),
    )
    val splitSeq2 = splitStrategy.splitTrainingSample(
      sample = sampleNalpNoPosEdge,
      datasetSplit = DatasetSplits.TRAIN,
    )
    assert(splitSeq2.length == 0) // No posEdgeList to begin with; this is more sanity check

    val sampleNalpFiltersSomeSupervisionEdges = createNALPSample(
      rootNodeId = 1,
      edgeList = Seq((1, 2), (1, 5), (3, 2), (10, 5)),
      posEdgeList = Seq((1, 2)),
      negEdgeList = Seq((3, 2)),
    )
    val expectedSampleAfterFiltering = createNALPSample(
      rootNodeId = 1,
      edgeList = Seq((1, 5), (10, 5)),
      posEdgeList = Seq((1, 2)),
      negEdgeList = Seq((3, 2)),
    )
    val splitSeq3 = splitStrategy.splitTrainingSample(
      sample = sampleNalpFiltersSomeSupervisionEdges,
      datasetSplit = DatasetSplits.TRAIN,
    )
    assert(splitSeq3.length == 1)
    println("splitSeq3: ", splitSeq3)
    println("expectedSampleAfterFiltering: ", expectedSampleAfterFiltering)
    assert(splitSeq3(0) == expectedSampleAfterFiltering)

    val testSeq = splitStrategy.splitTrainingSample(
      sample = sampleNalpFiltersSomeSupervisionEdges,
      datasetSplit = DatasetSplits.TEST,
    )
    assert(testSeq.length == 0) // "train_split" -> "1.0"
  }

  test("test rooted node neighborhood split is valid for user defined labels") {

    val sampleRNN = createRNN(rootNodeId = 1, edgeList = Seq((1, 2)))

    val splitSeq = splitStrategy.splitRootedNodeNeighborhoodTrainingSample(
      sample = sampleRNN,
      datasetSplit = DatasetSplits.TRAIN,
    )
    assert(splitSeq.length == 1)
    assert(splitSeq(0) == sampleRNN) // No changes expected

    val testSeq = splitStrategy.splitRootedNodeNeighborhoodTrainingSample(
      sample = sampleRNN,
      datasetSplit = DatasetSplits.TEST,
    )
    // We output test RNN even though "test_split"   -> "0.0",
    assert(splitSeq.length == 1)
    assert(splitSeq(0) == sampleRNN)
  }
}
