package splitgenerator.test

import common.types.pb_wrappers.GraphMetadataPbWrapper
import common.types.pb_wrappers.GraphPbWrappers.EdgePbWrapper
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import snapchat.research.gbml.graph_schema.Edge
import snapchat.research.gbml.graph_schema.Graph
import snapchat.research.gbml.graph_schema.Node
import snapchat.research.gbml.training_samples_schema.NodeAnchorBasedLinkPredictionSample
import snapchat.research.gbml.training_samples_schema.RootedNodeNeighborhood
import splitgenerator.lib.Types.DatasetSplits
import splitgenerator.lib.Types.LinkSplitType
import splitgenerator.lib.Types.LinkUsageTypes
import splitgenerator.lib.assigners.AbstractAssigners.Assigner
import splitgenerator.lib.split_strategies.TransductiveNodeAnchorBasedLinkPredictionSplitStrategy

import SplitGeneratorTestUtils.{getMockMainInputSample, getMockRootedNodeNeighborhoodSample}

class TransductiveNodeAnchorBasedLinkPredictionSplitStrategyTest
    extends AnyFunSuite
    with BeforeAndAfterAll {

  val mainSample: NodeAnchorBasedLinkPredictionSample      = getMockMainInputSample
  val rootedNodeNeighborhoodSample: RootedNodeNeighborhood = getMockRootedNodeNeighborhoodSample

  test("split main sample is valid") {
    val splitStrategy = new TransductiveNodeAnchorBasedLinkPredictionSplitStrategy(
      splitStrategyArgs = Map[String, String](),
      edgeToLinkSplitAssigner = new MockAssigner(
        posEdge = EdgePbWrapper(mainSample.posEdges(0)),
        forcePosEdgeToTrain = true, // forcing positive supervision to TRAIN split
      ),
    )

    val trainSample: NodeAnchorBasedLinkPredictionSample =
      splitStrategy.splitTrainingSample(
        sample = mainSample,
        datasetSplit = DatasetSplits.TRAIN,
      )(0)

    val valSample: NodeAnchorBasedLinkPredictionSample =
      splitStrategy.splitTrainingSample(
        sample = mainSample,
        datasetSplit = DatasetSplits.VAL,
      )(0)

    val testSample: NodeAnchorBasedLinkPredictionSample =
      splitStrategy.splitTrainingSample(
        sample = mainSample,
        datasetSplit = DatasetSplits.TEST,
      )(0)

    // train split has the posEdge
    assert(trainSample.posEdges == mainSample.posEdges)

    // test and val split has no pos edge
    assert(valSample.posEdges.size == 0)
    assert(testSample.posEdges.size == 0)

    // edges in train and val Neighborhoods are the same
    assert(trainSample.neighborhood.get.edges == valSample.neighborhood.get.edges)

    // test split has more neighborhood edges that train/val
    assert(valSample.neighborhood.get.edges.toSet.subsetOf(testSample.neighborhood.get.edges.toSet))
    assert(trainSample.neighborhood.get.edges.size < testSample.neighborhood.get.edges.size)

    // test split has less number of edges than input sample as it only has TRAIN and VAL message passing edges
    assert(
      testSample.neighborhood.get.edges.toSet.subsetOf(mainSample.neighborhood.get.edges.toSet),
    )
    assert(testSample.neighborhood.get.edges.size < mainSample.neighborhood.get.edges.size)

    // supervision nodes are present in neighborhood nodes
    assert(trainSample.neighborhood.get.nodes.exists(_.nodeId == trainSample.posEdges(0).srcNodeId))
    assert(trainSample.neighborhood.get.nodes.exists(_.nodeId == trainSample.posEdges(0).dstNodeId))

    // root node always present in neighborhood nodes
    assert(trainSample.neighborhood.get.nodes.exists(_.nodeId == trainSample.rootNode.get.nodeId))

    // test all nodes have features
    Seq(trainSample, valSample, testSample)
      .flatMap(_.neighborhood.get.nodes)
      .foreach(node => {
        assert(node.featureValues.size > 0)
      })
  }

  test("when train pos edges is empty") {
    val splitStrategy = new TransductiveNodeAnchorBasedLinkPredictionSplitStrategy(
      splitStrategyArgs = Map[String, String](),
      edgeToLinkSplitAssigner = new MockAssigner(
        posEdge = EdgePbWrapper(mainSample.posEdges(0)),
        forcePosEdgeToTrain = false, // in this case the PosEdge goes to VAL
      ),
    )

    // test there is no train sample as Train samples with no posEdges are filtered out
    assert(
      splitStrategy
        .splitTrainingSample(
          sample = mainSample,
          datasetSplit = DatasetSplits.TRAIN,
        )
        .size == 0,
    )

    // val and test sample still exist
    assert(
      splitStrategy
        .splitTrainingSample(
          sample = mainSample,
          datasetSplit = DatasetSplits.VAL,
        )
        .size == 1,
    )
    assert(
      splitStrategy
        .splitTrainingSample(
          sample = mainSample,
          datasetSplit = DatasetSplits.TEST,
        )
        .size == 1,
    )
  }

  test("Rooted Neighborhood split is valid") {
    val splitStrategy = new TransductiveNodeAnchorBasedLinkPredictionSplitStrategy(
      splitStrategyArgs = Map[String, String](),
      edgeToLinkSplitAssigner = new MockAssigner(),
    )

    val trainSample: RootedNodeNeighborhood =
      splitStrategy.splitRootedNodeNeighborhoodTrainingSample(
        sample = rootedNodeNeighborhoodSample,
        datasetSplit = DatasetSplits.TRAIN,
      )(0)

    val valSample: RootedNodeNeighborhood =
      splitStrategy.splitRootedNodeNeighborhoodTrainingSample(
        sample = rootedNodeNeighborhoodSample,
        datasetSplit = DatasetSplits.VAL,
      )(0)

    val testSample: RootedNodeNeighborhood =
      splitStrategy.splitRootedNodeNeighborhoodTrainingSample(
        sample = rootedNodeNeighborhoodSample,
        datasetSplit = DatasetSplits.TEST,
      )(0)

    // edges in train and val Neighborhoods are the same
    assert(trainSample.neighborhood.get.edges == valSample.neighborhood.get.edges)

    // test split has more neighborhood edges that train/val
    assert(valSample.neighborhood.get.edges.toSet.subsetOf(testSample.neighborhood.get.edges.toSet))
    assert(trainSample.neighborhood.get.edges.size < testSample.neighborhood.get.edges.size)

    // root node always present in neighborhood nodes
    assert(
      trainSample.neighborhood.get.nodes
        .exists(_.nodeId == rootedNodeNeighborhoodSample.rootNode.get.nodeId),
    )
    assert(
      valSample.neighborhood.get.nodes
        .exists(_.nodeId == rootedNodeNeighborhoodSample.rootNode.get.nodeId),
    )
    assert(
      testSample.neighborhood.get.nodes
        .exists(_.nodeId == rootedNodeNeighborhoodSample.rootNode.get.nodeId),
    )
  }

  test("main samples split is valid for disjoint training") {
    val splitStrategy =
      new TransductiveNodeAnchorBasedLinkPredictionSplitStrategy(
        splitStrategyArgs = Map[String, String]("is_disjoint_mode" -> "true"),
        edgeToLinkSplitAssigner = new DisjointModeMockAssigner(
          posEdge = EdgePbWrapper(mainSample.posEdges(0)),
          forcePosEdgeToTrainSupervision = false, // will go to TRAIN SUPERVISION
        ),
      )

    // test there is no train sample as Train samples with no posEdges are filtered out
    assert(
      splitStrategy
        .splitTrainingSample(
          sample = mainSample,
          datasetSplit = DatasetSplits.TRAIN,
        )
        .size == 0,
    )

    // val and test sample still exist
    assert(
      splitStrategy
        .splitTrainingSample(
          sample = mainSample,
          datasetSplit = DatasetSplits.VAL,
        )
        .size == 1,
    )
    assert(
      splitStrategy
        .splitTrainingSample(
          sample = mainSample,
          datasetSplit = DatasetSplits.TEST,
        )
        .size == 1,
    )

  }

  test("rooted node neighborhood split is valid for disjoint training") {
    val splitStrategy =
      new TransductiveNodeAnchorBasedLinkPredictionSplitStrategy(
        splitStrategyArgs = Map[String, String]("is_disjoint_mode" -> "true"),
        edgeToLinkSplitAssigner = new DisjointModeMockAssigner,
      )

    val trainSample: RootedNodeNeighborhood =
      splitStrategy.splitRootedNodeNeighborhoodTrainingSample(
        sample = rootedNodeNeighborhoodSample,
        datasetSplit = DatasetSplits.TRAIN,
      )(0)

    val valSample: RootedNodeNeighborhood =
      splitStrategy.splitRootedNodeNeighborhoodTrainingSample(
        sample = rootedNodeNeighborhoodSample,
        datasetSplit = DatasetSplits.VAL,
      )(0)

    val testSample: RootedNodeNeighborhood =
      splitStrategy.splitRootedNodeNeighborhoodTrainingSample(
        sample = rootedNodeNeighborhoodSample,
        datasetSplit = DatasetSplits.TEST,
      )(0)

    // train has lesser message edges than val as some are filtered out because they are supervision edges
    assert(
      trainSample.neighborhood.get.edges.toSet.subsetOf(valSample.neighborhood.get.edges.toSet),
    )
    assert(trainSample.neighborhood.get.edges.size < valSample.neighborhood.get.edges.size)

    // test split has more neighborhood edges that train/val
    assert(valSample.neighborhood.get.edges.toSet.subsetOf(testSample.neighborhood.get.edges.toSet))
    assert(valSample.neighborhood.get.edges.size < testSample.neighborhood.get.edges.size)

    // root node always present in neighborhood nodes
    assert(trainSample.neighborhood.get.nodes.exists(_.nodeId == trainSample.rootNode.get.nodeId))

    // test all nodes have features
    Seq(trainSample, valSample, testSample)
      .flatMap(_.neighborhood.get.nodes)
      .foreach(node => {
        assert(node.featureValues.size > 0)
      })
  }

  test("split of sample with isolated node is valid") {
    val rootNode          = Node(nodeId = 0, condensedNodeType = Some(0), featureValues = Seq(1.0f))
    val neighborhoodNodes = Seq(rootNode)
    // build isolated node samples
    val mainSample = NodeAnchorBasedLinkPredictionSample(
      rootNode = Some(rootNode),
      posEdges = Seq.empty[Edge],
      neighborhood = Some(Graph(nodes = neighborhoodNodes, edges = Seq.empty[Edge])),
    )
    val rootedNodeNeighborhoodSample = RootedNodeNeighborhood(
      rootNode = Some(rootNode),
      neighborhood = Some(Graph(nodes = neighborhoodNodes, edges = Seq.empty[Edge])),
    )

    val splitStrategy = new TransductiveNodeAnchorBasedLinkPredictionSplitStrategy(
      splitStrategyArgs = Map[String, String](),
      edgeToLinkSplitAssigner = new MockAssigner(),
    )

    // test split strategy for every split
    Seq(DatasetSplits.TRAIN, DatasetSplits.VAL, DatasetSplits.TEST)
      .foreach(datasetSplit => {
        // test main sample split does not fail
        splitStrategy.splitTrainingSample(sample = mainSample, datasetSplit = datasetSplit)

        // test rooted node neighborhood split does not fail
        splitStrategy.splitRootedNodeNeighborhoodTrainingSample(
          sample = rootedNodeNeighborhoodSample,
          datasetSplit = datasetSplit,
        )
      })
  }

  class MockAssigner(
    posEdge: EdgePbWrapper = EdgePbWrapper(Edge()),
    forcePosEdgeToTrain: Boolean = true)
      extends Assigner[EdgePbWrapper, LinkSplitType] {
    val graphMetadataPbWrapper: GraphMetadataPbWrapper =
      SplitGeneratorTestUtils.getGraphMetadataWrapper

    def assign(obj: EdgePbWrapper): LinkSplitType = {
      if (forcePosEdgeToTrain && obj == posEdge) {
        LinkSplitType(DatasetSplits.TRAIN, LinkUsageTypes.MESSAGE_AND_SUPERVISION)
      } else {
        // simple assign strategy based on just srcNodeId (Easy to reason about)
        obj.srcNodeId % 3 match {
          case 0 =>
            LinkSplitType(DatasetSplits.TRAIN, LinkUsageTypes.MESSAGE_AND_SUPERVISION)
          case 1 =>
            LinkSplitType(DatasetSplits.VAL, LinkUsageTypes.MESSAGE_AND_SUPERVISION)
          case 2 =>
            LinkSplitType(DatasetSplits.TEST, LinkUsageTypes.MESSAGE_AND_SUPERVISION)
        }
      }
    }
  }

  class DisjointModeMockAssigner(
    posEdge: EdgePbWrapper = EdgePbWrapper(Edge()),
    forcePosEdgeToTrainSupervision: Boolean = true)
      extends Assigner[EdgePbWrapper, LinkSplitType] {
    val graphMetadataPbWrapper: GraphMetadataPbWrapper =
      SplitGeneratorTestUtils.getGraphMetadataWrapper

    def assign(obj: EdgePbWrapper): LinkSplitType = {
      if (forcePosEdgeToTrainSupervision && obj == posEdge) {
        LinkSplitType(DatasetSplits.TRAIN, LinkUsageTypes.SUPERVISION)
      } else {
        // simple assign strategy based on just srcNodeId (Easy to reason about)
        obj.srcNodeId % 4 match {
          case 0 =>
            LinkSplitType(DatasetSplits.TRAIN, LinkUsageTypes.SUPERVISION)
          case 1 =>
            LinkSplitType(DatasetSplits.TRAIN, LinkUsageTypes.MESSAGE)
          case 2 =>
            LinkSplitType(DatasetSplits.VAL, LinkUsageTypes.MESSAGE_AND_SUPERVISION)
          case 3 =>
            LinkSplitType(DatasetSplits.TEST, LinkUsageTypes.MESSAGE_AND_SUPERVISION)
        }
      }
    }
  }
}
