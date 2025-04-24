package splitgenerator.test

import common.types.pb_wrappers.GraphPbWrappers.EdgePbWrapper
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers._
import snapchat.research.gbml.training_samples_schema.NodeAnchorBasedLinkPredictionSample
import snapchat.research.gbml.training_samples_schema.RootedNodeNeighborhood
import splitgenerator.lib.Types.DatasetSplits
import splitgenerator.lib.Types.LinkSplitType
import splitgenerator.lib.Types.LinkUsageTypes
import splitgenerator.lib.assigners.UserDefinedLabelsEdgeToLinkSplitHashingAssigner
import splitgenerator.lib.split_strategies.UserDefinedLabelsNodeAnchorBasedLinkPredictionSplitStrategy

import SplitGeneratorTestUtils.{getMockMainInputSample, getMockRootedNodeNeighborhoodSample}

class UserDefinedLabelsNodeAnchorBasedLinkPredictionSplitStrategyTest
    extends AnyFunSuite
    with BeforeAndAfterAll {

  val mainSample: NodeAnchorBasedLinkPredictionSample      = getMockMainInputSample
  val rootedNodeNeighborhoodSample: RootedNodeNeighborhood = getMockRootedNodeNeighborhoodSample
  val splitStrategy =
    new UserDefinedLabelsNodeAnchorBasedLinkPredictionSplitStrategy(
      splitStrategyArgs = Map[String, String](),
      edgeToLinkSplitAssigner = new UserDefinedLabelsEdgeToLinkSplitHashingAssigner(
        assignerArgs = Map[String, String](
          "train_split" -> "0.8", // we use these ratios to enforce pos edge in train split
          "val_split"   -> "0.1",
          "test_split"  -> "0.1",
        ),
        graphMetadataPbWrapper = SplitGeneratorTestUtils.getGraphMetadataWrapper(),
      ),
    )

  test("main samples split is valid for user defined labels") {
    val neighborhoodEdges = mainSample.neighborhood.get.edges
    // test message passing edges are the same across all splits
    val trainSample: NodeAnchorBasedLinkPredictionSample =
      splitStrategy.splitTrainingSample(
        sample = mainSample,
        datasetSplit = DatasetSplits.TRAIN,
      )(0)
    val trainEdges = trainSample.neighborhood.get.edges

    val valSample: NodeAnchorBasedLinkPredictionSample =
      splitStrategy.splitTrainingSample(
        sample = mainSample,
        datasetSplit = DatasetSplits.VAL,
      )(0)
    val valEdges = valSample.neighborhood.get.edges

    val testSample: NodeAnchorBasedLinkPredictionSample =
      splitStrategy.splitTrainingSample(
        sample = mainSample,
        datasetSplit = DatasetSplits.TEST,
      )(0)
    val testEdges = testSample.neighborhood.get.edges

    neighborhoodEdges should contain theSameElementsAs trainEdges
    trainEdges should contain theSameElementsAs valEdges
    valEdges should contain theSameElementsAs testEdges

  }

  test("rooted node samples split is valid for user defined labels with disjoint training") {

    val neighborhoodEdges = rootedNodeNeighborhoodSample.neighborhood.get.edges
    // test message passing edges are the same across all splits
    val trainSample: RootedNodeNeighborhood =
      splitStrategy.splitRootedNodeNeighborhoodTrainingSample(
        sample = rootedNodeNeighborhoodSample,
        datasetSplit = DatasetSplits.TRAIN,
      )(0)
    val trainEdges = trainSample.neighborhood.get.edges

    val valSample: RootedNodeNeighborhood =
      splitStrategy.splitRootedNodeNeighborhoodTrainingSample(
        sample = rootedNodeNeighborhoodSample,
        datasetSplit = DatasetSplits.VAL,
      )(0)
    val valEdges = valSample.neighborhood.get.edges

    val testSample: RootedNodeNeighborhood =
      splitStrategy.splitRootedNodeNeighborhoodTrainingSample(
        sample = rootedNodeNeighborhoodSample,
        datasetSplit = DatasetSplits.TEST,
      )(0)
    val testEdges = testSample.neighborhood.get.edges

    neighborhoodEdges should contain theSameElementsAs trainEdges
    trainEdges should contain theSameElementsAs valEdges
    valEdges should contain theSameElementsAs testEdges

  }

  // TODO: since this test for user def assigner is very small it does not worth adding a new file for it
  // If in future we decide to add more complex logic, we can move this test to a new file
  test("user define label assigner has only SUPERVISION edge type") {
    val userDefinedLabelsAssigner = new UserDefinedLabelsEdgeToLinkSplitHashingAssigner(
      assignerArgs = Map[String, String](
        "train_split" -> "0.5",
        "val_split"   -> "0.25",
        "test_split"  -> "0.25",
      ),
      graphMetadataPbWrapper = SplitGeneratorTestUtils.getGraphMetadataWrapper(),
    )
    var edgePbWrappers: Seq[EdgePbWrapper] = SplitGeneratorTestUtils.getMockEdgePbWrappers()
    val assignedBuckets: Seq[LinkSplitType] =
      edgePbWrappers.map(userDefinedLabelsAssigner.assign(_))
    val allAssignedBuckets: Set[LinkSplitType] = assignedBuckets.toSet
    val validBuckets = Set(
      LinkSplitType(
        DatasetSplits.TRAIN,
        LinkUsageTypes.SUPERVISION,
      ),
      LinkSplitType(
        DatasetSplits.VAL,
        LinkUsageTypes.SUPERVISION,
      ),
      LinkSplitType(
        DatasetSplits.TEST,
        LinkUsageTypes.SUPERVISION,
      ),
    )

    allAssignedBuckets should contain theSameElementsAs validBuckets
  }
}
