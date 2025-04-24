package splitgenerator.test

import common.types.GraphTypes.NodeId
import common.types.pb_wrappers.GraphMetadataPbWrapper
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import snapchat.research.gbml.graph_schema.Edge
import snapchat.research.gbml.graph_schema.Graph
import snapchat.research.gbml.graph_schema.Node
import snapchat.research.gbml.training_samples_schema.Label
import snapchat.research.gbml.training_samples_schema.SupervisedNodeClassificationSample
import splitgenerator.lib.Types.DatasetSplits
import splitgenerator.lib.Types.DatasetSplits.DatasetSplit
import splitgenerator.lib.assigners.AbstractAssigners.Assigner
import splitgenerator.lib.split_strategies.InductiveSupervisedNodeClassificationSplitStrategy
import splitgenerator.lib.split_strategies.TransductiveSupervisedNodeClassificationSplitStrategy

class SupervisedNodeClassificationSplitStrategyTest extends AnyFunSuite with BeforeAndAfterAll {

  val supervisedNodeClassificationSample = getMockSupervisedNodeClassificationSample
  val nodeToDatasetSplitAssigner         = new MockNodeToDatasetSplitAssignerByIdModulus

  test("inductive split is valid") {
    val inductiveSupervisedNodeClassificationSplitStrategy =
      new InductiveSupervisedNodeClassificationSplitStrategy(
        splitStrategyArgs = Map[String, String](),
        nodeToDatasetSplitAssigner = nodeToDatasetSplitAssigner,
      )

    val trainSplit: Seq[SupervisedNodeClassificationSample] =
      inductiveSupervisedNodeClassificationSplitStrategy.splitTrainingSample(
        sample = supervisedNodeClassificationSample,
        datasetSplit = DatasetSplits.TRAIN,
      )

    val valSplit: Seq[SupervisedNodeClassificationSample] =
      inductiveSupervisedNodeClassificationSplitStrategy.splitTrainingSample(
        sample = supervisedNodeClassificationSample,
        datasetSplit = DatasetSplits.VAL,
      )

    val testSplit: Seq[SupervisedNodeClassificationSample] =
      inductiveSupervisedNodeClassificationSplitStrategy.splitTrainingSample(
        sample = supervisedNodeClassificationSample,
        datasetSplit = DatasetSplits.TEST,
      )

    val splitSample: Seq[SupervisedNodeClassificationSample] = trainSplit ++ valSplit ++ testSplit

    // test the input sample only goes to one of train/test/val
    assert(splitSample.size == 1)

    // all the nodes in the split should have been assigned to the same split as the root node, by definition of the inductive split strategy
    // this can be verified in this test because the the assigner is deterministic (reassignment will go the same split)
    val datasetSplitForRootNode: DatasetSplit =
      nodeToDatasetSplitAssigner.assign(splitSample(0).rootNode.get)

    splitSample(0).neighborhood.get.nodes.forall(node =>
      nodeToDatasetSplitAssigner.assign(node) == datasetSplitForRootNode,
    )

    // all the edges should have src and dst nodes in the same split as root node
    // to verify these we test the srcnodeid and dstnodeid present in neighborhood nodes
    val neighborhoodNodesSet: Set[NodeId] =
      splitSample(0).neighborhood.get.nodes.map(_.nodeId).toSet
    assert(
      splitSample(0).neighborhood.get.edges.forall(edge =>
        neighborhoodNodesSet.contains(edge.srcNodeId) &&
          neighborhoodNodesSet.contains(edge.dstNodeId),
      ),
    )
  }

  test("transductive split is valid") {
    val transductiveSupervisedNodeClassificationSplitStrategy =
      new TransductiveSupervisedNodeClassificationSplitStrategy(
        splitStrategyArgs = Map[String, String](),
        nodeToDatasetSplitAssigner = nodeToDatasetSplitAssigner,
      )

    val trainSplit: Seq[SupervisedNodeClassificationSample] =
      transductiveSupervisedNodeClassificationSplitStrategy.splitTrainingSample(
        sample = supervisedNodeClassificationSample,
        datasetSplit = DatasetSplits.TRAIN,
      )

    val valSplit: Seq[SupervisedNodeClassificationSample] =
      transductiveSupervisedNodeClassificationSplitStrategy.splitTrainingSample(
        sample = supervisedNodeClassificationSample,
        datasetSplit = DatasetSplits.VAL,
      )

    val testSplit: Seq[SupervisedNodeClassificationSample] =
      transductiveSupervisedNodeClassificationSplitStrategy.splitTrainingSample(
        sample = supervisedNodeClassificationSample,
        datasetSplit = DatasetSplits.TEST,
      )

    val splitSample: Seq[SupervisedNodeClassificationSample] = trainSplit ++ valSplit ++ testSplit
    // test the input sample only goes to one of train/test/val
    assert(splitSample.size == 1)

    // test that the neighborhood is the same for the split sample and the input sample
    assert(
      splitSample(
        0,
      ).neighborhood.get.nodes.toSet == supervisedNodeClassificationSample.neighborhood.get.nodes.toSet,
    )
    assert(
      splitSample(
        0,
      ).neighborhood.get.edges.toSet == supervisedNodeClassificationSample.neighborhood.get.edges.toSet,
    )
  }

  def getMockSupervisedNodeClassificationSample: SupervisedNodeClassificationSample = {
    // Create the Node for root_node
    val rootNode = Node(
      nodeId = 5,
      condensedNodeType = Some(0),
      featureValues = Seq(1.0f),
    )

    // Create the nodes for neighborhood
    val neighborhoodNodes = Seq(
      Node(nodeId = 1, condensedNodeType = Some(0), featureValues = Seq(1.0f)),
      Node(nodeId = 3, condensedNodeType = Some(0), featureValues = Seq(1.0f)),
      Node(nodeId = 0, condensedNodeType = Some(0), featureValues = Seq(1.0f)),
      Node(nodeId = 8, condensedNodeType = Some(0), featureValues = Seq(1.0f)),
      Node(nodeId = 7, condensedNodeType = Some(0), featureValues = Seq(1.0f)),
      Node(nodeId = 5, condensedNodeType = Some(0), featureValues = Seq(1.0f)),
      Node(nodeId = 6, condensedNodeType = Some(0), featureValues = Seq(1.0f)),
    )

    // Create the edges for neighborhood
    val neighborhoodEdges = Seq(
      Edge(srcNodeId = 1, dstNodeId = 5, condensedEdgeType = Some(0)),
      Edge(srcNodeId = 3, dstNodeId = 5, condensedEdgeType = Some(0)),
      Edge(srcNodeId = 0, dstNodeId = 1, condensedEdgeType = Some(0)),
      Edge(srcNodeId = 8, dstNodeId = 1, condensedEdgeType = Some(0)),
      Edge(srcNodeId = 7, dstNodeId = 3, condensedEdgeType = Some(0)),
      Edge(srcNodeId = 0, dstNodeId = 3, condensedEdgeType = Some(0)),
      Edge(srcNodeId = 5, dstNodeId = 1, condensedEdgeType = Some(0)),
      Edge(srcNodeId = 6, dstNodeId = 8, condensedEdgeType = Some(0)),
      Edge(srcNodeId = 1, dstNodeId = 8, condensedEdgeType = Some(0)),
    )

    // Create the SupervisedNodeClassificationSample
    SupervisedNodeClassificationSample(
      rootNode = Some(rootNode),
      neighborhood = Some(Graph(nodes = neighborhoodNodes, edges = neighborhoodEdges)),
      rootNodeLabels = Seq(Label()), // labels not a requirement for split generator
    )
  }

  /**
    * Assigner class that assigns nodes based on nodeid
    */
  class MockNodeToDatasetSplitAssignerByIdModulus extends Assigner[Node, DatasetSplit] {

    val graphMetadataPbWrapper: GraphMetadataPbWrapper =
      SplitGeneratorTestUtils.getGraphMetadataWrapper

    def assign(obj: Node): DatasetSplit = {
      obj.nodeId % 3 match {
        case 0 =>
          DatasetSplits.TRAIN
        case 1 =>
          DatasetSplits.TEST
        case 2 =>
          DatasetSplits.VAL
      }
    }
  }
}
