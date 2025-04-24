package splitgenerator.test

import common.types.pb_wrappers.GraphMetadataPbWrapper
import common.types.pb_wrappers.GraphPbWrappers.EdgePbWrapper
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import splitgenerator.lib.Types.DatasetSplits
import splitgenerator.lib.Types.LinkSplitType
import splitgenerator.lib.Types.LinkUsageTypes
import splitgenerator.lib.assigners.TransductiveEdgeToLinkSplitHashingAssigner

class TransductiveEdgeToLinkSplitHashingAssignerTest extends AnyFunSuite with BeforeAndAfterAll {
  var assignerArgs: Map[String, String]              = _
  var graphMetadataPbWrapper: GraphMetadataPbWrapper = _
  var edgePbWrappers: Seq[EdgePbWrapper] = SplitGeneratorTestUtils.getMockEdgePbWrappers()

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    assignerArgs = Map[String, String](
      "train_split" -> "0.5",
      "val_split"   -> "0.25",
      "test_split"  -> "0.25",
    )

    graphMetadataPbWrapper = SplitGeneratorTestUtils.getGraphMetadataWrapper()
  }

  test("undirected / non disjoint training assignment") {
    // default values are undirected and non disjoint
    val assigner = new TransductiveEdgeToLinkSplitHashingAssigner(
      assignerArgs = assignerArgs,
      graphMetadataPbWrapper = graphMetadataPbWrapper,
    )

    val assignedBuckets: Seq[LinkSplitType] = edgePbWrappers.map(assigner.assign(_))

    // Test all expected linkSplitTypes present in assigned buckets
    // Here we have 100 edges hence a very high chance of all buckets being used.
    val allAssignedBuckets: Set[LinkSplitType] = assignedBuckets.toSet
    assert(
      allAssignedBuckets.contains(
        LinkSplitType(
          DatasetSplits.TRAIN,
          LinkUsageTypes.MESSAGE_AND_SUPERVISION,
        ),
      ),
    )
    assert(
      allAssignedBuckets.contains(
        LinkSplitType(
          DatasetSplits.VAL,
          LinkUsageTypes.MESSAGE_AND_SUPERVISION,
        ),
      ),
    )
    assert(
      allAssignedBuckets.contains(
        LinkSplitType(
          DatasetSplits.TEST,
          LinkUsageTypes.MESSAGE_AND_SUPERVISION,
        ),
      ),
    )
    assert(allAssignedBuckets.size == 3) // No other linksplittype

    // Test assigning again produces the same output
    val reassignedBuckets: Seq[LinkSplitType] = edgePbWrappers.map(assigner.assign(_))
    assert(assignedBuckets == reassignedBuckets)

    // Test train splits have more edges than val/test (because of 50:25:25 split)
    val trainEdges = assignedBuckets.filter(_.datasetSplit == DatasetSplits.TRAIN)
    val valEdges   = assignedBuckets.filter(_.datasetSplit == DatasetSplits.VAL)
    val testEdges  = assignedBuckets.filter(_.datasetSplit == DatasetSplits.TEST)

    assert(trainEdges.size > valEdges.size)
    assert(trainEdges.size > testEdges.size)

    // Test bi-directional edges is assigned to same bucket (a->b and b->a is assigned together)
    // order of the edges here always a->b, b->a, c->d, d->c, ...
    for (Seq(first, second) <- assignedBuckets.grouped(2)) {
      assert(first == second)
    }
  }

  test("directed / non disjoint training") {
    val assignerArgsForAsymmetricSplit =
      assignerArgs + ("should_split_edges_symmetrically" -> "False")
    val assigner = new TransductiveEdgeToLinkSplitHashingAssigner(
      assignerArgs = assignerArgsForAsymmetricSplit,
      graphMetadataPbWrapper = graphMetadataPbWrapper,
    )

    val assignedBuckets: Seq[LinkSplitType] = edgePbWrappers.map(assigner.assign(_))

    // test there may be a bidirectional edge that is not assigned to the same bucket
    // order of the edges here always a->b, b->a, c->d, d->c, ...
    assert(
      assignedBuckets.grouped(2).exists { case Seq(first, second) =>
        first != second
      },
    )
  }

  test("undirected / disjoint training assignment") {
    val assignerArgsForDisjointTraining = assignerArgs + ("disjoint_train_ratio" -> "0.5")
    val assigner = new TransductiveEdgeToLinkSplitHashingAssigner(
      assignerArgs = assignerArgsForDisjointTraining,
      graphMetadataPbWrapper = graphMetadataPbWrapper,
    )
    val assignedBuckets: Seq[LinkSplitType] = edgePbWrappers.map(assigner.assign(_))

    // Test all expected linkSplitTypes present in assigned buckets
    val allAssignedBuckets: Set[LinkSplitType] = assignedBuckets.toSet
    assert(
      allAssignedBuckets.contains(
        LinkSplitType(
          DatasetSplits.TRAIN,
          LinkUsageTypes.MESSAGE,
        ),
      ),
    )
    assert(
      allAssignedBuckets.contains(
        LinkSplitType(
          DatasetSplits.TRAIN,
          LinkUsageTypes.SUPERVISION,
        ),
      ),
    )
    assert(
      allAssignedBuckets.contains(
        LinkSplitType(
          DatasetSplits.VAL,
          LinkUsageTypes.MESSAGE_AND_SUPERVISION,
        ),
      ),
    )
    assert(
      allAssignedBuckets.contains(
        LinkSplitType(
          DatasetSplits.TEST,
          LinkUsageTypes.MESSAGE_AND_SUPERVISION,
        ),
      ),
    )
    assert(allAssignedBuckets.size == 4) // No other linksplittype
  }
}
