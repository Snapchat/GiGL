package splitgenerator.lib.assigners

import common.types.pb_wrappers.GraphMetadataPbWrapper
import common.types.pb_wrappers.GraphPbWrappers.EdgePbWrapper
import splitgenerator.lib.Types.DatasetSplits
import splitgenerator.lib.Types.LinkSplitType
import splitgenerator.lib.Types.LinkUsageTypes
import splitgenerator.lib.assigners.AbstractAssigners.HashingAssigner

/**
  *  Concrete class used to assign edge pbs to link splits in transductive setting.
  */
class TransductiveEdgeToLinkSplitHashingAssigner(
  assignerArgs: Map[String, String],
  val graphMetadataPbWrapper: GraphMetadataPbWrapper)
    extends HashingAssigner[EdgePbWrapper, LinkSplitType] {

  private var __trainRatio: Option[Float] = Some(
    assignerArgs.getOrElse("train_split", "0.8").toFloat,
  ) // Ratio dedicated for train split
  private val __valRatio =
    assignerArgs.getOrElse("val_split", "0.1").toFloat // Ratio dedicated for val split
  private val __testRatio =
    assignerArgs.getOrElse("test_split", "0.1").toFloat // Ratio dedicated for test split

  private val __disjointTrainRatio =
    assignerArgs
      .getOrElse("disjoint_train_ratio", "0.0")
      .toFloat // Ratio of split between train mesage passing and supervision
  private val __shouldSplitEdgesSymmetrically =
    assignerArgs
      .getOrElse("should_split_edges_symmetrically", "True")
      .toBoolean // Flag to see if edges in both directions between 2 nodes should be hashed to same bucket (a->b and b->a)

  private var __trainMessageRatio: Option[Float] =
    None // We will not use this property is non-disjoint training mode.
  private var __trainSupervisionRatio: Option[Float] =
    None // We will not use this property is non-disjoint training mode.

  if (__disjointTrainRatio > 0) {
    // disjoint training mode.
    __trainSupervisionRatio = Some(__disjointTrainRatio * __trainRatio.get)
    __trainMessageRatio = Some(__trainRatio.get - __trainSupervisionRatio.get)
    __trainRatio = None // We will not use this property in disjoint training mode.
  }

  val bucketWeights: Map[LinkSplitType, Float] = {
    if (__disjointTrainRatio > 0) {
      Map[LinkSplitType, Float](
        LinkSplitType(DatasetSplits.TRAIN, LinkUsageTypes.MESSAGE) -> __trainMessageRatio.get,
        LinkSplitType(
          DatasetSplits.TRAIN,
          LinkUsageTypes.SUPERVISION,
        ) -> __trainSupervisionRatio.get,
        LinkSplitType(DatasetSplits.VAL, LinkUsageTypes.MESSAGE_AND_SUPERVISION)  -> __valRatio,
        LinkSplitType(DatasetSplits.TEST, LinkUsageTypes.MESSAGE_AND_SUPERVISION) -> __testRatio,
      )
    } else {
      Map[LinkSplitType, Float](
        LinkSplitType(
          DatasetSplits.TRAIN,
          LinkUsageTypes.MESSAGE_AND_SUPERVISION,
        ) -> __trainRatio.get,
        LinkSplitType(DatasetSplits.VAL, LinkUsageTypes.MESSAGE_AND_SUPERVISION)  -> __valRatio,
        LinkSplitType(DatasetSplits.TEST, LinkUsageTypes.MESSAGE_AND_SUPERVISION) -> __testRatio,
      )
    }
  }

  def coder(obj: EdgePbWrapper): Array[Byte] = {
    if (__shouldSplitEdgesSymmetrically) {
      val isCanonicallyOrdered = obj.srcNodeId <= obj.dstNodeId
      if (isCanonicallyOrdered) {
        obj.uniqueId
      } else {
        obj.reverseEdgePb.uniqueId
      }
    } else {
      obj.uniqueId
    }
  }
}
