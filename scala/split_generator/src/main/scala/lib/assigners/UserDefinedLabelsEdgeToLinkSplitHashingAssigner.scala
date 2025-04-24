package splitgenerator.lib.assigners

import common.types.pb_wrappers.GraphMetadataPbWrapper
import common.types.pb_wrappers.GraphPbWrappers.EdgePbWrapper
import splitgenerator.lib.Types.DatasetSplits
import splitgenerator.lib.Types.LinkSplitType
import splitgenerator.lib.Types.LinkUsageTypes
import splitgenerator.lib.assigners.AbstractAssigners.HashingAssigner

/**
  *  Concrete class used to assign edge pbs to link splits in User Defined Label setting.
  *  This class is used for assigning user-provided pos_edge and hard_neg edges to train/val/test splits
  *  such that we will always have link split types as TRAIN_SUPERVISION, VAL_SUPERVISION, TEST_SUPERVISION.
  */
class UserDefinedLabelsEdgeToLinkSplitHashingAssigner(
  assignerArgs: Map[String, String],
  val graphMetadataPbWrapper: GraphMetadataPbWrapper)
    extends HashingAssigner[EdgePbWrapper, LinkSplitType] {

  private var __trainRatio =
    assignerArgs.getOrElse("train_split", "0.8").toFloat // Ratio dedicated for train split
  private val __valRatio =
    assignerArgs.getOrElse("val_split", "0.1").toFloat // Ratio dedicated for val split
  private val __testRatio =
    assignerArgs.getOrElse("test_split", "0.1").toFloat // Ratio dedicated for test split

  private val __shouldSplitEdgesSymmetrically =
    assignerArgs
      .getOrElse("should_split_edges_symmetrically", "False")
      .toBoolean // Flag to see if edges in both directions between 2 nodes should be hashed to same bucket (a->b and b->a)

  val bucketWeights: Map[LinkSplitType, Float] = {

    Map[LinkSplitType, Float](
      LinkSplitType(
        DatasetSplits.TRAIN,
        LinkUsageTypes.SUPERVISION,
      )                                                             -> __trainRatio,
      LinkSplitType(DatasetSplits.VAL, LinkUsageTypes.SUPERVISION)  -> __valRatio,
      LinkSplitType(DatasetSplits.TEST, LinkUsageTypes.SUPERVISION) -> __testRatio,
    )
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
