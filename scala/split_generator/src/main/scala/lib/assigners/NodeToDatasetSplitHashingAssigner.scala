package splitgenerator.lib.assigners

import common.types.pb_wrappers.GraphMetadataPbWrapper
import common.types.pb_wrappers.GraphPbWrappers.NodePbWrapper
import snapchat.research.gbml.graph_schema.Node
import splitgenerator.lib.Types.DatasetSplits
import splitgenerator.lib.Types.DatasetSplits.DatasetSplit

class NodeToDatasetSplitHashingAssigner(
  assignerArgs: Map[String, String],
  val graphMetadataPbWrapper: GraphMetadataPbWrapper)
    extends AbstractAssigners.HashingAssigner[Node, DatasetSplit] {

  private var __trainRatio: Float =
    assignerArgs.getOrElse("train_split", "0.8").toFloat // Ratio dedicated for train split
  private val __valRatio: Float =
    assignerArgs.getOrElse("val_split", "0.1").toFloat // Ratio dedicated for val split
  private val __testRatio: Float =
    assignerArgs.getOrElse("test_split", "0.1").toFloat // Ratio dedicated for test split

  val bucketWeights: Map[DatasetSplit, Float] = {
    Map[DatasetSplit, Float](
      DatasetSplits.TRAIN -> __trainRatio,
      DatasetSplits.VAL   -> __valRatio,
      DatasetSplits.TEST  -> __testRatio,
    )
  }

  def coder(obj: Node): Array[Byte] = {
    NodePbWrapper(node = obj).uniqueId
  }
}
