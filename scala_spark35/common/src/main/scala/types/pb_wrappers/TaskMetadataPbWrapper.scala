package common.types.pb_wrappers

import common.types.GraphTypes.NodeType
import snapchat.research.gbml.gbml_config.GbmlConfig.TaskMetadata

object TaskMetadataType extends Enumeration {
  type TaskMetadataType = Value
  val NODE_BASED_TASK: Value = Value("node_based_task")
  val NODE_ANCHOR_BASED_LINK_PREDICTION_TASK: Value = Value(
    "node_anchor_based_link_prediction_task",
  )
  val LINK_BASED_TASK: Value = Value("link_based_task")
}

import TaskMetadataType._

case class TaskMetadataPbWrapper(taskMetadataPb: TaskMetadata) {
  def taskMetadata: TaskMetadata.TaskMetadata = {
    taskMetadataPb.taskMetadata
  }

  def taskMetadataType: TaskMetadataType = {
    taskMetadataPb.taskMetadata match {
      case TaskMetadata.TaskMetadata.NodeBasedTaskMetadata(metadata) =>
        TaskMetadataType.NODE_BASED_TASK
      case TaskMetadata.TaskMetadata.NodeAnchorBasedLinkPredictionTaskMetadata(metadata) =>
        TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK
      case TaskMetadata.TaskMetadata.LinkBasedTaskMetadata(metadata) =>
        TaskMetadataType.LINK_BASED_TASK
      case _ => throw new Exception("Unknown task metadata type")
    }
  }

  /**
   * This method retrieves the distinct destination node types from the supervision edge types
   * of a NodeAnchorBasedLinkPredictionTaskMetadata. It is used to identify the anchor node types
   * for a given task. The destination node types are considered as we are looking at in-edges.
   *
   * * lazy val to only generate the list of anchor node types when needed,
   * *  since exception will be thrown if accessed when incorrect taskmetadata type is provided
   *
   * @throws Exception if the task metadata is not of type NodeAnchorBasedLinkPredictionTaskMetadata.
   * @return A sequence of distinct destination node types.
   */
  lazy val anchorNodeTypes: Seq[NodeType] = {
    taskMetadataPb.taskMetadata match {
      case metadata: TaskMetadata.TaskMetadata.NodeAnchorBasedLinkPredictionTaskMetadata =>
        taskMetadataPb.getNodeAnchorBasedLinkPredictionTaskMetadata.supervisionEdgeTypes
          .flatMap(edgeType => Seq(edgeType.srcNodeType, edgeType.dstNodeType))
          .distinct
      case _ =>
        throw new Exception(
          f"Only ${TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK} support anchor node types",
        )
    }
  }

  /**
   * This method retrieves the distinct source node types from the supervision edge types
   * of a NodeAnchorBasedLinkPredictionTaskMetadata. It is used to identify the target node types
   * for a given task. The source node types are considered as we are looking at in-edges.
   *
   * * lazy val to only generate the list of anchor node types when needed,
   * *  since exception will be thrown if accessed when incorrect taskmetadata type is provided
   *
   * @throws Exception if the task metadata is not of type NodeAnchorBasedLinkPredictionTaskMetadata.
   * @return A sequence of distinct source node types.
   */
  lazy val targetNodeTypes: Seq[NodeType] = {
    taskMetadataPb.taskMetadata match {
      case metadata: TaskMetadata.TaskMetadata.NodeAnchorBasedLinkPredictionTaskMetadata =>
        taskMetadataPb.getNodeAnchorBasedLinkPredictionTaskMetadata.supervisionEdgeTypes
          .map(_.dstNodeType)
          .distinct
      case _ =>
        throw new Exception(
          f"Only ${TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK} support target node types",
        )
    }
  }
}
