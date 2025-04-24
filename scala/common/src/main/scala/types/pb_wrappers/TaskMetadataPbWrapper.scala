package common.types.pb_wrappers

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

case class TaskMetadataPbWrapper(task_metadata_pb: TaskMetadata) {
  def taskMetadata: TaskMetadata.TaskMetadata = {
    task_metadata_pb.taskMetadata
  }

  def taskMetadataType: TaskMetadataType = {
    task_metadata_pb.taskMetadata match {
      case TaskMetadata.TaskMetadata.NodeBasedTaskMetadata(metadata) =>
        TaskMetadataType.NODE_BASED_TASK
      case TaskMetadata.TaskMetadata.NodeAnchorBasedLinkPredictionTaskMetadata(metadata) =>
        TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK
      case TaskMetadata.TaskMetadata.LinkBasedTaskMetadata(metadata) =>
        TaskMetadataType.LINK_BASED_TASK
      case _ => throw new Exception("Unknown task metadata type")
    }
  }
}
