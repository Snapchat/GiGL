import common.types.pb_wrappers.TaskMetadataPbWrapper
import org.scalatest.funsuite.AnyFunSuite
import snapchat.research.gbml.gbml_config.GbmlConfig.TaskMetadata
import snapchat.research.gbml.graph_schema.EdgeType

class TaskMetadataPbWrapperSpec extends AnyFunSuite {
  test("taskMetadataType should return the correct anchor node types") {
    val authorNodeType = "AUTHOR"
    val paperNodeType  = "PAPER"
    val termNodeType   = "TERM"
    val taskMetadataPbWrapper = TaskMetadataPbWrapper(
      TaskMetadata(
        taskMetadata = TaskMetadata.TaskMetadata.NodeAnchorBasedLinkPredictionTaskMetadata(
          TaskMetadata.NodeAnchorBasedLinkPredictionTaskMetadata(
            supervisionEdgeTypes = Seq(
              EdgeType(srcNodeType = paperNodeType, dstNodeType = authorNodeType),
              EdgeType(srcNodeType = authorNodeType, dstNodeType = termNodeType),
              EdgeType(srcNodeType = termNodeType, dstNodeType = authorNodeType),
            ),
          ),
        ),
      ),
    )
    assert(
      taskMetadataPbWrapper.anchorNodeTypes.toSeq.sorted
        .sameElements(Seq(authorNodeType, termNodeType, paperNodeType).sorted),
    )
  }

  test("taskMetadata should error task type does not support anchor node types") {
    assertThrows[Exception] {
      val taskMetadataPbWrapper = TaskMetadataPbWrapper(
        TaskMetadata(
          taskMetadata = TaskMetadata.TaskMetadata.NodeBasedTaskMetadata(
            TaskMetadata.NodeBasedTaskMetadata(),
          ),
        ),
      )
      taskMetadataPbWrapper.anchorNodeTypes
    }
  }
}
