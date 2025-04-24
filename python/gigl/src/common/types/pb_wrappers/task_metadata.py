from dataclasses import dataclass
from typing import List, Set, cast

from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.src.common.types.pb_wrappers.types import TaskMetadataPb
from gigl.src.common.types.task_metadata import TaskMetadataType
from snapchat.research.gbml import gbml_config_pb2


@dataclass
class TaskMetadataPbWrapper:
    task_metadata_pb: gbml_config_pb2.GbmlConfig.TaskMetadata

    @property
    def task_metadata(self) -> TaskMetadataPb:
        """
        Returns the relevant TaskMetadataPb instance
        (e.g. an instance of TaskMetadata.NodeClassificationTaskMetadata).

        Returns:
            TaskMetadataPb: The TaskMetadata proto
        """
        field = cast(str, self.task_metadata_pb.WhichOneof("task_metadata"))
        output = getattr(self.task_metadata_pb, field)
        return output

    @property
    def task_metadata_type(self) -> TaskMetadataType:
        """
        Returns the type of the Task Metadata instance

        Returns:
            TaskMetadataType: The type of the Task Metadata instance
        """
        return TaskMetadataType(self.task_metadata.__class__.__name__)

    def get_supervision_node_types(self) -> List[NodeType]:
        """
        Returns supervision node types for the Node Based Task

        Returns:
            List[NodeType]: The node types in supervision node types
        """
        if self.task_metadata_type == TaskMetadataType.NODE_BASED_TASK:
            supervision_node_types_pb = (
                self.task_metadata_pb.node_based_task_metadata.supervision_node_types
            )
        else:
            raise ValueError(
                f"Can only get supervision node types for Node Based Task, got {self.task_metadata_type}"
            )

        supervision_node_types: List[NodeType] = []
        for supervision_node_type_pb in supervision_node_types_pb:
            supervision_node_types.append(NodeType(supervision_node_type_pb))
        return supervision_node_types

    def get_supervision_edge_types(self) -> List[EdgeType]:
        """
        Returns supervision edge types for the Node Anchor Based Link Prediction and Link Based Task

        Returns:
            List[EdgeType]: The edge types in supervision edge types
        """
        if (
            self.task_metadata_type
            == TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK
        ):
            supervision_edge_types_pb = (
                self.task_metadata_pb.node_anchor_based_link_prediction_task_metadata.supervision_edge_types
            )
        elif self.task_metadata_type == TaskMetadataType.LINK_BASED_TASK:
            supervision_edge_types_pb = (
                self.task_metadata_pb.link_based_task_metadata.supervision_edge_types
            )
        else:
            raise ValueError(
                f"Can only get supervision edge types for Node Anchor Based Link Prediction and Link Based Tasks, got {self.task_metadata_type}"
            )

        supervision_edge_types: List[EdgeType] = []
        for supervision_edge_type_pb in supervision_edge_types_pb:
            supervision_edge_types.append(
                EdgeType(
                    src_node_type=NodeType(supervision_edge_type_pb.src_node_type),
                    relation=Relation(supervision_edge_type_pb.relation),
                    dst_node_type=NodeType(supervision_edge_type_pb.dst_node_type),
                )
            )
        return supervision_edge_types

    def get_supervision_edge_node_types(
        self,
        should_include_src_nodes: bool,
        should_include_dst_nodes: bool,
    ) -> Set[NodeType]:
        """
        Returns node types in supervision edge types for the Node Anchor Based Link Prediction and Link Based Task

        Args:
            should_include_src_nodes (bool): Whether to include source node types in the output
            should_include_dst_nodes (bool): Whether to include destination node types in the output
        Returns:
            Set[str]: The node types in supervision edge types
        """
        if not should_include_src_nodes and not should_include_dst_nodes:
            raise ValueError(
                "Expected one of should_include_src_nodes or should_include_dst_nodes to be set to true."
            )

        supervision_edge_node_types: Set[NodeType] = set()
        for supervision_edge_type in self.get_supervision_edge_types():
            if should_include_src_nodes:
                supervision_edge_node_types.add(
                    NodeType(supervision_edge_type.src_node_type)
                )
            if should_include_dst_nodes:
                supervision_edge_node_types.add(
                    NodeType(supervision_edge_type.dst_node_type)
                )
        return supervision_edge_node_types

    def get_task_root_node_types(self) -> Set[NodeType]:
        """
        Returns all root node types for the task.
        For tasks with supervision edges, this returns all node types contained within supervision edge types.
        For tasks with supervision nodes, this returns all supervision node types.

        Returns:
            Set[NodeType]: The root node types in supervision edge types
        """
        if (
            self.task_metadata_type
            == TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK
            or self.task_metadata_type == TaskMetadataType.LINK_BASED_TASK
        ):
            return self.get_supervision_edge_node_types(
                should_include_src_nodes=True, should_include_dst_nodes=True
            )
        elif self.task_metadata_type == TaskMetadataType.NODE_BASED_TASK:
            return set(self.get_supervision_node_types())
        else:
            raise ValueError(
                "Invalid task metadata type found when getting task root node types"
            )
