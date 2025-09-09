import unittest

from gigl.src.validation_check.libs.template_config_checks import (
    check_if_task_metadata_valid,
)
from snapchat.research.gbml import gbml_config_pb2, graph_schema_pb2
from tests.test_assets.graph_metadata_constants import (
    DEFAULT_HOMOGENEOUS_GRAPH_METADATA_PB,
    DEFAULT_HOMOGENEOUS_NODE_TYPE_STR,
)


class TaskMetadataIsValidTest(unittest.TestCase):
    """
    Tests for the check_if_task_metadata_valid function.
    Tests edge validation behavior for link prediction tasks.
    """

    def _create_link_prediction_task_config(
        self,
        supervision_edge_types: list[graph_schema_pb2.EdgeType],
        graph_metadata: graph_schema_pb2.GraphMetadata,
    ) -> gbml_config_pb2.GbmlConfig:
        """Helper method to create a node-anchor-based link prediction task configuration."""

        return gbml_config_pb2.GbmlConfig(
            task_metadata=gbml_config_pb2.GbmlConfig.TaskMetadata(
                node_anchor_based_link_prediction_task_metadata=gbml_config_pb2.GbmlConfig.TaskMetadata.NodeAnchorBasedLinkPredictionTaskMetadata(
                    supervision_edge_types=supervision_edge_types
                )
            ),
            graph_metadata=graph_metadata,
        )

    def test_link_prediction_task_edge_with_invalid_node_types_raises_error(self):
        """Test that error is raised when supervision edge has node types not in graph metadata."""
        # Create an edge type with node types that don't exist in graph metadata
        edge_with_invalid_nodes = graph_schema_pb2.EdgeType(
            src_node_type=DEFAULT_HOMOGENEOUS_NODE_TYPE_STR,  # valid node type
            relation="to",
            dst_node_type="nonexistent_dst_node_type",  # invalid destination node type
        )
        config = self._create_link_prediction_task_config(
            supervision_edge_types=[edge_with_invalid_nodes],
            graph_metadata=DEFAULT_HOMOGENEOUS_GRAPH_METADATA_PB,
        )

        with self.assertRaises(AssertionError):
            check_if_task_metadata_valid(config)

    def test_link_prediction_task_edge_not_in_graph_metadata_but_nodes_valid_passes(
        self,
    ):
        """Test that no error is raised when edge type is not in graph metadata but node types are valid."""
        # Create an edge type with valid node types but a relation that doesn't exist in graph metadata
        edge_with_new_relation = graph_schema_pb2.EdgeType(
            src_node_type=DEFAULT_HOMOGENEOUS_NODE_TYPE_STR,  # Valid node type
            relation="completely_new_relation",  # This relation doesn't exist in graph metadata
            dst_node_type=DEFAULT_HOMOGENEOUS_NODE_TYPE_STR,  # Valid node type
        )
        config = self._create_link_prediction_task_config(
            supervision_edge_types=[edge_with_new_relation],
            graph_metadata=DEFAULT_HOMOGENEOUS_GRAPH_METADATA_PB,
        )

        # This should not raise any errors
        check_if_task_metadata_valid(config)


if __name__ == "__main__":
    unittest.main()
