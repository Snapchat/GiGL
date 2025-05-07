import unittest

import torch
from parameterized import param, parameterized

from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.types.graph import (
    DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    DEFAULT_HOMOGENEOUS_NODE_TYPE,
    LoadedGraphTensors,
    message_passing_to_negative_label,
    message_passing_to_positive_label,
    select_label_edge_types,
    to_heterogeneous_edge,
    to_heterogeneous_node,
    to_homogeneous,
)


class GraphTypesTyest(unittest.TestCase):
    @parameterized.expand(
        [
            param("none_input", None, None),
            param(
                "custom_node_type",
                {"custom_node_type": "value"},
                {"custom_node_type": "value"},
            ),
            param(
                "default_node_type", "value", {DEFAULT_HOMOGENEOUS_NODE_TYPE: "value"}
            ),
        ]
    )
    def test_to_hetergeneous_node(self, _, input_value, expected_output):
        self.assertEqual(to_heterogeneous_node(input_value), expected_output)

    @parameterized.expand(
        [
            param("none_input", None, None),
            param(
                "custom_edge_type",
                {EdgeType(NodeType("src"), Relation("rel"), NodeType("dst")): "value"},
                {EdgeType(NodeType("src"), Relation("rel"), NodeType("dst")): "value"},
            ),
            param(
                "default_edge_type", "value", {DEFAULT_HOMOGENEOUS_EDGE_TYPE: "value"}
            ),
        ]
    )
    def test_to_hetergeneous_edge(self, _, input_value, expected_output):
        self.assertEqual(to_heterogeneous_edge(input_value), expected_output)

    @parameterized.expand(
        [
            param("none_input", None, None),
            param(
                "single_value_input",
                {EdgeType(NodeType("src"), Relation("rel"), NodeType("dst")): "value"},
                "value",
            ),
            param("direct_value_input", "value", "value"),
        ]
    )
    def test_from_heterogeneous(self, _, input_value, expected_output):
        self.assertEqual(to_homogeneous(input_value), expected_output)

    @parameterized.expand(
        [
            param(
                "multiple_keys_input",
                {NodeType("src"): "src_value", NodeType("dst"): "dst_value"},
            ),
            param(
                "empty_dict_input",
                {},
            ),
        ]
    )
    def test_from_heterogeneous_invalid(self, _, input_value):
        with self.assertRaises(ValueError):
            to_homogeneous(input_value)

    @parameterized.expand(
        [
            param(
                "valid_inputs",
                node_ids=torch.tensor([0, 1, 2]),
                node_features=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                edge_index=torch.tensor([[0, 1], [1, 2]]),
                edge_features=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
                positive_label=torch.tensor([[0, 2]]),
                negative_label=torch.tensor([[1, 0]]),
                expected_edge_index={
                    DEFAULT_HOMOGENEOUS_EDGE_TYPE: torch.tensor([[0, 1], [1, 2]]),
                    message_passing_to_positive_label(
                        DEFAULT_HOMOGENEOUS_EDGE_TYPE
                    ): torch.tensor([[0, 2]]),
                    message_passing_to_negative_label(
                        DEFAULT_HOMOGENEOUS_EDGE_TYPE
                    ): torch.tensor([[1, 0]]),
                },
            ),
        ]
    )
    def test_treat_labels_as_edges_success(
        self,
        name,
        node_ids,
        node_features,
        edge_index,
        edge_features,
        positive_label,
        negative_label,
        expected_edge_index,
    ):
        graph_tensors = LoadedGraphTensors(
            node_ids=node_ids,
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            positive_label=positive_label,
            negative_label=negative_label,
        )

        graph_tensors.treat_labels_as_edges()
        self.assertIsNone(graph_tensors.positive_label)
        self.assertIsNone(graph_tensors.negative_label)
        assert isinstance(graph_tensors.edge_index, dict)
        self.assertEqual(graph_tensors.edge_index.keys(), expected_edge_index.keys())
        for edge_type, expected_tensor in expected_edge_index.items():
            torch.testing.assert_close(
                graph_tensors.edge_index[edge_type], expected_tensor
            )

    @parameterized.expand(
        [
            param(
                "missing_labels",
                node_ids=torch.tensor([0, 1, 2]),
                node_features=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                edge_index=torch.tensor([[0, 1], [1, 2]]),
                edge_features=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
                positive_label=None,
                negative_label=None,
                raises=ValueError,
            ),
            param(
                "heterogeneous_inputs",
                node_ids={NodeType("type1"): torch.tensor([0, 1])},
                node_features=None,
                edge_index={
                    EdgeType(
                        NodeType("node1"), Relation("relation"), NodeType("node2")
                    ): torch.tensor([[0, 1]])
                },
                edge_features=None,
                positive_label=torch.tensor([[0, 2]]),
                negative_label=torch.tensor([[1, 0]]),
                raises=ValueError,
            ),
        ]
    )
    def test_treat_labels_as_edges_errors(
        self,
        name,
        node_ids,
        node_features,
        edge_index,
        edge_features,
        positive_label,
        negative_label,
        raises,
    ):
        graph_tensors = LoadedGraphTensors(
            node_ids=node_ids,
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            positive_label=positive_label,
            negative_label=negative_label,
        )

        with self.assertRaises(raises):
            graph_tensors.treat_labels_as_edges()

    def test_select_label_edge_types(self):
        message_passing_edge_type = DEFAULT_HOMOGENEOUS_EDGE_TYPE
        edge_types = [
            message_passing_edge_type,
            message_passing_to_positive_label(message_passing_edge_type),
            message_passing_to_negative_label(message_passing_edge_type),
            EdgeType(NodeType("foo"), Relation("bar"), NodeType("baz")),
            EdgeType(
                DEFAULT_HOMOGENEOUS_NODE_TYPE,
                Relation("bar"),
                DEFAULT_HOMOGENEOUS_NODE_TYPE,
            ),
        ]

        self.assertEqual(
            (
                message_passing_to_positive_label(message_passing_edge_type),
                message_passing_to_negative_label(message_passing_edge_type),
            ),
            select_label_edge_types(message_passing_edge_type, edge_types),
        )

    def test_select_label_edge_types_pyg(self):
        message_passing_edge_type = ("node", "to", "node")
        edge_types = [
            message_passing_edge_type,
            message_passing_to_positive_label(message_passing_edge_type),
            message_passing_to_negative_label(message_passing_edge_type),
            ("other", "to", "node"),
            ("other", "to", "other"),
        ]

        self.assertEqual(
            (
                message_passing_to_positive_label(message_passing_edge_type),
                message_passing_to_negative_label(message_passing_edge_type),
            ),
            select_label_edge_types(message_passing_edge_type, edge_types),
        )


if __name__ == "__main__":
    unittest.main()
