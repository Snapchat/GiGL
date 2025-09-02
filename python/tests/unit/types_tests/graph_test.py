import unittest
from typing import Literal, Union

import torch
from parameterized import param, parameterized

from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.types.graph import (
    DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    DEFAULT_HOMOGENEOUS_NODE_TYPE,
    LoadedGraphTensors,
    is_label_edge_type,
    message_passing_to_negative_supervision_edges,
    message_passing_to_positive_supervision_edges,
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
                "valid_inputs, edge_dir=in",
                node_ids=torch.tensor([0, 1, 2]),
                node_features=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                edge_index=torch.tensor([[0, 1], [1, 2]]),
                edge_features=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
                positive_supervision_edges=torch.tensor([[0], [2]]),
                negative_supervision_edges=torch.tensor([[1], [0]]),
                expected_edge_index={
                    DEFAULT_HOMOGENEOUS_EDGE_TYPE: torch.tensor([[0, 1], [1, 2]]),
                    message_passing_to_positive_supervision_edges(
                        DEFAULT_HOMOGENEOUS_EDGE_TYPE
                    ): torch.tensor([[2], [0]]),
                    message_passing_to_negative_supervision_edges(
                        DEFAULT_HOMOGENEOUS_EDGE_TYPE
                    ): torch.tensor([[0], [1]]),
                },
                edge_dir="in",
            ),
            param(
                "valid_inputs, edge_dir=out",
                node_ids=torch.tensor([0, 1, 2]),
                node_features=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                edge_index=torch.tensor([[0, 1], [1, 2]]),
                edge_features=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
                positive_supervision_edges=torch.tensor([[0], [2]]),
                negative_supervision_edges=torch.tensor([[1], [0]]),
                expected_edge_index={
                    DEFAULT_HOMOGENEOUS_EDGE_TYPE: torch.tensor([[0, 1], [1, 2]]),
                    message_passing_to_positive_supervision_edges(
                        DEFAULT_HOMOGENEOUS_EDGE_TYPE
                    ): torch.tensor([[0], [2]]),
                    message_passing_to_negative_supervision_edges(
                        DEFAULT_HOMOGENEOUS_EDGE_TYPE
                    ): torch.tensor([[1], [0]]),
                },
                edge_dir="out",
            ),
            param(
                "heterogeneous_inputs, positive and negative provided, edge_dir=out",
                node_ids={
                    NodeType("foo"): torch.tensor([0, 1]),
                    NodeType("bar"): torch.tensor([2, 3]),
                },
                node_features={
                    NodeType("foo"): torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                    NodeType("bar"): torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
                },
                edge_index={
                    EdgeType(
                        NodeType("foo"), Relation("to"), NodeType("bar")
                    ): torch.tensor([[0, 1], [2, 3]]),
                    EdgeType(
                        NodeType("bar"), Relation("to"), NodeType("foo")
                    ): torch.tensor([[2, 3], [0, 1]]),
                },
                edge_features={
                    EdgeType(
                        NodeType("foo"), Relation("to"), NodeType("bar")
                    ): torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
                    EdgeType(
                        NodeType("bar"), Relation("to"), NodeType("foo")
                    ): torch.tensor([[0.5, 0.6], [0.7, 0.8]]),
                },
                positive_supervision_edges={
                    EdgeType(
                        NodeType("foo"), Relation("labels"), NodeType("bar")
                    ): torch.tensor([[0], [2]])
                },
                negative_supervision_edges={
                    EdgeType(
                        NodeType("bar"), Relation("labels"), NodeType("foo")
                    ): torch.tensor([[1], [0]])
                },
                expected_edge_index={
                    EdgeType(
                        NodeType("foo"), Relation("to"), NodeType("bar")
                    ): torch.tensor([[0, 1], [2, 3]]),
                    EdgeType(
                        NodeType("bar"), Relation("to"), NodeType("foo")
                    ): torch.tensor([[2, 3], [0, 1]]),
                    message_passing_to_positive_supervision_edges(
                        EdgeType(NodeType("foo"), Relation("labels"), NodeType("bar"))
                    ): torch.tensor([[0], [2]]),
                    message_passing_to_negative_supervision_edges(
                        EdgeType(NodeType("bar"), Relation("labels"), NodeType("foo"))
                    ): torch.tensor([[1], [0]]),
                },
                edge_dir="out",
            ),
            param(
                "heterogeneous_inputs, positive and negative provided, edge_dir=in",
                node_ids={
                    NodeType("foo"): torch.tensor([0, 1]),
                    NodeType("bar"): torch.tensor([2, 3]),
                },
                node_features={
                    NodeType("foo"): torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                    NodeType("bar"): torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
                },
                edge_index={
                    EdgeType(
                        NodeType("foo"), Relation("to"), NodeType("bar")
                    ): torch.tensor([[0, 1], [2, 3]]),
                    EdgeType(
                        NodeType("bar"), Relation("to"), NodeType("foo")
                    ): torch.tensor([[2, 3], [0, 1]]),
                },
                edge_features={
                    EdgeType(
                        NodeType("foo"), Relation("to"), NodeType("bar")
                    ): torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
                    EdgeType(
                        NodeType("bar"), Relation("to"), NodeType("foo")
                    ): torch.tensor([[0.5, 0.6], [0.7, 0.8]]),
                },
                positive_supervision_edges={
                    EdgeType(
                        NodeType("foo"), Relation("labels"), NodeType("bar")
                    ): torch.tensor([[0], [2]])
                },
                negative_supervision_edges={
                    EdgeType(
                        NodeType("bar"), Relation("labels"), NodeType("foo")
                    ): torch.tensor([[1], [0]])
                },
                expected_edge_index={
                    EdgeType(
                        NodeType("foo"), Relation("to"), NodeType("bar")
                    ): torch.tensor([[0, 1], [2, 3]]),
                    EdgeType(
                        NodeType("bar"), Relation("to"), NodeType("foo")
                    ): torch.tensor([[2, 3], [0, 1]]),
                    message_passing_to_positive_supervision_edges(
                        EdgeType(NodeType("bar"), Relation("labels"), NodeType("foo"))
                    ): torch.tensor([[2], [0]]),
                    message_passing_to_negative_supervision_edges(
                        EdgeType(NodeType("foo"), Relation("labels"), NodeType("bar"))
                    ): torch.tensor([[0], [1]]),
                },
                edge_dir="in",
            ),
            param(
                "heterogeneous_inputs, only positive label provided, edge_dir=out",
                node_ids={
                    NodeType("foo"): torch.tensor([0, 1]),
                    NodeType("bar"): torch.tensor([2, 3]),
                },
                node_features={
                    NodeType("foo"): torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                    NodeType("bar"): torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
                },
                edge_index={
                    EdgeType(
                        NodeType("foo"), Relation("to"), NodeType("bar")
                    ): torch.tensor([[0, 1], [2, 3]]),
                    EdgeType(
                        NodeType("bar"), Relation("to"), NodeType("foo")
                    ): torch.tensor([[2, 3], [0, 1]]),
                },
                edge_features={
                    EdgeType(
                        NodeType("foo"), Relation("to"), NodeType("bar")
                    ): torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
                    EdgeType(
                        NodeType("bar"), Relation("to"), NodeType("foo")
                    ): torch.tensor([[0.5, 0.6], [0.7, 0.8]]),
                },
                positive_supervision_edges={
                    EdgeType(
                        NodeType("foo"), Relation("labels"), NodeType("bar")
                    ): torch.tensor([[0], [2]])
                },
                negative_supervision_edges=None,
                expected_edge_index={
                    EdgeType(
                        NodeType("foo"), Relation("to"), NodeType("bar")
                    ): torch.tensor([[0, 1], [2, 3]]),
                    EdgeType(
                        NodeType("bar"), Relation("to"), NodeType("foo")
                    ): torch.tensor([[2, 3], [0, 1]]),
                    message_passing_to_positive_supervision_edges(
                        EdgeType(NodeType("foo"), Relation("labels"), NodeType("bar"))
                    ): torch.tensor([[0], [2]]),
                },
                edge_dir="out",
            ),
        ]
    )
    def test_treat_labels_as_edges_success(
        self,
        _,
        node_ids: Union[torch.Tensor, dict[NodeType, torch.Tensor]],
        node_features: Union[torch.Tensor, dict[NodeType, torch.Tensor]],
        edge_index: Union[torch.Tensor, dict[EdgeType, torch.Tensor]],
        edge_features: Union[torch.Tensor, dict[EdgeType, torch.Tensor]],
        positive_supervision_edges: Union[torch.Tensor, dict[EdgeType, torch.Tensor]],
        negative_supervision_edges: Union[torch.Tensor, dict[EdgeType, torch.Tensor]],
        expected_edge_index: dict[EdgeType, torch.Tensor],
        edge_dir: Literal["in", "out"],
    ):
        graph_tensors = LoadedGraphTensors(
            node_ids=node_ids,
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            positive_supervision_edges=positive_supervision_edges,
            negative_supervision_edges=negative_supervision_edges,
        )
        graph_tensors.treat_labels_as_edges(edge_dir=edge_dir)
        self.assertIsNone(graph_tensors.positive_supervision_edges)
        self.assertIsNone(graph_tensors.negative_supervision_edges)
        assert isinstance(graph_tensors.edge_index, dict)
        self.assertEqual(graph_tensors.edge_index.keys(), expected_edge_index.keys())
        for edge_type, expected_tensor in expected_edge_index.items():
            torch.testing.assert_close(
                graph_tensors.edge_index[edge_type], expected_tensor
            )

    def test_select_label_edge_types(self):
        message_passing_edge_type = DEFAULT_HOMOGENEOUS_EDGE_TYPE
        edge_types = [
            message_passing_edge_type,
            message_passing_to_positive_supervision_edges(message_passing_edge_type),
            message_passing_to_negative_supervision_edges(message_passing_edge_type),
            EdgeType(NodeType("foo"), Relation("bar"), NodeType("baz")),
            EdgeType(
                DEFAULT_HOMOGENEOUS_NODE_TYPE,
                Relation("bar"),
                DEFAULT_HOMOGENEOUS_NODE_TYPE,
            ),
        ]

        self.assertEqual(
            (
                message_passing_to_positive_supervision_edges(
                    message_passing_edge_type
                ),
                message_passing_to_negative_supervision_edges(
                    message_passing_edge_type
                ),
            ),
            select_label_edge_types(message_passing_edge_type, edge_types),
        )

    def test_select_label_edge_types_pyg(self):
        message_passing_edge_type = ("node", "to", "node")
        edge_types = [
            message_passing_edge_type,
            message_passing_to_positive_supervision_edges(message_passing_edge_type),
            message_passing_to_negative_supervision_edges(message_passing_edge_type),
            ("other", "to", "node"),
            ("other", "to", "other"),
        ]

        self.assertEqual(
            (
                message_passing_to_positive_supervision_edges(
                    message_passing_edge_type
                ),
                message_passing_to_negative_supervision_edges(
                    message_passing_edge_type
                ),
            ),
            select_label_edge_types(message_passing_edge_type, edge_types),
        )

    @parameterized.expand(
        [
            param(
                "valid positive label edge type",
                input_edge_type=EdgeType(
                    NodeType("foo"), Relation("to_gigl_positive"), NodeType("bar")
                ),
                is_expected_label=True,
            ),
            param(
                "valid negative label edge type",
                input_edge_type=EdgeType(
                    NodeType("bar"), Relation("to_gigl_negative"), NodeType("foo")
                ),
                is_expected_label=True,
            ),
            param(
                "invalid label edge type",
                input_edge_type=EdgeType(
                    NodeType("foo"), Relation("to"), NodeType("bar")
                ),
                is_expected_label=False,
            ),
        ]
    )
    def test_is_label_edge_type(
        self, _, input_edge_type: EdgeType, is_expected_label: bool
    ):
        if is_expected_label:
            self.assertTrue(is_label_edge_type(edge_type=input_edge_type))
        else:
            self.assertFalse(is_label_edge_type(edge_type=input_edge_type))


if __name__ == "__main__":
    unittest.main()
