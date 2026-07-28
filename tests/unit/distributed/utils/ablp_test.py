"""Tests for vectorized ABLP label remapping."""

import unittest

import torch
from parameterized import param, parameterized
from torch_geometric.typing import EdgeType as PyGEdgeType

from gigl.distributed.utils.ablp import (
    label_edge_index_to_dict,
    remap_labels_to_local_edge_indices,
)
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.types.graph import (
    DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    DEFAULT_HOMOGENEOUS_NODE_TYPE,
    message_passing_to_negative_label,
    message_passing_to_positive_label,
)
from tests.test_assets.test_case import TestCase

_USER = NodeType("user")
_STORY = NodeType("story")
_USER_TO_STORY = EdgeType(_USER, Relation("to"), _STORY)
_A = NodeType("a")
_B = NodeType("b")
_C = NodeType("c")
_A_TO_B = EdgeType(_A, Relation("to"), _B)
_A_TO_C = EdgeType(_A, Relation("to"), _C)


def _positive_label_edge_type(edge_type: EdgeType) -> EdgeType:
    return message_passing_to_positive_label(edge_type)


def _negative_label_edge_type(edge_type: EdgeType) -> EdgeType:
    return message_passing_to_negative_label(edge_type)


def _assert_label_sets_equal(
    actual: dict[PyGEdgeType, torch.Tensor],
    expected: dict[PyGEdgeType, dict[int, list[int]]],
) -> None:
    assert set(actual.keys()) == set(expected.keys())
    for edge_type, expected_by_anchor in expected.items():
        actual_by_anchor = label_edge_index_to_dict(
            label_edge_index=actual[edge_type],
            num_anchors=len(expected_by_anchor),
        )
        assert set(actual_by_anchor.keys()) == set(expected_by_anchor.keys())
        for anchor, expected_labels in expected_by_anchor.items():
            actual_labels = actual_by_anchor[anchor]
            assert actual_labels.dtype == torch.long
            assert sorted(actual_labels.tolist()) == sorted(expected_labels)


class LabelEdgeIndexToDictTest(TestCase):
    def test_preserves_empty_and_multi_label_anchors(self) -> None:
        label_edge_index = torch.tensor([[0, 2, 2], [5, 7, 8]])

        labels_by_anchor = label_edge_index_to_dict(
            label_edge_index=label_edge_index,
            num_anchors=3,
        )

        self.assertEqual(set(labels_by_anchor.keys()), {0, 1, 2})
        torch.testing.assert_close(
            labels_by_anchor[0], torch.tensor([5], dtype=torch.long)
        )
        torch.testing.assert_close(
            labels_by_anchor[1], torch.empty(0, dtype=torch.long)
        )
        torch.testing.assert_close(
            labels_by_anchor[2], torch.tensor([7, 8], dtype=torch.long)
        )

    def test_all_anchors_empty(self) -> None:
        labels_by_anchor = label_edge_index_to_dict(
            label_edge_index=torch.empty((2, 0), dtype=torch.long),
            num_anchors=2,
        )

        self.assertEqual(set(labels_by_anchor.keys()), {0, 1})
        torch.testing.assert_close(
            labels_by_anchor[0], torch.empty(0, dtype=torch.long)
        )
        torch.testing.assert_close(
            labels_by_anchor[1], torch.empty(0, dtype=torch.long)
        )


class RemapLabelsToLocalEdgeIndicesTest(TestCase):
    @parameterized.expand(
        [
            param(
                "sorted_present_empty_and_padded",
                local_id_to_global_id_by_node_type={
                    _STORY: torch.tensor([10, 11, 12, 13, 14, 15, 16, 17])
                },
                positive_labels={
                    _positive_label_edge_type(_USER_TO_STORY): torch.tensor(
                        [[15, -1], [15, 16], [-1, -1], [99, -1]]
                    )
                },
                negative_labels={},
                expected_positive={_USER_TO_STORY: {0: [5], 1: [5, 6], 2: [], 3: []}},
                expected_negative={},
            ),
            param(
                "duplicate_labels_preserve_multiplicity",
                local_id_to_global_id_by_node_type={
                    _STORY: torch.tensor([10, 11, 12, 13, 14, 15])
                },
                positive_labels={
                    _positive_label_edge_type(_USER_TO_STORY): torch.tensor(
                        [[15, 15], [11, 11]]
                    )
                },
                negative_labels={},
                expected_positive={_USER_TO_STORY: {0: [5, 5], 1: [1, 1]}},
                expected_negative={},
            ),
            param(
                "default_homogeneous_keying",
                local_id_to_global_id_by_node_type={
                    DEFAULT_HOMOGENEOUS_NODE_TYPE: torch.tensor([20, 10, 30, 11, 15])
                },
                positive_labels={
                    message_passing_to_positive_label(
                        DEFAULT_HOMOGENEOUS_EDGE_TYPE
                    ): torch.tensor([[30, 10], [-1, -1]])
                },
                negative_labels={},
                expected_positive={DEFAULT_HOMOGENEOUS_EDGE_TYPE: {0: [2, 1], 1: []}},
                expected_negative={},
            ),
            param(
                "positive_and_negative_labels",
                local_id_to_global_id_by_node_type={
                    _STORY: torch.tensor([10, 11, 12, 13, 14, 15, 16, 17])
                },
                positive_labels={
                    _positive_label_edge_type(_USER_TO_STORY): torch.tensor(
                        [[15], [16]]
                    )
                },
                negative_labels={
                    _negative_label_edge_type(_USER_TO_STORY): torch.tensor(
                        [[13, 16], [17, -1]]
                    )
                },
                expected_positive={_USER_TO_STORY: {0: [5], 1: [6]}},
                expected_negative={_USER_TO_STORY: {0: [3, 6], 1: [7]}},
            ),
            param(
                "multiple_supervision_edge_types",
                local_id_to_global_id_by_node_type={
                    _B: torch.tensor([11, 12, 13, 14, 15, 16]),
                    _C: torch.tensor([20, 21, 22, 23, 24, 25]),
                },
                positive_labels={
                    _positive_label_edge_type(_A_TO_B): torch.tensor([[13, 14]]),
                    _positive_label_edge_type(_A_TO_C): torch.tensor([[22, 23]]),
                },
                negative_labels={
                    _negative_label_edge_type(_A_TO_B): torch.tensor([[15, 16]]),
                    _negative_label_edge_type(_A_TO_C): torch.tensor([[24, 25]]),
                },
                expected_positive={
                    _A_TO_B: {0: [2, 3]},
                    _A_TO_C: {0: [2, 3]},
                },
                expected_negative={
                    _A_TO_B: {0: [4, 5]},
                    _A_TO_C: {0: [4, 5]},
                },
            ),
            param(
                "all_anchors_empty",
                local_id_to_global_id_by_node_type={_STORY: torch.tensor([10, 11, 12])},
                positive_labels={
                    _positive_label_edge_type(_USER_TO_STORY): torch.tensor(
                        [[-1, -1], [99, 98]]
                    )
                },
                negative_labels={},
                expected_positive={_USER_TO_STORY: {0: [], 1: []}},
                expected_negative={},
            ),
        ]
    )
    def test_matches_constructed_labels(
        self,
        _,
        local_id_to_global_id_by_node_type: dict[NodeType, torch.Tensor],
        positive_labels: dict[EdgeType, torch.Tensor],
        negative_labels: dict[EdgeType, torch.Tensor],
        expected_positive: dict[PyGEdgeType, dict[int, list[int]]],
        expected_negative: dict[PyGEdgeType, dict[int, list[int]]],
    ) -> None:
        positive_edge_indices, negative_edge_indices = (
            remap_labels_to_local_edge_indices(
                local_id_to_global_id_by_node_type=local_id_to_global_id_by_node_type,
                positive_labels_by_edge_type=positive_labels,
                negative_labels_by_edge_type=negative_labels,
            )
        )

        _assert_label_sets_equal(positive_edge_indices, expected_positive)
        _assert_label_sets_equal(negative_edge_indices, expected_negative)

    def test_pins_exact_edge_index(self) -> None:
        positive_edge_indices, _ = remap_labels_to_local_edge_indices(
            local_id_to_global_id_by_node_type={
                _STORY: torch.tensor([10, 11, 12, 13, 14, 15, 16, 17])
            },
            positive_labels_by_edge_type={
                _positive_label_edge_type(_USER_TO_STORY): torch.tensor(
                    [[15, -1], [15, 16], [-1, -1]]
                )
            },
            negative_labels_by_edge_type={},
        )

        torch.testing.assert_close(
            positive_edge_indices[_USER_TO_STORY],
            torch.tensor([[0, 1, 1], [5, 5, 6]], dtype=torch.long),
        )

    def test_unsorted_node_map_returns_local_indices(self) -> None:
        positive_edge_indices, _ = remap_labels_to_local_edge_indices(
            local_id_to_global_id_by_node_type={_STORY: torch.tensor([15, 10, 16, 11])},
            positive_labels_by_edge_type={
                _positive_label_edge_type(_USER_TO_STORY): torch.tensor([[16, 15]])
            },
            negative_labels_by_edge_type={},
        )

        labels_by_anchor = label_edge_index_to_dict(
            positive_edge_indices[_USER_TO_STORY], num_anchors=1
        )
        self.assertEqual(sorted(labels_by_anchor[0].tolist()), [0, 2])

    def test_zero_anchor_tensor_yields_no_edge_type_key(self) -> None:
        positive_edge_indices, negative_edge_indices = (
            remap_labels_to_local_edge_indices(
                local_id_to_global_id_by_node_type={_STORY: torch.tensor([10, 11, 12])},
                positive_labels_by_edge_type={
                    _positive_label_edge_type(_USER_TO_STORY): torch.empty((0, 0))
                },
                negative_labels_by_edge_type={},
            )
        )

        self.assertEqual(positive_edge_indices, {})
        self.assertEqual(negative_edge_indices, {})

    def test_output_follows_input_device_and_dtype(self) -> None:
        positive_edge_indices, _ = remap_labels_to_local_edge_indices(
            local_id_to_global_id_by_node_type={
                _STORY: torch.tensor([10, 11, 12, 13, 14, 15])
            },
            positive_labels_by_edge_type={
                _positive_label_edge_type(_USER_TO_STORY): torch.tensor([[15], [11]])
            },
            negative_labels_by_edge_type={},
        )

        label_edge_index = positive_edge_indices[_USER_TO_STORY]
        self.assertEqual(label_edge_index.device.type, "cpu")
        self.assertEqual(label_edge_index.dtype, torch.long)

    def test_duplicate_node_map_raises(self) -> None:
        with self.assertRaises(ValueError):
            remap_labels_to_local_edge_indices(
                local_id_to_global_id_by_node_type={_STORY: torch.tensor([10, 10, 11])},
                positive_labels_by_edge_type={
                    _positive_label_edge_type(_USER_TO_STORY): torch.tensor([[10, 11]])
                },
                negative_labels_by_edge_type={},
            )


if __name__ == "__main__":
    unittest.main()
