"""Unit tests for the ABLP label-remap loop oracle and vectorized kernel.

These exercise the pure-tensor label-remap logic directly (no GLT, no
distributed runtime), so they run in-process without ``mp.spawn``.
"""

import unittest

import torch
from parameterized import param, parameterized

from gigl.distributed.dist_ablp_neighborloader import (
    _loop_set_labels,
    vectorized_set_labels,
)
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.types.graph import (
    DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    DEFAULT_HOMOGENEOUS_NODE_TYPE,
    message_passing_to_negative_label,
    message_passing_to_positive_label,
)
from tests.test_assets.test_case import TestCase

_CPU = torch.device("cpu")
_USER = NodeType("user")
_STORY = NodeType("story")
_USER_TO_STORY = EdgeType(_USER, Relation("to"), _STORY)

_A = NodeType("a")
_B = NodeType("b")
_C = NodeType("c")
_A_TO_B = EdgeType(_A, Relation("to"), _B)
_A_TO_C = EdgeType(_A, Relation("to"), _C)


def _pos(edge_type: EdgeType) -> EdgeType:
    return message_passing_to_positive_label(edge_type)


def _neg(edge_type: EdgeType) -> EdgeType:
    return message_passing_to_negative_label(edge_type)


def _assert_label_dicts_equal(
    actual: dict[EdgeType, dict[int, torch.Tensor]],
    expected: dict[EdgeType, dict[int, torch.Tensor]],
) -> None:
    assert set(actual.keys()) == set(expected.keys()), (
        f"{set(actual.keys())} != {set(expected.keys())}"
    )
    for edge_type, inner in expected.items():
        actual_inner = actual[edge_type]
        assert set(actual_inner.keys()) == set(inner.keys()), (
            f"{edge_type}: {set(actual_inner.keys())} != {set(inner.keys())}"
        )
        for anchor, expected_tensor in inner.items():
            got = actual_inner[anchor]
            assert got.dtype == torch.long, f"{edge_type}[{anchor}] dtype {got.dtype}"
            torch.testing.assert_close(got, expected_tensor)


class LoopSetLabelsContractTest(TestCase):
    def test_homogeneous_with_empty_and_padded_anchors(self) -> None:
        # node holds global ids; index = local id. Supervision node type is
        # _STORY (edge_type[2] of the positive-label edge type).
        node = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17])
        node_map = {_STORY: node}
        # anchor 0 -> global 15 (local 5); anchor 1 -> {15,16} (local 5,6);
        # anchor 2 -> fully padded (empty); anchor 3 -> global 99 (absent -> empty).
        positives = {
            _pos(_USER_TO_STORY): torch.tensor(
                [[15, -1], [15, 16], [-1, -1], [99, -1]], dtype=torch.long
            )
        }
        y_pos, y_neg = _loop_set_labels(
            node_local_to_global_by_type=node_map,
            positive_labels_by_edge_type=positives,
            negative_labels_by_edge_type={},
            supervision_edge_types=[_USER_TO_STORY],
            to_device=_CPU,
        )
        expected = {
            _USER_TO_STORY: {
                0: torch.tensor([5]),
                1: torch.tensor([5, 6]),
                2: torch.tensor([], dtype=torch.long),
                3: torch.tensor([], dtype=torch.long),
            }
        }
        _assert_label_dicts_equal(y_pos, expected)
        self.assertEqual(y_neg, {})

    def test_duplicate_label_columns_preserve_multiplicity(self) -> None:
        # torch.nonzero over [N, M] yields one row index per matching column,
        # so a node matching two identical label columns appears twice.
        node = torch.tensor([10, 11, 12, 13, 14, 15])
        node_map = {_STORY: node}
        positives = {_pos(_USER_TO_STORY): torch.tensor([[15, 15]], dtype=torch.long)}
        y_pos, _ = _loop_set_labels(
            node_local_to_global_by_type=node_map,
            positive_labels_by_edge_type=positives,
            negative_labels_by_edge_type={},
            supervision_edge_types=[_USER_TO_STORY],
            to_device=_CPU,
        )
        torch.testing.assert_close(y_pos[_USER_TO_STORY][0], torch.tensor([5, 5]))


if __name__ == "__main__":
    unittest.main()
