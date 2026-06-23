"""Unit tests for the vectorized ABLP label-remap kernel and the collate-impl flag.

These tests exercise the pure-tensor label-remap logic directly (no GLT, no
distributed runtime), so they run in-process without ``mp.spawn``.
"""

import os
from unittest import mock

import torch
from absl.testing import absltest, parameterized

from gigl.distributed.dist_ablp_neighborloader import _loop_set_labels
from gigl.distributed.utils.neighborloader import (
    COLLATE_IMPL_ENV_VAR,
    resolve_collate_impl,
)
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.types.graph import (
    message_passing_to_positive_label,
)


class ResolveCollateImplTest(parameterized.TestCase):
    @parameterized.parameters(
        ("python", "python"),
        ("vectorized", "vectorized"),
        ("cpp", "cpp"),
        ("VECTORIZED", "vectorized"),  # case-insensitive
    )
    def test_valid_values(self, env_value: str, expected: str) -> None:
        with mock.patch.dict(os.environ, {COLLATE_IMPL_ENV_VAR: env_value}):
            self.assertEqual(resolve_collate_impl(), expected)

    def test_unset_defaults_to_python(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_collate_impl(), "python")

    def test_invalid_value_raises(self) -> None:
        with mock.patch.dict(os.environ, {COLLATE_IMPL_ENV_VAR: "rust"}):
            with self.assertRaises(ValueError):
                resolve_collate_impl()


_CPU = torch.device("cpu")
_USER = NodeType("user")
_STORY = NodeType("story")
_USER_TO_STORY = EdgeType(_USER, Relation("to"), _STORY)


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
            f"{edge_type}: anchor keys {set(actual_inner.keys())} != {set(inner.keys())}"
        )
        for anchor, expected_tensor in inner.items():
            got = actual_inner[anchor]
            assert got.dtype == torch.long, f"{edge_type}[{anchor}] dtype {got.dtype}"
            torch.testing.assert_close(got, expected_tensor)


class LoopSetLabelsContractTest(absltest.TestCase):
    def test_homogeneous_with_empty_and_padded_anchors(self) -> None:
        # node holds global ids; index = local id.
        # The supervision node type is _STORY (edge_type[2] of pos_label_et).
        node = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17])
        node_map = {_STORY: node}
        # Anchor 0 -> global 15 (local 5); anchor 1 -> {15,16} (local 5,6);
        # anchor 2 -> fully padded (empty); anchor 3 -> global 99 (absent -> empty).
        pos_label_et = message_passing_to_positive_label(_USER_TO_STORY)
        positives = {
            pos_label_et: torch.tensor(
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
        # torch.nonzero over [N, M] yields a row index per matching column,
        # so a node matching two identical label columns appears twice.
        # The supervision node type is _STORY (edge_type[2] of pos_label_et).
        node = torch.tensor([10, 11, 12, 13, 14, 15])
        node_map = {_STORY: node}
        pos_label_et = message_passing_to_positive_label(_USER_TO_STORY)
        positives = {pos_label_et: torch.tensor([[15, 15]], dtype=torch.long)}
        y_pos, _ = _loop_set_labels(
            node_local_to_global_by_type=node_map,
            positive_labels_by_edge_type=positives,
            negative_labels_by_edge_type={},
            supervision_edge_types=[_USER_TO_STORY],
            to_device=_CPU,
        )
        torch.testing.assert_close(y_pos[_USER_TO_STORY][0], torch.tensor([5, 5]))


if __name__ == "__main__":
    absltest.main()
