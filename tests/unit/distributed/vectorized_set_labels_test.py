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


class VectorizedSetLabelsEquivalenceTest(TestCase):
    @parameterized.expand(
        [
            param(
                "homogeneous_present_empty_and_padded",
                node_map={_STORY: torch.tensor([10, 11, 12, 13, 14, 15, 16, 17])},
                positives={
                    _pos(_USER_TO_STORY): torch.tensor(
                        [[15, -1], [15, 16], [-1, -1], [99, -1]], dtype=torch.long
                    )
                },
                negatives={},
                supervision_edge_types=[_USER_TO_STORY],
            ),
            param(
                "homogeneous_duplicate_labels",
                node_map={_STORY: torch.tensor([10, 11, 12, 13, 14, 15])},
                positives={
                    _pos(_USER_TO_STORY): torch.tensor(
                        [[15, 15], [11, 11]], dtype=torch.long
                    )
                },
                negatives={},
                supervision_edge_types=[_USER_TO_STORY],
            ),
            # UNSORTED node map + reversed label columns. With a sorted map
            # `sort_perm` is the identity and a broken port that emits sorted (or
            # global) order would still pass; here `torch.sort` permutes nontrivially
            # (15->local0, 10->local1, 16->local2, 11->local3), and the label row is
            # given high-id-first ([16, 15]). The loop oracle emits ascending LOCAL
            # index regardless of column order, i.e. local 0 (g15) before local 2
            # (g16) -> [0, 2]; a kernel that forgot to map through `sort_perm`, or
            # that preserved column order, would diverge.
            param(
                "unsorted_node_map_reversed_columns",
                node_map={_STORY: torch.tensor([15, 10, 16, 11])},
                positives={
                    _pos(_USER_TO_STORY): torch.tensor([[16, 15]], dtype=torch.long)
                },
                negatives={},
                supervision_edge_types=[_USER_TO_STORY],
            ),
            # Real homogeneous keying: the loader keys the node map by
            # DEFAULT_HOMOGENEOUS_NODE_TYPE and uses DEFAULT_HOMOGENEOUS_EDGE_TYPE
            # for the supervision edge type (see _set_labels). Exercise that exact
            # keying at the kernel level, not just a custom NodeType.
            param(
                "default_homogeneous_keying",
                node_map={
                    DEFAULT_HOMOGENEOUS_NODE_TYPE: torch.tensor([20, 10, 30, 11, 15])
                },
                positives={
                    message_passing_to_positive_label(
                        DEFAULT_HOMOGENEOUS_EDGE_TYPE
                    ): torch.tensor([[30, 10], [-1, -1]], dtype=torch.long)
                },
                negatives={},
                supervision_edge_types=[DEFAULT_HOMOGENEOUS_EDGE_TYPE],
            ),
            param(
                "homogeneous_with_negatives",
                node_map={_STORY: torch.tensor([10, 11, 12, 13, 14, 15, 16, 17])},
                positives={
                    _pos(_USER_TO_STORY): torch.tensor([[15], [16]], dtype=torch.long)
                },
                negatives={
                    _neg(_USER_TO_STORY): torch.tensor(
                        [[13, 16], [17, -1]], dtype=torch.long
                    )
                },
                supervision_edge_types=[_USER_TO_STORY],
            ),
            param(
                "heterogeneous_multi_edge_type",
                node_map={
                    _A: torch.tensor([10]),
                    _B: torch.tensor([11, 12, 13, 14, 20, 21]),
                    _C: torch.tensor([20, 21, 22, 23]),
                },
                positives={
                    _pos(_A_TO_B): torch.tensor([[13, 14]], dtype=torch.long),
                    _pos(_A_TO_C): torch.tensor([[22, 23]], dtype=torch.long),
                },
                negatives={},
                supervision_edge_types=[_A_TO_B, _A_TO_C],
            ),
            param(
                "heterogeneous_multi_edge_type_with_negatives",
                node_map={
                    _A: torch.tensor([10]),
                    _B: torch.tensor([11, 12, 13, 14, 15, 16]),
                    _C: torch.tensor([20, 21, 22, 23, 24, 25]),
                },
                positives={
                    _pos(_A_TO_B): torch.tensor([[13, 14]], dtype=torch.long),
                    _pos(_A_TO_C): torch.tensor([[22, 23]], dtype=torch.long),
                },
                negatives={
                    _neg(_A_TO_B): torch.tensor([[15, 16]], dtype=torch.long),
                    _neg(_A_TO_C): torch.tensor([[24, 25]], dtype=torch.long),
                },
                supervision_edge_types=[_A_TO_B, _A_TO_C],
            ),
            param(
                "all_anchors_empty",
                node_map={_STORY: torch.tensor([10, 11, 12])},
                positives={
                    _pos(_USER_TO_STORY): torch.tensor(
                        [[-1, -1], [99, 98]], dtype=torch.long
                    )
                },
                negatives={},
                supervision_edge_types=[_USER_TO_STORY],
            ),
            param(
                "zero_anchors",
                node_map={_STORY: torch.tensor([10, 11, 12])},
                positives={_pos(_USER_TO_STORY): torch.empty((0, 0), dtype=torch.long)},
                negatives={},
                supervision_edge_types=[_USER_TO_STORY],
            ),
        ]
    )
    def test_matches_loop(
        self,
        _,
        node_map: dict[NodeType, torch.Tensor],
        positives: dict[EdgeType, torch.Tensor],
        negatives: dict[EdgeType, torch.Tensor],
        supervision_edge_types: list[EdgeType],
    ) -> None:
        loop_pos, loop_neg = _loop_set_labels(
            node_local_to_global_by_type=node_map,
            positive_labels_by_edge_type=positives,
            negative_labels_by_edge_type=negatives,
            supervision_edge_types=supervision_edge_types,
            to_device=_CPU,
        )
        vec_pos, vec_neg = vectorized_set_labels(
            node_local_to_global_by_type=node_map,
            positive_labels_by_edge_type=positives,
            negative_labels_by_edge_type=negatives,
            supervision_edge_types=supervision_edge_types,
            to_device=_CPU,
        )
        _assert_label_dicts_equal(vec_pos, loop_pos)
        _assert_label_dicts_equal(vec_neg, loop_neg)

    def test_unsorted_node_map_exact_order(self) -> None:
        # Belt-and-suspenders on top of the parameterized case: assert the EXACT
        # tensor (not just equality-to-loop) so the expected ascending-local order
        # is visible and a regression to sorted/column order is unmistakable.
        node_map = {_STORY: torch.tensor([15, 10, 16, 11])}
        positives = {_pos(_USER_TO_STORY): torch.tensor([[16, 15]], dtype=torch.long)}
        vec_pos, _ = vectorized_set_labels(
            node_local_to_global_by_type=node_map,
            positive_labels_by_edge_type=positives,
            negative_labels_by_edge_type={},
            supervision_edge_types=[_USER_TO_STORY],
            to_device=_CPU,
        )
        # g15 is local 0, g16 is local 2 -> ascending local order is [0, 2].
        torch.testing.assert_close(
            vec_pos[_USER_TO_STORY][0], torch.tensor([0, 2], dtype=torch.long)
        )

    def test_duplicate_node_map_raises_assertion(self) -> None:
        # NOTE: the uniqueness check is gated on `__debug__`, so this assertion is
        # a no-op under `python -O` / `PYTHONOPTIMIZE`. GiGL node maps are unique by
        # construction (each local index is a distinct subgraph node), so the guard
        # exists only to catch misuse; the test asserts the guard fires under the
        # default (non-optimized) interpreter used by the test suite.
        node_map = {_STORY: torch.tensor([10, 10, 11])}
        positives = {_pos(_USER_TO_STORY): torch.tensor([[10, 11]], dtype=torch.long)}
        with self.assertRaises(AssertionError):
            vectorized_set_labels(
                node_local_to_global_by_type=node_map,
                positive_labels_by_edge_type=positives,
                negative_labels_by_edge_type={},
                supervision_edge_types=[_USER_TO_STORY],
                to_device=_CPU,
            )


if __name__ == "__main__":
    unittest.main()
