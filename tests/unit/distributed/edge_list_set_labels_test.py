"""Unit tests for the dense edge-list ABLP label container and kernel.

These exercise the pure-tensor label-remap logic directly (no GLT, no
distributed runtime), so they run in-process without ``mp.spawn``.
"""

import unittest

import torch
from parameterized import param, parameterized
from torch_geometric.typing import EdgeType as PyGEdgeType

from gigl.distributed.dist_ablp_neighborloader import (
    AnchorLabels,
    _loop_set_labels,
    _remap_one_label_tensor_edge_list,
    edge_list_set_labels,
)
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.types.graph import (
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


def _dict_from_edge_list(
    edge_list_by_type: dict[PyGEdgeType, AnchorLabels],
) -> dict[PyGEdgeType, dict[int, torch.Tensor]]:
    return {et: labels.to_dict() for et, labels in edge_list_by_type.items()}


def _assert_label_dicts_equal(
    actual: dict[PyGEdgeType, dict[int, torch.Tensor]],
    expected: dict[PyGEdgeType, dict[int, torch.Tensor]],
) -> None:
    assert set(actual.keys()) == set(expected.keys()), (
        f"{set(actual.keys())} != {set(expected.keys())}"
    )
    for edge_type, inner in expected.items():
        actual_inner = actual[edge_type]
        assert set(actual_inner.keys()) == set(inner.keys())
        for anchor, expected_tensor in inner.items():
            got = actual_inner[anchor]
            assert got.dtype == torch.long
            torch.testing.assert_close(got, expected_tensor)


class AnchorLabelsTest(TestCase):
    def test_to_dict_round_trips_empty_and_multi(self) -> None:
        # 3 anchors: anchor 0 -> [5], anchor 1 -> [] (empty), anchor 2 -> [7, 8].
        labels = AnchorLabels(
            anchor_index=torch.tensor([0, 2, 2], dtype=torch.long),
            label_index=torch.tensor([5, 7, 8], dtype=torch.long),
            num_anchors=3,
        )
        as_dict = labels.to_dict()
        self.assertEqual(set(as_dict.keys()), {0, 1, 2})
        torch.testing.assert_close(as_dict[0], torch.tensor([5], dtype=torch.long))
        torch.testing.assert_close(as_dict[1], torch.empty(0, dtype=torch.long))
        torch.testing.assert_close(as_dict[2], torch.tensor([7, 8], dtype=torch.long))


class RemapOneEdgeListTest(TestCase):
    def test_matches_nonzero_order_with_padding_and_empty(self) -> None:
        node = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17])
        sorted_node, sort_perm = torch.sort(node)
        # anchor 0: [15, -1] -> local 5 ; anchor 1: [15, 16] -> local 5,6 ;
        # anchor 2: [-1, -1] -> empty ; anchor 3: [99, -1] -> empty (99 absent).
        label_tensor = torch.tensor(
            [[15, -1], [15, 16], [-1, -1], [99, -1]], dtype=torch.long
        )
        result = _remap_one_label_tensor_edge_list(
            label_tensor, sorted_node, sort_perm, _CPU
        )
        self.assertEqual(result.num_anchors, 4)
        torch.testing.assert_close(
            result.anchor_index, torch.tensor([0, 1, 1], dtype=torch.long)
        )
        torch.testing.assert_close(
            result.label_index, torch.tensor([5, 5, 6], dtype=torch.long)
        )

    def test_unsorted_node_map_nontrivial_sort_perm(self) -> None:
        # node map is UNSORTED, so torch.sort yields a non-identity sort_perm.
        # node[i]=global id of local i: g15->local0, g10->local1, g16->local2,
        # g11->local3. The label row is high-id-first ([16, 15]); the edge-list
        # must emit ascending LOCAL index (g15=local0 before g16=local2), proving
        # the result is mapped through sort_perm and is not in column or sorted
        # order. A port that dropped the sort_perm gather would emit [0, 2] mapped
        # to the wrong locals (or sorted-position indices 1 and 3) and fail here.
        node = torch.tensor([15, 10, 16, 11])
        sorted_node, sort_perm = torch.sort(node)
        label_tensor = torch.tensor([[16, 15]], dtype=torch.long)
        result = _remap_one_label_tensor_edge_list(
            label_tensor, sorted_node, sort_perm, _CPU
        )
        self.assertEqual(result.num_anchors, 1)
        torch.testing.assert_close(
            result.anchor_index, torch.tensor([0, 0], dtype=torch.long)
        )
        torch.testing.assert_close(
            result.label_index, torch.tensor([0, 2], dtype=torch.long)
        )


class EdgeListSetLabelsEquivalenceTest(TestCase):
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
            # UNSORTED node map + reversed label columns -- non-identity sort_perm
            # (see the vectorized test for the rationale). Guards against a port
            # that emits sorted/column order instead of ascending local index.
            param(
                "unsorted_node_map_reversed_columns",
                node_map={_STORY: torch.tensor([15, 10, 16, 11])},
                positives={
                    _pos(_USER_TO_STORY): torch.tensor([[16, 15]], dtype=torch.long)
                },
                negatives={},
                supervision_edge_types=[_USER_TO_STORY],
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
    def test_edge_list_to_dict_matches_loop(
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
        el_pos, el_neg = edge_list_set_labels(
            node_local_to_global_by_type=node_map,
            positive_labels_by_edge_type=positives,
            negative_labels_by_edge_type=negatives,
            supervision_edge_types=supervision_edge_types,
            to_device=_CPU,
        )
        _assert_label_dicts_equal(_dict_from_edge_list(el_pos), loop_pos)
        _assert_label_dicts_equal(_dict_from_edge_list(el_neg), loop_neg)

    def test_duplicate_node_map_raises_assertion(self) -> None:
        # NOTE: the uniqueness check is gated on `__debug__`, so this is a no-op
        # under `python -O` / `PYTHONOPTIMIZE`. GiGL node maps are unique by
        # construction; the guard catches misuse only. The test asserts it fires
        # under the default (non-optimized) interpreter used by the test suite.
        node_map = {_STORY: torch.tensor([10, 10, 11])}
        positives = {_pos(_USER_TO_STORY): torch.tensor([[10, 11]], dtype=torch.long)}
        with self.assertRaises(AssertionError):
            edge_list_set_labels(
                node_local_to_global_by_type=node_map,
                positive_labels_by_edge_type=positives,
                negative_labels_by_edge_type={},
                supervision_edge_types=[_USER_TO_STORY],
                to_device=_CPU,
            )


if __name__ == "__main__":
    unittest.main()
