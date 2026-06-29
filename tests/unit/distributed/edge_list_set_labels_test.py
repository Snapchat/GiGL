"""Unit tests for the ABLP label-remap kernel and its edge-list container.

These exercise the pure-tensor label-remap logic directly (no GLT, no
distributed runtime), so they run in-process without ``mp.spawn``.

``edge_list_set_labels`` is the loader's label-remap path; it turns padded blocks
of global label ids into per-edge-type :class:`AnchorLabels` edge lists. We assert
against constructed expected values, and we check both the edge-list tensors and
their :meth:`AnchorLabels.to_dict` view, since the loader uses both forms.

Within an anchor the kernel emits labels in column-visit order, which is left
unspecified by contract (the ABLP loss is order-invariant). We therefore pin
exact tensors only where the node map is sorted -- there column order coincides
with ascending-local order -- and otherwise compare per-anchor label *sets*.
"""

import unittest

import torch
from parameterized import param, parameterized
from torch_geometric.typing import EdgeType as PyGEdgeType

from gigl.distributed.dist_ablp_neighborloader import (
    AnchorLabels,
    edge_list_set_labels,
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


def _assert_dict_sets_equal(
    actual: dict[PyGEdgeType, dict[int, torch.Tensor]],
    expected: dict[PyGEdgeType, dict[int, torch.Tensor]],
) -> None:
    """Assert per-anchor SET equality (with multiplicity) between two label dicts.

    Comparison is order-independent within an anchor (the kernel does not pin
    within-anchor order) but still catches missing/extra labels and multiplicity
    (duplicate label columns), and asserts each value tensor is ``long``.
    """
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
            assert sorted(got.tolist()) == sorted(expected_tensor.tolist()), (
                f"{edge_type}[{anchor}]: got {got.tolist()}, "
                f"expected {expected_tensor.tolist()} (as sets)"
            )


class AnchorLabelsTest(TestCase):
    def test_to_dict_expands_empty_and_multi_label_anchors(self) -> None:
        # 3 anchors: anchor 0 -> [5], anchor 1 -> [] (empty), anchor 2 -> [7, 8].
        # Every anchor must get a key, including the empty one in the middle.
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

    def test_to_dict_all_empty(self) -> None:
        # No pairs at all: every anchor still gets an empty tensor.
        labels = AnchorLabels(
            anchor_index=torch.empty(0, dtype=torch.long),
            label_index=torch.empty(0, dtype=torch.long),
            num_anchors=2,
        )
        as_dict = labels.to_dict()
        self.assertEqual(set(as_dict.keys()), {0, 1})
        torch.testing.assert_close(as_dict[0], torch.empty(0, dtype=torch.long))
        torch.testing.assert_close(as_dict[1], torch.empty(0, dtype=torch.long))


class EdgeListSetLabelsTest(TestCase):
    @parameterized.expand(
        [
            # Sorted node map -> exact tensors are pinned. Covers present labels,
            # a fully-padded (empty) anchor, and an absent global id (also empty).
            param(
                "sorted_present_empty_and_padded",
                node_map={_STORY: torch.tensor([10, 11, 12, 13, 14, 15, 16, 17])},
                positives={
                    _pos(_USER_TO_STORY): torch.tensor(
                        [[15, -1], [15, 16], [-1, -1], [99, -1]], dtype=torch.long
                    )
                },
                negatives={},
                expected_positive={
                    _USER_TO_STORY: {
                        0: [5],
                        1: [5, 6],
                        2: [],
                        3: [],
                    }
                },
                expected_negative={},
            ),
            # Duplicate label columns must keep multiplicity: a node matching two
            # identical columns appears twice in its anchor's labels.
            param(
                "duplicate_label_columns_keep_multiplicity",
                node_map={_STORY: torch.tensor([10, 11, 12, 13, 14, 15])},
                positives={
                    _pos(_USER_TO_STORY): torch.tensor(
                        [[15, 15], [11, 11]], dtype=torch.long
                    )
                },
                negatives={},
                expected_positive={_USER_TO_STORY: {0: [5, 5], 1: [1, 1]}},
                expected_negative={},
            ),
            # Real homogeneous keying: the loader keys the node map by
            # DEFAULT_HOMOGENEOUS_NODE_TYPE and uses DEFAULT_HOMOGENEOUS_EDGE_TYPE.
            # Exercise that exact keying, not just a custom NodeType. Node map is
            # unsorted here, so we compare per-anchor sets.
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
                # g30->local2, g10->local1 ; second anchor fully padded.
                expected_positive={DEFAULT_HOMOGENEOUS_EDGE_TYPE: {0: [2, 1], 1: []}},
                expected_negative={},
            ),
            # Negatives travel the same path and must remap independently of the
            # positives. Sorted node map -> exact sets are unambiguous.
            param(
                "with_negatives",
                node_map={_STORY: torch.tensor([10, 11, 12, 13, 14, 15, 16, 17])},
                positives={
                    _pos(_USER_TO_STORY): torch.tensor([[15], [16]], dtype=torch.long)
                },
                negatives={
                    _neg(_USER_TO_STORY): torch.tensor(
                        [[13, 16], [17, -1]], dtype=torch.long
                    )
                },
                expected_positive={_USER_TO_STORY: {0: [5], 1: [6]}},
                expected_negative={_USER_TO_STORY: {0: [3, 6], 1: [7]}},
            ),
            # Multiple supervision edge types: each gets its own key and remaps
            # against its own supervision node type's map.
            param(
                "multi_edge_type",
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
                expected_positive={
                    _A_TO_B: {0: [2, 3]},  # g13->local2, g14->local3 in _B
                    _A_TO_C: {0: [2, 3]},  # g22->local2, g23->local3 in _C
                },
                expected_negative={},
            ),
            # Multiple edge types WITH negatives on each.
            param(
                "multi_edge_type_with_negatives",
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
                expected_positive={
                    _A_TO_B: {0: [2, 3]},
                    _A_TO_C: {0: [2, 3]},
                },
                expected_negative={
                    _A_TO_B: {0: [4, 5]},  # g15->local4, g16->local5 in _B
                    _A_TO_C: {0: [4, 5]},  # g24->local4, g25->local5 in _C
                },
            ),
            # Every anchor empty (one fully padded, one with only absent ids):
            # the edge type still appears, with an empty tensor per anchor.
            param(
                "all_anchors_empty",
                node_map={_STORY: torch.tensor([10, 11, 12])},
                positives={
                    _pos(_USER_TO_STORY): torch.tensor(
                        [[-1, -1], [99, 98]], dtype=torch.long
                    )
                },
                negatives={},
                expected_positive={_USER_TO_STORY: {0: [], 1: []}},
                expected_negative={},
            ),
        ]
    )
    def test_to_dict_matches_constructed_expected(
        self,
        _,
        node_map: dict[NodeType, torch.Tensor],
        positives: dict[EdgeType, torch.Tensor],
        negatives: dict[EdgeType, torch.Tensor],
        expected_positive: dict[PyGEdgeType, dict[int, list[int]]],
        expected_negative: dict[PyGEdgeType, dict[int, list[int]]],
    ) -> None:
        pos, neg = edge_list_set_labels(
            node_local_to_global_by_type=node_map,
            positive_labels_by_edge_type=positives,
            negative_labels_by_edge_type=negatives,
            to_device=_CPU,
        )
        for labels in pos.values():
            self.assertIsInstance(labels, AnchorLabels)
        for labels in neg.values():
            self.assertIsInstance(labels, AnchorLabels)
        expected_pos_tensors = {
            et: {a: torch.tensor(v, dtype=torch.long) for a, v in inner.items()}
            for et, inner in expected_positive.items()
        }
        expected_neg_tensors = {
            et: {a: torch.tensor(v, dtype=torch.long) for a, v in inner.items()}
            for et, inner in expected_negative.items()
        }
        _assert_dict_sets_equal(
            {et: labels.to_dict() for et, labels in pos.items()},
            expected_pos_tensors,
        )
        _assert_dict_sets_equal(
            {et: labels.to_dict() for et, labels in neg.items()},
            expected_neg_tensors,
        )

    def test_sorted_node_map_pins_exact_edge_list_tensors(self) -> None:
        # With a sorted node map column-visit order coincides with ascending-local
        # order, so the kernel's AnchorLabels tensors are fully determined: pin
        # them exactly. This locks the (anchor_index, label_index) co-indexing the
        # loss relies on. Covers present labels, a multi-label anchor, and a
        # fully-padded (empty) anchor.
        node_map = {_STORY: torch.tensor([10, 11, 12, 13, 14, 15, 16, 17])}
        positives = {
            _pos(_USER_TO_STORY): torch.tensor(
                [[15, -1], [15, 16], [-1, -1]], dtype=torch.long
            )
        }
        pos, _ = edge_list_set_labels(
            node_local_to_global_by_type=node_map,
            positive_labels_by_edge_type=positives,
            negative_labels_by_edge_type={},
            to_device=_CPU,
        )
        labels = pos[_USER_TO_STORY]
        self.assertEqual(labels.num_anchors, 3)
        torch.testing.assert_close(
            labels.anchor_index, torch.tensor([0, 1, 1], dtype=torch.long)
        )
        torch.testing.assert_close(
            labels.label_index, torch.tensor([5, 5, 6], dtype=torch.long)
        )

    def test_unsorted_node_map_correct_membership(self) -> None:
        # The node map is UNSORTED, so torch.sort yields a non-identity sort_perm
        # and the kernel must map sorted positions back through it to recover real
        # local indices. We assert the per-anchor SET (within-anchor order is
        # unspecified by contract): a regression that emitted sorted-position
        # indices instead of sort_perm-mapped locals would produce {1, 3} here and
        # fail, so this proves the local indices are real node indices.
        # node = [15, 10, 16, 11] -> g15=local0, g10=local1, g16=local2, g11=local3.
        # Labels [16, 15] -> SET {local0, local2} = {0, 2}.
        node_map = {_STORY: torch.tensor([15, 10, 16, 11])}
        positives = {_pos(_USER_TO_STORY): torch.tensor([[16, 15]], dtype=torch.long)}
        pos, _ = edge_list_set_labels(
            node_local_to_global_by_type=node_map,
            positive_labels_by_edge_type=positives,
            negative_labels_by_edge_type={},
            to_device=_CPU,
        )
        self.assertEqual(sorted(pos[_USER_TO_STORY].to_dict()[0].tolist()), [0, 2])

    def test_zero_anchor_tensor_yields_no_edge_type_key(self) -> None:
        # A zero-anchor label tensor means there were no anchors for that edge
        # type this batch, so it must NOT appear in the output at all (not as an
        # empty entry).
        node_map = {_STORY: torch.tensor([10, 11, 12])}
        positives = {_pos(_USER_TO_STORY): torch.empty((0, 0), dtype=torch.long)}
        pos, neg = edge_list_set_labels(
            node_local_to_global_by_type=node_map,
            positive_labels_by_edge_type=positives,
            negative_labels_by_edge_type={},
            to_device=_CPU,
        )
        self.assertEqual(pos, {})
        self.assertEqual(neg, {})

    def test_device_placement_cpu(self) -> None:
        # The output index tensors must land on the requested device. (CPU here;
        # the CUDA counterpart lives in label_remap_cuda_device_test.py.)
        node_map = {_STORY: torch.tensor([10, 11, 12, 13, 14, 15])}
        positives = {_pos(_USER_TO_STORY): torch.tensor([[15], [11]], dtype=torch.long)}
        pos, _ = edge_list_set_labels(
            node_local_to_global_by_type=node_map,
            positive_labels_by_edge_type=positives,
            negative_labels_by_edge_type={},
            to_device=_CPU,
        )
        labels = pos[_USER_TO_STORY]
        self.assertEqual(labels.anchor_index.device.type, "cpu")
        self.assertEqual(labels.label_index.device.type, "cpu")
        self.assertEqual(labels.anchor_index.dtype, torch.long)
        self.assertEqual(labels.label_index.dtype, torch.long)

    def test_duplicate_node_map_raises(self) -> None:
        # The membership lookup requires unique global ids; a duplicate would
        # silently drop a local index and corrupt the loss, so the kernel raises
        # ``ValueError``. GiGL node maps are unique by construction, so this only
        # guards misuse of the public kernel.
        node_map = {_STORY: torch.tensor([10, 10, 11])}
        positives = {_pos(_USER_TO_STORY): torch.tensor([[10, 11]], dtype=torch.long)}
        with self.assertRaises(ValueError):
            edge_list_set_labels(
                node_local_to_global_by_type=node_map,
                positive_labels_by_edge_type=positives,
                negative_labels_by_edge_type={},
                to_device=_CPU,
            )


if __name__ == "__main__":
    unittest.main()
