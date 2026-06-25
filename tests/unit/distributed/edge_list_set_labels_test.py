"""Unit tests for the dense edge-list ABLP label kernel and its format selector.

These exercise the pure-tensor label-remap logic directly (no GLT, no
distributed runtime), so they run in-process without ``mp.spawn``.
"""

import os
from unittest import mock

from absl.testing import absltest, parameterized

from gigl.distributed.utils.neighborloader import (
    ABLP_LABEL_FORMAT_ENV_VAR,
    resolve_ablp_label_format,
)


class ResolveAblpLabelFormatTest(parameterized.TestCase):
    @parameterized.parameters(
        ("dict", "dict"),
        ("edge_list", "edge_list"),
        ("EDGE_LIST", "edge_list"),
    )
    def test_valid_values(self, env_value: str, expected: str) -> None:
        with mock.patch.dict(os.environ, {ABLP_LABEL_FORMAT_ENV_VAR: env_value}):
            self.assertEqual(resolve_ablp_label_format(), expected)

    def test_unset_defaults_to_dict(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_ablp_label_format(), "dict")

    def test_invalid_value_raises(self) -> None:
        with mock.patch.dict(os.environ, {ABLP_LABEL_FORMAT_ENV_VAR: "ragged"}):
            with self.assertRaises(ValueError):
                resolve_ablp_label_format()


import torch

from gigl.distributed.dist_ablp_neighborloader import (
    AnchorLabels,
    _remap_one_label_tensor_edge_list,
)

_CPU = torch.device("cpu")


class AnchorLabelsTest(absltest.TestCase):
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


class RemapOneEdgeListTest(absltest.TestCase):
    def test_matches_nonzero_order_with_padding_and_empty(self) -> None:
        # node holds global ids; index = local id.
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


from torch_geometric.typing import EdgeType as PyGEdgeType

from gigl.distributed.dist_ablp_neighborloader import (
    _loop_set_labels,
    edge_list_set_labels,
)
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.types.graph import (
    message_passing_to_negative_label,
    message_passing_to_positive_label,
)

_USER = NodeType("user")
_STORY = NodeType("story")
_USER_TO_STORY = EdgeType(_USER, Relation("to"), _STORY)
_A = NodeType("a")
_B = NodeType("b")
_C = NodeType("c")
_A_TO_B = EdgeType(_A, Relation("to"), _B)
_A_TO_C = EdgeType(_A, Relation("to"), _C)


def _pos(et: EdgeType) -> EdgeType:
    return message_passing_to_positive_label(et)


def _neg(et: EdgeType) -> EdgeType:
    return message_passing_to_negative_label(et)


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


class EdgeListSetLabelsEquivalenceTest(parameterized.TestCase):
    @parameterized.named_parameters(
        dict(
            testcase_name="homogeneous_present_empty_and_padded",
            node_map={_STORY: torch.tensor([10, 11, 12, 13, 14, 15, 16, 17])},
            positives={
                _pos(_USER_TO_STORY): torch.tensor(
                    [[15, -1], [15, 16], [-1, -1], [99, -1]], dtype=torch.long
                )
            },
            negatives={},
            supervision_edge_types=[_USER_TO_STORY],
        ),
        dict(
            testcase_name="homogeneous_duplicate_labels",
            node_map={_STORY: torch.tensor([10, 11, 12, 13, 14, 15])},
            positives={
                _pos(_USER_TO_STORY): torch.tensor(
                    [[15, 15], [11, 11]], dtype=torch.long
                )
            },
            negatives={},
            supervision_edge_types=[_USER_TO_STORY],
        ),
        dict(
            testcase_name="homogeneous_with_negatives",
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
        dict(
            testcase_name="heterogeneous_multi_edge_type",
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
        dict(
            testcase_name="all_anchors_empty",
            node_map={_STORY: torch.tensor([10, 11, 12])},
            positives={
                _pos(_USER_TO_STORY): torch.tensor(
                    [[-1, -1], [99, 98]], dtype=torch.long
                )
            },
            negatives={},
            supervision_edge_types=[_USER_TO_STORY],
        ),
        dict(
            testcase_name="zero_anchors",
            node_map={_STORY: torch.tensor([10, 11, 12])},
            positives={_pos(_USER_TO_STORY): torch.empty((0, 0), dtype=torch.long)},
            negatives={},
            supervision_edge_types=[_USER_TO_STORY],
        ),
    )
    def test_edge_list_to_dict_matches_loop(
        self,
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
    absltest.main()
