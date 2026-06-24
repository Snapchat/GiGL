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


if __name__ == "__main__":
    absltest.main()
