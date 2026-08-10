"""CUDA device-placement regression test for ABLP label remapping."""

import unittest

import torch

from gigl.distributed.utils.ablp import remap_labels_to_local_edge_indices
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.types.graph import message_passing_to_positive_label
from tests.test_assets.test_case import TestCase

_USER = NodeType("user")
_STORY = NodeType("story")
_USER_TO_STORY = EdgeType(_USER, Relation("to"), _STORY)


def _inputs(
    device: torch.device,
) -> tuple[dict[NodeType, torch.Tensor], dict[EdgeType, torch.Tensor]]:
    """Build an unsorted node map and padded labels on one device."""
    node_map = {_STORY: torch.tensor([15, 10, 16, 11, 12], device=device)}
    positive_labels = {
        message_passing_to_positive_label(_USER_TO_STORY): torch.tensor(
            [[15, 15], [16, -1], [-1, -1]],
            dtype=torch.long,
            device=device,
        )
    }
    return node_map, positive_labels


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class LabelRemapCudaDeviceTest(TestCase):
    def test_cuda_output_matches_cpu(self) -> None:
        cpu_node_map, cpu_positive_labels = _inputs(torch.device("cpu"))
        expected_positive, _ = remap_labels_to_local_edge_indices(
            local_id_to_global_id_by_node_type=cpu_node_map,
            positive_labels_by_edge_type=cpu_positive_labels,
            negative_labels_by_edge_type={},
        )

        cuda_node_map, cuda_positive_labels = _inputs(torch.device("cuda"))
        actual_positive, _ = remap_labels_to_local_edge_indices(
            local_id_to_global_id_by_node_type=cuda_node_map,
            positive_labels_by_edge_type=cuda_positive_labels,
            negative_labels_by_edge_type={},
        )

        self.assertEqual(set(actual_positive.keys()), set(expected_positive.keys()))
        for edge_type, expected_edge_index in expected_positive.items():
            actual_edge_index = actual_positive[edge_type]
            self.assertEqual(actual_edge_index.device.type, "cuda")
            torch.testing.assert_close(
                actual_edge_index.cpu(),
                expected_edge_index,
            )


if __name__ == "__main__":
    unittest.main()
