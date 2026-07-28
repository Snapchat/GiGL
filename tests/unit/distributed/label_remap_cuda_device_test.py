"""CUDA device-placement regression test for the ABLP label-remap kernels.

``vectorized_set_labels`` and ``edge_list_set_labels`` build an internal
``anchor_of_entry`` index and then select it with a mask derived from the input
``label_tensor``. If that index is created on CPU while ``label_tensor`` is on
GPU, the masked select raises ``"indices should be either on cpu or on the same
device as the indexed tensor"``. CPU-only unit tests cannot observe this, so the
bug only surfaces on a real GPU training run.

These tests run the kernels with all inputs on CUDA and assert the result equals
the CPU result. They are skipped when no GPU is present (e.g. CPU CI); run them
on a CUDA host to guard the device placement.
"""

import unittest

import torch
from absl.testing import absltest

from gigl.distributed.dist_ablp_neighborloader import (
    edge_list_set_labels,
    vectorized_set_labels,
)
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.types.graph import message_passing_to_positive_label

_USER = NodeType("user")
_STORY = NodeType("story")
_USER_TO_STORY = EdgeType(_USER, Relation("to"), _STORY)


def _inputs(device: torch.device):
    """A small case with a multi-label, a padded, and an empty anchor."""
    node_map = {_STORY: torch.tensor([10, 11, 12, 13, 14, 15], device=device)}
    positives = {
        message_passing_to_positive_label(_USER_TO_STORY): torch.tensor(
            [[10, 12], [15, -1], [-1, -1]], dtype=torch.long, device=device
        )
    }
    return node_map, positives


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class LabelRemapCudaDeviceTest(absltest.TestCase):
    def test_vectorized_set_labels_cuda_matches_cpu(self) -> None:
        cpu_node, cpu_pos = _inputs(torch.device("cpu"))
        expected_pos, _ = vectorized_set_labels(
            node_local_to_global_by_type=cpu_node,
            positive_labels_by_edge_type=cpu_pos,
            negative_labels_by_edge_type={},
            supervision_edge_types=[_USER_TO_STORY],
            to_device=torch.device("cpu"),
        )

        cuda = torch.device("cuda")
        cuda_node, cuda_pos = _inputs(cuda)
        # Must not raise the CPU/GPU index mismatch.
        got_pos, _ = vectorized_set_labels(
            node_local_to_global_by_type=cuda_node,
            positive_labels_by_edge_type=cuda_pos,
            negative_labels_by_edge_type={},
            supervision_edge_types=[_USER_TO_STORY],
            to_device=cuda,
        )

        self.assertEqual(set(got_pos.keys()), set(expected_pos.keys()))
        for edge_type, inner in expected_pos.items():
            got_inner = got_pos[edge_type]
            self.assertEqual(set(got_inner.keys()), set(inner.keys()))
            for anchor, expected_tensor in inner.items():
                got = got_inner[anchor]
                self.assertEqual(got.device.type, "cuda")
                torch.testing.assert_close(got.cpu(), expected_tensor)

    def test_edge_list_set_labels_cuda_matches_cpu(self) -> None:
        cpu_node, cpu_pos = _inputs(torch.device("cpu"))
        expected_pos, _ = edge_list_set_labels(
            node_local_to_global_by_type=cpu_node,
            positive_labels_by_edge_type=cpu_pos,
            negative_labels_by_edge_type={},
            supervision_edge_types=[_USER_TO_STORY],
            to_device=torch.device("cpu"),
        )

        cuda = torch.device("cuda")
        cuda_node, cuda_pos = _inputs(cuda)
        got_pos, _ = edge_list_set_labels(
            node_local_to_global_by_type=cuda_node,
            positive_labels_by_edge_type=cuda_pos,
            negative_labels_by_edge_type={},
            supervision_edge_types=[_USER_TO_STORY],
            to_device=cuda,
        )

        self.assertEqual(set(got_pos.keys()), set(expected_pos.keys()))
        for edge_type, expected_labels in expected_pos.items():
            got_labels = got_pos[edge_type]
            self.assertEqual(got_labels.anchor_index.device.type, "cuda")
            # Expanding to the ragged dict must reproduce the CPU result.
            expected_dict = expected_labels.to_dict()
            got_dict = got_labels.to_dict()
            self.assertEqual(set(got_dict.keys()), set(expected_dict.keys()))
            for anchor, expected_tensor in expected_dict.items():
                torch.testing.assert_close(got_dict[anchor].cpu(), expected_tensor)


if __name__ == "__main__":
    absltest.main()
