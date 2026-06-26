"""CUDA device-placement regression test for the ABLP label-remap kernel.

``edge_list_set_labels`` builds an internal ``anchor_of_entry`` index and then
selects it with a mask derived from the input ``label_tensor``. If that index is
created on CPU while ``label_tensor`` is on GPU, the masked select raises
``"indices should be either on cpu or on the same device as the indexed
tensor"``. CPU-only unit tests cannot observe this, so the bug only surfaces on a
real GPU training run.

This test runs the kernel with all inputs on CUDA and asserts the result matches
the CPU result. It is skipped when no GPU is present (e.g. CPU CI); run it on a
CUDA host to guard the device placement.
"""

import unittest

import torch

from gigl.distributed.dist_ablp_neighborloader import edge_list_set_labels
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.types.graph import message_passing_to_positive_label
from tests.test_assets.test_case import TestCase

_USER = NodeType("user")
_STORY = NodeType("story")
_USER_TO_STORY = EdgeType(_USER, Relation("to"), _STORY)


def _inputs(device: torch.device):
    """A small case exercising every on-device code path.

    The node map is UNSORTED (so ``torch.sort`` yields a non-identity
    ``sort_perm`` -- the gather through it must run on-device). Anchor 0 has a
    DUPLICATE label column ([15, 15]) to exercise duplicate handling and
    verify multiplicity is preserved (local 0 appears twice). Local layout:
    g15->0, g10->1, g16->2, g11->3, g12->4. Anchor rows: [15, 15] -> local 0
    twice; [16, -1] -> local 2; [-1, -1] -> empty.
    """
    node_map = {_STORY: torch.tensor([15, 10, 16, 11, 12], device=device)}
    positives = {
        message_passing_to_positive_label(_USER_TO_STORY): torch.tensor(
            [[15, 15], [16, -1], [-1, -1]], dtype=torch.long, device=device
        )
    }
    return node_map, positives


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class LabelRemapCudaDeviceTest(TestCase):
    def test_edge_list_set_labels_cuda_matches_cpu(self) -> None:
        cpu_node, cpu_pos = _inputs(torch.device("cpu"))
        expected_pos, _ = edge_list_set_labels(
            node_local_to_global_by_type=cpu_node,
            positive_labels_by_edge_type=cpu_pos,
            negative_labels_by_edge_type={},
            to_device=torch.device("cpu"),
        )

        cuda = torch.device("cuda")
        cuda_node, cuda_pos = _inputs(cuda)
        # Must not raise the CPU/GPU index mismatch.
        got_pos, _ = edge_list_set_labels(
            node_local_to_global_by_type=cuda_node,
            positive_labels_by_edge_type=cuda_pos,
            negative_labels_by_edge_type={},
            to_device=cuda,
        )

        self.assertEqual(set(got_pos.keys()), set(expected_pos.keys()))
        for edge_type, expected_labels in expected_pos.items():
            got_labels = got_pos[edge_type]
            # The output tensors must land on the requested CUDA device.
            self.assertEqual(got_labels.anchor_index.device.type, "cuda")
            self.assertEqual(got_labels.label_index.device.type, "cuda")
            expected_dict = expected_labels.to_dict()
            got_dict = got_labels.to_dict()
            self.assertEqual(set(got_dict.keys()), set(expected_dict.keys()))
            for anchor, expected_tensor in expected_dict.items():
                # Multiplicity (the duplicate [15, 15] column) must survive too.
                torch.testing.assert_close(got_dict[anchor].cpu(), expected_tensor)


if __name__ == "__main__":
    unittest.main()
