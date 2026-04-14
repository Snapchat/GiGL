import torch

from gigl.distributed.graph_store.messages import (
    FetchABLPInputRequest,
    FetchNodesRequest,
)
from gigl.distributed.graph_store.sharding import ServerSlice
from tests.test_assets.distributed.test_dataset import USER, USER_TO_STORY
from tests.test_assets.test_case import TestCase


class TestFetchNodesRequest(TestCase):
    def test_defaults(self) -> None:
        """Request can be constructed with all defaults."""
        request = FetchNodesRequest()
        self.assertIsNone(request.split)
        self.assertIsNone(request.node_type)
        self.assertIsNone(request.server_slice)

    def test_with_server_slice(self) -> None:
        """Request can include a server slice."""
        server_slice = ServerSlice(
            server_rank=0, start_numerator=0, end_numerator=1, denominator=2
        )
        request = FetchNodesRequest(
            split="train", node_type=USER, server_slice=server_slice
        )
        self.assertEqual(request.split, "train")
        self.assertEqual(request.node_type, USER)
        self.assertEqual(request.server_slice, server_slice)


class TestFetchABLPInputRequest(TestCase):
    def test_construction(self) -> None:
        """Request can be constructed with required fields."""
        request = FetchABLPInputRequest(
            split="train",
            node_type=USER,
            supervision_edge_type=USER_TO_STORY,
        )
        self.assertEqual(request.split, "train")
        self.assertIsNone(request.server_slice)

    def test_with_server_slice(self) -> None:
        """Request can include a server slice."""
        server_slice = ServerSlice(
            server_rank=0, start_numerator=0, end_numerator=1, denominator=2
        )
        request = FetchABLPInputRequest(
            split="train",
            node_type=USER,
            supervision_edge_type=USER_TO_STORY,
            server_slice=server_slice,
        )
        self.assertEqual(request.server_slice, server_slice)


class TestServerSlice(TestCase):
    def test_full_tensor_returns_same_object(self) -> None:
        tensor = torch.arange(10)
        server_slice = ServerSlice(
            server_rank=0,
            start_numerator=0,
            end_numerator=1,
            denominator=1,
        )
        result = server_slice.slice_tensor(tensor)
        self.assertEqual(result.data_ptr(), tensor.data_ptr())

    def test_partial_slice_returns_requested_range(self) -> None:
        tensor = torch.arange(10)
        server_slice = ServerSlice(
            server_rank=0,
            start_numerator=0,
            end_numerator=1,
            denominator=2,
        )
        result = server_slice.slice_tensor(tensor)
        self.assert_tensor_equality(result, torch.arange(5))
