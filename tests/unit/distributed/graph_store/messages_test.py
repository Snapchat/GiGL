import torch
from parameterized import param, parameterized

from gigl.distributed.graph_store.messages import (
    FetchABLPInputRequest,
    FetchNodesRequest,
)
from gigl.distributed.graph_store.sharding import ServerSlice
from tests.test_assets.distributed.test_dataset import USER, USER_TO_STORY
from tests.test_assets.test_case import TestCase


class TestFetchNodesRequestValidation(TestCase):
    @parameterized.expand(
        [
            param("both_provided", FetchNodesRequest(rank=0, world_size=4)),
            param("both_none", FetchNodesRequest(rank=None, world_size=None)),
            param("defaults", FetchNodesRequest()),
            param(
                "server_slice_provided",
                FetchNodesRequest(
                    server_slice=ServerSlice(
                        server_rank=0,
                        start_num=0,
                        start_den=1,
                        end_num=1,
                        end_den=1,
                    ),
                ),
            ),
        ]
    )
    def test_validate_passes(self, _: str, request: FetchNodesRequest) -> None:
        """Validation passes when rank and world_size are both provided or both absent."""
        request.validate()

    @parameterized.expand(
        [
            param(
                "rank_without_world_size", FetchNodesRequest(rank=0, world_size=None)
            ),
            param(
                "world_size_without_rank", FetchNodesRequest(rank=None, world_size=4)
            ),
            param(
                "server_slice_with_rank_world_size",
                FetchNodesRequest(
                    rank=0,
                    world_size=4,
                    server_slice=ServerSlice(
                        server_rank=0,
                        start_num=0,
                        start_den=1,
                        end_num=1,
                        end_den=1,
                    ),
                ),
            ),
        ]
    )
    def test_validate_fails(self, _: str, request: FetchNodesRequest) -> None:
        """Validation fails when only one of rank/world_size is provided."""
        with self.assertRaises(ValueError):
            request.validate()


class TestFetchABLPRequestValidation(TestCase):
    @parameterized.expand(
        [
            param(
                "both_provided",
                FetchABLPInputRequest(
                    split="train",
                    node_type=USER,
                    supervision_edge_type=USER_TO_STORY,
                    rank=0,
                    world_size=4,
                ),
            ),
            param(
                "both_none",
                FetchABLPInputRequest(
                    split="train",
                    node_type=USER,
                    supervision_edge_type=USER_TO_STORY,
                    rank=None,
                    world_size=None,
                ),
            ),
            param(
                "defaults",
                FetchABLPInputRequest(
                    split="train",
                    node_type=USER,
                    supervision_edge_type=USER_TO_STORY,
                ),
            ),
            param(
                "server_slice_provided",
                FetchABLPInputRequest(
                    split="train",
                    node_type=USER,
                    supervision_edge_type=USER_TO_STORY,
                    server_slice=ServerSlice(
                        server_rank=0,
                        start_num=0,
                        start_den=1,
                        end_num=1,
                        end_den=1,
                    ),
                ),
            ),
        ]
    )
    def test_validate_passes(self, _: str, request: FetchABLPInputRequest) -> None:
        """Validation passes when rank and world_size are both provided or both absent."""
        request.validate()

    @parameterized.expand(
        [
            param(
                "rank_without_world_size",
                FetchABLPInputRequest(
                    split="train",
                    node_type=USER,
                    supervision_edge_type=USER_TO_STORY,
                    rank=0,
                    world_size=None,
                ),
            ),
            param(
                "world_size_without_rank",
                FetchABLPInputRequest(
                    split="train",
                    node_type=USER,
                    supervision_edge_type=USER_TO_STORY,
                    rank=None,
                    world_size=4,
                ),
            ),
            param(
                "server_slice_with_rank_world_size",
                FetchABLPInputRequest(
                    split="train",
                    node_type=USER,
                    supervision_edge_type=USER_TO_STORY,
                    rank=0,
                    world_size=4,
                    server_slice=ServerSlice(
                        server_rank=0,
                        start_num=0,
                        start_den=1,
                        end_num=1,
                        end_den=1,
                    ),
                ),
            ),
        ]
    )
    def test_validate_fails(self, _: str, request: FetchABLPInputRequest) -> None:
        """Validation fails when only one of rank/world_size is provided."""
        with self.assertRaises(ValueError):
            request.validate()


class TestServerSlice(TestCase):
    def test_full_tensor_returns_same_object(self) -> None:
        tensor = torch.arange(10)
        server_slice = ServerSlice(
            server_rank=0, start_num=0, start_den=1, end_num=1, end_den=1
        )
        result = server_slice.slice_tensor(tensor)
        self.assertEqual(result.data_ptr(), tensor.data_ptr())

    def test_partial_slice_returns_requested_range(self) -> None:
        tensor = torch.arange(10)
        server_slice = ServerSlice(
            server_rank=0, start_num=0, start_den=2, end_num=1, end_den=2
        )
        result = server_slice.slice_tensor(tensor)
        self.assert_tensor_equality(result, torch.arange(5))
