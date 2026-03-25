from parameterized import param, parameterized

from gigl.distributed.graph_store.messages import (
    FetchABLPInputRequest,
    FetchNodesRequest,
)
from tests.test_assets.distributed.test_dataset import USER, USER_TO_STORY
from tests.test_assets.test_case import TestCase


class TestFetchNodesRequestValidation(TestCase):
    @parameterized.expand(
        [
            param("both_provided", FetchNodesRequest(rank=0, world_size=4)),
            param("both_none", FetchNodesRequest(rank=None, world_size=None)),
            param("defaults", FetchNodesRequest()),
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
        ]
    )
    def test_validate_fails_when_rank_world_size_mismatch(
        self, _: str, request: FetchNodesRequest
    ) -> None:
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
        ]
    )
    def test_validate_fails_when_rank_world_size_mismatch(
        self, _: str, request: FetchABLPInputRequest
    ) -> None:
        """Validation fails when only one of rank/world_size is provided."""
        with self.assertRaises(ValueError):
            request.validate()

    def test_frozen(self) -> None:
        """FetchABLPRequest is immutable."""
        request = FetchABLPInputRequest(
            split="train",
            node_type=USER,
            supervision_edge_type=USER_TO_STORY,
        )
        with self.assertRaises(AttributeError):
            request.split = "val"  # type: ignore[misc]
