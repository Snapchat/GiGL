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
            param("both_provided", FetchNodesRequest(split_idx=0, num_splits=4)),
            param("both_none", FetchNodesRequest(split_idx=None, num_splits=None)),
            param("defaults", FetchNodesRequest()),
        ]
    )
    def test_validate_passes(self, _: str, request: FetchNodesRequest) -> None:
        """Validation passes when split_idx and num_splits are both provided or both absent."""
        request.validate()

    @parameterized.expand(
        [
            param(
                "split_idx_without_num_splits",
                FetchNodesRequest(split_idx=0, num_splits=None),
            ),
            param(
                "num_splits_without_split_idx",
                FetchNodesRequest(split_idx=None, num_splits=4),
            ),
        ]
    )
    def test_validate_fails_when_split_params_mismatch(
        self, _: str, request: FetchNodesRequest
    ) -> None:
        """Validation fails when only one of split_idx/num_splits is provided."""
        with self.assertRaises(ValueError):
            request.validate()

    @parameterized.expand(
        [
            param(
                "num_splits_zero",
                FetchNodesRequest(split_idx=0, num_splits=0),
            ),
            param(
                "num_splits_negative",
                FetchNodesRequest(split_idx=0, num_splits=-1),
            ),
            param(
                "split_idx_negative",
                FetchNodesRequest(split_idx=-1, num_splits=4),
            ),
            param(
                "split_idx_equals_num_splits",
                FetchNodesRequest(split_idx=4, num_splits=4),
            ),
            param(
                "split_idx_exceeds_num_splits",
                FetchNodesRequest(split_idx=5, num_splits=4),
            ),
        ]
    )
    def test_validate_fails_when_split_params_out_of_range(
        self, _: str, request: FetchNodesRequest
    ) -> None:
        """Validation fails when split_idx or num_splits are out of valid range."""
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
                    split_idx=0,
                    num_splits=4,
                ),
            ),
            param(
                "both_none",
                FetchABLPInputRequest(
                    split="train",
                    node_type=USER,
                    supervision_edge_type=USER_TO_STORY,
                    split_idx=None,
                    num_splits=None,
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
        """Validation passes when split_idx and num_splits are both provided or both absent."""
        request.validate()

    @parameterized.expand(
        [
            param(
                "split_idx_without_num_splits",
                FetchABLPInputRequest(
                    split="train",
                    node_type=USER,
                    supervision_edge_type=USER_TO_STORY,
                    split_idx=0,
                    num_splits=None,
                ),
            ),
            param(
                "num_splits_without_split_idx",
                FetchABLPInputRequest(
                    split="train",
                    node_type=USER,
                    supervision_edge_type=USER_TO_STORY,
                    split_idx=None,
                    num_splits=4,
                ),
            ),
        ]
    )
    def test_validate_fails_when_split_params_mismatch(
        self, _: str, request: FetchABLPInputRequest
    ) -> None:
        """Validation fails when only one of split_idx/num_splits is provided."""
        with self.assertRaises(ValueError):
            request.validate()

    @parameterized.expand(
        [
            param(
                "num_splits_zero",
                FetchABLPInputRequest(
                    split="train",
                    node_type=USER,
                    supervision_edge_type=USER_TO_STORY,
                    split_idx=0,
                    num_splits=0,
                ),
            ),
            param(
                "split_idx_negative",
                FetchABLPInputRequest(
                    split="train",
                    node_type=USER,
                    supervision_edge_type=USER_TO_STORY,
                    split_idx=-1,
                    num_splits=4,
                ),
            ),
            param(
                "split_idx_equals_num_splits",
                FetchABLPInputRequest(
                    split="train",
                    node_type=USER,
                    supervision_edge_type=USER_TO_STORY,
                    split_idx=4,
                    num_splits=4,
                ),
            ),
        ]
    )
    def test_validate_fails_when_split_params_out_of_range(
        self, _: str, request: FetchABLPInputRequest
    ) -> None:
        """Validation fails when split_idx or num_splits are out of valid range."""
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
