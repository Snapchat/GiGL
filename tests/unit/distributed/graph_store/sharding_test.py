import torch
from parameterized import param, parameterized

from gigl.distributed.graph_store.sharding import (
    ServerSlice,
    compute_server_assignments,
)
from tests.test_assets.test_case import TestCase


class TestComputeServerAssignments(TestCase):
    @parameterized.expand(
        [
            param(
                "rank_0",
                compute_rank=0,
                expected_assignments={
                    0: ServerSlice(
                        server_rank=0,
                        start_num=0,
                        start_den=2,
                        end_num=2,
                        end_den=2,
                    ),
                    1: ServerSlice(
                        server_rank=1,
                        start_num=0,
                        start_den=2,
                        end_num=1,
                        end_den=2,
                    ),
                },
            ),
            param(
                "rank_1",
                compute_rank=1,
                expected_assignments={
                    1: ServerSlice(
                        server_rank=1,
                        start_num=1,
                        start_den=2,
                        end_num=2,
                        end_den=2,
                    ),
                    2: ServerSlice(
                        server_rank=2,
                        start_num=0,
                        start_den=2,
                        end_num=2,
                        end_den=2,
                    ),
                },
            ),
        ]
    )
    def test_fractional_boundary_assignment(
        self, _, compute_rank: int, expected_assignments: dict[int, ServerSlice]
    ) -> None:
        assignments = compute_server_assignments(
            num_servers=3, num_compute_nodes=2, compute_rank=compute_rank
        )
        self.assertEqual(assignments, expected_assignments)

    def test_assignments_recombine_server_data(self) -> None:
        tensor = torch.arange(7)
        all_assignments = [
            compute_server_assignments(
                num_servers=2, num_compute_nodes=5, compute_rank=rank
            )
            for rank in range(5)
        ]

        for server_rank in range(2):
            combined = torch.cat(
                [
                    assignments[server_rank].slice_tensor(tensor)
                    for assignments in all_assignments
                    if server_rank in assignments
                ]
            )
            self.assert_tensor_equality(combined, tensor)

    @parameterized.expand(
        [
            param(
                "negative_servers",
                num_servers=-1,
                num_compute_nodes=2,
                compute_rank=0,
            ),
            param(
                "zero_servers",
                num_servers=0,
                num_compute_nodes=2,
                compute_rank=0,
            ),
            param(
                "negative_compute_nodes",
                num_servers=2,
                num_compute_nodes=-1,
                compute_rank=0,
            ),
            param(
                "zero_compute_nodes",
                num_servers=2,
                num_compute_nodes=0,
                compute_rank=0,
            ),
            param(
                "rank_too_large",
                num_servers=2,
                num_compute_nodes=2,
                compute_rank=2,
            ),
            param(
                "negative_rank",
                num_servers=2,
                num_compute_nodes=2,
                compute_rank=-1,
            ),
        ]
    )
    def test_validates_arguments(
        self, _, num_servers: int, num_compute_nodes: int, compute_rank: int
    ) -> None:
        with self.assertRaises(ValueError):
            compute_server_assignments(
                num_servers=num_servers,
                num_compute_nodes=num_compute_nodes,
                compute_rank=compute_rank,
            )


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
