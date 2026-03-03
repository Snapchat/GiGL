import torch
from absl.testing import absltest
from parameterized import param, parameterized
from torch_geometric.typing import EdgeType

from gigl.distributed.utils.degree import (
    _clamp_to_int16,
    _compute_degrees_from_indptr,
    _pad_to_size,
    _sum_tensors_with_padding,
    compute_and_broadcast_degree_tensors,
)
from tests.test_assets.distributed.test_dataset import (
    DEFAULT_HETEROGENEOUS_EDGE_INDICES,
    DEFAULT_HOMOGENEOUS_EDGE_INDEX,
    STORY_TO_USER,
    USER_TO_STORY,
    create_heterogeneous_dataset,
    create_homogeneous_dataset,
)
from tests.test_assets.distributed.utils import (
    MockRemoteDistDataset,
    create_test_process_group,
)
from tests.test_assets.test_case import TestCase


def _compute_expected_degrees_from_edge_index(
    edge_index: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """Compute expected out-degrees from COO edge index."""
    src_nodes = edge_index[0]
    degrees = torch.zeros(num_nodes, dtype=torch.int16)
    for src in src_nodes:
        degrees[src] += 1
    return degrees


class TestLocalDegreeComputation(TestCase):
    """Tests for local DistDataset degree computation."""

    def test_homogeneous_graph(self):
        """Test degree computation for a homogeneous graph using real DistDataset."""
        edge_index = DEFAULT_HOMOGENEOUS_EDGE_INDEX
        num_nodes = int(edge_index.max().item() + 1)

        dataset = create_homogeneous_dataset(edge_index=edge_index)
        result = compute_and_broadcast_degree_tensors(dataset)

        assert isinstance(result, torch.Tensor)
        expected = _compute_expected_degrees_from_edge_index(edge_index, num_nodes)
        self.assertEqual(result.shape[0], num_nodes)
        self.assert_tensor_equality(result, expected)

    def test_homogeneous_graph_custom_edges(self):
        """Test degree computation with a custom edge index."""
        edge_index = torch.tensor([[0, 0, 1, 2, 2, 2, 3], [1, 2, 2, 0, 1, 3, 0]])
        num_nodes = int(edge_index[0].max().item() + 1)

        dataset = create_homogeneous_dataset(edge_index=edge_index)
        result = compute_and_broadcast_degree_tensors(dataset)

        assert isinstance(result, torch.Tensor)
        expected = _compute_expected_degrees_from_edge_index(edge_index, num_nodes)
        self.assert_tensor_equality(result, expected)

    def test_heterogeneous_graph(self):
        """Test degree computation for a heterogeneous graph using real DistDataset."""
        edge_indices = DEFAULT_HETEROGENEOUS_EDGE_INDICES
        dataset = create_heterogeneous_dataset(edge_indices=edge_indices)

        result = compute_and_broadcast_degree_tensors(dataset)

        assert isinstance(result, dict)
        self.assertEqual(len(result), len(edge_indices))

        for edge_type, edge_index in edge_indices.items():
            num_nodes = int(edge_index[0].max().item() + 1)
            expected = _compute_expected_degrees_from_edge_index(edge_index, num_nodes)
            self.assertIn(edge_type, result)
            self.assert_tensor_equality(result[edge_type], expected)


class TestRemoteDegreeComputation(TestCase):
    """Tests for remote RemoteDistDataset degree computation.

    RemoteDistDataset requires RPC connections to storage nodes,
    so these tests use MockRemoteDistDataset from test_assets to simulate
    the remote degree fetching.
    """

    def test_homogeneous_aggregation(self):
        """Test degree aggregation for homogeneous remote graph."""
        num_nodes, num_servers = 100, 3
        partition_degrees = [
            torch.randint(0, 5, (num_nodes,), dtype=torch.int32)
            for _ in range(num_servers)
        ]
        expected = sum(d.to(torch.int64) for d in partition_degrees)
        local_degrees: dict[int, torch.Tensor] = {
            i: d for i, d in enumerate(partition_degrees)
        }
        dataset = MockRemoteDistDataset(
            num_storage_nodes=num_servers, local_degrees=local_degrees
        )
        result = compute_and_broadcast_degree_tensors(dataset)
        self.assertIsInstance(result, torch.Tensor)
        assert isinstance(result, torch.Tensor)
        assert isinstance(expected, torch.Tensor)
        self.assert_tensor_equality(result.to(torch.int64), expected)

    def test_heterogeneous_aggregation(self):
        """Test degree aggregation for heterogeneous remote graph."""
        edge_types = [USER_TO_STORY, STORY_TO_USER]
        num_servers = 2
        partition_data: dict[int, dict[EdgeType, torch.Tensor]] = {
            i: {} for i in range(num_servers)
        }
        expected_by_type: dict[EdgeType, torch.Tensor] = {}
        for et in edge_types:
            num_nodes = 50
            total = torch.zeros(num_nodes, dtype=torch.int64)
            for i in range(num_servers):
                d = torch.randint(0, 5, (num_nodes,), dtype=torch.int32)
                partition_data[i][et] = d
                total += d.to(torch.int64)
            expected_by_type[et] = total
        dataset = MockRemoteDistDataset(
            num_storage_nodes=num_servers,
            edge_types=edge_types,
            local_degrees=partition_data,
        )
        result = compute_and_broadcast_degree_tensors(dataset)
        assert isinstance(result, dict)
        for et in edge_types:
            self.assert_tensor_equality(
                result[et].to(torch.int64), expected_by_type[et]
            )

    def test_empty_fetch_result_raises_error(self):
        """Test that ValueError is raised when no degrees returned."""
        dataset = MockRemoteDistDataset(num_storage_nodes=1, local_degrees={})
        with self.assertRaises(ValueError):
            compute_and_broadcast_degree_tensors(dataset)


class TestDistributedDegreeComputation(TestCase):
    """Tests for degree computation with torch.distributed initialized.

    These tests verify that the all-reduce path works correctly when
    torch.distributed is initialized. Uses a single-node process group.
    """

    def setUp(self):
        """Set up distributed process group before each test."""
        super().setUp()
        create_test_process_group()

    def tearDown(self):
        """Clean up distributed process group after each test."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        super().tearDown()

    def test_homogeneous_graph_distributed(self):
        """Test degree computation with distributed initialized."""
        edge_index = DEFAULT_HOMOGENEOUS_EDGE_INDEX
        num_nodes = int(edge_index.max().item() + 1)

        dataset = create_homogeneous_dataset(edge_index=edge_index)
        result = compute_and_broadcast_degree_tensors(dataset)

        assert isinstance(result, torch.Tensor)
        expected = _compute_expected_degrees_from_edge_index(edge_index, num_nodes)
        self.assertEqual(result.shape[0], num_nodes)
        self.assert_tensor_equality(result, expected)

    def test_heterogeneous_graph_distributed(self):
        """Test heterogeneous degree computation with distributed initialized."""
        edge_indices = DEFAULT_HETEROGENEOUS_EDGE_INDICES
        dataset = create_heterogeneous_dataset(edge_indices=edge_indices)

        result = compute_and_broadcast_degree_tensors(dataset)

        assert isinstance(result, dict)
        self.assertEqual(len(result), len(edge_indices))

        for edge_type, edge_index in edge_indices.items():
            num_nodes = int(edge_index[0].max().item() + 1)
            expected = _compute_expected_degrees_from_edge_index(edge_index, num_nodes)
            self.assertIn(edge_type, result)
            self.assert_tensor_equality(result[edge_type], expected)


class TestHelperFunctions(TestCase):
    """Tests for internal helper functions."""

    @parameterized.expand(
        [
            param(
                "sum_equal_length_tensors",
                tensors=[
                    torch.tensor([1, 2, 3], dtype=torch.int32),
                    torch.tensor([4, 5, 6], dtype=torch.int32),
                ],
                expected=torch.tensor([5, 7, 9], dtype=torch.int16),
            ),
            param(
                "sum_different_length_tensors",
                tensors=[
                    torch.tensor([1, 2], dtype=torch.int32),
                    torch.tensor([4, 5, 6], dtype=torch.int32),
                ],
                expected=torch.tensor([5, 7, 6], dtype=torch.int16),
            ),
            param(
                "sum_empty_list",
                tensors=[],
                expected=torch.tensor([], dtype=torch.int16),
            ),
        ]
    )
    def test_sum_tensors_with_padding(self, _, tensors, expected):
        """Test _sum_tensors_with_padding helper function."""
        result = _sum_tensors_with_padding(tensors)
        self.assert_tensor_equality(result, expected)

    @parameterized.expand(
        [
            param(
                "pad_smaller_tensor",
                tensor=torch.tensor([1, 2, 3], dtype=torch.int32),
                target_size=5,
                expected=torch.tensor([1, 2, 3, 0, 0], dtype=torch.int32),
            ),
            param(
                "no_padding_needed",
                tensor=torch.tensor([1, 2, 3], dtype=torch.int32),
                target_size=3,
                expected=torch.tensor([1, 2, 3], dtype=torch.int32),
            ),
            param(
                "tensor_larger_than_target",
                tensor=torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32),
                target_size=3,
                expected=torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32),
            ),
        ]
    )
    def test_pad_to_size(self, _, tensor, target_size, expected):
        """Test _pad_to_size helper function."""
        result = _pad_to_size(tensor, target_size)
        self.assert_tensor_equality(result, expected)

    def test_compute_degrees_from_indptr(self):
        """Test _compute_degrees_from_indptr helper function."""
        indptr = torch.tensor([0, 3, 5, 10, 12], dtype=torch.int64)
        expected = torch.tensor([3, 2, 5, 2], dtype=torch.int16)
        result = _compute_degrees_from_indptr(indptr)
        self.assert_tensor_equality(result, expected)

    def test_clamp_to_int16(self):
        """Test _clamp_to_int16 helper function."""
        max_int16 = torch.iinfo(torch.int16).max
        tensor = torch.tensor([1, max_int16 + 100, 5], dtype=torch.int64)
        expected = torch.tensor([1, max_int16, 5], dtype=torch.int16)
        result = _clamp_to_int16(tensor)
        self.assert_tensor_equality(result, expected)


if __name__ == "__main__":
    absltest.main()
