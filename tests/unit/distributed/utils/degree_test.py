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

    In graph store mode, the graph is partitioned across multiple storage nodes.
    Each storage node holds a subset of the edges and computes local degrees
    (the degree of each node based only on edges in that partition).

    To get the true global degree of each node, we must fetch local degrees
    from all storage nodes and sum them together. These tests verify that
    aggregation logic using MockRemoteDistDataset to simulate the RPC calls.
    """

    def test_homogeneous_aggregation(self):
        """Test degree aggregation for homogeneous remote graph.

        Simulates 3 storage nodes, each with partial degree counts for 100 nodes.
        The expected result is the element-wise sum of all partition degrees.
        """
        num_nodes, num_servers = 100, 3

        # Simulate each storage node having computed local degrees for its edges.
        # In a real scenario, these would come from CSR indptr on each partition.
        partition_degrees = [
            torch.randint(0, 5, (num_nodes,), dtype=torch.int32)
            for _ in range(num_servers)
        ]

        # The global degree for each node is the sum across all partitions,
        # since edges for any node may be spread across multiple storage nodes.
        expected = sum(d.to(torch.int64) for d in partition_degrees)

        # Create mock dataset that returns these partition degrees when
        # fetch_local_degrees() is called (instead of making real RPC calls).
        local_degrees: dict[int, torch.Tensor] = {
            i: d for i, d in enumerate(partition_degrees)
        }
        dataset = MockRemoteDistDataset(
            num_storage_nodes=num_servers, local_degrees=local_degrees
        )

        result = compute_and_broadcast_degree_tensors(dataset)

        assert isinstance(result, torch.Tensor)
        assert isinstance(expected, torch.Tensor)
        self.assert_tensor_equality(result.to(torch.int64), expected)

    def test_heterogeneous_aggregation(self):
        """Test degree aggregation for heterogeneous remote graph.

        Simulates 2 storage nodes with a heterogeneous graph containing
        multiple edge types. Each edge type's degrees are aggregated
        independently across storage nodes.
        """
        edge_types = [USER_TO_STORY, STORY_TO_USER]
        num_servers = 2

        # Build partition data: each server has a dict of edge_type -> degrees.
        # This mirrors the structure returned by DistServer.get_local_degrees()
        # for heterogeneous graphs.
        partition_data: dict[int, dict[EdgeType, torch.Tensor]] = {
            i: {} for i in range(num_servers)
        }
        expected_by_type: dict[EdgeType, torch.Tensor] = {}

        # For each edge type, generate random local degrees per server
        # and compute the expected sum.
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

        # Result should be a dict with aggregated degrees per edge type.
        assert isinstance(result, dict)
        for et in edge_types:
            self.assert_tensor_equality(
                result[et].to(torch.int64), expected_by_type[et]
            )

    def test_empty_fetch_result_raises_error(self):
        """Test that ValueError is raised when no degrees returned.

        If fetch_local_degrees() returns an empty dict (e.g., no storage nodes
        responded or all failed), compute_and_broadcast_degree_tensors should
        raise a ValueError rather than returning an invalid result.
        """
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
