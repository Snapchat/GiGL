import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from absl.testing import absltest
from parameterized import param, parameterized

from gigl.distributed.utils.degree import (
    _clamp_to_int16,
    _compute_degrees_from_indptr,
    _pad_to_size,
    compute_and_broadcast_degree_tensor,
)
from tests.test_assets.distributed.test_dataset import (
    DEFAULT_HETEROGENEOUS_EDGE_INDICES,
    DEFAULT_HOMOGENEOUS_EDGE_INDEX,
    create_heterogeneous_dataset,
    create_homogeneous_dataset,
)
from tests.test_assets.distributed.utils import (
    assert_tensor_equality,
    create_test_process_group,
    get_process_group_init_method,
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


class TestDegreeComputation(TestCase):
    """Tests for degree computation with torch.distributed initialized.

    These tests verify that the all-reduce path works correctly.
    Uses a single-node process group.
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

    def test_homogeneous_graph(self):
        """Test degree computation for a homogeneous graph."""
        edge_index = DEFAULT_HOMOGENEOUS_EDGE_INDEX
        num_nodes = int(edge_index.max().item() + 1)

        dataset = create_homogeneous_dataset(edge_index=edge_index)
        assert dataset.graph is not None
        result = compute_and_broadcast_degree_tensor(dataset.graph)

        assert isinstance(result, torch.Tensor)
        expected = _compute_expected_degrees_from_edge_index(edge_index, num_nodes)
        self.assertEqual(result.shape[0], num_nodes)
        self.assert_tensor_equality(result, expected)

    def test_heterogeneous_graph(self):
        """Test degree computation for a heterogeneous graph."""
        edge_indices = DEFAULT_HETEROGENEOUS_EDGE_INDICES
        dataset = create_heterogeneous_dataset(edge_indices=edge_indices)

        assert dataset.graph is not None
        result = compute_and_broadcast_degree_tensor(dataset.graph)

        assert isinstance(result, dict)
        self.assertEqual(set(result.keys()), set(edge_indices.keys()))

        for edge_type, edge_index in edge_indices.items():
            num_nodes = int(edge_index[0].max().item() + 1)
            expected = _compute_expected_degrees_from_edge_index(edge_index, num_nodes)
            self.assert_tensor_equality(result[edge_type], expected)

    def test_heterogeneous_graph_with_missing_topology(self):
        """Test that edge types with missing topology get empty tensors.

        This test creates a real heterogeneous dataset, then manually sets one
        edge type's topology to None to simulate the edge case where topology
        is unavailable.
        """
        edge_indices = DEFAULT_HETEROGENEOUS_EDGE_INDICES
        dataset = create_heterogeneous_dataset(edge_indices=edge_indices)

        assert dataset.graph is not None
        assert isinstance(dataset.graph, dict)

        # Get edge types from the dataset
        edge_types = list(dataset.graph.keys())

        edge_type_with_topo = edge_types[0]
        edge_type_without_topo = edge_types[1]

        # Save the original topology for computing expected degrees
        original_graph = dataset.graph[edge_type_with_topo]
        assert original_graph.topo is not None
        expected_degrees = _compute_expected_degrees_from_edge_index(
            edge_indices[edge_type_with_topo],
            int(edge_indices[edge_type_with_topo][0].max().item() + 1),
        )

        # Manually set one graph's topology to None to test the edge case
        dataset.graph[edge_type_without_topo].topo = None

        result = compute_and_broadcast_degree_tensor(dataset.graph)

        assert isinstance(result, dict)
        self.assertEqual(set(result.keys()), set(edge_types))

        # Edge type with topology should have computed degrees
        self.assert_tensor_equality(result[edge_type_with_topo], expected_degrees)

        # Edge type without topology should have empty tensor
        self.assertEqual(result[edge_type_without_topo].numel(), 0)


def _run_local_world_size_correction_homogeneous(
    rank: int,
    world_size: int,
    init_method: str,
    edge_index: torch.Tensor,
    expected_degrees: torch.Tensor,
) -> None:
    """Worker function for multi-process local_world_size correction test (homogeneous)."""
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )
    try:
        dataset = create_homogeneous_dataset(edge_index=edge_index)
        assert dataset.graph is not None
        result = compute_and_broadcast_degree_tensor(dataset.graph)

        assert isinstance(result, torch.Tensor)
        assert_tensor_equality(result, expected_degrees)
    finally:
        dist.destroy_process_group()


def _run_local_world_size_correction_heterogeneous(
    rank: int,
    world_size: int,
    init_method: str,
    edge_indices: dict,
    expected_degrees: dict,
) -> None:
    """Worker function for multi-process local_world_size correction test (heterogeneous)."""
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )
    try:
        dataset = create_heterogeneous_dataset(edge_indices=edge_indices)
        assert dataset.graph is not None
        result = compute_and_broadcast_degree_tensor(dataset.graph)

        assert isinstance(result, dict)
        assert set(result.keys()) == set(expected_degrees.keys())
        for edge_type, expected in expected_degrees.items():
            assert_tensor_equality(result[edge_type], expected)
    finally:
        dist.destroy_process_group()


class TestLocalWorldSizeCorrection(TestCase):
    """Tests for over-counting correction with multiple processes on the same machine.

    These tests spawn 2 real processes that share the same graph data, simulating
    the scenario where multiple training processes on one machine share a partition.
    The all-reduce should correctly divide by local_world_size=2.
    """

    def test_local_world_size_correction_homogeneous(self):
        """Test over-counting correction with 2 processes sharing the same data."""
        edge_index = DEFAULT_HOMOGENEOUS_EDGE_INDEX
        num_nodes = int(edge_index.max().item() + 1)

        raw_degrees = _compute_expected_degrees_from_edge_index(edge_index, num_nodes)
        expected_degrees = raw_degrees  # After correction: (2*raw) / 2 = raw

        init_method = get_process_group_init_method()
        mp.spawn(
            fn=_run_local_world_size_correction_homogeneous,
            args=(2, init_method, edge_index, expected_degrees),
            nprocs=2,
        )

    def test_local_world_size_correction_heterogeneous(self):
        """Test over-counting correction for heterogeneous graphs with 2 processes."""
        edge_indices = DEFAULT_HETEROGENEOUS_EDGE_INDICES

        expected_degrees = {}
        for edge_type, edge_index in edge_indices.items():
            num_nodes = int(edge_index[0].max().item() + 1)
            raw_degrees = _compute_expected_degrees_from_edge_index(
                edge_index, num_nodes
            )
            expected_degrees[edge_type] = raw_degrees

        init_method = get_process_group_init_method()
        mp.spawn(
            fn=_run_local_world_size_correction_heterogeneous,
            args=(2, init_method, edge_indices, expected_degrees),
            nprocs=2,
        )


class TestDatasetDegreeProperty(TestCase):
    """Tests for DistDataset.degree_tensor property."""

    def setUp(self):
        """Set up distributed process group before each test."""
        super().setUp()
        create_test_process_group()

    def tearDown(self):
        """Clean up distributed process group after each test."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        super().tearDown()

    def test_degree_tensor_homogeneous(self):
        """Test degree_tensor property for a homogeneous graph."""
        edge_index = DEFAULT_HOMOGENEOUS_EDGE_INDEX
        num_nodes = int(edge_index.max().item() + 1)

        dataset = create_homogeneous_dataset(edge_index=edge_index)
        result = dataset.degree_tensor

        assert isinstance(result, torch.Tensor)
        expected = _compute_expected_degrees_from_edge_index(edge_index, num_nodes)
        self.assert_tensor_equality(result, expected)

    def test_degree_tensor_caches_result(self):
        """Test that degree_tensor property caches the result."""
        dataset = create_homogeneous_dataset(edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX)

        result1 = dataset.degree_tensor
        result2 = dataset.degree_tensor

        self.assertIs(result1, result2)

    def test_degree_tensor_heterogeneous(self):
        """Test degree_tensor property for a heterogeneous graph."""
        edge_indices = DEFAULT_HETEROGENEOUS_EDGE_INDICES
        dataset = create_heterogeneous_dataset(edge_indices=edge_indices)

        result = dataset.degree_tensor

        assert isinstance(result, dict)
        self.assertEqual(set(result.keys()), set(edge_indices.keys()))

        for edge_type, edge_index in edge_indices.items():
            num_nodes = int(edge_index[0].max().item() + 1)
            expected = _compute_expected_degrees_from_edge_index(edge_index, num_nodes)
            self.assert_tensor_equality(result[edge_type], expected)


class TestHelperFunctions(TestCase):
    """Tests for internal helper functions."""

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

    def test_compute_degrees_from_indptr_all_zeros(self):
        """Test _compute_degrees_from_indptr with all-zero indptr (no edges)."""
        # All-zero indptr means no outgoing edges for any node
        indptr = torch.tensor([0, 0, 0, 0, 0], dtype=torch.int64)
        expected = torch.tensor([0, 0, 0, 0], dtype=torch.int16)
        result = _compute_degrees_from_indptr(indptr)
        self.assert_tensor_equality(result, expected)

    def test_compute_degrees_from_indptr_empty(self):
        """Test _compute_degrees_from_indptr with empty indptr (no nodes)."""
        # indptr of [0] means 0 nodes
        indptr = torch.empty(0, dtype=torch.int64)
        expected = torch.empty(0, dtype=torch.int16)
        result = _compute_degrees_from_indptr(indptr)
        self.assert_tensor_equality(result, expected)
        self.assertEqual(result.numel(), 0)

    def test_clamp_to_int16(self):
        """Test _clamp_to_int16 helper function."""
        max_int16 = torch.iinfo(torch.int16).max
        tensor = torch.tensor([1, max_int16 + 100, 5], dtype=torch.int64)
        expected = torch.tensor([1, max_int16, 5], dtype=torch.int16)
        result = _clamp_to_int16(tensor)
        self.assert_tensor_equality(result, expected)


if __name__ == "__main__":
    absltest.main()
