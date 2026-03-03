from unittest.mock import patch

import torch
from absl.testing import absltest
from parameterized import param, parameterized

import gigl.distributed.graph_store.dist_server as dist_server_module
from gigl.distributed.graph_store.dist_server import DistServer, _call_func_on_server
from gigl.distributed.graph_store.remote_dist_dataset import RemoteDistDataset
from gigl.distributed.utils.degree import (
    _clamp_to_int16,
    _compute_degrees_from_indptr,
    _pad_to_size,
    _sum_tensors_with_padding,
    compute_and_broadcast_degree_tensors,
)
from gigl.env.distributed import GraphStoreInfo
from tests.test_assets.distributed.test_dataset import (
    DEFAULT_HETEROGENEOUS_EDGE_INDICES,
    DEFAULT_HOMOGENEOUS_EDGE_INDEX,
    create_heterogeneous_dataset,
    create_homogeneous_dataset,
)
from tests.test_assets.distributed.utils import create_test_process_group
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


# =============================================================================
# Remote Degree Computation Tests (using real RemoteDistDataset + DistServer)
# =============================================================================


def _mock_async_request_server(server_rank, func, *args, **kwargs):
    """Mock async_request_server that returns a completed Future with the result.

    In a real distributed setup, this would make an async RPC call to the server.
    For testing, we call the server method synchronously and wrap the result
    in a Future to match the expected interface.
    """
    result = _call_func_on_server(func, *args, **kwargs)
    future: torch.futures.Future = torch.futures.Future()
    future.set_result(result)
    return future


def _create_mock_graph_store_info(num_storage_nodes: int = 1) -> GraphStoreInfo:
    """Create a mock GraphStoreInfo for testing."""
    return GraphStoreInfo(
        num_storage_nodes=num_storage_nodes,
        num_compute_nodes=1,
        cluster_master_ip="127.0.0.1",
        storage_cluster_master_ip="127.0.0.1",
        compute_cluster_master_ip="127.0.0.1",
        cluster_master_port=12345,
        storage_cluster_master_port=12346,
        compute_cluster_master_port=12347,
        num_processes_per_compute=1,
        rpc_master_port=12348,
        rpc_wait_port=12349,
    )


class TestRemoteHomogeneousDegreeComputation(TestCase):
    """Tests for remote degree computation on homogeneous graphs.

    Uses a real DistServer and RemoteDistDataset, with RPC calls mocked to route
    through the local server. This verifies the full integration path:
    RemoteDistDataset.fetch_local_degrees() -> DistServer.get_local_degrees()
    -> degree computation utilities.
    """

    def setUp(self) -> None:
        """Set up a DistServer with a homogeneous graph."""
        super().setUp()
        dataset = create_homogeneous_dataset(edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX)
        self._server = DistServer(dataset)
        dist_server_module._dist_server = self._server

    def tearDown(self) -> None:
        """Clean up server state."""
        dist_server_module._dist_server = None
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        super().tearDown()

    @patch(
        "gigl.distributed.graph_store.remote_dist_dataset.async_request_server",
        side_effect=_mock_async_request_server,
    )
    def test_degree_computation(self, mock_async_request):
        """Test degree computation for homogeneous graph via RemoteDistDataset."""
        edge_index = DEFAULT_HOMOGENEOUS_EDGE_INDEX
        num_nodes = int(edge_index.max().item() + 1)

        cluster_info = _create_mock_graph_store_info()
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)
        result = compute_and_broadcast_degree_tensors(remote_dataset)

        assert isinstance(result, torch.Tensor)
        expected = _compute_expected_degrees_from_edge_index(edge_index, num_nodes)
        self.assertEqual(result.shape[0], num_nodes)
        self.assert_tensor_equality(result, expected)


class TestRemoteHeterogeneousDegreeComputation(TestCase):
    """Tests for remote degree computation on heterogeneous graphs.

    Uses a real DistServer and RemoteDistDataset, with RPC calls mocked to route
    through the local server. Each edge type's degrees are computed independently.
    """

    def setUp(self) -> None:
        """Set up a DistServer with a heterogeneous graph."""
        super().setUp()
        dataset = create_heterogeneous_dataset(
            edge_indices=DEFAULT_HETEROGENEOUS_EDGE_INDICES
        )
        self._server = DistServer(dataset)
        dist_server_module._dist_server = self._server

    def tearDown(self) -> None:
        """Clean up server state."""
        dist_server_module._dist_server = None
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        super().tearDown()

    @patch(
        "gigl.distributed.graph_store.remote_dist_dataset.async_request_server",
        side_effect=_mock_async_request_server,
    )
    def test_degree_computation(self, mock_async_request):
        """Test degree computation for heterogeneous graph via RemoteDistDataset."""
        edge_indices = DEFAULT_HETEROGENEOUS_EDGE_INDICES

        cluster_info = _create_mock_graph_store_info()
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)
        result = compute_and_broadcast_degree_tensors(remote_dataset)

        assert isinstance(result, dict)
        self.assertEqual(len(result), len(edge_indices))

        for edge_type, edge_index in edge_indices.items():
            num_nodes = int(edge_index[0].max().item() + 1)
            expected = _compute_expected_degrees_from_edge_index(edge_index, num_nodes)
            self.assertIn(edge_type, result)
            self.assert_tensor_equality(result[edge_type], expected)


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


class TestDatasetDegreeProperty(TestCase):
    """Tests for DistDataset.compute_degree_tensors() and degree_tensors property."""

    def test_degree_tensors_initially_none(self):
        """Test that degree_tensors is None before compute_degree_tensors is called."""
        dataset = create_homogeneous_dataset(edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX)
        self.assertIsNone(dataset.degree_tensors)

    def test_compute_degree_tensors_homogeneous(self):
        """Test compute_degree_tensors for a homogeneous graph."""
        edge_index = DEFAULT_HOMOGENEOUS_EDGE_INDEX
        num_nodes = int(edge_index.max().item() + 1)

        dataset = create_homogeneous_dataset(edge_index=edge_index)
        result = dataset.compute_degree_tensors()

        assert isinstance(result, torch.Tensor)
        expected = _compute_expected_degrees_from_edge_index(edge_index, num_nodes)
        self.assert_tensor_equality(result, expected)

    def test_compute_degree_tensors_caches_result(self):
        """Test that compute_degree_tensors caches the result."""
        dataset = create_homogeneous_dataset(edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX)

        result1 = dataset.compute_degree_tensors()
        result2 = dataset.compute_degree_tensors()

        self.assertIs(result1, result2)

    def test_degree_tensors_property_after_compute(self):
        """Test that degree_tensors property returns cached result after compute."""
        edge_index = DEFAULT_HOMOGENEOUS_EDGE_INDEX
        num_nodes = int(edge_index.max().item() + 1)

        dataset = create_homogeneous_dataset(edge_index=edge_index)
        computed = dataset.compute_degree_tensors()
        from_property = dataset.degree_tensors

        self.assertIs(computed, from_property)
        assert isinstance(from_property, torch.Tensor)
        expected = _compute_expected_degrees_from_edge_index(edge_index, num_nodes)
        self.assert_tensor_equality(from_property, expected)

    def test_compute_degree_tensors_heterogeneous(self):
        """Test compute_degree_tensors for a heterogeneous graph."""
        edge_indices = DEFAULT_HETEROGENEOUS_EDGE_INDICES
        dataset = create_heterogeneous_dataset(edge_indices=edge_indices)

        result = dataset.compute_degree_tensors()

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
