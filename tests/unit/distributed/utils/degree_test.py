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
    compute_and_broadcast_degree_tensor,
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
        result = compute_and_broadcast_degree_tensor(dataset)

        assert isinstance(result, torch.Tensor)
        expected = _compute_expected_degrees_from_edge_index(edge_index, num_nodes)
        self.assertEqual(result.shape[0], num_nodes)
        self.assert_tensor_equality(result, expected)

    def test_homogeneous_graph_custom_edges(self):
        """Test degree computation with a custom edge index."""
        edge_index = torch.tensor([[0, 0, 1, 2, 2, 2, 3], [1, 2, 2, 0, 1, 3, 0]])
        num_nodes = int(edge_index[0].max().item() + 1)

        dataset = create_homogeneous_dataset(edge_index=edge_index)
        result = compute_and_broadcast_degree_tensor(dataset)

        assert isinstance(result, torch.Tensor)
        expected = _compute_expected_degrees_from_edge_index(edge_index, num_nodes)
        self.assert_tensor_equality(result, expected)

    def test_heterogeneous_graph(self):
        """Test degree computation for a heterogeneous graph using real DistDataset."""
        edge_indices = DEFAULT_HETEROGENEOUS_EDGE_INDICES
        dataset = create_heterogeneous_dataset(edge_indices=edge_indices)

        result = compute_and_broadcast_degree_tensor(dataset)

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
        result = compute_and_broadcast_degree_tensor(dataset)

        assert isinstance(result, torch.Tensor)
        expected = _compute_expected_degrees_from_edge_index(edge_index, num_nodes)
        self.assertEqual(result.shape[0], num_nodes)
        self.assert_tensor_equality(result, expected)

    def test_heterogeneous_graph_distributed(self):
        """Test heterogeneous degree computation with distributed initialized."""
        edge_indices = DEFAULT_HETEROGENEOUS_EDGE_INDICES
        dataset = create_heterogeneous_dataset(edge_indices=edge_indices)

        result = compute_and_broadcast_degree_tensor(dataset)

        assert isinstance(result, dict)
        self.assertEqual(len(result), len(edge_indices))

        for edge_type, edge_index in edge_indices.items():
            num_nodes = int(edge_index[0].max().item() + 1)
            expected = _compute_expected_degrees_from_edge_index(edge_index, num_nodes)
            self.assertIn(edge_type, result)
            self.assert_tensor_equality(result[edge_type], expected)

    @patch("gigl.distributed.utils.degree.get_internal_ip_from_all_ranks")
    def test_local_world_size_correction_homogeneous(self, mock_get_ips):
        """Test over-counting correction when local_world_size > 1.

        Mocks get_internal_ip_from_all_ranks to simulate 2 processes on the same
        machine (both reporting the same IP). This should cause local_world_size=2,
        which divides the all-reduced degrees by 2.

        The mock returns a list with 2 identical IPs, simulating 2 ranks that
        share the same machine. Since my_rank=0 in a single-process test,
        my_ip=all_ips[0] and Counter(all_ips)[my_ip]=2, giving local_world_size=2.
        """
        mock_get_ips.return_value = ["192.168.1.1", "192.168.1.1"]

        edge_index = DEFAULT_HOMOGENEOUS_EDGE_INDEX
        num_nodes = int(edge_index.max().item() + 1)
        dataset = create_homogeneous_dataset(edge_index=edge_index)

        result = compute_and_broadcast_degree_tensor(dataset)

        assert isinstance(result, torch.Tensor)
        raw_expected = _compute_expected_degrees_from_edge_index(edge_index, num_nodes)
        expected = raw_expected // 2
        self.assert_tensor_equality(result, expected)

    @patch("gigl.distributed.utils.degree.get_internal_ip_from_all_ranks")
    def test_local_world_size_correction_heterogeneous(self, mock_get_ips):
        """Test over-counting correction for heterogeneous graphs with local_world_size > 1."""
        mock_get_ips.return_value = ["192.168.1.1", "192.168.1.1"]

        edge_indices = DEFAULT_HETEROGENEOUS_EDGE_INDICES
        dataset = create_heterogeneous_dataset(edge_indices=edge_indices)

        result = compute_and_broadcast_degree_tensor(dataset)

        assert isinstance(result, dict)
        for edge_type, edge_index in edge_indices.items():
            num_nodes = int(edge_index[0].max().item() + 1)
            raw_expected = _compute_expected_degrees_from_edge_index(
                edge_index, num_nodes
            )
            expected = raw_expected // 2
            self.assertIn(edge_type, result)
            self.assert_tensor_equality(result[edge_type], expected)


class TestDatasetDegreeProperty(TestCase):
    """Tests for DistDataset.compute_degree_tensor() and degree_tensor property."""

    def test_degree_tensor_initially_none(self):
        """Test that degree_tensor is None before compute_degree_tensor is called."""
        dataset = create_homogeneous_dataset(edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX)
        self.assertIsNone(dataset.degree_tensor)

    def test_compute_degree_tensor_homogeneous(self):
        """Test compute_degree_tensor for a homogeneous graph."""
        edge_index = DEFAULT_HOMOGENEOUS_EDGE_INDEX
        num_nodes = int(edge_index.max().item() + 1)

        dataset = create_homogeneous_dataset(edge_index=edge_index)
        result = dataset.compute_degree_tensor()

        assert isinstance(result, torch.Tensor)
        expected = _compute_expected_degrees_from_edge_index(edge_index, num_nodes)
        self.assert_tensor_equality(result, expected)

    def test_compute_degree_tensor_caches_result(self):
        """Test that compute_degree_tensor caches the result."""
        dataset = create_homogeneous_dataset(edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX)

        result1 = dataset.compute_degree_tensor()
        result2 = dataset.compute_degree_tensor()

        self.assertIs(result1, result2)

    def test_degree_tensor_property_after_compute(self):
        """Test that degree_tensor property returns cached result after compute."""
        edge_index = DEFAULT_HOMOGENEOUS_EDGE_INDEX
        num_nodes = int(edge_index.max().item() + 1)

        dataset = create_homogeneous_dataset(edge_index=edge_index)
        computed = dataset.compute_degree_tensor()
        from_property = dataset.degree_tensor

        self.assertIs(computed, from_property)
        assert isinstance(from_property, torch.Tensor)
        expected = _compute_expected_degrees_from_edge_index(edge_index, num_nodes)
        self.assert_tensor_equality(from_property, expected)

    def test_compute_degree_tensor_heterogeneous(self):
        """Test compute_degree_tensor for a heterogeneous graph."""
        edge_indices = DEFAULT_HETEROGENEOUS_EDGE_INDICES
        dataset = create_heterogeneous_dataset(edge_indices=edge_indices)

        result = dataset.compute_degree_tensor()

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


# =============================================================================
# Remote Degree Computation Tests
# =============================================================================


def _mock_async_request_server(server_rank, func, *args, **kwargs):
    """Mock async_request_server that calls the server method synchronously.

    Returns a completed Future with the result.
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
    RemoteDistDataset.compute_degree_tensor() -> DistServer.compute_and_distribute_global_degrees()
    -> degree computation utilities.

    The architecture uses all-reduce among storage servers:
    1. Client calls remote_dataset.compute_degree_tensor()
    2. This triggers all servers to participate in all-reduce
    3. Each server computes local degrees, all-reduces, and stores global degrees
    """

    def setUp(self) -> None:
        """Set up a DistServer with a homogeneous graph and initialize distributed."""
        super().setUp()
        create_test_process_group()
        dataset = create_homogeneous_dataset(edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX)
        self._server = DistServer(dataset)
        dist_server_module._dist_server = self._server

    def tearDown(self) -> None:
        """Clean up server state and distributed."""
        dist_server_module._dist_server = None
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        super().tearDown()

    @patch(
        "gigl.distributed.graph_store.remote_dist_dataset.async_request_server",
        side_effect=_mock_async_request_server,
    )
    def test_compute_degree_tensor_pushes_to_server(self, mock_async_request):
        """Test that RemoteDistDataset.compute_degree_tensor() pushes degrees to server."""
        edge_index = DEFAULT_HOMOGENEOUS_EDGE_INDEX
        num_nodes = int(edge_index.max().item() + 1)

        self.assertIsNone(self._server.dataset.degree_tensor)

        cluster_info = _create_mock_graph_store_info()
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)
        remote_dataset.compute_degree_tensor()

        server_degrees = self._server.dataset.degree_tensor
        self.assertIsNotNone(server_degrees)
        assert isinstance(server_degrees, torch.Tensor)
        expected = _compute_expected_degrees_from_edge_index(edge_index, num_nodes)
        self.assert_tensor_equality(server_degrees, expected)


class TestRemoteHeterogeneousDegreeComputation(TestCase):
    """Tests for remote degree computation on heterogeneous graphs.

    Uses a real DistServer and RemoteDistDataset, with RPC calls mocked to route
    through the local server. Each edge type's degrees are computed independently.

    The architecture uses all-reduce among storage servers.
    """

    def setUp(self) -> None:
        """Set up a DistServer with a heterogeneous graph and initialize distributed."""
        super().setUp()
        create_test_process_group()
        dataset = create_heterogeneous_dataset(
            edge_indices=DEFAULT_HETEROGENEOUS_EDGE_INDICES
        )
        self._server = DistServer(dataset)
        dist_server_module._dist_server = self._server

    def tearDown(self) -> None:
        """Clean up server state and distributed."""
        dist_server_module._dist_server = None
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        super().tearDown()

    @patch(
        "gigl.distributed.graph_store.remote_dist_dataset.async_request_server",
        side_effect=_mock_async_request_server,
    )
    def test_compute_degree_tensor_pushes_to_server(self, mock_async_request):
        """Test that RemoteDistDataset.compute_degree_tensor() pushes degrees to server."""
        edge_indices = DEFAULT_HETEROGENEOUS_EDGE_INDICES

        self.assertIsNone(self._server.dataset.degree_tensor)

        cluster_info = _create_mock_graph_store_info()
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)
        remote_dataset.compute_degree_tensor()

        server_degrees = self._server.dataset.degree_tensor
        self.assertIsNotNone(server_degrees)
        assert isinstance(server_degrees, dict)
        self.assertEqual(len(server_degrees), len(edge_indices))

        for edge_type, edge_index in edge_indices.items():
            num_nodes = int(edge_index[0].max().item() + 1)
            expected = _compute_expected_degrees_from_edge_index(edge_index, num_nodes)
            self.assertIn(edge_type, server_degrees)
            self.assert_tensor_equality(server_degrees[edge_type], expected)


if __name__ == "__main__":
    absltest.main()
