import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.graph_store import storage_utils
from gigl.distributed.graph_store.remote_dist_dataset import RemoteDistDataset
from gigl.env.distributed import GraphStoreInfo
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.types.graph import (
    FeatureInfo,
    FeaturePartitionData,
    GraphPartitionData,
    PartitionOutput,
)
from tests.test_assets.distributed.utils import (
    MockGraphStoreInfo,
    assert_tensor_equality,
    get_process_group_init_method,
)

_USER = NodeType("user")
_STORY = NodeType("story")
_USER_TO_STORY = EdgeType(_USER, Relation("to"), _STORY)
_STORY_TO_USER = EdgeType(_STORY, Relation("to"), _USER)

# TODO(kmonte): Add tests for get_node_ids with a split.


def _mock_request_server(server_rank, func, *args, **kwargs):
    """Mock request_server that directly calls the function."""
    return func(*args, **kwargs)


def _mock_async_request_server(server_rank, func, *args, **kwargs):
    """Mock async_request_server that returns a completed future with the function result."""
    future: torch.futures.Future = torch.futures.Future()
    future.set_result(func(*args, **kwargs))
    return future


def _create_mock_graph_store_info(
    num_storage_nodes: int = 1,
    num_compute_nodes: int = 1,
    compute_node_rank: int = 0,
    num_processes_per_compute: int = 1,
) -> GraphStoreInfo:
    """Create a mock GraphStoreInfo with placeholder values."""
    real_info = GraphStoreInfo(
        num_storage_nodes=num_storage_nodes,
        num_compute_nodes=num_compute_nodes,
        cluster_master_ip="127.0.0.1",
        storage_cluster_master_ip="127.0.0.1",
        compute_cluster_master_ip="127.0.0.1",
        cluster_master_port=12345,
        storage_cluster_master_port=12346,
        compute_cluster_master_port=12347,
        num_processes_per_compute=num_processes_per_compute,
        rpc_master_port=12348,
        rpc_wait_port=12349,
    )
    return MockGraphStoreInfo(real_info, compute_node_rank)


# TODO(kmonte): Move this to shared util.
def _create_homogeneous_dataset() -> DistDataset:
    """Helper method to create a homogeneous test dataset."""
    partition_output = PartitionOutput(
        node_partition_book=torch.zeros(10, dtype=torch.int64),
        edge_partition_book=torch.zeros(10, dtype=torch.int64),
        partitioned_edge_index=GraphPartitionData(
            edge_index=torch.tensor(
                [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]
            ),
            edge_ids=None,
        ),
        partitioned_node_features=FeaturePartitionData(
            feats=torch.zeros(10, 3), ids=torch.arange(10)
        ),
        partitioned_edge_features=None,
        partitioned_positive_labels=None,
        partitioned_negative_labels=None,
        partitioned_node_labels=None,
    )
    dataset = DistDataset(rank=0, world_size=1, edge_dir="out")
    dataset.build(partition_output=partition_output)
    return dataset


def _create_heterogeneous_dataset() -> DistDataset:
    """Helper method to create a heterogeneous test dataset."""
    partition_output = PartitionOutput(
        node_partition_book={
            _USER: torch.zeros(5, dtype=torch.int64),
            _STORY: torch.zeros(5, dtype=torch.int64),
        },
        edge_partition_book={
            _USER_TO_STORY: torch.zeros(5, dtype=torch.int64),
            _STORY_TO_USER: torch.zeros(5, dtype=torch.int64),
        },
        partitioned_edge_index={
            _USER_TO_STORY: GraphPartitionData(
                edge_index=torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]),
                edge_ids=None,
            ),
            _STORY_TO_USER: GraphPartitionData(
                edge_index=torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]),
                edge_ids=None,
            ),
        },
        partitioned_node_features={
            _USER: FeaturePartitionData(feats=torch.zeros(5, 2), ids=torch.arange(5)),
            _STORY: FeaturePartitionData(feats=torch.zeros(5, 2), ids=torch.arange(5)),
        },
        partitioned_edge_features=None,
        partitioned_positive_labels=None,
        partitioned_negative_labels=None,
        partitioned_node_labels=None,
    )
    dataset = DistDataset(rank=0, world_size=1, edge_dir="in")
    dataset.build(partition_output=partition_output)
    return dataset


class TestRemoteDistDataset(unittest.TestCase):
    def setUp(self) -> None:
        storage_utils._dataset = None
        storage_utils.register_dataset(_create_homogeneous_dataset())

    def tearDown(self) -> None:
        storage_utils._dataset = None
        if dist.is_initialized():
            dist.destroy_process_group()

    @patch(
        "gigl.distributed.graph_store.remote_dist_dataset.request_server",
        side_effect=_mock_request_server,
    )
    def test_get_node_feature_info(self, mock_request):
        cluster_info = _create_mock_graph_store_info()
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        result = remote_dataset.get_node_feature_info()

        self.assertEqual(result, FeatureInfo(dim=3, dtype=torch.float32))

    @patch(
        "gigl.distributed.graph_store.remote_dist_dataset.request_server",
        side_effect=_mock_request_server,
    )
    def test_get_edge_feature_info(self, mock_request):
        cluster_info = _create_mock_graph_store_info()
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        result = remote_dataset.get_edge_feature_info()

        self.assertIsNone(result)

    @patch(
        "gigl.distributed.graph_store.remote_dist_dataset.request_server",
        side_effect=_mock_request_server,
    )
    def test_get_edge_dir(self, mock_request):
        cluster_info = _create_mock_graph_store_info()
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        result = remote_dataset.get_edge_dir()

        self.assertEqual(result, "out")

    @patch(
        "gigl.distributed.graph_store.remote_dist_dataset.request_server",
        side_effect=_mock_request_server,
    )
    def test_get_edge_types_homogeneous(self, mock_request):
        cluster_info = _create_mock_graph_store_info()
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        result = remote_dataset.get_edge_types()

        self.assertIsNone(result)

    @patch(
        "gigl.distributed.graph_store.remote_dist_dataset.request_server",
        side_effect=_mock_request_server,
    )
    def test_get_edge_types_heterogeneous(self, mock_request):
        storage_utils._dataset = None
        storage_utils.register_dataset(_create_heterogeneous_dataset())

        cluster_info = _create_mock_graph_store_info()
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        result = remote_dataset.get_edge_types()

        self.assertEqual(result, [_USER_TO_STORY, _STORY_TO_USER])

    def test_init_rejects_non_dict_proxy_for_mp_sharing_dict(self):
        cluster_info = _create_mock_graph_store_info()

        with self.assertRaises(ValueError):
            RemoteDistDataset(
                cluster_info=cluster_info,
                local_rank=0,
                mp_sharing_dict=dict(),  # Regular dict should fail
            )

    def test_cluster_info_property(self):
        cluster_info = _create_mock_graph_store_info(
            num_storage_nodes=3, num_compute_nodes=2
        )
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        self.assertEqual(remote_dataset.cluster_info.num_storage_nodes, 3)
        self.assertEqual(remote_dataset.cluster_info.num_compute_nodes, 2)

    @patch(
        "gigl.distributed.graph_store.remote_dist_dataset.async_request_server",
        side_effect=_mock_async_request_server,
    )
    def test_get_node_ids_basic(self, mock_async_request):
        """Test get_node_ids returns node ids from all storage nodes."""
        cluster_info = _create_mock_graph_store_info(num_storage_nodes=1)
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        result = remote_dataset.get_node_ids()

        self.assertIn(0, result)
        assert_tensor_equality(result[0], torch.arange(10))

    @patch(
        "gigl.distributed.graph_store.remote_dist_dataset.async_request_server",
        side_effect=_mock_async_request_server,
    )
    def test_get_node_ids_with_rank_world_size(self, mock_async_request):
        """Test get_node_ids with rank/world_size for sharding."""
        cluster_info = _create_mock_graph_store_info(num_storage_nodes=1)
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        # Request first half of nodes (rank 0 of 2)
        result = remote_dataset.get_node_ids(rank=0, world_size=2)
        assert_tensor_equality(result[0], torch.arange(5))

        # Request second half of nodes (rank 1 of 2)
        result = remote_dataset.get_node_ids(rank=1, world_size=2)
        assert_tensor_equality(result[0], torch.arange(5, 10))


class TestRemoteDistDatasetHeterogeneous(unittest.TestCase):
    def setUp(self) -> None:
        storage_utils._dataset = None
        storage_utils.register_dataset(_create_heterogeneous_dataset())

    def tearDown(self) -> None:
        storage_utils._dataset = None
        if dist.is_initialized():
            dist.destroy_process_group()

    @patch(
        "gigl.distributed.graph_store.remote_dist_dataset.request_server",
        side_effect=_mock_request_server,
    )
    def test_get_node_feature_info_heterogeneous(self, mock_request):
        cluster_info = _create_mock_graph_store_info()
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        result = remote_dataset.get_node_feature_info()

        expected = {
            _USER: FeatureInfo(dim=2, dtype=torch.float32),
            _STORY: FeatureInfo(dim=2, dtype=torch.float32),
        }
        self.assertEqual(result, expected)

    @patch(
        "gigl.distributed.graph_store.remote_dist_dataset.request_server",
        side_effect=_mock_request_server,
    )
    def test_get_edge_dir_heterogeneous(self, mock_request):
        cluster_info = _create_mock_graph_store_info()
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        result = remote_dataset.get_edge_dir()

        self.assertEqual(result, "in")

    @patch(
        "gigl.distributed.graph_store.remote_dist_dataset.async_request_server",
        side_effect=_mock_async_request_server,
    )
    def test_get_node_ids_with_node_type(self, mock_async_request):
        """Test get_node_ids with node_type for heterogeneous graphs."""
        cluster_info = _create_mock_graph_store_info(num_storage_nodes=1)
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        # Get user nodes
        result = remote_dataset.get_node_ids(node_type=_USER)
        assert_tensor_equality(result[0], torch.arange(5))

        # Get story nodes
        result = remote_dataset.get_node_ids(node_type=_STORY)
        assert_tensor_equality(result[0], torch.arange(5))

    @patch(
        "gigl.distributed.graph_store.remote_dist_dataset.async_request_server",
        side_effect=_mock_async_request_server,
    )
    def test_get_node_ids_with_node_type_and_sharding(self, mock_async_request):
        """Test get_node_ids with node_type and rank/world_size."""
        cluster_info = _create_mock_graph_store_info(num_storage_nodes=1)
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        # Get first half of user nodes
        result = remote_dataset.get_node_ids(rank=0, world_size=2, node_type=_USER)
        assert_tensor_equality(result[0], torch.arange(2))

        # Get second half of user nodes
        result = remote_dataset.get_node_ids(rank=1, world_size=2, node_type=_USER)
        assert_tensor_equality(result[0], torch.arange(2, 5))


def _test_get_free_ports_on_storage_cluster(
    rank: int,
    world_size: int,
    init_process_group_init_method: str,
    num_ports: int,
    mock_ports: list[int],
):
    """Test function to run in spawned processes."""
    dist.init_process_group(
        backend="gloo",
        init_method=init_process_group_init_method,
        world_size=world_size,
        rank=rank,
    )
    try:
        cluster_info = _create_mock_graph_store_info(
            num_compute_nodes=world_size,
            num_processes_per_compute=1,
            compute_node_rank=rank,
        )
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        with patch(
            "gigl.distributed.graph_store.remote_dist_dataset.request_server",
            return_value=mock_ports,
        ):
            ports = remote_dataset.get_free_ports_on_storage_cluster(num_ports)

        assert len(ports) == num_ports, f"Expected {num_ports} ports, got {len(ports)}"

        # Verify all ranks get the same ports via all_gather
        gathered_ports = [None] * world_size
        dist.all_gather_object(gathered_ports, ports)

        for i, rank_ports in enumerate(gathered_ports):
            assert (
                rank_ports == mock_ports
            ), f"Rank {i} got {rank_ports}, expected {mock_ports}"
    finally:
        dist.destroy_process_group()


class TestGetFreePortsOnStorageCluster(unittest.TestCase):
    def setUp(self) -> None:
        storage_utils._dataset = None
        storage_utils.register_dataset(_create_homogeneous_dataset())

    def tearDown(self) -> None:
        storage_utils._dataset = None
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_get_free_ports_on_storage_cluster_distributed(self):
        """Test that free ports are correctly broadcast across all ranks."""
        init_method = get_process_group_init_method()
        world_size = 2
        num_ports = 3
        mock_ports = [10000, 10001, 10002]

        mp.spawn(
            fn=_test_get_free_ports_on_storage_cluster,
            args=(world_size, init_method, num_ports, mock_ports),
            nprocs=world_size,
        )

    def test_get_free_ports_fails_without_process_group(self):
        """Test that get_free_ports_on_storage_cluster raises when dist not initialized."""
        cluster_info = _create_mock_graph_store_info()
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        with self.assertRaises(ValueError):
            remote_dataset.get_free_ports_on_storage_cluster(num_ports=1)


if __name__ == "__main__":
    unittest.main()
