import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from gigl.distributed.graph_store import storage_utils
from gigl.distributed.graph_store.remote_dist_dataset import RemoteDistDataset
from gigl.env.distributed import GraphStoreInfo
from gigl.types.graph import FeatureInfo
from tests.test_assets.distributed.test_dataset import (
    DEFAULT_HETEROGENEOUS_EDGE_INDICES,
    DEFAULT_HOMOGENEOUS_EDGE_INDEX,
    STORY,
    STORY_TO_USER,
    USER,
    USER_TO_STORY,
    create_heterogeneous_dataset,
    create_heterogeneous_dataset_for_ablp,
    create_homogeneous_dataset,
)
from tests.test_assets.distributed.utils import (
    MockGraphStoreInfo,
    assert_tensor_equality,
    create_test_process_group,
    get_process_group_init_method,
)


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


class TestRemoteDistDataset(unittest.TestCase):
    def setUp(self) -> None:
        storage_utils._dataset = None
        # 10 nodes in DEFAULT_HOMOGENEOUS_EDGE_INDEX ring graph
        node_features = torch.zeros(10, 3)
        storage_utils.register_dataset(
            create_homogeneous_dataset(
                edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX,
                node_features=node_features,
            )
        )

    def tearDown(self) -> None:
        storage_utils._dataset = None
        if dist.is_initialized():
            dist.destroy_process_group()

    @patch(
        "gigl.distributed.graph_store.remote_dist_dataset.request_server",
        side_effect=_mock_request_server,
    )
    def test_graph_metadata_getters_homogeneous(self, mock_request):
        """Test get_node_feature_info, get_edge_feature_info, get_edge_dir, get_edge_types for homogeneous graphs."""
        cluster_info = _create_mock_graph_store_info()
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        self.assertEqual(
            remote_dataset.get_node_feature_info(),
            FeatureInfo(dim=3, dtype=torch.float32),
        )
        self.assertIsNone(remote_dataset.get_edge_feature_info())
        self.assertEqual(remote_dataset.get_edge_dir(), "out")
        self.assertIsNone(remote_dataset.get_edge_types())

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
    def test_get_node_ids(self, mock_async_request):
        """Test get_node_ids returns node ids, with optional sharding via rank/world_size."""
        cluster_info = _create_mock_graph_store_info(num_storage_nodes=1)
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        # Basic: all nodes
        result = remote_dataset.get_node_ids()
        self.assertIn(0, result)
        assert_tensor_equality(result[0], torch.arange(10))

        # With sharding: first half (rank 0 of 2)
        result = remote_dataset.get_node_ids(rank=0, world_size=2)
        assert_tensor_equality(result[0], torch.arange(5))

        # With sharding: second half (rank 1 of 2)
        result = remote_dataset.get_node_ids(rank=1, world_size=2)
        assert_tensor_equality(result[0], torch.arange(5, 10))


class TestRemoteDistDatasetHeterogeneous(unittest.TestCase):
    def setUp(self) -> None:
        storage_utils._dataset = None
        # 5 users, 5 stories in DEFAULT_HETEROGENEOUS_EDGE_INDICES
        node_features = {
            USER: torch.zeros(5, 2),
            STORY: torch.zeros(5, 2),
        }
        storage_utils.register_dataset(
            create_heterogeneous_dataset(
                edge_indices=DEFAULT_HETEROGENEOUS_EDGE_INDICES,
                node_features=node_features,
            )
        )

    def tearDown(self) -> None:
        storage_utils._dataset = None
        if dist.is_initialized():
            dist.destroy_process_group()

    @patch(
        "gigl.distributed.graph_store.remote_dist_dataset.request_server",
        side_effect=_mock_request_server,
    )
    def test_graph_metadata_getters_heterogeneous(self, mock_request):
        """Test get_node_feature_info, get_edge_dir, get_edge_types for heterogeneous graphs."""
        cluster_info = _create_mock_graph_store_info()
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        self.assertEqual(
            remote_dataset.get_node_feature_info(),
            {
                USER: FeatureInfo(dim=2, dtype=torch.float32),
                STORY: FeatureInfo(dim=2, dtype=torch.float32),
            },
        )
        self.assertEqual(remote_dataset.get_edge_dir(), "out")
        self.assertEqual(
            remote_dataset.get_edge_types(), [USER_TO_STORY, STORY_TO_USER]
        )

    @patch(
        "gigl.distributed.graph_store.remote_dist_dataset.async_request_server",
        side_effect=_mock_async_request_server,
    )
    def test_get_node_ids_with_node_type(self, mock_async_request):
        """Test get_node_ids with node_type for heterogeneous graphs, with optional sharding."""
        cluster_info = _create_mock_graph_store_info(num_storage_nodes=1)
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        # Get user nodes
        result = remote_dataset.get_node_ids(node_type=USER)
        assert_tensor_equality(result[0], torch.arange(5))

        # Get story nodes
        result = remote_dataset.get_node_ids(node_type=STORY)
        assert_tensor_equality(result[0], torch.arange(5))

        # With sharding: first half of user nodes (rank 0 of 2)
        result = remote_dataset.get_node_ids(rank=0, world_size=2, node_type=USER)
        assert_tensor_equality(result[0], torch.arange(2))

        # With sharding: second half of user nodes (rank 1 of 2)
        result = remote_dataset.get_node_ids(rank=1, world_size=2, node_type=USER)
        assert_tensor_equality(result[0], torch.arange(2, 5))


class TestRemoteDistDatasetWithSplits(unittest.TestCase):
    """Tests for get_node_ids with train/val/test splits."""

    def setUp(self) -> None:
        storage_utils._dataset = None

    def tearDown(self) -> None:
        storage_utils._dataset = None
        if dist.is_initialized():
            dist.destroy_process_group()

    def _create_and_register_dataset_with_splits(self) -> None:
        """Create and register a dataset with train/val/test splits."""
        create_test_process_group()

        positive_labels = {
            0: [0, 1],
            1: [1, 2],
            2: [2, 3],
            3: [3, 4],
            4: [4, 0],
        }
        negative_labels = {
            0: [2],
            1: [3],
            2: [4],
            3: [0],
            4: [1],
        }

        dataset = create_heterogeneous_dataset_for_ablp(
            positive_labels=positive_labels,
            negative_labels=negative_labels,
            train_node_ids=[0, 1, 2],
            val_node_ids=[3],
            test_node_ids=[4],
            edge_indices=DEFAULT_HETEROGENEOUS_EDGE_INDICES,
        )
        storage_utils.register_dataset(dataset)

    @patch(
        "gigl.distributed.graph_store.remote_dist_dataset.async_request_server",
        side_effect=_mock_async_request_server,
    )
    def test_get_node_ids_with_splits(self, mock_async_request):
        """Test get_node_ids with train/val/test splits and optional sharding."""
        self._create_and_register_dataset_with_splits()

        cluster_info = _create_mock_graph_store_info(num_storage_nodes=1)
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        # Test each split returns correct nodes
        assert_tensor_equality(
            remote_dataset.get_node_ids(node_type=USER, split="train")[0],
            torch.tensor([0, 1, 2]),
        )
        assert_tensor_equality(
            remote_dataset.get_node_ids(node_type=USER, split="val")[0],
            torch.tensor([3]),
        )
        assert_tensor_equality(
            remote_dataset.get_node_ids(node_type=USER, split="test")[0],
            torch.tensor([4]),
        )

        # No split returns all nodes
        assert_tensor_equality(
            remote_dataset.get_node_ids(node_type=USER, split=None)[0],
            torch.arange(5),
        )

        # With sharding: train split [0, 1, 2] across 2 ranks
        assert_tensor_equality(
            remote_dataset.get_node_ids(
                rank=0, world_size=2, node_type=USER, split="train"
            )[0],
            torch.tensor([0]),
        )
        assert_tensor_equality(
            remote_dataset.get_node_ids(
                rank=1, world_size=2, node_type=USER, split="train"
            )[0],
            torch.tensor([1, 2]),
        )

    @patch(
        "gigl.distributed.graph_store.remote_dist_dataset.async_request_server",
        side_effect=_mock_async_request_server,
    )
    def test_get_ablp_input(self, mock_async_request):
        """Test get_ablp_input with train/val/test splits."""
        self._create_and_register_dataset_with_splits()

        cluster_info = _create_mock_graph_store_info(num_storage_nodes=1)
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        # Train split: nodes [0, 1, 2]
        result = remote_dataset.get_ablp_input(
            split="train", node_type=USER, supervision_edge_type=USER_TO_STORY
        )
        self.assertIn(0, result)
        anchors, positive_labels, negative_labels = result[0]
        assert_tensor_equality(anchors, torch.tensor([0, 1, 2]))
        assert_tensor_equality(positive_labels, torch.tensor([[0, 1], [1, 2], [2, 3]]))
        assert negative_labels is not None
        assert_tensor_equality(negative_labels, torch.tensor([[2], [3], [4]]))

        # Val split: node [3]
        result = remote_dataset.get_ablp_input(
            split="val", node_type=USER, supervision_edge_type=USER_TO_STORY
        )
        anchors, positive_labels, negative_labels = result[0]
        assert_tensor_equality(anchors, torch.tensor([3]))
        assert_tensor_equality(positive_labels, torch.tensor([[3, 4]]))
        assert negative_labels is not None
        assert_tensor_equality(negative_labels, torch.tensor([[0]]))

        # Test split: node [4]
        # Note: Labels are stored in CSR format which sorts by destination indices,
        # so [4, 0] from the input becomes [0, 4] in the stored format.
        result = remote_dataset.get_ablp_input(
            split="test", node_type=USER, supervision_edge_type=USER_TO_STORY
        )
        anchors, positive_labels, negative_labels = result[0]
        assert_tensor_equality(anchors, torch.tensor([4]))
        assert_tensor_equality(positive_labels, torch.tensor([[0, 4]]))
        assert negative_labels is not None
        assert_tensor_equality(negative_labels, torch.tensor([[1]]))

    @patch(
        "gigl.distributed.graph_store.remote_dist_dataset.async_request_server",
        side_effect=_mock_async_request_server,
    )
    def test_get_ablp_input_with_sharding(self, mock_async_request):
        """Test get_ablp_input with sharding across compute nodes."""
        self._create_and_register_dataset_with_splits()

        cluster_info = _create_mock_graph_store_info(num_storage_nodes=1)
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        # With sharding: train split [0, 1, 2] across 2 ranks
        result_rank0 = remote_dataset.get_ablp_input(
            split="train",
            rank=0,
            world_size=2,
            node_type=USER,
            supervision_edge_type=USER_TO_STORY,
        )
        anchors_0, positive_labels_0, negative_labels_0 = result_rank0[0]

        # Rank 0 should get node 0
        assert_tensor_equality(anchors_0, torch.tensor([0]))
        assert_tensor_equality(positive_labels_0, torch.tensor([[0, 1]]))
        assert negative_labels_0 is not None
        assert_tensor_equality(negative_labels_0, torch.tensor([[2]]))

        result_rank1 = remote_dataset.get_ablp_input(
            split="train",
            rank=1,
            world_size=2,
            node_type=USER,
            supervision_edge_type=USER_TO_STORY,
        )
        anchors_1, positive_labels_1, negative_labels_1 = result_rank1[0]

        # Rank 1 should get nodes 1, 2
        assert_tensor_equality(anchors_1, torch.tensor([1, 2]))
        assert_tensor_equality(positive_labels_1, torch.tensor([[1, 2], [2, 3]]))
        assert negative_labels_1 is not None
        assert_tensor_equality(negative_labels_1, torch.tensor([[3], [4]]))


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
        storage_utils.register_dataset(
            create_homogeneous_dataset(edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX)
        )

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
