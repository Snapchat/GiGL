from typing import Optional
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from absl.testing import absltest

from gigl.distributed.graph_store.dist_server import DistServer
from gigl.distributed.graph_store.remote_dist_dataset import RemoteDistDataset
from gigl.env.distributed import GraphStoreInfo
from gigl.types.graph import FeatureInfo
from gigl.utils.sampling import ABLPInputNodes
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
    create_test_process_group,
    get_process_group_init_method,
)
from tests.test_assets.test_case import TestCase

# Module-level test server instance used by mock functions
_test_server: Optional[DistServer] = None


def _mock_request_server(server_rank, func, *args, **kwargs):
    """Mock request_server that calls the method on the test DistServer instance."""
    if _test_server is None:
        raise RuntimeError("Test server not initialized")
    return func(_test_server, *args, **kwargs)


def _mock_async_request_server(server_rank, func, *args, **kwargs):
    """Mock async_request_server that returns a completed future with the method result."""
    if _test_server is None:
        raise RuntimeError("Test server not initialized")
    future: torch.futures.Future = torch.futures.Future()
    future.set_result(func(_test_server, *args, **kwargs))
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


class TestRemoteDistDataset(TestCase):
    def setUp(self) -> None:
        global _test_server
        # 10 nodes in DEFAULT_HOMOGENEOUS_EDGE_INDEX ring graph
        node_features = torch.zeros(10, 3)
        dataset = create_homogeneous_dataset(
            edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX,
            node_features=node_features,
        )
        _test_server = DistServer(dataset)

    def tearDown(self) -> None:
        global _test_server
        _test_server = None
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
        self.assert_tensor_equality(result[0], torch.arange(10))

        # With sharding: first half (rank 0 of 2)
        result = remote_dataset.get_node_ids(rank=0, world_size=2)
        self.assert_tensor_equality(result[0], torch.arange(5))

        # With sharding: second half (rank 1 of 2)
        result = remote_dataset.get_node_ids(rank=1, world_size=2)
        self.assert_tensor_equality(result[0], torch.arange(5, 10))


class TestRemoteDistDatasetHeterogeneous(TestCase):
    def setUp(self) -> None:
        global _test_server
        # 5 users, 5 stories in DEFAULT_HETEROGENEOUS_EDGE_INDICES
        node_features = {
            USER: torch.zeros(5, 2),
            STORY: torch.zeros(5, 2),
        }
        dataset = create_heterogeneous_dataset(
            edge_indices=DEFAULT_HETEROGENEOUS_EDGE_INDICES,
            node_features=node_features,
        )
        _test_server = DistServer(dataset)

    def tearDown(self) -> None:
        global _test_server
        _test_server = None
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
        self.assert_tensor_equality(result[0], torch.arange(5))

        # Get story nodes
        result = remote_dataset.get_node_ids(node_type=STORY)
        self.assert_tensor_equality(result[0], torch.arange(5))

        # With sharding: first half of user nodes (rank 0 of 2)
        result = remote_dataset.get_node_ids(rank=0, world_size=2, node_type=USER)
        self.assert_tensor_equality(result[0], torch.arange(2))

        # With sharding: second half of user nodes (rank 1 of 2)
        result = remote_dataset.get_node_ids(rank=1, world_size=2, node_type=USER)
        self.assert_tensor_equality(result[0], torch.arange(2, 5))


class TestRemoteDistDatasetWithSplits(TestCase):
    """Tests for get_node_ids with train/val/test splits."""

    def tearDown(self) -> None:
        global _test_server
        _test_server = None
        if dist.is_initialized():
            dist.destroy_process_group()

    def _create_server_with_splits(self) -> None:
        """Create a DistServer with a dataset that has train/val/test splits."""
        global _test_server
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
        _test_server = DistServer(dataset)

    @patch(
        "gigl.distributed.graph_store.remote_dist_dataset.async_request_server",
        side_effect=_mock_async_request_server,
    )
    def test_get_node_ids_with_splits(self, mock_async_request):
        """Test get_node_ids with train/val/test splits and optional sharding."""
        self._create_server_with_splits()

        cluster_info = _create_mock_graph_store_info(num_storage_nodes=1)
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        # Test each split returns correct nodes
        self.assert_tensor_equality(
            remote_dataset.get_node_ids(node_type=USER, split="train")[0],
            torch.tensor([0, 1, 2]),
        )
        self.assert_tensor_equality(
            remote_dataset.get_node_ids(node_type=USER, split="val")[0],
            torch.tensor([3]),
        )
        self.assert_tensor_equality(
            remote_dataset.get_node_ids(node_type=USER, split="test")[0],
            torch.tensor([4]),
        )

        # No split returns all nodes
        self.assert_tensor_equality(
            remote_dataset.get_node_ids(node_type=USER, split=None)[0],
            torch.arange(5),
        )

        # With sharding: train split [0, 1, 2] across 2 ranks
        self.assert_tensor_equality(
            remote_dataset.get_node_ids(
                rank=0, world_size=2, node_type=USER, split="train"
            )[0],
            torch.tensor([0]),
        )
        self.assert_tensor_equality(
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
        self._create_server_with_splits()

        cluster_info = _create_mock_graph_store_info(num_storage_nodes=1)
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        from gigl.types.graph import (
            message_passing_to_negative_label,
            message_passing_to_positive_label,
        )

        positive_label_edge_type = message_passing_to_positive_label(USER_TO_STORY)
        negative_label_edge_type = message_passing_to_negative_label(USER_TO_STORY)

        # Train split: nodes [0, 1, 2]
        result = remote_dataset.get_ablp_input(
            split="train", node_type=USER, supervision_edge_type=USER_TO_STORY
        )
        self.assertIn(0, result)
        ablp_input = result[0]
        self.assertIsInstance(ablp_input, ABLPInputNodes)
        self.assertEqual(ablp_input.anchor_node_type, USER)
        self.assert_tensor_equality(ablp_input.anchor_nodes, torch.tensor([0, 1, 2]))
        self.assertIn(positive_label_edge_type, ablp_input.positive_labels)
        self.assert_tensor_equality(
            ablp_input.positive_labels[positive_label_edge_type],
            torch.tensor([[0, 1], [1, 2], [2, 3]]),
        )
        assert ablp_input.negative_labels is not None
        self.assertIn(negative_label_edge_type, ablp_input.negative_labels)
        self.assert_tensor_equality(
            ablp_input.negative_labels[negative_label_edge_type],
            torch.tensor([[2], [3], [4]]),
        )

        # Val split: node [3]
        result = remote_dataset.get_ablp_input(
            split="val", node_type=USER, supervision_edge_type=USER_TO_STORY
        )
        ablp_input = result[0]
        self.assert_tensor_equality(ablp_input.anchor_nodes, torch.tensor([3]))
        self.assert_tensor_equality(
            ablp_input.positive_labels[positive_label_edge_type],
            torch.tensor([[3, 4]]),
        )
        assert ablp_input.negative_labels is not None
        self.assert_tensor_equality(
            ablp_input.negative_labels[negative_label_edge_type],
            torch.tensor([[0]]),
        )

        # Test split: node [4]
        # Note: Labels are stored in CSR format which sorts by destination indices,
        # so [4, 0] from the input becomes [0, 4] in the stored format.
        result = remote_dataset.get_ablp_input(
            split="test", node_type=USER, supervision_edge_type=USER_TO_STORY
        )
        ablp_input = result[0]
        self.assert_tensor_equality(ablp_input.anchor_nodes, torch.tensor([4]))
        self.assert_tensor_equality(
            ablp_input.positive_labels[positive_label_edge_type],
            torch.tensor([[0, 4]]),
        )
        assert ablp_input.negative_labels is not None
        self.assert_tensor_equality(
            ablp_input.negative_labels[negative_label_edge_type],
            torch.tensor([[1]]),
        )

    @patch(
        "gigl.distributed.graph_store.remote_dist_dataset.async_request_server",
        side_effect=_mock_async_request_server,
    )
    def test_get_ablp_input_with_sharding(self, mock_async_request):
        """Test get_ablp_input with sharding across compute nodes."""
        self._create_server_with_splits()

        cluster_info = _create_mock_graph_store_info(num_storage_nodes=1)
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        from gigl.types.graph import (
            message_passing_to_negative_label,
            message_passing_to_positive_label,
        )

        positive_label_edge_type = message_passing_to_positive_label(USER_TO_STORY)
        negative_label_edge_type = message_passing_to_negative_label(USER_TO_STORY)

        # With sharding: train split [0, 1, 2] across 2 ranks
        result_rank0 = remote_dataset.get_ablp_input(
            split="train",
            rank=0,
            world_size=2,
            node_type=USER,
            supervision_edge_type=USER_TO_STORY,
        )
        ablp_0 = result_rank0[0]
        self.assertIsInstance(ablp_0, ABLPInputNodes)
        self.assertEqual(ablp_0.anchor_node_type, USER)

        # Rank 0 should get node 0
        self.assert_tensor_equality(ablp_0.anchor_nodes, torch.tensor([0]))
        self.assert_tensor_equality(
            ablp_0.positive_labels[positive_label_edge_type],
            torch.tensor([[0, 1]]),
        )
        assert ablp_0.negative_labels is not None
        self.assert_tensor_equality(
            ablp_0.negative_labels[negative_label_edge_type],
            torch.tensor([[2]]),
        )

        result_rank1 = remote_dataset.get_ablp_input(
            split="train",
            rank=1,
            world_size=2,
            node_type=USER,
            supervision_edge_type=USER_TO_STORY,
        )
        ablp_1 = result_rank1[0]
        self.assertIsInstance(ablp_1, ABLPInputNodes)

        # Rank 1 should get nodes 1, 2
        self.assert_tensor_equality(ablp_1.anchor_nodes, torch.tensor([1, 2]))
        self.assert_tensor_equality(
            ablp_1.positive_labels[positive_label_edge_type],
            torch.tensor([[1, 2], [2, 3]]),
        )
        assert ablp_1.negative_labels is not None
        self.assert_tensor_equality(
            ablp_1.negative_labels[negative_label_edge_type],
            torch.tensor([[3], [4]]),
        )


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


class TestGetFreePortsOnStorageCluster(TestCase):
    def setUp(self) -> None:
        global _test_server
        dataset = create_homogeneous_dataset(edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX)
        _test_server = DistServer(dataset)

    def tearDown(self) -> None:
        global _test_server
        _test_server = None
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
    absltest.main()
