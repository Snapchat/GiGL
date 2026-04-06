from unittest.mock import MagicMock, patch

import torch
from absl.testing import absltest
from graphlearn_torch.sampler import SamplingConfig, SamplingType

from gigl.distributed.graph_store import dist_server
from gigl.distributed.graph_store.messages import (
    FetchABLPInputRequest,
    FetchNodesRequest,
    InitSamplingBackendRequest,
    RegisterBackendRequest,
)
from gigl.src.common.types.graph_data import Relation
from tests.test_assets.distributed.test_dataset import (
    DEFAULT_HETEROGENEOUS_EDGE_INDICES,
    DEFAULT_HOMOGENEOUS_EDGE_INDEX,
    STORY,
    USER,
    USER_TO_STORY,
    create_heterogeneous_dataset,
    create_heterogeneous_dataset_for_ablp,
    create_homogeneous_dataset,
)
from tests.test_assets.distributed.utils import create_test_process_group
from tests.test_assets.test_case import TestCase


class TestRemoteDataset(TestCase):
    def setUp(self) -> None:
        """Reset the global dataset before each test."""
        dist_server._dist_server = None

    def tearDown(self) -> None:
        """Clean up after each test."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def test_get_node_feature_info_with_heterogeneous_dataset(self) -> None:
        """Test get_node_feature_info with a heterogeneous dataset."""
        dataset = create_heterogeneous_dataset(
            edge_indices=DEFAULT_HETEROGENEOUS_EDGE_INDICES,
        )
        server = dist_server.DistServer(dataset)

        node_feature_info = server.get_node_feature_info()

        # Verify it returns the correct feature info
        self.assertIsNone(node_feature_info)

    def test_get_node_feature_info_with_homogeneous_dataset(self) -> None:
        """Test get_node_feature_info with a homogeneous dataset."""
        dataset = create_homogeneous_dataset(
            edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX,
        )
        server = dist_server.DistServer(dataset)

        node_feature_info = server.get_node_feature_info()

        # Verify it returns the correct feature info
        self.assertIsNone(node_feature_info)

    def test_get_edge_feature_info_with_heterogeneous_dataset(self) -> None:
        """Test get_edge_feature_info with a heterogeneous dataset."""
        dataset = create_heterogeneous_dataset(
            edge_indices=DEFAULT_HETEROGENEOUS_EDGE_INDICES,
        )
        server = dist_server.DistServer(dataset)

        edge_feature_info = server.get_edge_feature_info()

        # For this test dataset, edge features are None
        self.assertIsNone(edge_feature_info)

    def test_get_edge_feature_info_with_homogeneous_dataset(self) -> None:
        """Test get_edge_feature_info with a homogeneous dataset."""
        dataset = create_homogeneous_dataset(
            edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX,
        )
        server = dist_server.DistServer(dataset)

        edge_feature_info = server.get_edge_feature_info()

        # For this test dataset, edge features are None
        self.assertIsNone(edge_feature_info)

    def test_get_node_ids_with_homogeneous_dataset(self) -> None:
        """Test get_node_ids with a homogeneous dataset."""
        dataset = create_homogeneous_dataset(
            edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX,
        )
        server = dist_server.DistServer(dataset)

        # Test with world_size=1, rank=0 (should get all nodes)
        node_ids = server.get_node_ids(
            FetchNodesRequest(rank=0, world_size=1, node_type=None)
        )
        self.assertIsInstance(node_ids, torch.Tensor)
        self.assertEqual(node_ids.shape[0], 10)
        self.assert_tensor_equality(node_ids, torch.arange(10))

    def test_get_node_ids_with_heterogeneous_dataset(self) -> None:
        """Test get_node_ids with a heterogeneous dataset."""
        dataset = create_heterogeneous_dataset(
            edge_indices=DEFAULT_HETEROGENEOUS_EDGE_INDICES,
        )
        server = dist_server.DistServer(dataset)

        # Test with USER node type
        user_node_ids = server.get_node_ids(
            FetchNodesRequest(rank=0, world_size=1, node_type=USER)
        )
        self.assertIsInstance(user_node_ids, torch.Tensor)
        self.assertEqual(user_node_ids.shape[0], 5)
        self.assert_tensor_equality(user_node_ids, torch.arange(5))

        # Test with STORY node type
        story_node_ids = server.get_node_ids(
            FetchNodesRequest(rank=0, world_size=1, node_type=STORY)
        )
        self.assertIsInstance(story_node_ids, torch.Tensor)
        self.assertEqual(story_node_ids.shape[0], 5)
        self.assert_tensor_equality(story_node_ids, torch.arange(5))

    def test_get_node_ids_with_multiple_ranks(self) -> None:
        """Test get_node_ids with multiple ranks to verify sharding."""
        dataset = create_homogeneous_dataset(
            edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX,
        )
        server = dist_server.DistServer(dataset)

        # Test with world_size=2
        rank_0_nodes = server.get_node_ids(
            FetchNodesRequest(rank=0, world_size=2, node_type=None)
        )
        rank_1_nodes = server.get_node_ids(
            FetchNodesRequest(rank=1, world_size=2, node_type=None)
        )

        # Verify each rank gets different nodes
        self.assert_tensor_equality(rank_0_nodes, torch.arange(5))
        self.assert_tensor_equality(rank_1_nodes, torch.arange(5, 10))

        # Test with world_size=3 (uneven split)
        rank_0_nodes = server.get_node_ids(
            FetchNodesRequest(rank=0, world_size=3, node_type=None)
        )
        rank_1_nodes = server.get_node_ids(
            FetchNodesRequest(rank=1, world_size=3, node_type=None)
        )
        rank_2_nodes = server.get_node_ids(
            FetchNodesRequest(rank=2, world_size=3, node_type=None)
        )

        self.assert_tensor_equality(rank_0_nodes, torch.arange(3))
        self.assert_tensor_equality(rank_1_nodes, torch.arange(3, 6))
        self.assert_tensor_equality(rank_2_nodes, torch.arange(6, 10))

    def test_get_node_ids_rank_world_size_must_be_provided_together(self) -> None:
        """Test get_node_ids raises ValueError when rank/world_size not provided together."""
        dataset = create_homogeneous_dataset(
            edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX,
        )
        server = dist_server.DistServer(dataset)

        with self.assertRaises(ValueError):
            server.get_node_ids(FetchNodesRequest(rank=0, world_size=None))

        with self.assertRaises(ValueError):
            server.get_node_ids(FetchNodesRequest(rank=None, world_size=1))

    def test_get_node_ids_with_homogeneous_dataset_and_node_type(self) -> None:
        """Test get_node_ids with a homogeneous dataset and a node type raises error."""
        dataset = create_homogeneous_dataset(
            edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX,
        )
        server = dist_server.DistServer(dataset)
        with self.assertRaises(ValueError):
            server.get_node_ids(FetchNodesRequest(rank=0, world_size=1, node_type=USER))

    def test_get_node_ids_with_heterogeneous_dataset_and_no_node_type(
        self,
    ) -> None:
        """Test get_node_ids with a heterogeneous dataset and no node type raises error."""
        dataset = create_heterogeneous_dataset(
            edge_indices=DEFAULT_HETEROGENEOUS_EDGE_INDICES,
        )
        server = dist_server.DistServer(dataset)
        with self.assertRaises(ValueError):
            server.get_node_ids(FetchNodesRequest(rank=0, world_size=1, node_type=None))

    def test_get_node_ids_with_train_split(self) -> None:
        """Test get_node_ids returns only training nodes when split='train'."""
        create_test_process_group()

        positive_labels = {0: [0], 1: [1], 2: [2], 3: [3], 4: [4]}
        dataset = create_heterogeneous_dataset_for_ablp(
            positive_labels=positive_labels,
            train_node_ids=[0, 1, 2],
            val_node_ids=[3],
            test_node_ids=[4],
            edge_indices=DEFAULT_HETEROGENEOUS_EDGE_INDICES,
        )
        server = dist_server.DistServer(dataset)

        train_nodes = server.get_node_ids(
            FetchNodesRequest(node_type=USER, split="train")
        )
        self.assert_tensor_equality(train_nodes, torch.tensor([0, 1, 2]))

    def test_get_node_ids_with_val_split(self) -> None:
        """Test get_node_ids returns only validation nodes when split='val'."""
        create_test_process_group()

        positive_labels = {0: [0], 1: [1], 2: [2], 3: [3], 4: [4]}
        dataset = create_heterogeneous_dataset_for_ablp(
            positive_labels=positive_labels,
            train_node_ids=[0, 1, 2],
            val_node_ids=[3],
            test_node_ids=[4],
            edge_indices=DEFAULT_HETEROGENEOUS_EDGE_INDICES,
        )
        server = dist_server.DistServer(dataset)

        val_nodes = server.get_node_ids(FetchNodesRequest(node_type=USER, split="val"))
        self.assert_tensor_equality(val_nodes, torch.tensor([3]))

    def test_get_node_ids_with_test_split(self) -> None:
        """Test get_node_ids returns only test nodes when split='test'."""
        create_test_process_group()

        positive_labels = {0: [0], 1: [1], 2: [2], 3: [3], 4: [4]}
        dataset = create_heterogeneous_dataset_for_ablp(
            positive_labels=positive_labels,
            train_node_ids=[0, 1, 2],
            val_node_ids=[3],
            test_node_ids=[4],
            edge_indices=DEFAULT_HETEROGENEOUS_EDGE_INDICES,
        )
        server = dist_server.DistServer(dataset)

        test_nodes = server.get_node_ids(
            FetchNodesRequest(node_type=USER, split="test")
        )
        self.assert_tensor_equality(test_nodes, torch.tensor([4]))

    def test_get_node_ids_with_split_and_sharding(self) -> None:
        """Test get_node_ids with split and rank/world_size for sharding."""
        create_test_process_group()

        positive_labels = {0: [0], 1: [1], 2: [2], 3: [3], 4: [4]}
        dataset = create_heterogeneous_dataset_for_ablp(
            positive_labels=positive_labels,
            train_node_ids=[0, 1, 2],
            val_node_ids=[3],
            test_node_ids=[4],
            edge_indices=DEFAULT_HETEROGENEOUS_EDGE_INDICES,
        )
        server = dist_server.DistServer(dataset)

        # Train split has [0, 1, 2], shard across 2 ranks
        rank_0_nodes = server.get_node_ids(
            FetchNodesRequest(rank=0, world_size=2, node_type=USER, split="train")
        )
        rank_1_nodes = server.get_node_ids(
            FetchNodesRequest(rank=1, world_size=2, node_type=USER, split="train")
        )

        self.assert_tensor_equality(rank_0_nodes, torch.tensor([0]))
        self.assert_tensor_equality(rank_1_nodes, torch.tensor([1, 2]))

    def test_get_node_ids_invalid_split(self) -> None:
        """Test get_node_ids raises ValueError with invalid split."""
        dataset = create_homogeneous_dataset(
            edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX,
        )
        server = dist_server.DistServer(dataset)

        with self.assertRaises(ValueError):
            server.get_node_ids(FetchNodesRequest(split="invalid"))

    def test_get_edge_dir(self) -> None:
        """Test get_edge_dir with a dataset."""
        dataset = create_homogeneous_dataset(
            edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX,
        )
        server = dist_server.DistServer(dataset)
        edge_dir = server.get_edge_dir()
        self.assertEqual(edge_dir, dataset.edge_dir)

    def test_get_node_feature_info(self) -> None:
        """Test get_node_feature_info with a dataset."""
        dataset = create_homogeneous_dataset(
            edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX,
        )
        server = dist_server.DistServer(dataset)
        node_feature_info = server.get_node_feature_info()
        self.assertEqual(node_feature_info, dataset.node_feature_info)

    def test_get_edge_feature_info(self) -> None:
        """Test get_edge_feature_info with a dataset."""
        dataset = create_homogeneous_dataset(
            edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX,
        )
        server = dist_server.DistServer(dataset)
        edge_feature_info = server.get_edge_feature_info()
        self.assertEqual(edge_feature_info, dataset.edge_feature_info)

    def test_get_edge_types_homogeneous(self) -> None:
        """Test get_edge_types with a homogeneous dataset."""
        dataset = create_homogeneous_dataset(
            edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX,
        )
        server = dist_server.DistServer(dataset)
        edge_types = server.get_edge_types()
        self.assertIsNone(edge_types)

    def test_get_edge_types_heterogeneous(self) -> None:
        """Test get_edge_types with a heterogeneous dataset."""
        dataset = create_heterogeneous_dataset(
            edge_indices=DEFAULT_HETEROGENEOUS_EDGE_INDICES,
        )
        server = dist_server.DistServer(dataset)
        edge_types = server.get_edge_types()
        self.assertEqual(
            edge_types,
            [(USER, Relation("to"), STORY), (STORY, Relation("to"), USER)],
        )

    def test_get_ablp_input(self) -> None:
        """Test get_ablp_input returns correct labels for each split."""
        create_test_process_group()
        # Define the labels explicitly: user_id -> list of story_ids
        positive_labels = {
            0: [0, 1],  # User 0 likes Story 0 and Story 1
            1: [1, 2],  # User 1 likes Story 1 and Story 2
            2: [2, 3],  # User 2 likes Story 2 and Story 3
            3: [3, 4],  # User 3 likes Story 3 and Story 4
            4: [4, 0],  # User 4 likes Story 4 and Story 0
        }
        negative_labels = {
            0: [2],  # User 0 dislikes Story 2
            1: [3],  # User 1 dislikes Story 3
            2: [4],  # User 2 dislikes Story 4
            3: [0],  # User 3 dislikes Story 0
            4: [1],  # User 4 dislikes Story 1
        }

        # Define user IDs for each split
        split_to_user_ids = {
            "train": [0, 1, 2],
            "val": [3],
            "test": [4],
        }

        dataset = create_heterogeneous_dataset_for_ablp(
            positive_labels=positive_labels,
            negative_labels=negative_labels,
            train_node_ids=split_to_user_ids["train"],
            val_node_ids=split_to_user_ids["val"],
            test_node_ids=split_to_user_ids["test"],
            edge_indices=DEFAULT_HETEROGENEOUS_EDGE_INDICES,
        )
        server = dist_server.DistServer(dataset)

        for split, expected_user_ids in split_to_user_ids.items():
            with self.subTest(split=split):
                anchor_nodes, pos_labels, neg_labels = server.get_ablp_input(
                    FetchABLPInputRequest(
                        split=split,
                        rank=0,
                        world_size=1,
                        node_type=USER,
                        supervision_edge_type=USER_TO_STORY,
                    )
                )

                # Verify anchor nodes match expected users
                self.assert_tensor_equality(
                    anchor_nodes, torch.tensor(expected_user_ids)
                )

                # Verify positive labels (order may vary due to CSR representation)
                expected_positive = [positive_labels[uid] for uid in expected_user_ids]
                self.assert_tensor_equality(
                    pos_labels, torch.tensor(expected_positive), dim=1
                )

                # Verify negative labels
                expected_negative = [negative_labels[uid] for uid in expected_user_ids]
                assert neg_labels is not None
                self.assert_tensor_equality(neg_labels, torch.tensor(expected_negative))

    def test_get_ablp_input_multiple_ranks(self) -> None:
        """Test get_ablp_input with multiple ranks to verify sharding."""
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
        train_user_ids = [0, 1, 2, 3]

        dataset = create_heterogeneous_dataset_for_ablp(
            positive_labels=positive_labels,
            negative_labels=negative_labels,
            train_node_ids=train_user_ids,
            val_node_ids=[4],
            test_node_ids=[],
            edge_indices=DEFAULT_HETEROGENEOUS_EDGE_INDICES,
        )
        server = dist_server.DistServer(dataset)

        # Get training input for rank 0 of 2

        # Note that the rank and world size here are for the process group we're *fetching for*, not the process group we're *fetching from*.
        # e.g. if our compute cluster is of world size 4, and we have 2 storage nodes, then the world size this gets called with is 4, not 2.
        anchor_nodes_0, pos_labels_0, neg_labels_0 = server.get_ablp_input(
            FetchABLPInputRequest(
                split="train",
                rank=0,
                world_size=2,
                node_type=USER,
                supervision_edge_type=USER_TO_STORY,
            )
        )

        # Get training input for rank 1 of 2
        anchor_nodes_1, pos_labels_1, neg_labels_1 = server.get_ablp_input(
            FetchABLPInputRequest(
                split="train",
                rank=1,
                world_size=2,
                node_type=USER,
                supervision_edge_type=USER_TO_STORY,
            )
        )

        # Train nodes [0, 1, 2, 3] should be split across ranks
        rank_0_user_ids = [0, 1]
        rank_1_user_ids = [2, 3]
        self.assert_tensor_equality(anchor_nodes_0, torch.tensor(rank_0_user_ids))
        self.assert_tensor_equality(anchor_nodes_1, torch.tensor(rank_1_user_ids))

        # Verify positive labels for each rank (order may vary due to CSR representation)
        expected_positive_0 = [positive_labels[uid] for uid in rank_0_user_ids]
        expected_positive_1 = [positive_labels[uid] for uid in rank_1_user_ids]
        self.assert_tensor_equality(
            pos_labels_0, torch.tensor(expected_positive_0), dim=1
        )
        self.assert_tensor_equality(
            pos_labels_1, torch.tensor(expected_positive_1), dim=1
        )

        # Verify negative labels for each rank
        expected_negative_0 = [negative_labels[uid] for uid in rank_0_user_ids]
        expected_negative_1 = [negative_labels[uid] for uid in rank_1_user_ids]
        assert neg_labels_0 is not None
        assert neg_labels_1 is not None
        self.assert_tensor_equality(neg_labels_0, torch.tensor(expected_negative_0))
        self.assert_tensor_equality(neg_labels_1, torch.tensor(expected_negative_1))

    def test_get_ablp_input_invalid_split(self) -> None:
        """Test get_ablp_input raises ValueError with invalid split."""
        create_test_process_group()
        positive_labels = {0: [0], 1: [1], 2: [2], 3: [3], 4: [4]}
        negative_labels = {0: [1], 1: [2], 2: [3], 3: [4], 4: [0]}

        dataset = create_heterogeneous_dataset_for_ablp(
            positive_labels=positive_labels,
            negative_labels=negative_labels,
            train_node_ids=[0, 1, 2],
            val_node_ids=[3],
            test_node_ids=[4],
            edge_indices=DEFAULT_HETEROGENEOUS_EDGE_INDICES,
        )
        server = dist_server.DistServer(dataset)

        with self.assertRaises(ValueError):
            server.get_ablp_input(
                FetchABLPInputRequest(
                    split="invalid",
                    rank=0,
                    world_size=1,
                    node_type=USER,
                    supervision_edge_type=USER_TO_STORY,
                )
            )

    def test_get_ablp_input_without_negative_labels(self) -> None:
        """Test get_ablp_input when no negative labels exist in the dataset."""
        create_test_process_group()
        # Define only positive labels, no negative labels
        positive_labels = {
            0: [0, 1],  # User 0 likes Story 0 and Story 1
            1: [1, 2],  # User 1 likes Story 1 and Story 2
            2: [2, 3],  # User 2 likes Story 2 and Story 3
            3: [3, 4],  # User 3 likes Story 3 and Story 4
            4: [4, 0],  # User 4 likes Story 4 and Story 0
        }
        train_user_ids = [0, 1, 2]

        dataset = create_heterogeneous_dataset_for_ablp(
            positive_labels=positive_labels,
            negative_labels=None,  # No negative labels
            train_node_ids=train_user_ids,
            val_node_ids=[3],
            test_node_ids=[4],
            edge_indices=DEFAULT_HETEROGENEOUS_EDGE_INDICES,
        )
        server = dist_server.DistServer(dataset)

        anchor_nodes, pos_labels, neg_labels = server.get_ablp_input(
            FetchABLPInputRequest(
                split="train",
                rank=0,
                world_size=1,
                node_type=USER,
                supervision_edge_type=USER_TO_STORY,
            )
        )

        # Verify train split returns the expected users
        self.assert_tensor_equality(anchor_nodes, torch.tensor(train_user_ids))

        # Positive labels should still work
        expected_positive = [positive_labels[uid] for uid in train_user_ids]
        self.assert_tensor_equality(pos_labels, torch.tensor(expected_positive), dim=1)

        # Negative labels should be None
        self.assertIsNone(neg_labels)


def _make_sampling_config() -> SamplingConfig:
    return SamplingConfig(
        sampling_type=SamplingType.NODE,
        num_neighbors=[2],
        batch_size=2,
        shuffle=False,
        drop_last=False,
        with_edge=True,
        collect_features=True,
        with_neg=False,
        with_weight=False,
        edge_dir="out",
        seed=None,
    )


class TestDistServerSampling(TestCase):
    def setUp(self) -> None:
        dist_server._dist_server = None
        self.dataset = create_homogeneous_dataset(
            edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX,
        )
        self.server = dist_server.DistServer(self.dataset)
        self.worker_options = MagicMock()
        self.worker_options.buffer_capacity = 2
        self.worker_options.buffer_size = "1MB"
        self.sampling_config = _make_sampling_config()
        self.sampler_options = MagicMock()

    def tearDown(self) -> None:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    @patch("gigl.distributed.graph_store.dist_server.SharedDistSamplingBackend")
    def test_init_sampling_backend_idempotent(
        self, mock_backend_cls: MagicMock
    ) -> None:
        runtime = mock_backend_cls.return_value

        backend_id_1 = self.server.init_sampling_backend(
            InitSamplingBackendRequest(
                backend_key="neighbor_loader_0",
                worker_options=self.worker_options,
                sampler_options=self.sampler_options,
                sampling_config=self.sampling_config,
            )
        )
        backend_id_2 = self.server.init_sampling_backend(
            InitSamplingBackendRequest(
                backend_key="neighbor_loader_0",
                worker_options=self.worker_options,
                sampler_options=self.sampler_options,
                sampling_config=self.sampling_config,
            )
        )

        self.assertEqual(backend_id_1, backend_id_2)
        mock_backend_cls.assert_called_once()
        runtime.init_backend.assert_called_once()

    @patch("gigl.distributed.graph_store.dist_server.ShmChannel")
    @patch("gigl.distributed.graph_store.dist_server.SharedDistSamplingBackend")
    def test_register_creates_channel(
        self,
        mock_backend_cls: MagicMock,
        mock_channel_cls: MagicMock,
    ) -> None:
        runtime = mock_backend_cls.return_value
        backend_id = self.server.init_sampling_backend(
            InitSamplingBackendRequest(
                backend_key="neighbor_loader_0",
                worker_options=self.worker_options,
                sampler_options=self.sampler_options,
                sampling_config=self.sampling_config,
            )
        )

        channel_id = self.server.register_sampling_input(
            RegisterBackendRequest(
                backend_id=backend_id,
                worker_key="neighbor_loader_0_compute_rank_0",
                sampler_input=MagicMock(),
                sampling_config=self.sampling_config,
                buffer_capacity=2,
                buffer_size="1MB",
            )
        )

        self.assertEqual(channel_id, 0)
        runtime.register_input.assert_called_once()
        mock_channel_cls.assert_called_once_with(2, "1MB")

    @patch("gigl.distributed.graph_store.dist_server.ShmChannel")
    @patch("gigl.distributed.graph_store.dist_server.SharedDistSamplingBackend")
    def test_destroy_last_channel_shuts_down_backend(
        self,
        mock_backend_cls: MagicMock,
        _mock_channel_cls: MagicMock,
    ) -> None:
        runtime = mock_backend_cls.return_value
        backend_id = self.server.init_sampling_backend(
            InitSamplingBackendRequest(
                backend_key="neighbor_loader_0",
                worker_options=self.worker_options,
                sampler_options=self.sampler_options,
                sampling_config=self.sampling_config,
            )
        )
        channel_id = self.server.register_sampling_input(
            RegisterBackendRequest(
                backend_id=backend_id,
                worker_key="neighbor_loader_0_compute_rank_0",
                sampler_input=MagicMock(),
                sampling_config=self.sampling_config,
                buffer_capacity=2,
                buffer_size="1MB",
            )
        )

        self.server.destroy_sampling_input(channel_id)

        runtime.unregister_input.assert_called_once_with(channel_id)
        runtime.shutdown.assert_called_once()
        self.assertEqual(self.server._backend_state_by_id, {})

    def test_destroy_unknown_channel_noop(self) -> None:
        self.server.destroy_sampling_input(999)
        self.assertEqual(self.server._backend_state_by_id, {})

    @patch("gigl.distributed.graph_store.dist_server.ShmChannel")
    @patch("gigl.distributed.graph_store.dist_server.SharedDistSamplingBackend")
    def test_start_epoch_idempotent(
        self,
        mock_backend_cls: MagicMock,
        _mock_channel_cls: MagicMock,
    ) -> None:
        runtime = mock_backend_cls.return_value
        backend_id = self.server.init_sampling_backend(
            InitSamplingBackendRequest(
                backend_key="neighbor_loader_0",
                worker_options=self.worker_options,
                sampler_options=self.sampler_options,
                sampling_config=self.sampling_config,
            )
        )
        channel_id = self.server.register_sampling_input(
            RegisterBackendRequest(
                backend_id=backend_id,
                worker_key="neighbor_loader_0_compute_rank_0",
                sampler_input=MagicMock(),
                sampling_config=self.sampling_config,
                buffer_capacity=2,
                buffer_size="1MB",
            )
        )

        self.server.start_new_epoch_sampling(channel_id, 0)
        self.server.start_new_epoch_sampling(channel_id, 0)

        runtime.start_new_epoch_sampling.assert_called_once_with(channel_id, 0)

    def test_shutdown_cleans_all_backends(self) -> None:
        runtime_1 = MagicMock()
        runtime_2 = MagicMock()
        self.server._backend_state_by_id = {
            0: dist_server.SamplingBackendState(
                backend_id=0,
                backend_key="neighbor_loader_0",
                runtime=runtime_1,
            ),
            1: dist_server.SamplingBackendState(
                backend_id=1,
                backend_key="neighbor_loader_1",
                runtime=runtime_2,
            ),
        }
        self.server._backend_key_to_id = {
            "neighbor_loader_0": 0,
            "neighbor_loader_1": 1,
        }

        self.server.shutdown()

        runtime_1.shutdown.assert_called_once()
        runtime_2.shutdown.assert_called_once()
        self.assertEqual(self.server._backend_state_by_id, {})

    def test_create_sampling_producer_removed(self) -> None:
        self.assertFalse(hasattr(dist_server.DistServer, "create_sampling_producer"))
        self.assertFalse(hasattr(dist_server.DistServer, "destroy_sampling_producer"))


if __name__ == "__main__":
    absltest.main()
