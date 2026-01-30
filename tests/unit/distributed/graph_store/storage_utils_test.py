import unittest

import torch

from gigl.distributed.graph_store import storage_utils
from gigl.src.common.types.graph_data import Relation
from gigl.types.graph import FeatureInfo
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
from tests.test_assets.distributed.utils import (
    assert_tensor_equality,
    create_test_process_group,
)


class TestRemoteDataset(unittest.TestCase):
    def setUp(self) -> None:
        """Reset the global dataset before each test."""
        storage_utils._dataset = None

    def tearDown(self) -> None:
        """Clean up after each test."""
        storage_utils._dataset = None
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def test_register_dataset(self) -> None:
        """Test that register_dataset correctly sets the global dataset."""
        dataset = create_heterogeneous_dataset(
            edge_indices=DEFAULT_HETEROGENEOUS_EDGE_INDICES,
        )
        storage_utils.register_dataset(dataset)

        # Verify the dataset was registered
        self.assertIsNotNone(storage_utils._dataset)
        self.assertEqual(storage_utils._dataset, dataset)

    def test_reregister_dataset_raises_error(self) -> None:
        """Test that reregistering a dataset raises an error."""
        dataset = create_heterogeneous_dataset(
            edge_indices=DEFAULT_HETEROGENEOUS_EDGE_INDICES,
        )
        storage_utils.register_dataset(dataset)
        with self.assertRaises(ValueError) as context:
            storage_utils.register_dataset(dataset)
        self.assertIn("Dataset already registered!", str(context.exception))

    def test_get_node_feature_info_with_heterogeneous_dataset(self) -> None:
        """Test get_node_feature_info with a registered heterogeneous dataset."""
        dataset = create_heterogeneous_dataset(
            edge_indices=DEFAULT_HETEROGENEOUS_EDGE_INDICES,
        )
        storage_utils.register_dataset(dataset)

        node_feature_info = storage_utils.get_node_feature_info()

        # Verify it returns the correct feature info
        expected = {
            USER: FeatureInfo(dim=2, dtype=torch.float32),
            STORY: FeatureInfo(dim=2, dtype=torch.float32),
        }
        self.assertEqual(node_feature_info, expected)

    def test_get_node_feature_info_with_homogeneous_dataset(self) -> None:
        """Test get_node_feature_info with a registered homogeneous dataset."""
        dataset = create_homogeneous_dataset(
            edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX,
        )
        storage_utils.register_dataset(dataset)

        node_feature_info = storage_utils.get_node_feature_info()

        # Verify it returns the correct feature info
        expected = FeatureInfo(dim=3, dtype=torch.float32)
        self.assertEqual(node_feature_info, expected)

    def test_get_node_feature_info_without_registered_dataset(self) -> None:
        """Test get_node_feature_info raises ValueError when no dataset is registered."""
        with self.assertRaises(ValueError) as context:
            storage_utils.get_node_feature_info()

        self.assertIn("Dataset not registered", str(context.exception))
        self.assertIn("register_dataset", str(context.exception))

    def test_get_edge_feature_info_with_heterogeneous_dataset(self) -> None:
        """Test get_edge_feature_info with a registered heterogeneous dataset."""
        dataset = create_heterogeneous_dataset(
            edge_indices=DEFAULT_HETEROGENEOUS_EDGE_INDICES,
        )
        storage_utils.register_dataset(dataset)

        edge_feature_info = storage_utils.get_edge_feature_info()

        # For this test dataset, edge features are None
        self.assertIsNone(edge_feature_info)

    def test_get_edge_feature_info_with_homogeneous_dataset(self) -> None:
        """Test get_edge_feature_info with a registered homogeneous dataset."""
        dataset = create_homogeneous_dataset(
            edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX,
        )
        storage_utils.register_dataset(dataset)

        edge_feature_info = storage_utils.get_edge_feature_info()

        # For this test dataset, edge features are None
        self.assertIsNone(edge_feature_info)

    def test_get_edge_feature_info_without_registered_dataset(self) -> None:
        """Test get_edge_feature_info raises ValueError when no dataset is registered."""
        with self.assertRaises(ValueError) as context:
            storage_utils.get_edge_feature_info()

        self.assertIn("Dataset not registered", str(context.exception))
        self.assertIn("register_dataset", str(context.exception))

    def get_node_ids(self) -> None:
        """Test get_node_ids with a registered dataset."""
        dataset = create_homogeneous_dataset(
            edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX,
        )
        storage_utils.register_dataset(dataset)
        node_ids = storage_utils.get_node_ids()
        self.assertIsInstance(node_ids, torch.Tensor)
        self.assertEqual(node_ids.shape[0], 10)
        assert_tensor_equality(node_ids, torch.arange(10))

    def get_node_ids_heterogeneous(self) -> None:
        """Test get_node_ids with a registered heterogeneous dataset."""
        dataset = create_heterogeneous_dataset(
            edge_indices=DEFAULT_HETEROGENEOUS_EDGE_INDICES,
        )
        storage_utils.register_dataset(dataset)
        node_ids = storage_utils.get_node_ids(node_type=USER)
        self.assertIsInstance(node_ids, torch.Tensor)
        self.assertEqual(node_ids.shape[0], 5)
        assert_tensor_equality(node_ids, torch.arange(5))

    def test_get_node_ids_with_homogeneous_dataset(self) -> None:
        """Test get_node_ids with a homogeneous dataset."""
        dataset = create_homogeneous_dataset(
            edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX,
        )
        storage_utils.register_dataset(dataset)

        # Test with world_size=1, rank=0 (should get all nodes)
        node_ids = storage_utils.get_node_ids(rank=0, world_size=1, node_type=None)
        self.assertIsInstance(node_ids, torch.Tensor)
        self.assertEqual(node_ids.shape[0], 10)
        assert_tensor_equality(node_ids, torch.arange(10))

    def test_get_node_ids_with_heterogeneous_dataset(self) -> None:
        """Test get_node_ids with a heterogeneous dataset."""
        dataset = create_heterogeneous_dataset(
            edge_indices=DEFAULT_HETEROGENEOUS_EDGE_INDICES,
        )
        storage_utils.register_dataset(dataset)

        # Test with USER node type
        user_node_ids = storage_utils.get_node_ids(rank=0, world_size=1, node_type=USER)
        self.assertIsInstance(user_node_ids, torch.Tensor)
        self.assertEqual(user_node_ids.shape[0], 5)
        assert_tensor_equality(user_node_ids, torch.arange(5))

        # Test with STORY node type
        story_node_ids = storage_utils.get_node_ids(
            rank=0, world_size=1, node_type=STORY
        )
        self.assertIsInstance(story_node_ids, torch.Tensor)
        self.assertEqual(story_node_ids.shape[0], 5)
        assert_tensor_equality(story_node_ids, torch.arange(5))

    def test_get_node_ids_with_multiple_ranks(self) -> None:
        """Test get_node_ids with multiple ranks to verify sharding."""
        dataset = create_homogeneous_dataset(
            edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX,
        )
        storage_utils.register_dataset(dataset)

        # Test with world_size=2
        rank_0_nodes = storage_utils.get_node_ids(rank=0, world_size=2, node_type=None)
        rank_1_nodes = storage_utils.get_node_ids(rank=1, world_size=2, node_type=None)

        # Verify each rank gets different nodes
        assert_tensor_equality(rank_0_nodes, torch.arange(5))
        assert_tensor_equality(rank_1_nodes, torch.arange(5, 10))

        # Test with world_size=3 (uneven split)
        rank_0_nodes = storage_utils.get_node_ids(rank=0, world_size=3, node_type=None)
        rank_1_nodes = storage_utils.get_node_ids(rank=1, world_size=3, node_type=None)
        rank_2_nodes = storage_utils.get_node_ids(rank=2, world_size=3, node_type=None)

        assert_tensor_equality(rank_0_nodes, torch.arange(3))
        assert_tensor_equality(rank_1_nodes, torch.arange(3, 6))
        assert_tensor_equality(rank_2_nodes, torch.arange(6, 10))

    def test_get_node_ids_without_registered_dataset(self) -> None:
        """Test get_node_ids raises ValueError when no dataset is registered."""
        with self.assertRaises(ValueError) as context:
            storage_utils.get_node_ids(rank=0, world_size=1)

        self.assertIn("Dataset not registered", str(context.exception))
        self.assertIn("register_dataset", str(context.exception))

    def test_get_node_ids_with_homogeneous_dataset_and_node_type(self) -> None:
        """Test get_node_ids with a homogeneous dataset and a node type."""
        dataset = create_homogeneous_dataset(
            edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX,
        )
        storage_utils.register_dataset(dataset)
        with self.assertRaises(ValueError):
            storage_utils.get_node_ids(rank=0, world_size=1, node_type=USER)

    def test_get_node_ids_with_heterogeneous_dataset_and_no_node_type(
        self,
    ) -> None:
        """Test get_node_ids with a heterogeneous dataset and no node type."""
        dataset = create_heterogeneous_dataset(
            edge_indices=DEFAULT_HETEROGENEOUS_EDGE_INDICES,
        )
        storage_utils.register_dataset(dataset)
        with self.assertRaises(ValueError):
            storage_utils.get_node_ids(rank=0, world_size=1, node_type=None)

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
        storage_utils.register_dataset(dataset)

        train_nodes = storage_utils.get_node_ids(node_type=USER, split="train")
        assert_tensor_equality(train_nodes, torch.tensor([0, 1, 2]))

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
        storage_utils.register_dataset(dataset)

        val_nodes = storage_utils.get_node_ids(node_type=USER, split="val")
        assert_tensor_equality(val_nodes, torch.tensor([3]))

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
        storage_utils.register_dataset(dataset)

        test_nodes = storage_utils.get_node_ids(node_type=USER, split="test")
        assert_tensor_equality(test_nodes, torch.tensor([4]))

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
        storage_utils.register_dataset(dataset)

        # Train split has [0, 1, 2], shard across 2 ranks
        rank_0_nodes = storage_utils.get_node_ids(
            rank=0, world_size=2, node_type=USER, split="train"
        )
        rank_1_nodes = storage_utils.get_node_ids(
            rank=1, world_size=2, node_type=USER, split="train"
        )

        assert_tensor_equality(rank_0_nodes, torch.tensor([0]))
        assert_tensor_equality(rank_1_nodes, torch.tensor([1, 2]))

    def test_get_edge_dir(self) -> None:
        """Test get_edge_dir with a registered dataset."""
        dataset = create_homogeneous_dataset(
            edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX,
        )
        storage_utils.register_dataset(dataset)
        edge_dir = storage_utils.get_edge_dir()
        self.assertEqual(edge_dir, dataset.edge_dir)

    def test_get_node_feature_info(self) -> None:
        """Test get_node_feature_info with a registered dataset."""
        dataset = create_homogeneous_dataset(
            edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX,
        )
        storage_utils.register_dataset(dataset)
        node_feature_info = storage_utils.get_node_feature_info()
        self.assertEqual(node_feature_info, dataset.node_feature_info)

    def test_get_edge_feature_info(self) -> None:
        """Test get_edge_feature_info with a registered dataset."""
        dataset = create_homogeneous_dataset(
            edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX,
        )
        storage_utils.register_dataset(dataset)
        edge_feature_info = storage_utils.get_edge_feature_info()
        self.assertEqual(edge_feature_info, dataset.edge_feature_info)

    def test_get_edge_types_homogeneous(self) -> None:
        """Test get_edge_types with a homogeneous dataset."""
        dataset = create_homogeneous_dataset(
            edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX,
        )
        storage_utils.register_dataset(dataset)
        edge_types = storage_utils.get_edge_types()
        self.assertIsNone(edge_types)

    def test_get_edge_types_heterogeneous(self) -> None:
        """Test get_edge_types with a heterogeneous dataset."""
        dataset = create_heterogeneous_dataset(
            edge_indices=DEFAULT_HETEROGENEOUS_EDGE_INDICES,
        )
        storage_utils.register_dataset(dataset)
        edge_types = storage_utils.get_edge_types()
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
        storage_utils.register_dataset(dataset)

        for split, expected_user_ids in split_to_user_ids.items():
            with self.subTest(split=split):
                anchor_nodes, pos_labels, neg_labels = storage_utils.get_ablp_input(
                    split=split,
                    rank=0,
                    world_size=1,
                    node_type=USER,
                    supervision_edge_type=USER_TO_STORY,
                )

                # Verify anchor nodes match expected users
                assert_tensor_equality(anchor_nodes, torch.tensor(expected_user_ids))

                # Verify positive labels (order may vary due to CSR representation)
                expected_positive = [positive_labels[uid] for uid in expected_user_ids]
                assert_tensor_equality(
                    pos_labels, torch.tensor(expected_positive), dim=1
                )

                # Verify negative labels
                expected_negative = [negative_labels[uid] for uid in expected_user_ids]
                assert neg_labels is not None
                assert_tensor_equality(neg_labels, torch.tensor(expected_negative))

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
        storage_utils.register_dataset(dataset)

        # Get training input for rank 0 of 2

        # Note that the rank and world size here are for the process group we're *fetching for*, not the process group we're *fetching from*.
        # e.g. if our compute cluster is of world size 4, and we have 2 storage nodes, then the world size this gets called with is 4, not 2.
        anchor_nodes_0, pos_labels_0, neg_labels_0 = storage_utils.get_ablp_input(
            split="train",
            rank=0,
            world_size=2,
            node_type=USER,
            supervision_edge_type=USER_TO_STORY,
        )

        # Get training input for rank 1 of 2
        anchor_nodes_1, pos_labels_1, neg_labels_1 = storage_utils.get_ablp_input(
            split="train",
            rank=1,
            world_size=2,
            node_type=USER,
            supervision_edge_type=USER_TO_STORY,
        )

        # Train nodes [0, 1, 2, 3] should be split across ranks
        rank_0_user_ids = [0, 1]
        rank_1_user_ids = [2, 3]
        assert_tensor_equality(anchor_nodes_0, torch.tensor(rank_0_user_ids))
        assert_tensor_equality(anchor_nodes_1, torch.tensor(rank_1_user_ids))

        # Verify positive labels for each rank (order may vary due to CSR representation)
        expected_positive_0 = [positive_labels[uid] for uid in rank_0_user_ids]
        expected_positive_1 = [positive_labels[uid] for uid in rank_1_user_ids]
        assert_tensor_equality(pos_labels_0, torch.tensor(expected_positive_0), dim=1)
        assert_tensor_equality(pos_labels_1, torch.tensor(expected_positive_1), dim=1)

        # Verify negative labels for each rank
        expected_negative_0 = [negative_labels[uid] for uid in rank_0_user_ids]
        expected_negative_1 = [negative_labels[uid] for uid in rank_1_user_ids]
        assert neg_labels_0 is not None
        assert neg_labels_1 is not None
        assert_tensor_equality(neg_labels_0, torch.tensor(expected_negative_0))
        assert_tensor_equality(neg_labels_1, torch.tensor(expected_negative_1))

    def test_get_training_input_without_registered_dataset(self) -> None:
        """Test get_training_input raises ValueError when no dataset is registered."""
        with self.assertRaises(ValueError):
            storage_utils.get_ablp_input(
                split="train",
                rank=0,
                world_size=1,
                node_type=USER,
                supervision_edge_type=USER_TO_STORY,
            )

    def test_get_ablp_input_invalid_split(self) -> None:
        """Test get_training_input raises ValueError with invalid split."""
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
        storage_utils.register_dataset(dataset)

        with self.assertRaises(ValueError):
            storage_utils.get_ablp_input(
                split="invalid",
                rank=0,
                world_size=1,
                node_type=USER,
                supervision_edge_type=USER_TO_STORY,
            )

    def test_get_training_input_without_negative_labels(self) -> None:
        """Test get_training_input when no negative labels exist in the dataset."""
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
        storage_utils.register_dataset(dataset)

        anchor_nodes, pos_labels, neg_labels = storage_utils.get_ablp_input(
            split="train",
            rank=0,
            world_size=1,
            node_type=USER,
            supervision_edge_type=USER_TO_STORY,
        )

        # Verify train split returns the expected users
        assert_tensor_equality(anchor_nodes, torch.tensor(train_user_ids))

        # Positive labels should still work
        expected_positive = [positive_labels[uid] for uid in train_user_ids]
        assert_tensor_equality(pos_labels, torch.tensor(expected_positive), dim=1)

        # Negative labels should be None
        self.assertIsNone(neg_labels)


if __name__ == "__main__":
    unittest.main()
