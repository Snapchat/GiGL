import unittest

import torch
from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.graph_store import remote_dataset
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.types.graph import (
    FeatureInfo,
    FeaturePartitionData,
    GraphPartitionData,
    PartitionOutput,
)
from tests.test_assets.distributed.utils import assert_tensor_equality

_USER = NodeType("user")
_STORY = NodeType("story")
_USER_TO_STORY = EdgeType(_USER, Relation("to"), _STORY)
_STORY_TO_USER = EdgeType(_STORY, Relation("to"), _USER)


class TestRemoteDataset(unittest.TestCase):
    def setUp(self) -> None:
        """Reset the global dataset before each test."""
        remote_dataset._dataset = None

    def tearDown(self) -> None:
        """Clean up after each test."""
        remote_dataset._dataset = None

    def _create_heterogeneous_dataset(self) -> DistDataset:
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
                _USER: FeaturePartitionData(
                    feats=torch.zeros(5, 2), ids=torch.arange(5)
                ),
                _STORY: FeaturePartitionData(
                    feats=torch.zeros(5, 2), ids=torch.arange(5)
                ),
            },
            partitioned_edge_features=None,
            partitioned_positive_labels=None,
            partitioned_negative_labels=None,
            partitioned_node_labels={
                _USER: FeaturePartitionData(
                    feats=torch.arange(5).unsqueeze(1), ids=torch.arange(5)
                ),
                _STORY: FeaturePartitionData(
                    feats=torch.arange(5).unsqueeze(1), ids=torch.arange(5)
                ),
            },
        )
        dataset = DistDataset(rank=0, world_size=1, edge_dir="out")
        dataset.build(partition_output=partition_output)
        return dataset

    def _create_homogeneous_dataset(self) -> DistDataset:
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

    def test_register_dataset(self) -> None:
        """Test that register_dataset correctly sets the global dataset."""
        dataset = self._create_heterogeneous_dataset()
        remote_dataset.register_dataset(dataset)

        # Verify the dataset was registered
        self.assertIsNotNone(remote_dataset._dataset)
        self.assertEqual(remote_dataset._dataset, dataset)

    def test_reregister_dataset_raises_error(self) -> None:
        """Test that reregistering a dataset raises an error."""
        dataset = self._create_heterogeneous_dataset()
        remote_dataset.register_dataset(dataset)
        with self.assertRaises(ValueError) as context:
            remote_dataset.register_dataset(dataset)
        self.assertIn("Dataset already registered!", str(context.exception))

    def test_get_node_feature_info_with_heterogeneous_dataset(self) -> None:
        """Test get_node_feature_info with a registered heterogeneous dataset."""
        dataset = self._create_heterogeneous_dataset()
        remote_dataset.register_dataset(dataset)

        node_feature_info = remote_dataset.get_node_feature_info()

        # Verify it returns the correct feature info
        expected = {
            _USER: FeatureInfo(dim=2, dtype=torch.float32),
            _STORY: FeatureInfo(dim=2, dtype=torch.float32),
        }
        self.assertEqual(node_feature_info, expected)

    def test_get_node_feature_info_with_homogeneous_dataset(self) -> None:
        """Test get_node_feature_info with a registered homogeneous dataset."""
        dataset = self._create_homogeneous_dataset()
        remote_dataset.register_dataset(dataset)

        node_feature_info = remote_dataset.get_node_feature_info()

        # Verify it returns the correct feature info
        expected = FeatureInfo(dim=3, dtype=torch.float32)
        self.assertEqual(node_feature_info, expected)

    def test_get_node_feature_info_without_registered_dataset(self) -> None:
        """Test get_node_feature_info raises ValueError when no dataset is registered."""
        with self.assertRaises(ValueError) as context:
            remote_dataset.get_node_feature_info()

        self.assertIn("Dataset not registered", str(context.exception))
        self.assertIn("register_dataset", str(context.exception))

    def test_get_edge_feature_info_with_heterogeneous_dataset(self) -> None:
        """Test get_edge_feature_info with a registered heterogeneous dataset."""
        dataset = self._create_heterogeneous_dataset()
        remote_dataset.register_dataset(dataset)

        edge_feature_info = remote_dataset.get_edge_feature_info()

        # For this test dataset, edge features are None
        self.assertIsNone(edge_feature_info)

    def test_get_edge_feature_info_with_homogeneous_dataset(self) -> None:
        """Test get_edge_feature_info with a registered homogeneous dataset."""
        dataset = self._create_homogeneous_dataset()
        remote_dataset.register_dataset(dataset)

        edge_feature_info = remote_dataset.get_edge_feature_info()

        # For this test dataset, edge features are None
        self.assertIsNone(edge_feature_info)

    def test_get_edge_feature_info_without_registered_dataset(self) -> None:
        """Test get_edge_feature_info raises ValueError when no dataset is registered."""
        with self.assertRaises(ValueError) as context:
            remote_dataset.get_edge_feature_info()

        self.assertIn("Dataset not registered", str(context.exception))
        self.assertIn("register_dataset", str(context.exception))

    def test_get_node_ids_for_rank_with_homogeneous_dataset(self) -> None:
        """Test get_node_ids_for_rank with a homogeneous dataset."""
        dataset = self._create_homogeneous_dataset()
        remote_dataset.register_dataset(dataset)

        # Test with world_size=1, rank=0 (should get all nodes)
        node_ids = remote_dataset.get_node_ids_for_rank(
            rank=0, world_size=1, node_type=None
        )
        self.assertIsInstance(node_ids, torch.Tensor)
        self.assertEqual(node_ids.shape[0], 10)
        assert_tensor_equality(node_ids, torch.arange(10))

    def test_get_node_ids_for_rank_with_heterogeneous_dataset(self) -> None:
        """Test get_node_ids_for_rank with a heterogeneous dataset."""
        dataset = self._create_heterogeneous_dataset()
        remote_dataset.register_dataset(dataset)

        # Test with USER node type
        user_node_ids = remote_dataset.get_node_ids_for_rank(
            rank=0, world_size=1, node_type=_USER
        )
        self.assertIsInstance(user_node_ids, torch.Tensor)
        self.assertEqual(user_node_ids.shape[0], 5)
        assert_tensor_equality(user_node_ids, torch.arange(5))

        # Test with STORY node type
        story_node_ids = remote_dataset.get_node_ids_for_rank(
            rank=0, world_size=1, node_type=_STORY
        )
        self.assertIsInstance(story_node_ids, torch.Tensor)
        self.assertEqual(story_node_ids.shape[0], 5)
        assert_tensor_equality(story_node_ids, torch.arange(5))

    def test_get_node_ids_for_rank_with_multiple_ranks(self) -> None:
        """Test get_node_ids_for_rank with multiple ranks to verify sharding."""
        dataset = self._create_homogeneous_dataset()
        remote_dataset.register_dataset(dataset)

        # Test with world_size=2
        rank_0_nodes = remote_dataset.get_node_ids_for_rank(
            rank=0, world_size=2, node_type=None
        )
        rank_1_nodes = remote_dataset.get_node_ids_for_rank(
            rank=1, world_size=2, node_type=None
        )

        # Verify each rank gets different nodes
        assert_tensor_equality(rank_0_nodes, torch.arange(5))
        assert_tensor_equality(rank_1_nodes, torch.arange(5, 10))

        # Test with world_size=3 (uneven split)
        rank_0_nodes = remote_dataset.get_node_ids_for_rank(
            rank=0, world_size=3, node_type=None
        )
        rank_1_nodes = remote_dataset.get_node_ids_for_rank(
            rank=1, world_size=3, node_type=None
        )
        rank_2_nodes = remote_dataset.get_node_ids_for_rank(
            rank=2, world_size=3, node_type=None
        )

        assert_tensor_equality(rank_0_nodes, torch.arange(3))
        assert_tensor_equality(rank_1_nodes, torch.arange(3, 6))
        assert_tensor_equality(rank_2_nodes, torch.arange(6, 10))

    def test_get_node_ids_for_rank_without_registered_dataset(self) -> None:
        """Test get_node_ids_for_rank raises ValueError when no dataset is registered."""
        with self.assertRaises(ValueError) as context:
            remote_dataset.get_node_ids_for_rank(rank=0, world_size=1)

        self.assertIn("Dataset not registered", str(context.exception))
        self.assertIn("register_dataset", str(context.exception))

    def test_get_node_ids_for_rank_with_homogeneous_dataset_and_node_type(self) -> None:
        """Test get_node_ids_for_rank with a homogeneous dataset and a node type."""
        dataset = self._create_homogeneous_dataset()
        remote_dataset.register_dataset(dataset)
        with self.assertRaises(ValueError) as context:
            remote_dataset.get_node_ids_for_rank(rank=0, world_size=1, node_type=_USER)
        self.assertIn(
            "node_type must be None for a homogeneous dataset. Got user.",
            str(context.exception),
        )

    def test_get_node_ids_for_rank_with_heterogeneous_dataset_and_no_node_type(
        self,
    ) -> None:
        """Test get_node_ids_for_rank with a heterogeneous dataset and no node type."""
        dataset = self._create_heterogeneous_dataset()
        remote_dataset.register_dataset(dataset)
        with self.assertRaises(ValueError) as context:
            remote_dataset.get_node_ids_for_rank(rank=0, world_size=1, node_type=None)
        self.assertIn(
            "node_type must be not None for a heterogeneous dataset. Got None.",
            str(context.exception),
        )


if __name__ == "__main__":
    unittest.main()
