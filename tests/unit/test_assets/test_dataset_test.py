"""Unit tests for test_dataset factory functions."""

import unittest

import torch

from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.types.graph import FeatureInfo
from tests.test_assets.distributed.test_dataset import (
    DEFAULT_HETEROGENEOUS_EDGE_INDICES,
    DEFAULT_HETEROGENEOUS_NODE_FEATURE_DIM,
    DEFAULT_HOMOGENEOUS_EDGE_INDEX,
    DEFAULT_HOMOGENEOUS_NODE_FEATURE_DIM,
    STORY,
    STORY_TO_USER,
    USER,
    USER_TO_STORY,
    create_heterogeneous_dataset,
    create_heterogeneous_dataset_with_labels,
    create_homogeneous_dataset,
)
from tests.test_assets.distributed.utils import (
    assert_tensor_equality,
    create_test_process_group,
)


class TestCreateHomogeneousDataset(unittest.TestCase):
    """Tests for create_homogeneous_dataset function."""

    def test_default_dataset(self) -> None:
        """Test creating a dataset with default parameters."""
        dataset = create_homogeneous_dataset()

        # Verify node count (10 nodes in default ring graph)
        node_ids = dataset.node_ids
        assert isinstance(node_ids, torch.Tensor)
        self.assertEqual(node_ids.shape[0], 10)

        # Verify feature info
        expected_feature_info = FeatureInfo(
            dim=DEFAULT_HOMOGENEOUS_NODE_FEATURE_DIM, dtype=torch.float32
        )
        self.assertEqual(dataset.node_feature_info, expected_feature_info)

        # Verify edge direction
        self.assertEqual(dataset.edge_dir, "out")

    def test_custom_edge_index(self) -> None:
        """Test creating a dataset with a custom edge index."""
        # Create a simple 3-node graph
        custom_edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        dataset = create_homogeneous_dataset(edge_index=custom_edge_index)

        # Verify node count
        node_ids = dataset.node_ids
        assert isinstance(node_ids, torch.Tensor)
        self.assertEqual(node_ids.shape[0], 3)
        assert_tensor_equality(node_ids, torch.arange(3))

    def test_custom_node_features(self) -> None:
        """Test creating a dataset with custom node features."""
        custom_features = torch.ones(10, 5)
        dataset = create_homogeneous_dataset(node_features=custom_features)

        # Verify feature dimension
        expected_feature_info = FeatureInfo(dim=5, dtype=torch.float32)
        self.assertEqual(dataset.node_feature_info, expected_feature_info)

    def test_custom_feature_dim(self) -> None:
        """Test creating a dataset with custom feature dimension."""
        dataset = create_homogeneous_dataset(node_feature_dim=7)

        expected_feature_info = FeatureInfo(dim=7, dtype=torch.float32)
        self.assertEqual(dataset.node_feature_info, expected_feature_info)

    def test_edge_dir_in(self) -> None:
        """Test creating a dataset with 'in' edge direction."""
        dataset = create_homogeneous_dataset(edge_dir="in")

        self.assertEqual(dataset.edge_dir, "in")

    def test_default_edge_index_not_modified(self) -> None:
        """Test that creating a dataset doesn't modify the default edge index."""
        original = DEFAULT_HOMOGENEOUS_EDGE_INDEX.clone()
        _ = create_homogeneous_dataset()

        assert_tensor_equality(DEFAULT_HOMOGENEOUS_EDGE_INDEX, original)


class TestCreateHeterogeneousDataset(unittest.TestCase):
    """Tests for create_heterogeneous_dataset function."""

    def test_default_dataset(self) -> None:
        """Test creating a dataset with default parameters."""
        dataset = create_heterogeneous_dataset()

        # Verify node counts (5 users, 5 stories in default graph)
        node_ids = dataset.node_ids
        assert isinstance(node_ids, dict)
        self.assertEqual(node_ids[USER].shape[0], 5)
        self.assertEqual(node_ids[STORY].shape[0], 5)

        # Verify feature info
        expected_feature_info = {
            USER: FeatureInfo(
                dim=DEFAULT_HETEROGENEOUS_NODE_FEATURE_DIM, dtype=torch.float32
            ),
            STORY: FeatureInfo(
                dim=DEFAULT_HETEROGENEOUS_NODE_FEATURE_DIM, dtype=torch.float32
            ),
        }
        self.assertEqual(dataset.node_feature_info, expected_feature_info)

        # Verify edge direction
        self.assertEqual(dataset.edge_dir, "out")

    def test_custom_edge_indices(self) -> None:
        """Test creating a dataset with custom edge indices."""
        custom_edges = {
            USER_TO_STORY: torch.tensor([[0, 1, 2], [0, 1, 2]]),
            STORY_TO_USER: torch.tensor([[0, 1, 2], [0, 1, 2]]),
        }
        dataset = create_heterogeneous_dataset(edge_indices=custom_edges)

        # Verify node counts (3 users, 3 stories)
        node_ids = dataset.node_ids
        assert isinstance(node_ids, dict)
        self.assertEqual(node_ids[USER].shape[0], 3)
        self.assertEqual(node_ids[STORY].shape[0], 3)

    def test_custom_node_features(self) -> None:
        """Test creating a dataset with custom node features."""
        custom_features = {
            USER: torch.ones(5, 4),
            STORY: torch.ones(5, 4),
        }
        dataset = create_heterogeneous_dataset(node_features=custom_features)

        expected_feature_info = {
            USER: FeatureInfo(dim=4, dtype=torch.float32),
            STORY: FeatureInfo(dim=4, dtype=torch.float32),
        }
        self.assertEqual(dataset.node_feature_info, expected_feature_info)

    def test_custom_node_labels(self) -> None:
        """Test creating a dataset with custom node labels."""
        custom_labels = {
            USER: torch.tensor([[10], [20], [30], [40], [50]]),
            STORY: torch.tensor([[100], [200], [300], [400], [500]]),
        }
        dataset = create_heterogeneous_dataset(node_labels=custom_labels)

        # Verify node labels are set (node_labels returns Feature objects, not raw tensors)
        node_labels = dataset.node_labels
        assert isinstance(node_labels, dict)
        # The node_labels property returns Feature objects which wrap the tensors
        self.assertIn(USER, node_labels)
        self.assertIn(STORY, node_labels)

    def test_edge_dir_in(self) -> None:
        """Test creating a dataset with 'in' edge direction."""
        dataset = create_heterogeneous_dataset(edge_dir="in")

        self.assertEqual(dataset.edge_dir, "in")

    def test_default_edge_indices_not_modified(self) -> None:
        """Test that creating a dataset doesn't modify the default edge indices."""
        original = {
            edge_type: edge_index.clone()
            for edge_type, edge_index in DEFAULT_HETEROGENEOUS_EDGE_INDICES.items()
        }
        _ = create_heterogeneous_dataset()

        for edge_type, edge_index in DEFAULT_HETEROGENEOUS_EDGE_INDICES.items():
            assert_tensor_equality(edge_index, original[edge_type])


class TestCreateHeterogeneousDatasetWithLabels(unittest.TestCase):
    """Tests for create_heterogeneous_dataset_with_labels function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.positive_labels = {
            0: [0, 1],
            1: [1, 2],
            2: [2, 3],
            3: [3, 4],
            4: [4, 0],
        }
        self.negative_labels = {
            0: [2],
            1: [3],
            2: [4],
            3: [0],
            4: [1],
        }

    def tearDown(self) -> None:
        """Clean up after each test."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def test_basic_dataset_with_splits(self) -> None:
        """Test creating a dataset with train/val/test splits."""
        create_test_process_group()

        dataset = create_heterogeneous_dataset_with_labels(
            positive_labels=self.positive_labels,
            train_node_ids=[0, 1, 2],
            val_node_ids=[3],
            test_node_ids=[4],
        )

        # Verify train/val/test node IDs are set
        train_node_ids = dataset.train_node_ids
        val_node_ids = dataset.val_node_ids
        test_node_ids = dataset.test_node_ids
        assert isinstance(train_node_ids, dict)
        assert isinstance(val_node_ids, dict)
        assert isinstance(test_node_ids, dict)

        # Verify split sizes
        self.assertEqual(train_node_ids[USER].shape[0], 3)
        self.assertEqual(val_node_ids[USER].shape[0], 1)
        self.assertEqual(test_node_ids[USER].shape[0], 1)

    def test_dataset_with_negative_labels(self) -> None:
        """Test creating a dataset with both positive and negative labels."""
        create_test_process_group()

        dataset = create_heterogeneous_dataset_with_labels(
            positive_labels=self.positive_labels,
            negative_labels=self.negative_labels,
            train_node_ids=[0, 1, 2],
            val_node_ids=[3],
            test_node_ids=[4],
        )

        # Verify the dataset was created successfully
        self.assertIsNotNone(dataset.train_node_ids)

    def test_dataset_without_negative_labels(self) -> None:
        """Test creating a dataset without negative labels."""
        create_test_process_group()

        dataset = create_heterogeneous_dataset_with_labels(
            positive_labels=self.positive_labels,
            negative_labels=None,
            train_node_ids=[0, 1, 2],
            val_node_ids=[3],
            test_node_ids=[4],
        )

        # Verify the dataset was created successfully
        self.assertIsNotNone(dataset.train_node_ids)

    def test_missing_positive_labels_raises_error(self) -> None:
        """Test that missing positive labels for split nodes raises an error."""
        # Node 5 is in test split but not in positive_labels
        with self.assertRaises(ValueError) as context:
            create_heterogeneous_dataset_with_labels(
                positive_labels=self.positive_labels,
                train_node_ids=[0, 1, 2],
                val_node_ids=[3],
                test_node_ids=[5],  # Node 5 not in positive_labels
            )

        self.assertIn("5", str(context.exception))
        self.assertIn("positive_labels", str(context.exception))

    def test_custom_node_types(self) -> None:
        """Test creating a dataset with custom node types."""
        create_test_process_group()

        custom_src_type = NodeType("author")
        custom_dst_type = NodeType("article")
        custom_edge_type = EdgeType(custom_src_type, Relation("wrote"), custom_dst_type)
        reverse_edge_type = EdgeType(
            custom_dst_type, Relation("written_by"), custom_src_type
        )

        custom_edges = {
            custom_edge_type: torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]),
            reverse_edge_type: torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]),
        }

        dataset = create_heterogeneous_dataset_with_labels(
            positive_labels=self.positive_labels,
            train_node_ids=[0, 1, 2],
            val_node_ids=[3],
            test_node_ids=[4],
            edge_indices=custom_edges,
            src_node_type=custom_src_type,
            dst_node_type=custom_dst_type,
            supervision_edge_type=custom_edge_type,
        )

        # Verify node types
        node_ids = dataset.node_ids
        assert isinstance(node_ids, dict)
        self.assertIn(custom_src_type, node_ids)
        self.assertIn(custom_dst_type, node_ids)

    def test_custom_feature_dim(self) -> None:
        """Test creating a dataset with custom feature dimension."""
        create_test_process_group()

        dataset = create_heterogeneous_dataset_with_labels(
            positive_labels=self.positive_labels,
            train_node_ids=[0, 1, 2],
            val_node_ids=[3],
            test_node_ids=[4],
            node_feature_dim=8,
        )

        expected_feature_info = {
            USER: FeatureInfo(dim=8, dtype=torch.float32),
            STORY: FeatureInfo(dim=8, dtype=torch.float32),
        }
        self.assertEqual(dataset.node_feature_info, expected_feature_info)

    def test_all_train_split(self) -> None:
        """Test creating a dataset with most nodes in train split."""
        create_test_process_group()

        # The splitter requires at least some val and test nodes, so we use minimal splits
        dataset = create_heterogeneous_dataset_with_labels(
            positive_labels=self.positive_labels,
            train_node_ids=[0, 1, 2],
            val_node_ids=[3],
            test_node_ids=[4],
        )

        # Verify train has expected nodes
        train_node_ids = dataset.train_node_ids
        assert isinstance(train_node_ids, dict)
        self.assertEqual(train_node_ids[USER].shape[0], 3)


if __name__ == "__main__":
    unittest.main()
