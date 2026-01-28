import unittest
from typing import Final, Optional

import torch

from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.graph_store import storage_utils
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.types.graph import (
    FeatureInfo,
    FeaturePartitionData,
    GraphPartitionData,
    PartitionOutput,
    message_passing_to_negative_label,
    message_passing_to_positive_label,
)
from gigl.utils.data_splitters import DistNodeAnchorLinkSplitter
from tests.test_assets.distributed.utils import (
    assert_tensor_equality,
    create_test_process_group,
)

_USER = NodeType("user")
_STORY = NodeType("story")
_USER_TO_STORY = EdgeType(_USER, Relation("to"), _STORY)
_STORY_TO_USER = EdgeType(_STORY, Relation("to"), _USER)

# Default edge indices for test graphs (COO format: [2, num_edges])
# Homogeneous: 10-node ring graph where node i connects to node (i+1) % 10
_DEFAULT_HOMOGENEOUS_EDGE_INDEX: Final[torch.Tensor] = torch.tensor(
    [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]
)
# Heterogeneous: 5 users, 5 stories with identity mapping (user i <-> story i)
_DEFAULT_HETEROGENEOUS_EDGE_INDICES: Final[dict[EdgeType, torch.Tensor]] = {
    _USER_TO_STORY: torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]),
    _STORY_TO_USER: torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]),
}


# TODO(kmonte): Add tests for get_node_ids with a split.
class TestRemoteDataset(unittest.TestCase):
    def setUp(self) -> None:
        """Reset the global dataset before each test."""
        storage_utils._dataset = None

    def tearDown(self) -> None:
        """Clean up after each test."""
        storage_utils._dataset = None
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def _create_heterogeneous_dataset(
        self,
        edge_indices: dict[EdgeType, torch.Tensor],
    ) -> DistDataset:
        """Helper method to create a heterogeneous test dataset.

        Args:
            edge_indices: Mapping of EdgeType -> COO format edge index [2, num_edges].
        """
        # Derive node counts from edge indices by collecting max node ID per node type
        node_counts: dict[NodeType, int] = {}
        for edge_type, edge_index in edge_indices.items():
            src_type, _, dst_type = edge_type
            src_max = edge_index[0].max().item() + 1
            dst_max = edge_index[1].max().item() + 1
            node_counts[src_type] = int(max(node_counts.get(src_type, 0), src_max))
            node_counts[dst_type] = int(max(node_counts.get(dst_type, 0), dst_max))

        # Partition books filled with zeros assign all nodes/edges to partition 0 (rank 0 - we only have 1 rank in the test)
        node_partition_book = {
            node_type: torch.zeros(count, dtype=torch.int64)
            for node_type, count in node_counts.items()
        }
        edge_partition_book = {
            edge_type: torch.zeros(edge_index.shape[1], dtype=torch.int64)
            for edge_type, edge_index in edge_indices.items()
        }
        partitioned_edge_index = {
            edge_type: GraphPartitionData(edge_index=edge_index, edge_ids=None)
            for edge_type, edge_index in edge_indices.items()
        }
        partitioned_node_features = {
            node_type: FeaturePartitionData(
                feats=torch.zeros(count, 2), ids=torch.arange(count)
            )
            for node_type, count in node_counts.items()
        }
        partitioned_node_labels = {
            node_type: FeaturePartitionData(
                feats=torch.arange(count).unsqueeze(1), ids=torch.arange(count)
            )
            for node_type, count in node_counts.items()
        }

        partition_output = PartitionOutput(
            node_partition_book=node_partition_book,
            edge_partition_book=edge_partition_book,
            partitioned_edge_index=partitioned_edge_index,
            partitioned_node_features=partitioned_node_features,
            partitioned_edge_features=None,
            partitioned_positive_labels=None,
            partitioned_negative_labels=None,
            partitioned_node_labels=partitioned_node_labels,
        )
        dataset = DistDataset(rank=0, world_size=1, edge_dir="out")
        dataset.build(partition_output=partition_output)
        return dataset

    def _create_homogeneous_dataset(
        self,
        edge_index: torch.Tensor,
    ) -> DistDataset:
        """Helper method to create a homogeneous test dataset.

        Args:
            edge_index: COO format edge index [2, num_edges].
        """

        # Derive counts from edge index
        num_nodes = int(edge_index.max().item() + 1)
        num_edges = int(edge_index.shape[1])

        partition_output = PartitionOutput(
            # Partition books filled with zeros assign all nodes/edges to partition 0 (rank 0 - we only have 1 rank in the test)
            node_partition_book=torch.zeros(num_nodes, dtype=torch.int64),
            edge_partition_book=torch.zeros(num_edges, dtype=torch.int64),
            partitioned_edge_index=GraphPartitionData(
                edge_index=edge_index,
                edge_ids=None,
            ),
            partitioned_node_features=FeaturePartitionData(
                feats=torch.zeros(num_nodes, 3), ids=torch.arange(num_nodes)
            ),
            partitioned_edge_features=None,
            partitioned_positive_labels=None,
            partitioned_negative_labels=None,
            partitioned_node_labels=None,
        )
        dataset = DistDataset(rank=0, world_size=1, edge_dir="out")
        dataset.build(partition_output=partition_output)
        return dataset

    def _create_heterogeneous_dataset_with_labels(
        self,
        positive_labels: dict[int, list[int]],
        negative_labels: Optional[dict[int, list[int]]],
        train_user_ids: list[int],
        val_user_ids: list[int],
        test_user_ids: list[int],
        edge_indices: dict[EdgeType, torch.Tensor],
    ) -> DistDataset:
        """Helper method to create a heterogeneous test dataset with label edges and splits.

        Creates a dataset with:
        - USER nodes (count derived from edge indices)
        - STORY nodes (count derived from edge indices)
        - Message passing edges from edge_indices
        - Positive label edges: USER -[to_gigl_positive]-> STORY (from positive_labels)
        - Negative label edges (optional): USER -[to_gigl_negative]-> STORY (from negative_labels)
        - Train/val/test splits for USER nodes

        The splits are achieved using DistNodeAnchorLinkSplitter with an identity-like hash
        function (hash(x) = x + 1). This produces deterministic splits where:
        - Nodes with lower IDs go to train
        - Nodes with middle IDs go to val
        - Nodes with higher IDs go to test

        Args:
            positive_labels: Mapping of user_id -> list of positive story_ids.
            negative_labels: Mapping of user_id -> list of negative story_ids, or None.
            train_user_ids: List of user IDs in the train split (must be the lowest IDs).
            val_user_ids: List of user IDs in the val split (must be middle IDs).
            test_user_ids: List of user IDs in the test split (must be the highest IDs).
            edge_indices: Mapping of EdgeType -> COO format edge index [2, num_edges].

        Raises:
            ValueError: If any user ID in train/val/test is not in positive_labels.
        """
        # Validate that all split user IDs have positive labels
        all_split_user_ids = (
            set(train_user_ids) | set(val_user_ids) | set(test_user_ids)
        )
        missing_users = all_split_user_ids - set(positive_labels.keys())
        if missing_users:
            raise ValueError(
                f"User IDs {missing_users} are in train/val/test splits but not in positive_labels"
            )

        positive_label_edge_type = message_passing_to_positive_label(_USER_TO_STORY)
        negative_label_edge_type = message_passing_to_negative_label(_USER_TO_STORY)

        # Convert positive_labels dict to COO edge index
        pos_src, pos_dst = [], []
        for user_id, story_ids in positive_labels.items():
            for story_id in story_ids:
                pos_src.append(user_id)
                pos_dst.append(story_id)
        positive_label_edge_index = torch.tensor([pos_src, pos_dst])

        # Derive node counts from edge indices by collecting max node ID per node type
        node_counts: dict[NodeType, int] = {}
        for edge_type, edge_index in edge_indices.items():
            src_type, _, dst_type = edge_type
            src_max = edge_index[0].max().item() + 1
            dst_max = edge_index[1].max().item() + 1
            node_counts[src_type] = int(max(node_counts.get(src_type, 0), src_max))
            node_counts[dst_type] = int(max(node_counts.get(dst_type, 0), dst_max))
        # Also account for nodes in positive labels
        node_counts[_USER] = max(
            node_counts.get(_USER, 0), max(positive_labels.keys()) + 1
        )
        node_counts[_STORY] = max(
            node_counts.get(_STORY, 0),
            max(max(stories) for stories in positive_labels.values()) + 1,
        )

        # Set up edge partition books and edge indices
        # Partition books filled with zeros assign all edges to partition 0 (single machine)
        edge_partition_book = {
            edge_type: torch.zeros(edge_index.shape[1], dtype=torch.int64)
            for edge_type, edge_index in edge_indices.items()
        }
        edge_partition_book[positive_label_edge_type] = torch.zeros(
            len(pos_src), dtype=torch.int64
        )

        partitioned_edge_index = {
            edge_type: GraphPartitionData(edge_index=edge_index, edge_ids=None)
            for edge_type, edge_index in edge_indices.items()
        }
        partitioned_edge_index[positive_label_edge_type] = GraphPartitionData(
            edge_index=positive_label_edge_index,
            edge_ids=None,
        )

        if negative_labels is not None:
            # Convert negative_labels dict to COO edge index
            neg_src, neg_dst = [], []
            for user_id, story_ids in negative_labels.items():
                for story_id in story_ids:
                    neg_src.append(user_id)
                    neg_dst.append(story_id)
            negative_label_edge_index = torch.tensor([neg_src, neg_dst])
            edge_partition_book[negative_label_edge_type] = torch.zeros(
                len(neg_src), dtype=torch.int64
            )
            partitioned_edge_index[negative_label_edge_type] = GraphPartitionData(
                edge_index=negative_label_edge_index,
                edge_ids=None,
            )

        # Partition books filled with zeros assign all nodes to partition 0 (single machine)
        node_partition_book = {
            node_type: torch.zeros(count, dtype=torch.int64)
            for node_type, count in node_counts.items()
        }
        partitioned_node_features = {
            node_type: FeaturePartitionData(
                feats=torch.zeros(count, 2), ids=torch.arange(count)
            )
            for node_type, count in node_counts.items()
        }

        partition_output = PartitionOutput(
            node_partition_book=node_partition_book,
            edge_partition_book=edge_partition_book,
            partitioned_edge_index=partitioned_edge_index,
            partitioned_node_features=partitioned_node_features,
            partitioned_edge_features=None,
            partitioned_positive_labels=None,
            partitioned_negative_labels=None,
            partitioned_node_labels=None,
        )

        # Calculate split ratios based on provided user IDs.
        # With identity hash (x + 1), nodes are split by their ID values:
        # - Lower IDs -> train, middle IDs -> val, higher IDs -> test
        total_users = len(positive_labels)
        num_val = len(val_user_ids) / total_users
        num_test = len(test_user_ids) / total_users

        # Identity-like hash function for deterministic splits based on node ID ordering.
        # Adding 1 ensures hash(0) != 0 and creates proper normalization boundaries.
        def _identity_hash(x: torch.Tensor) -> torch.Tensor:
            return x.clone().to(torch.int64) + 1

        # Create splitter that will produce splits based on node ID ordering
        splitter = DistNodeAnchorLinkSplitter(
            sampling_direction="out",
            num_val=num_val,
            num_test=num_test,
            hash_function=_identity_hash,
            supervision_edge_types=[_USER_TO_STORY],
            should_convert_labels_to_edges=True,  # Derives positive/negative edge types from supervision edge type
        )

        dataset = DistDataset(rank=0, world_size=1, edge_dir="out")
        dataset.build(partition_output=partition_output, splitter=splitter)
        return dataset

    def test_register_dataset(self) -> None:
        """Test that register_dataset correctly sets the global dataset."""
        dataset = self._create_heterogeneous_dataset(
            edge_indices=_DEFAULT_HETEROGENEOUS_EDGE_INDICES,
        )
        storage_utils.register_dataset(dataset)

        # Verify the dataset was registered
        self.assertIsNotNone(storage_utils._dataset)
        self.assertEqual(storage_utils._dataset, dataset)

    def test_reregister_dataset_raises_error(self) -> None:
        """Test that reregistering a dataset raises an error."""
        dataset = self._create_heterogeneous_dataset(
            edge_indices=_DEFAULT_HETEROGENEOUS_EDGE_INDICES,
        )
        storage_utils.register_dataset(dataset)
        with self.assertRaises(ValueError) as context:
            storage_utils.register_dataset(dataset)
        self.assertIn("Dataset already registered!", str(context.exception))

    def test_get_node_feature_info_with_heterogeneous_dataset(self) -> None:
        """Test get_node_feature_info with a registered heterogeneous dataset."""
        dataset = self._create_heterogeneous_dataset(
            edge_indices=_DEFAULT_HETEROGENEOUS_EDGE_INDICES,
        )
        storage_utils.register_dataset(dataset)

        node_feature_info = storage_utils.get_node_feature_info()

        # Verify it returns the correct feature info
        expected = {
            _USER: FeatureInfo(dim=2, dtype=torch.float32),
            _STORY: FeatureInfo(dim=2, dtype=torch.float32),
        }
        self.assertEqual(node_feature_info, expected)

    def test_get_node_feature_info_with_homogeneous_dataset(self) -> None:
        """Test get_node_feature_info with a registered homogeneous dataset."""
        dataset = self._create_homogeneous_dataset(
            edge_index=_DEFAULT_HOMOGENEOUS_EDGE_INDEX,
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
        dataset = self._create_heterogeneous_dataset(
            edge_indices=_DEFAULT_HETEROGENEOUS_EDGE_INDICES,
        )
        storage_utils.register_dataset(dataset)

        edge_feature_info = storage_utils.get_edge_feature_info()

        # For this test dataset, edge features are None
        self.assertIsNone(edge_feature_info)

    def test_get_edge_feature_info_with_homogeneous_dataset(self) -> None:
        """Test get_edge_feature_info with a registered homogeneous dataset."""
        dataset = self._create_homogeneous_dataset(
            edge_index=_DEFAULT_HOMOGENEOUS_EDGE_INDEX,
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
        dataset = self._create_homogeneous_dataset(
            edge_index=_DEFAULT_HOMOGENEOUS_EDGE_INDEX,
        )
        storage_utils.register_dataset(dataset)
        node_ids = storage_utils.get_node_ids()
        self.assertIsInstance(node_ids, torch.Tensor)
        self.assertEqual(node_ids.shape[0], 10)
        assert_tensor_equality(node_ids, torch.arange(10))

    def get_node_ids_heterogeneous(self) -> None:
        """Test get_node_ids with a registered heterogeneous dataset."""
        dataset = self._create_heterogeneous_dataset(
            edge_indices=_DEFAULT_HETEROGENEOUS_EDGE_INDICES,
        )
        storage_utils.register_dataset(dataset)
        node_ids = storage_utils.get_node_ids(node_type=_USER)
        self.assertIsInstance(node_ids, torch.Tensor)
        self.assertEqual(node_ids.shape[0], 5)
        assert_tensor_equality(node_ids, torch.arange(5))

    def test_get_node_ids_for_rank_with_homogeneous_dataset(self) -> None:
        """Test get_node_ids_for_rank with a homogeneous dataset."""
        dataset = self._create_homogeneous_dataset(
            edge_index=_DEFAULT_HOMOGENEOUS_EDGE_INDEX,
        )
        storage_utils.register_dataset(dataset)

        # Test with world_size=1, rank=0 (should get all nodes)
        node_ids = storage_utils.get_node_ids(rank=0, world_size=1, node_type=None)
        self.assertIsInstance(node_ids, torch.Tensor)
        self.assertEqual(node_ids.shape[0], 10)
        assert_tensor_equality(node_ids, torch.arange(10))

    def test_get_node_ids_for_rank_with_heterogeneous_dataset(self) -> None:
        """Test get_node_ids_for_rank with a heterogeneous dataset."""
        dataset = self._create_heterogeneous_dataset(
            edge_indices=_DEFAULT_HETEROGENEOUS_EDGE_INDICES,
        )
        storage_utils.register_dataset(dataset)

        # Test with USER node type
        user_node_ids = storage_utils.get_node_ids(
            rank=0, world_size=1, node_type=_USER
        )
        self.assertIsInstance(user_node_ids, torch.Tensor)
        self.assertEqual(user_node_ids.shape[0], 5)
        assert_tensor_equality(user_node_ids, torch.arange(5))

        # Test with STORY node type
        story_node_ids = storage_utils.get_node_ids(
            rank=0, world_size=1, node_type=_STORY
        )
        self.assertIsInstance(story_node_ids, torch.Tensor)
        self.assertEqual(story_node_ids.shape[0], 5)
        assert_tensor_equality(story_node_ids, torch.arange(5))

    def test_get_node_ids_for_rank_with_multiple_ranks(self) -> None:
        """Test get_node_ids_for_rank with multiple ranks to verify sharding."""
        dataset = self._create_homogeneous_dataset(
            edge_index=_DEFAULT_HOMOGENEOUS_EDGE_INDEX,
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

    def test_get_node_ids_for_rank_without_registered_dataset(self) -> None:
        """Test get_node_ids_for_rank raises ValueError when no dataset is registered."""
        with self.assertRaises(ValueError) as context:
            storage_utils.get_node_ids(rank=0, world_size=1)

        self.assertIn("Dataset not registered", str(context.exception))
        self.assertIn("register_dataset", str(context.exception))

    def test_get_node_ids_for_rank_with_homogeneous_dataset_and_node_type(self) -> None:
        """Test get_node_ids_for_rank with a homogeneous dataset and a node type."""
        dataset = self._create_homogeneous_dataset(
            edge_index=_DEFAULT_HOMOGENEOUS_EDGE_INDEX,
        )
        storage_utils.register_dataset(dataset)
        with self.assertRaises(ValueError) as context:
            storage_utils.get_node_ids(rank=0, world_size=1, node_type=_USER)

    def test_get_node_ids_for_rank_with_heterogeneous_dataset_and_no_node_type(
        self,
    ) -> None:
        """Test get_node_ids_for_rank with a heterogeneous dataset and no node type."""
        dataset = self._create_heterogeneous_dataset(
            edge_indices=_DEFAULT_HETEROGENEOUS_EDGE_INDICES,
        )
        storage_utils.register_dataset(dataset)
        with self.assertRaises(ValueError) as context:
            storage_utils.get_node_ids(rank=0, world_size=1, node_type=None)

    def test_get_edge_dir(self) -> None:
        """Test get_edge_dir with a registered dataset."""
        dataset = self._create_homogeneous_dataset(
            edge_index=_DEFAULT_HOMOGENEOUS_EDGE_INDEX,
        )
        storage_utils.register_dataset(dataset)
        edge_dir = storage_utils.get_edge_dir()
        self.assertEqual(edge_dir, dataset.edge_dir)

    def test_get_node_feature_info(self) -> None:
        """Test get_node_feature_info with a registered dataset."""
        dataset = self._create_homogeneous_dataset(
            edge_index=_DEFAULT_HOMOGENEOUS_EDGE_INDEX,
        )
        storage_utils.register_dataset(dataset)
        node_feature_info = storage_utils.get_node_feature_info()
        self.assertEqual(node_feature_info, dataset.node_feature_info)

    def test_get_edge_feature_info(self) -> None:
        """Test get_edge_feature_info with a registered dataset."""
        dataset = self._create_homogeneous_dataset(
            edge_index=_DEFAULT_HOMOGENEOUS_EDGE_INDEX,
        )
        storage_utils.register_dataset(dataset)
        edge_feature_info = storage_utils.get_edge_feature_info()
        self.assertEqual(edge_feature_info, dataset.edge_feature_info)

    def test_get_edge_types_homogeneous(self) -> None:
        """Test get_edge_types with a homogeneous dataset."""
        dataset = self._create_homogeneous_dataset(
            edge_index=_DEFAULT_HOMOGENEOUS_EDGE_INDEX,
        )
        storage_utils.register_dataset(dataset)
        edge_types = storage_utils.get_edge_types()
        self.assertIsNone(edge_types)

    def test_get_edge_types_heterogeneous(self) -> None:
        """Test get_edge_types with a heterogeneous dataset."""
        dataset = self._create_heterogeneous_dataset(
            edge_indices=_DEFAULT_HETEROGENEOUS_EDGE_INDICES,
        )
        storage_utils.register_dataset(dataset)
        edge_types = storage_utils.get_edge_types()
        self.assertEqual(
            edge_types,
            [(_USER, Relation("to"), _STORY), (_STORY, Relation("to"), _USER)],
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

        dataset = self._create_heterogeneous_dataset_with_labels(
            positive_labels=positive_labels,
            negative_labels=negative_labels,
            train_user_ids=split_to_user_ids["train"],
            val_user_ids=split_to_user_ids["val"],
            test_user_ids=split_to_user_ids["test"],
            edge_indices=_DEFAULT_HETEROGENEOUS_EDGE_INDICES,
        )
        storage_utils.register_dataset(dataset)

        for split, expected_user_ids in split_to_user_ids.items():
            with self.subTest(split=split):
                anchor_nodes, pos_labels, neg_labels = storage_utils.get_ablp_input(
                    split=split,
                    rank=0,
                    world_size=1,
                    node_type=_USER,
                    supervision_edge_type=_USER_TO_STORY,
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

        dataset = self._create_heterogeneous_dataset_with_labels(
            positive_labels=positive_labels,
            negative_labels=negative_labels,
            train_user_ids=train_user_ids,
            val_user_ids=[4],
            test_user_ids=[],
            edge_indices=_DEFAULT_HETEROGENEOUS_EDGE_INDICES,
        )
        storage_utils.register_dataset(dataset)

        # Get training input for rank 0 of 2

        # Note that the rank and world size here are for the process group we're *fetching for*, not the process group we're *fetching from*.
        # e.g. if our compute cluster is of world size 4, and we have 2 storage nodes, then the world size this gets called with is 4, not 2.
        anchor_nodes_0, pos_labels_0, neg_labels_0 = storage_utils.get_ablp_input(
            split="train",
            rank=0,
            world_size=2,
            node_type=_USER,
            supervision_edge_type=_USER_TO_STORY,
        )

        # Get training input for rank 1 of 2
        anchor_nodes_1, pos_labels_1, neg_labels_1 = storage_utils.get_ablp_input(
            split="train",
            rank=1,
            world_size=2,
            node_type=_USER,
            supervision_edge_type=_USER_TO_STORY,
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
                node_type=_USER,
                supervision_edge_type=_USER_TO_STORY,
            )

    def test_get_ablp_input_invalid_split(self) -> None:
        """Test get_training_input raises ValueError with invalid split."""
        create_test_process_group()
        positive_labels = {0: [0], 1: [1], 2: [2], 3: [3], 4: [4]}
        negative_labels = {0: [1], 1: [2], 2: [3], 3: [4], 4: [0]}

        dataset = self._create_heterogeneous_dataset_with_labels(
            positive_labels=positive_labels,
            negative_labels=negative_labels,
            train_user_ids=[0, 1, 2],
            val_user_ids=[3],
            test_user_ids=[4],
            edge_indices=_DEFAULT_HETEROGENEOUS_EDGE_INDICES,
        )
        storage_utils.register_dataset(dataset)

        with self.assertRaises(ValueError):
            storage_utils.get_ablp_input(
                split="invalid",
                rank=0,
                world_size=1,
                node_type=_USER,
                supervision_edge_type=_USER_TO_STORY,
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

        dataset = self._create_heterogeneous_dataset_with_labels(
            positive_labels=positive_labels,
            negative_labels=None,  # No negative labels
            train_user_ids=train_user_ids,
            val_user_ids=[3],
            test_user_ids=[4],
            edge_indices=_DEFAULT_HETEROGENEOUS_EDGE_INDICES,
        )
        storage_utils.register_dataset(dataset)

        anchor_nodes, pos_labels, neg_labels = storage_utils.get_ablp_input(
            split="train",
            rank=0,
            world_size=1,
            node_type=_USER,
            supervision_edge_type=_USER_TO_STORY,
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
