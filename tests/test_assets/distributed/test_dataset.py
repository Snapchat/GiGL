"""Factory functions for creating test DistDataset instances.

This module provides utility functions to create DistDataset instances for unit testing.
The functions support both homogeneous and heterogeneous graphs with configurable features,
edge indices, and label splits.

Example usage:
    from tests.test_assets.distributed.test_dataset import (
        create_homogeneous_dataset,
        create_heterogeneous_dataset,
        create_heterogeneous_dataset_with_labels,
        DEFAULT_HOMOGENEOUS_EDGE_INDEX,
        DEFAULT_HETEROGENEOUS_EDGE_INDICES,
    )

    # Create a simple homogeneous dataset with default edge index
    dataset = create_homogeneous_dataset(edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX)

    # Create with custom edge index
    custom_edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    dataset = create_homogeneous_dataset(edge_index=custom_edge_index)
"""

from typing import Final, Literal, Optional

import torch

from gigl.distributed.dist_dataset import DistDataset
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.types.graph import (
    FeaturePartitionData,
    GraphPartitionData,
    PartitionOutput,
    message_passing_to_negative_label,
    message_passing_to_positive_label,
)
from gigl.utils.data_splitters import DistNodeAnchorLinkSplitter

# =============================================================================
# Default Node and Edge Types
# =============================================================================

USER: Final[NodeType] = NodeType("user")
STORY: Final[NodeType] = NodeType("story")
USER_TO_STORY: Final[EdgeType] = EdgeType(USER, Relation("to"), STORY)
STORY_TO_USER: Final[EdgeType] = EdgeType(STORY, Relation("to"), USER)

# =============================================================================
# Default Edge Indices
# =============================================================================

# Homogeneous: 10-node ring graph where node i connects to node (i+1) % 10
DEFAULT_HOMOGENEOUS_EDGE_INDEX: Final[torch.Tensor] = torch.tensor(
    [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]
)

# Heterogeneous: 5 users, 5 stories with identity mapping (user i <-> story i)
DEFAULT_HETEROGENEOUS_EDGE_INDICES: Final[dict[EdgeType, torch.Tensor]] = {
    USER_TO_STORY: torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]),
    STORY_TO_USER: torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]),
}


def create_homogeneous_dataset(
    edge_index: torch.Tensor,
    node_features: Optional[torch.Tensor] = None,
    edge_features: Optional[torch.Tensor] = None,
    node_labels: Optional[torch.Tensor] = None,
    rank: int = 0,
    world_size: int = 1,
    edge_dir: Literal["in", "out"] = "out",
) -> DistDataset:
    """Create a homogeneous test dataset.

    Creates a single-partition DistDataset with the specified edge index, node features,
    edge features, and node labels.

    Args:
        edge_index: COO format edge index [2, num_edges].
        node_features: Node feature tensor [num_nodes, feature_dim], or None.
        edge_features: Edge feature tensor [num_edges, feature_dim], or None.
        node_labels: Node label tensor [num_nodes, label_dim], or None.
        rank: Rank of the current process. Defaults to 0.
        world_size: Total number of processes. Defaults to 1.
        edge_dir: Edge direction ("in" or "out"). Defaults to "out".

    Returns:
        A DistDataset instance with the specified configuration.

    Example:
        >>> dataset = create_homogeneous_dataset(edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX)
        >>> dataset.node_ids.shape
        torch.Size([10])

        >>> custom_edge_index = torch.tensor([[0, 1], [1, 0]])
        >>> dataset = create_homogeneous_dataset(edge_index=custom_edge_index)
        >>> dataset.node_ids.shape
        torch.Size([2])
    """

    # Derive counts from edge index
    num_nodes = int(edge_index.max().item() + 1)
    num_edges = int(edge_index.shape[1])

    # Build partitioned features only if provided
    partitioned_node_features = None
    if node_features is not None:
        partitioned_node_features = FeaturePartitionData(
            feats=node_features, ids=torch.arange(num_nodes)
        )

    partitioned_edge_features = None
    if edge_features is not None:
        partitioned_edge_features = FeaturePartitionData(
            feats=edge_features, ids=torch.arange(num_edges)
        )

    partitioned_node_labels = None
    if node_labels is not None:
        partitioned_node_labels = FeaturePartitionData(
            feats=node_labels, ids=torch.arange(num_nodes)
        )

    partition_output = PartitionOutput(
        # Partition books filled with zeros assign all nodes/edges to partition 0
        node_partition_book=torch.zeros(num_nodes, dtype=torch.int64),
        edge_partition_book=torch.zeros(num_edges, dtype=torch.int64),
        partitioned_edge_index=GraphPartitionData(
            edge_index=edge_index,
            edge_ids=None,
        ),
        partitioned_node_features=partitioned_node_features,
        partitioned_edge_features=partitioned_edge_features,
        partitioned_positive_labels=None,
        partitioned_negative_labels=None,
        partitioned_node_labels=partitioned_node_labels,
    )
    dataset = DistDataset(rank=rank, world_size=world_size, edge_dir=edge_dir)
    dataset.build(partition_output=partition_output)
    return dataset


def create_heterogeneous_dataset(
    edge_indices: dict[EdgeType, torch.Tensor],
    node_features: Optional[dict[NodeType, torch.Tensor]] = None,
    node_labels: Optional[dict[NodeType, torch.Tensor]] = None,
    rank: int = 0,
    world_size: int = 1,
    edge_dir: Literal["in", "out"] = "out",
) -> DistDataset:
    """Create a heterogeneous test dataset.

    Creates a single-partition DistDataset with the specified edge indices and node features.

    Args:
        edge_indices: Mapping of EdgeType -> COO format edge index [2, num_edges].
        node_features: Mapping of NodeType -> feature tensor [num_nodes, feature_dim], or None.
        node_labels: Mapping of NodeType -> label tensor [num_nodes, label_dim], or None.
        rank: Rank of the current process. Defaults to 0.
        world_size: Total number of processes. Defaults to 1.
        edge_dir: Edge direction ("in" or "out"). Defaults to "out".

    Returns:
        A DistDataset instance with the specified configuration.

    Example:
        >>> dataset = create_heterogeneous_dataset(edge_indices=DEFAULT_HETEROGENEOUS_EDGE_INDICES)
        >>> dataset.node_ids[USER].shape
        torch.Size([5])

        >>> custom_edges = {USER_TO_STORY: torch.tensor([[0, 1], [0, 1]])}
        >>> dataset = create_heterogeneous_dataset(edge_indices=custom_edges)
    """

    # Derive node counts from edge indices by collecting max node ID per node type
    node_counts: dict[NodeType, int] = {}
    for edge_type, edge_index in edge_indices.items():
        src_type, _, dst_type = edge_type
        src_max = edge_index[0].max().item() + 1
        dst_max = edge_index[1].max().item() + 1
        node_counts[src_type] = int(max(node_counts.get(src_type, 0), src_max))
        node_counts[dst_type] = int(max(node_counts.get(dst_type, 0), dst_max))

    # Partition books filled with zeros assign all nodes/edges to partition 0
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

    # Build partitioned features only if provided
    partitioned_node_features = None
    if node_features is not None:
        partitioned_node_features = {
            node_type: FeaturePartitionData(
                feats=feats, ids=torch.arange(feats.shape[0])
            )
            for node_type, feats in node_features.items()
        }

    partitioned_node_labels = None
    if node_labels is not None:
        partitioned_node_labels = {
            node_type: FeaturePartitionData(
                feats=labels, ids=torch.arange(labels.shape[0])
            )
            for node_type, labels in node_labels.items()
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
    dataset = DistDataset(rank=rank, world_size=world_size, edge_dir=edge_dir)
    dataset.build(partition_output=partition_output)
    return dataset


def create_heterogeneous_dataset_for_ablp(
    positive_labels: dict[int, list[int]],
    train_node_ids: list[int],
    val_node_ids: list[int],
    test_node_ids: list[int],
    edge_indices: dict[EdgeType, torch.Tensor],
    negative_labels: Optional[dict[int, list[int]]] = None,
    node_features: Optional[dict[NodeType, torch.Tensor]] = None,
    src_node_type: NodeType = USER,
    dst_node_type: NodeType = STORY,
    supervision_edge_type: Optional[EdgeType] = None,
    rank: int = 0,
    world_size: int = 1,
    edge_dir: Literal["in", "out"] = "out",
) -> DistDataset:
    """Create a heterogeneous test dataset for ABLP with label edges and train/val/test splits.

    Creates a dataset with:
    - Source and destination nodes (default: USER and STORY)
    - Message passing edges from edge_indices
    - Positive label edges: src_node_type -[to_gigl_positive]-> dst_node_type
    - Negative label edges (optional): src_node_type -[to_gigl_negative]-> dst_node_type
    - Train/val/test splits for source nodes

    The splits are achieved using DistNodeAnchorLinkSplitter with an identity-like hash
    function (hash(x) = x + 1). This produces deterministic splits where:
    - Nodes with lower IDs go to train
    - Nodes with middle IDs go to val
    - Nodes with higher IDs go to test

    Args:
        positive_labels: Mapping of src_node_id -> list of positive dst_node_ids.
        train_node_ids: List of source node IDs in the train split (must be the lowest IDs).
        val_node_ids: List of source node IDs in the val split (must be middle IDs).
        test_node_ids: List of source node IDs in the test split (must be the highest IDs).
        edge_indices: Mapping of EdgeType -> COO format edge index [2, num_edges].
        negative_labels: Mapping of src_node_id -> list of negative dst_node_ids, or None.
        node_features: Mapping of NodeType -> feature tensor [num_nodes, feature_dim], or None.
        src_node_type: The source node type for labels. Defaults to USER.
        dst_node_type: The destination node type for labels. Defaults to STORY.
        supervision_edge_type: The edge type for supervision. If None, defaults to
            EdgeType(src_node_type, Relation("to"), dst_node_type).
        rank: Rank of the current process. Defaults to 0.
        world_size: Total number of processes. Defaults to 1.
        edge_dir: Edge direction ("in" or "out"). Defaults to "out".

    Returns:
        A DistDataset instance with the specified configuration and splits.

    Raises:
        ValueError: If any node ID in train/val/test is not in positive_labels.

    Example:
        >>> positive_labels = {0: [0, 1], 1: [1, 2], 2: [2, 3]}
        >>> dataset = create_heterogeneous_dataset_for_ablp(
        ...     positive_labels=positive_labels,
        ...     train_node_ids=[0, 1],
        ...     val_node_ids=[2],
        ...     test_node_ids=[],
        ...     edge_indices=DEFAULT_HETEROGENEOUS_EDGE_INDICES,
        ... )
    """
    # Set default supervision edge type
    if supervision_edge_type is None:
        supervision_edge_type = EdgeType(src_node_type, Relation("to"), dst_node_type)

    # Validate that all split node IDs have positive labels
    all_split_node_ids = set(train_node_ids) | set(val_node_ids) | set(test_node_ids)
    missing_nodes = all_split_node_ids - set(positive_labels.keys())
    if missing_nodes:
        raise ValueError(
            f"Node IDs {missing_nodes} are in train/val/test splits but not in positive_labels"
        )

    positive_label_edge_type = message_passing_to_positive_label(supervision_edge_type)
    negative_label_edge_type = message_passing_to_negative_label(supervision_edge_type)

    # Convert positive_labels dict to COO edge index
    pos_src, pos_dst = [], []
    for node_id, dst_ids in positive_labels.items():
        for dst_id in dst_ids:
            pos_src.append(node_id)
            pos_dst.append(dst_id)
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
    node_counts[src_node_type] = max(
        node_counts.get(src_node_type, 0), max(positive_labels.keys()) + 1
    )
    node_counts[dst_node_type] = max(
        node_counts.get(dst_node_type, 0),
        max(max(stories) for stories in positive_labels.values()) + 1,
    )

    # Set up edge partition books and edge indices
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
        for node_id, dst_ids in negative_labels.items():
            for dst_id in dst_ids:
                neg_src.append(node_id)
                neg_dst.append(dst_id)
        negative_label_edge_index = torch.tensor([neg_src, neg_dst])
        edge_partition_book[negative_label_edge_type] = torch.zeros(
            len(neg_src), dtype=torch.int64
        )
        partitioned_edge_index[negative_label_edge_type] = GraphPartitionData(
            edge_index=negative_label_edge_index,
            edge_ids=None,
        )

    # Partition books filled with zeros assign all nodes to partition 0
    node_partition_book = {
        node_type: torch.zeros(count, dtype=torch.int64)
        for node_type, count in node_counts.items()
    }

    # Build partitioned features only if provided
    partitioned_node_features = None
    if node_features is not None:
        partitioned_node_features = {
            node_type: FeaturePartitionData(
                feats=feats, ids=torch.arange(feats.shape[0])
            )
            for node_type, feats in node_features.items()
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

    # Calculate split ratios based on provided node IDs.
    # With identity hash (x + 1), nodes are split by their ID values:
    # - Lower IDs -> train, middle IDs -> val, higher IDs -> test
    total_nodes = len(positive_labels)
    num_val = len(val_node_ids) / total_nodes
    num_test = len(test_node_ids) / total_nodes

    # Identity-like hash function for deterministic splits based on node ID ordering.
    # Adding 1 ensures hash(0) != 0 and creates proper normalization boundaries.
    def _identity_hash(x: torch.Tensor) -> torch.Tensor:
        return x.clone().to(torch.int64) + 1

    # Create splitter that will produce splits based on node ID ordering
    splitter = DistNodeAnchorLinkSplitter(
        sampling_direction=edge_dir,
        num_val=num_val,
        num_test=num_test,
        hash_function=_identity_hash,
        supervision_edge_types=[supervision_edge_type],
        should_convert_labels_to_edges=True,
    )

    dataset = DistDataset(rank=rank, world_size=world_size, edge_dir=edge_dir)
    dataset.build(partition_output=partition_output, splitter=splitter)
    return dataset
