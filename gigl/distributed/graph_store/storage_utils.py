"""Utils for operating on a dataset remotely.

These are intended to be used in the context of a server-client architecture,
and with `graphlearn_torch.distributed.request_server`.

`register_dataset` must be called once per process in the server.

And then the client can do something like:

>>> edge_feature_info = graphlearn_torch.distributed.request_server(
>>>    server_rank,
>>>    gigl.distributed.graph_store.storage_utils.get_edge_feature_info,
>>> )


NOTE: Ideally these would be exposed via `DistServer` [1] so we could call them directly.
TOOD(kmonte): If we ever fork GLT, we should look into expanding DistServer instead.

[1]: https://github.com/alibaba/graphlearn-for-pytorch/blob/main/graphlearn_torch/python/distributed/dist_server.py#L38
"""
from collections import abc
from typing import Literal, Optional, Union

import torch

from gigl.common.logger import Logger
from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.utils.neighborloader import shard_nodes_by_process
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.types.graph import (
    DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    DEFAULT_HOMOGENEOUS_NODE_TYPE,
    FeatureInfo,
    select_label_edge_types,
)
from gigl.utils.data_splitters import get_labels_for_anchor_nodes

logger = Logger()

_NO_DATASET_ERROR = ValueError(
    "Dataset not registered! Register the dataset first with `gigl.distributed.graph_store.storage_utils.register_dataset`"
)
_dataset: Optional[DistDataset] = None


def register_dataset(dataset: DistDataset) -> None:
    """Register a dataset for remote access.

    This function must be called once per process in the server before any remote
    dataset operations can be performed.

    Args:
        dataset: The distributed dataset to register.

    Raises:
        ValueError: If a dataset has already been registered.
    """
    global _dataset
    if _dataset is not None:
        raise ValueError("Dataset already registered! Cannot register a new dataset.")
    _dataset = dataset


def get_node_feature_info() -> Union[FeatureInfo, dict[NodeType, FeatureInfo], None]:
    """Get node feature information from the registered dataset.

    Returns:
        Node feature information, which can be:
        - A single FeatureInfo object for homogeneous graphs
        - A dict mapping NodeType to FeatureInfo for heterogeneous graphs
        - None if no node features are available

    Raises:
        ValueError: If no dataset has been registered.
    """
    if _dataset is None:
        raise _NO_DATASET_ERROR
    return _dataset.node_feature_info


def get_edge_feature_info() -> Union[FeatureInfo, dict[EdgeType, FeatureInfo], None]:
    """Get edge feature information from the registered dataset.

    Returns:
        Edge feature information, which can be:
        - A single FeatureInfo object for homogeneous graphs
        - A dict mapping EdgeType to FeatureInfo for heterogeneous graphs
        - None if no edge features are available

    Raises:
        ValueError: If no dataset has been registered.
    """
    if _dataset is None:
        raise _NO_DATASET_ERROR
    return _dataset.edge_feature_info


def get_edge_dir() -> Literal["in", "out"]:
    """Get the edge direction from the registered dataset.

    Returns:
        The edge direction.
    """
    if _dataset is None:
        raise _NO_DATASET_ERROR
    return _dataset.edge_dir


def get_node_ids(
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    split: Optional[Union[Literal["train", "val", "test"], str]] = None,
    node_type: Optional[NodeType] = None,
) -> torch.Tensor:
    """
    Get the node ids from the registered dataset.

    Args:
        rank (Optional[int]): The rank of the process requesting node ids. Must be provided if world_size is provided.
        world_size (Optional[int]): The total number of processes in the distributed setup. Must be provided if rank is provided.
        split (Optional[Literal["train", "val", "test"]]): The split of the dataset to get node ids from. If provided, the dataset must have `train_node_ids`, `val_node_ids`, and `test_node_ids` properties.
        node_type (Optional[NodeType]): The type of nodes to get node ids for. Must be provided if the dataset is heterogeneous.

    Returns:
        The node ids.

    Raises:
        ValueError:
            * If no dataset has been registered
            * If the rank and world_size are not provided together
            * If the split is invalid
            * If the node ids are not a torch.Tensor or a dict[NodeType, torch.Tensor]
            * If the node type is provided for a homogeneous dataset
            * If the node ids are not a dict[NodeType, torch.Tensor] when no node type is provided

    Examples:
        Suppose the dataset has 100 nodes total: train=[0..59], val=[60..79], test=[80..99].

        Get all node ids (no split filtering):

        >>> get_node_ids()
        tensor([0, 1, 2, ..., 99])  # All 100 nodes

        Get only training nodes:

        >>> get_node_ids(split="train")
        tensor([0, 1, 2, ..., 59])  # 60 training nodes

        Shard all nodes across 4 processes (each gets ~25 nodes):

        >>> get_node_ids(rank=0, world_size=4)
        tensor([0, 1, 2, ..., 24])  # First 25 of all 100 nodes

        Shard training nodes across 4 processes (each gets ~15 nodes):

        >>> get_node_ids(rank=0, world_size=4, split="train")
        tensor([0, 1, 2, ..., 14])  # First 15 of the 60 training nodes

        Note: When `split=None`, all nodes are queryable. This means nodes from any
        split (train, val, or test) may be returned. This is useful when you need
        to sample neighbors during inference, as neighbor nodes may belong to any split.
    """
    if _dataset is None:
        raise _NO_DATASET_ERROR
    if (rank is None) ^ (world_size is None):
        raise ValueError(
            f"rank and world_size must be provided together. Received rank: {rank}, world_size: {world_size}"
        )
    if split == "train":
        nodes = _dataset.train_node_ids
    elif split == "val":
        nodes = _dataset.val_node_ids
    elif split == "test":
        nodes = _dataset.test_node_ids
    elif split is None:
        nodes = _dataset.node_ids
    else:
        raise ValueError(
            f"Invalid split: {split}. Must be one of 'train', 'val', 'test', or None."
        )

    if node_type is not None:
        if not isinstance(nodes, abc.Mapping):
            raise ValueError(
                f"node_type was provided as {node_type}, so node ids must be a dict[NodeType, torch.Tensor] (e.g. a heterogeneous dataset), got {type(nodes)}"
            )
        nodes = nodes[node_type]
    elif not isinstance(nodes, torch.Tensor):
        raise ValueError(
            f"node_type was not provided, so node ids must be a torch.Tensor (e.g. a homogeneous dataset), got {type(nodes)}."
        )

    if rank is not None and world_size is not None:
        return shard_nodes_by_process(nodes, rank, world_size)
    return nodes


def get_edge_types() -> Optional[list[EdgeType]]:
    """Get the edge types from the registered dataset.

    Returns:
        The edge types in the dataset, None if the dataset is homogeneous.
    """
    if _dataset is None:
        raise _NO_DATASET_ERROR
    if isinstance(_dataset.graph, dict):
        return list(_dataset.graph.keys())
    else:
        return None


def get_ablp_input(
    split: Union[Literal["train", "val", "test"], str],
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    node_type: NodeType = DEFAULT_HOMOGENEOUS_NODE_TYPE,
    supervision_edge_type: EdgeType = DEFAULT_HOMOGENEOUS_EDGE_TYPE,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Get the ABLP (Anchor Based Link Prediction) input for a specific rank in distributed processing.

    Note: rank and world_size here are for the process group we're *fetching for*, not the process group we're *fetching from*.
    e.g. if our compute cluster is of world size 4, and we have 2 storage nodes, then the world size this gets called with is 4, not 2.

    Args:
        split: The split to get the training input for.
        rank: The rank of the process requesting the training input. Defaults to None, in which case all nodes are returned. Must be provided if world_size is provided.
        world_size: The total number of processes in the distributed setup. Defaults to None, in which case all nodes are returned. Must be provided if rank is provided.
        node_type: The type of nodes to retrieve. Defaults to the default homogeneous node type.
        supervision_edge_type: The edge type to use for the supervision. Defaults to the default homogeneous edge type.
    Returns:
        A tuple containing the anchor nodes for the rank, the positive labels, and the negative labels.
        The positive labels are of shape [N, M], where N is the number of anchor nodes and M is the number of positive labels.
        The negative labels are of shape [N, M], where N is the number of anchor nodes and M is the number of negative labels.
        The negative labels may be None if no negative labels are available.
    Raises:
        ValueError: If no dataset has been registered or if the split is invalid.
    """

    # TODO(kmonte): Migrate this part to use some `get_node_ids` method on the dataset.
    if _dataset is None:
        raise _NO_DATASET_ERROR

    anchors = get_node_ids(
        split=split, rank=rank, world_size=world_size, node_type=node_type
    )
    positive_label_edge_type, negative_label_edge_type = select_label_edge_types(
        supervision_edge_type, _dataset.get_edge_types()
    )
    positive_labels, negative_labels = get_labels_for_anchor_nodes(
        _dataset, anchors, positive_label_edge_type, negative_label_edge_type
    )
    return anchors, positive_labels, negative_labels
