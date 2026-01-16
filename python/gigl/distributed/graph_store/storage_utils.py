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
    "Dataset not registered! Register the dataset first with `gigl.distributed.server_client.register_dataset`"
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


def get_node_ids_for_rank(
    rank: int,
    world_size: int,
    node_type: Optional[NodeType] = DEFAULT_HOMOGENEOUS_NODE_TYPE,
) -> torch.Tensor:
    """Get the node IDs assigned to a specific rank in distributed processing.

    Shards the node IDs across processes based on the rank and world size.

    Args:
        rank: The rank of the process requesting node IDs.
        world_size: The total number of processes in the distributed setup.
        node_type: The type of nodes to retrieve. Defaults to the default homogeneous node type.

    Returns:
        A tensor containing the node IDs assigned to the specified rank.

    Raises:
        ValueError: If no dataset has been registered or if node_ids format is invalid.
    """
    logger.info(
        f"Getting node ids for rank {rank} / {world_size} with node type {node_type}"
    )
    if _dataset is None:
        raise _NO_DATASET_ERROR
    if isinstance(_dataset.node_ids, torch.Tensor):
        if node_type is not None:
            raise ValueError(
                f"node_type must be None for a homogeneous dataset. Got {node_type}. In GiGL, we usually do not have a truly homogeneous dataset, this is an odd error!"
            )
        nodes = _dataset.node_ids
    elif isinstance(_dataset.node_ids, dict):
        if node_type is None:
            raise ValueError(
                f"node_type must be not None for a heterogeneous dataset. Got {node_type}."
            )
        nodes = _dataset.node_ids[node_type]
    else:
        raise ValueError(
            f"Node ids must be a torch.Tensor or a dict[NodeType, torch.Tensor], got {type(_dataset.node_ids)}"
        )
    return shard_nodes_by_process(nodes, rank, world_size)


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


def get_training_input(
    split: Union[Literal["train", "val", "test"], str],
    rank: int,
    world_size: int,
    node_type: NodeType = DEFAULT_HOMOGENEOUS_NODE_TYPE,
    supervision_edge_type: EdgeType = DEFAULT_HOMOGENEOUS_EDGE_TYPE,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Get the training input for a specific rank in distributed processing.

    Args:
        split: The split to get the training input for.
        rank: The rank of the process requesting the training input.
        world_size: The total number of processes in the distributed setup.
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
    if _dataset is None:
        raise _NO_DATASET_ERROR

    if split == "train":
        anchors = _dataset.train_node_ids
    elif split == "val":
        anchors = _dataset.val_node_ids
    elif split == "test":
        anchors = _dataset.test_node_ids
    else:
        raise ValueError(f"Invalid split: {split}")

    if isinstance(anchors, torch.Tensor):
        raise ValueError(
            f"dataset.node_ids should be a dict[NodeType, torch.Tensor] for getting training input for datasets, got a torch.Tensor for split {split}"
        )
    elif isinstance(anchors, dict):
        anchor_nodes = anchors[node_type]
    else:
        raise ValueError(
            f"Anchor nodes must be a torch.Tensor or a dict[NodeType, torch.Tensor], got {type(anchors)}"
        )

    anchors_for_rank = shard_nodes_by_process(anchor_nodes, rank, world_size)
    positive_label_edge_type, negative_label_edge_type = select_label_edge_types(
        supervision_edge_type, _dataset.get_edge_types()
    )
    positive_labels, negative_labels = get_labels_for_anchor_nodes(
        _dataset, anchors_for_rank, positive_label_edge_type, negative_label_edge_type
    )
    return anchors_for_rank, positive_labels, negative_labels
