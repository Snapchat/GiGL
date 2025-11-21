"""Utils for operating on a dataset remotely.

These are intended to be used in the context of a server-client architecture,
and with `graphlearn_torch.distributed.request_server`.

`register_dataset` must be called once per process in the server.

And then the client can do something like:

>>> edge_feature_info = graphlearn_torch.distributed.request_server(
>>>    server_rank,
>>>    gigl.distributed.graph_store.remote_dataset.get_edge_feature_info,
>>> )


NOTE: Ideally these would be exposed via `DistServer` [1] so we could call them directly.
TOOD(kmonte): If we ever fork GLT, we should look into expanding DistServer instead.

[1]: https://github.com/alibaba/graphlearn-for-pytorch/blob/main/graphlearn_torch/python/distributed/dist_server.py#L38
"""
from typing import Optional, Union

import torch

from gigl.common.logger import Logger
from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.utils.neighborloader import shard_nodes_by_process
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.types.graph import DEFAULT_HOMOGENEOUS_NODE_TYPE, FeatureInfo

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


def get_node_ids_for_rank(
    rank: int, world_size: int, node_type: NodeType = DEFAULT_HOMOGENEOUS_NODE_TYPE
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
        nodes = _dataset.node_ids
    elif isinstance(_dataset.node_ids, dict):
        nodes = _dataset.node_ids[node_type]
    else:
        raise ValueError(
            f"Node ids must be a torch.Tensor or a dict[NodeType, torch.Tensor], got {type(_dataset.node_ids)}"
        )
    return shard_nodes_by_process(nodes, rank, world_size)
