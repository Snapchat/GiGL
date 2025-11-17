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

_dataset: Optional[DistDataset] = None


def register_dataset(dataset: DistDataset) -> None:
    global _dataset
    if _dataset is not None:
        raise ValueError("Dataset already registered! Cannot register a new dataset.")
    _dataset = dataset


def get_node_feature_info() -> Union[FeatureInfo, dict[NodeType, FeatureInfo], None]:
    if _dataset is None:
        raise ValueError(
            "Dataset not registered! Register the dataset first with `gigl.distributed.server_client.register_dataset`"
        )
    return _dataset.node_feature_info


def get_edge_feature_info() -> Union[FeatureInfo, dict[EdgeType, FeatureInfo], None]:
    if _dataset is None:
        raise ValueError(
            "Dataset not registered! Register the dataset first with `gigl.distributed.server_client.register_dataset`"
        )
    return _dataset.edge_feature_info


def get_node_ids_for_rank(
    rank: int, world_size: int, node_type: NodeType = DEFAULT_HOMOGENEOUS_NODE_TYPE
) -> torch.Tensor:
    logger.info(
        f"Getting node ids for rank {rank} / {world_size} with node type {node_type}"
    )
    if _dataset is None:
        raise ValueError(
            "Dataset not registered! Register the dataset first with `gigl.distributed.server_client.register_dataset`"
        )
    if isinstance(_dataset.node_ids, torch.Tensor):
        nodes = _dataset.node_ids
    elif isinstance(_dataset.node_ids, dict):
        nodes = _dataset.node_ids[node_type]
    else:
        raise ValueError(
            f"Node ids must be a torch.Tensor or a dict[NodeType, torch.Tensor], got {type(_dataset.node_ids)}"
        )
    return shard_nodes_by_process(nodes, rank, world_size)
