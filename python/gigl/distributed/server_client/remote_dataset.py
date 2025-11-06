from typing import Optional, Union

import torch
from graphlearn_torch.distributed import request_server

from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.utils.neighborloader import shard_nodes_by_process
from gigl.env.distributed import GraphStoreInfo
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.types.graph import DEFAULT_HOMOGENEOUS_NODE_TYPE, FeatureInfo

_dataset: Optional[DistDataset] = None


def register_dataset(dataset: DistDataset) -> None:
    global _dataset
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


def _get_node_ids_for_rank(
    rank: int, world_size: int, node_type: NodeType = DEFAULT_HOMOGENEOUS_NODE_TYPE
) -> torch.Tensor:
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


def get_sampler_input_for_inference(
    rank: int,
    cluster_info: GraphStoreInfo,
    node_type: NodeType = DEFAULT_HOMOGENEOUS_NODE_TYPE,
) -> list[torch.Tensor]:
    sampler_input: list[torch.Tensor] = []
    for server_rank in range(
        cluster_info.num_storage_nodes * cluster_info.num_processes_per_storage
    ):
        node_ids = request_server(
            server_rank,
            _get_node_ids_for_rank,
            rank,
            cluster_info.num_compute_nodes * cluster_info.num_processes_per_compute,
            node_type,
        )
        sampler_input.append(node_ids)
    return sampler_input
