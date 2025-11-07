import torch
from graphlearn_torch.distributed import request_server

from gigl.distributed.server_client.remote_dataset import get_node_ids_for_rank
from gigl.env.distributed import GraphStoreInfo
from gigl.src.common.types.graph_data import NodeType
from gigl.types.graph import DEFAULT_HOMOGENEOUS_NODE_TYPE


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
            get_node_ids_for_rank,
            rank,
            cluster_info.num_compute_nodes * cluster_info.num_processes_per_compute,
            node_type,
        )
        sampler_input.append(node_ids)
    return sampler_input
