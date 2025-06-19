"""Utils for Neighbor loaders."""
import torch
import copy
from torch_geometric.typing import EdgeType
from typing import Union

from gigl.common.logger import Logger
from gigl.types.graph import is_label_edge_type, message_passing_to_negative_label, to_heterogeneous_edge

logger = Logger()


def patch_neighbors_with_zero_fanout(
    edge_types: list[EdgeType],
    num_neighbors: Union[list[int], dict[EdgeType, list[int]]],
) -> dict[EdgeType, list[int]]:
    """
    Sets the labeled edge type fanout to 0 if it is present in the edge types
    Args:
        edge_types (list[EdgeType]): List of edge types
        num_neighbors (dict[EdgeType, list[int]]): Specified fanout by the user
    Returns:
        dict[EdgeType, list[int]]: Modified fanout where the labeled edge type fanouts, if present, are set to 0.
    """
    if isinstance(num_neighbors, list):
        original_fanout = num_neighbors
        num_neighbors = {}
    else:
        original_fanout = next(iter(num_neighbors.values()))

    num_hop = len(original_fanout)
    zero_samples = [0 for _ in range(num_hop)]
    for edge_type in edge_types:
        if is_label_edge_type(edge_type):
            num_neighbors[edge_type] = zero_samples
        elif edge_type not in num_neighbors:
            num_neighbors[edge_type] = original_fanout

    logger.info(f"Overwrote num_neighbors to: {num_neighbors}.")
    return num_neighbors


def shard_nodes_by_process(
    input_nodes: torch.Tensor,
    local_process_rank: int,
    local_process_world_size: int,
) -> torch.Tensor:
    """
    Shards input nodes based on the local process rank
    Args:
        input_nodes (torch.Tensor): Nodes which are split across each training or inference process
        local_process_rank (int): Rank of the current local process
        local_process_world_size (int): Total number of local processes on the current machine
    Returns:
        torch.Tensor: The sharded nodes for the current local process
    """
    num_node_ids_per_process = input_nodes.size(0) // local_process_world_size
    start_index = local_process_rank * num_node_ids_per_process
    end_index = (
        input_nodes.size(0)
        if local_process_rank == local_process_world_size - 1
        else start_index + num_node_ids_per_process
    )
    nodes_for_current_process = input_nodes[start_index:end_index]
    return nodes_for_current_process
