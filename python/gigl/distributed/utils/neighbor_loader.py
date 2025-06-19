"""Utils for Neighbor loaders."""
import torch
import copy
from torch_geometric.typing import EdgeType

from gigl.common.logger import Logger
from gigl.types.graph import is_label_edge_type, message_passing_to_negative_label

logger = Logger()


def zero_label_edge_fanout(
    edge_types: list[EdgeType],
    num_neighbors: dict[EdgeType, list[int]],
) -> dict[EdgeType, list[int]]:
    """
    Sets the labeled edge type fanout to 0 if it is present in the edge types
    Args:
        edge_types (list[EdgeType]): List of edge types
        num_neighbors (dict[EdgeType, list[int]]): Specified fanout by the user
    Returns:
        dict[EdgeType, list[int]]: Modified fanout where the labeled edge type fanouts, if present, are set to 0.
    """
    zeroed_num_neighors = copy.deepcopy(num_neighbors)
    num_hop = len(list(num_neighbors.values())[0])
    zero_samples = [0 for _ in range(num_hop)]
    for edge_type in edge_types:
        if message_passing_to_negative_label(edge_type) in edge_types::
            if is_label_edge_type(edge_type):
                num_neighbors[edge_type] = zero_samples
                logger.info(
                    f"Overwrote labeled edge type {edge_type} fanout to: {zero_samples}."
                )
            else:
                logger.warning(
                    f"Edge type {edge_type} is not a labeled edge type, skipping zeroing."
                )
    logger.info(f"Overwrote num_neighbors to: {zeroed_num_neighbors}.")
    return zeroed_num_neighbors


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
