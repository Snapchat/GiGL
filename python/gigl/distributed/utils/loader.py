from typing import TypeVar, Union

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import EdgeType

from gigl.common.logger import Logger
from gigl.distributed.dist_link_prediction_dataset import DistLinkPredictionDataset
from gigl.types.graph import is_label_edge_type, to_heterogeneous_edge

logger = Logger()

_PyGData = TypeVar("_PyGData", Data, HeteroData)


def remove_labeled_edge_types(data: _PyGData) -> _PyGData:
    """
    Removes labeled edge types from an output HeteroData object
    Args:
        data (Union[Data, HeteroData]): Output subgraph
    Returns:
        Union[Data, HeteroData]: The subgraph with the labeled edge types removed
    """
    if isinstance(data, HeteroData):
        for edge_type in data.edge_types:
            if is_label_edge_type(edge_type):
                del data.num_sampled_edges[edge_type]
                del data._edge_store_dict[edge_type]
    return data


def set_labeled_edge_type_fanout(
    dataset: DistLinkPredictionDataset,
    num_neighbors: Union[list[int], dict[EdgeType, list[int]]],
) -> Union[list[int], dict[EdgeType, list[int]]]:
    """
    Sets the labeled edge type fanout to 0 if it is present in the dataset's edge types
    Args:
        dataset (DistLinkPredictionDataset): Link Prediction Dataset we are doing dataloading on
        num_neighbors (Union[list[int], dict[EdgeType, list[int]]]): Specified fanout by the user
    Returns:
        Union[list[int], dict[EdgeType, list[int]]]: Modified fanout where the labeled edge type fanouts, if present, are set to 0.
    """
    dataset_edge_types = dataset.get_edge_types()
    if dataset_edge_types is not None:
        if isinstance(num_neighbors, dict):
            num_hop = len(list(num_neighbors.values())[0])
        else:
            num_hop = len(num_neighbors)
        zero_samples = [0 for _ in range(num_hop)]
        # If there are no labeled edge types present, we don't need to overwrite num_neighbors
        if any([is_label_edge_type(edge_type) for edge_type in dataset_edge_types]):
            num_neighbors = to_heterogeneous_edge(num_neighbors)
            for edge_type in dataset_edge_types:
                if is_label_edge_type(edge_type):
                    num_neighbors[edge_type] = zero_samples
                elif edge_type not in num_neighbors:
                    num_neighbors[edge_type] = zero_samples
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
