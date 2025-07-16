"""Utils for Neighbor loaders."""
from collections import abc
from copy import deepcopy
from typing import Optional, Union

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import EdgeType

from gigl.common.logger import Logger
from gigl.types.graph import is_label_edge_type

logger = Logger()


def patch_fanout_for_sampling(
    edge_types: Optional[list[EdgeType]],
    num_neighbors: Union[list[int], dict[EdgeType, list[int]]],
) -> Union[list[int], dict[EdgeType, list[int]]]:
    """
    Sets up an approprirate fanout for sampling.

    Does the following:
    - For all label edge types, sets the fanout to be zero.
    - For all other edge types, if the fanout is not specified, uses the original fanout.

    Note that if fanout is provided as a dict, the keys (edges) in the fanout must be in `edge_types`.

    We add this because the existing sampling logic (below) makes strict assumptions that we need to conform to.
    https://github.com/alibaba/graphlearn-for-pytorch/blob/26fe3d4e050b081bc51a79dc9547f244f5d314da/graphlearn_torch/python/distributed/dist_neighbor_sampler.py#L317-L318

    Args:
        edge_types (Optional[list[EdgeType]]): List of all edge types in the graph, is None for homogeneous datasets
        num_neighbors (dict[EdgeType, list[int]]): Specified fanout by the user
    Returns:
        Union[list[int], dict[EdgeType, list[int]]]: Modified fanout that is appropriate for sampling. Is a list[int]
            if the dataset is homogeneous, otherwise is dict[EdgeType, list[int]]
    """
    if edge_types is None:
        if isinstance(num_neighbors, abc.Mapping):
            raise ValueError(
                "When dataset is homogeneous, the num_neighbors field cannot be a dictionary."
            )
        if not all(hop >= 0 for hop in num_neighbors):
            raise ValueError(f"Hops provided must be non-negative, got {num_neighbors}")
        return num_neighbors
    if isinstance(num_neighbors, list):
        original_fanout = num_neighbors
        should_broadcast_fanout = True
        num_neighbors = {}
    else:
        extra_edge_types = set(num_neighbors.keys()) - set(edge_types)
        if extra_edge_types:
            raise ValueError(
                f"Found extra edge types {extra_edge_types} in fanout which is not in dataset edge types {edge_types}."
            )
        original_fanout = next(iter(num_neighbors.values()))
        should_broadcast_fanout = False
        num_neighbors = deepcopy(num_neighbors)

    num_hop = len(original_fanout)
    zero_samples = [0 for _ in range(num_hop)]
    for edge_type in edge_types:
        # TODO(kmonte): stop setting fanout for positive/negative edges once GLT sampling correctly ignores those edges during fanout.
        if is_label_edge_type(edge_type):
            num_neighbors[edge_type] = zero_samples
        elif should_broadcast_fanout and edge_type not in num_neighbors:
            num_neighbors[edge_type] = original_fanout
        elif not should_broadcast_fanout and edge_type not in num_neighbors:
            raise ValueError(
                f"Found non-labeled edge type in dataset {edge_type} which is not in the provided fanout {num_neighbors.keys()}. \
                If fanout is provided as a dict, all edges must be present."
            )

    hops = len(next(iter(num_neighbors.values())))
    if not all(len(fanout) == hops for fanout in num_neighbors.values()):
        raise ValueError(
            f"num_neighbors must be a dict of edge types with the same number of hops. Received: {num_neighbors}"
        )
    if not all(
        hop >= 0 for edge_type in num_neighbors for hop in num_neighbors[edge_type]
    ):
        raise ValueError(f"Hops provided must be non-negative, got {num_neighbors}")

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


def labeled_to_homogeneous(supervision_edge_type: EdgeType, data: HeteroData) -> Data:
    """
    Returns a Data object with the label edges removed.

    Args:
        supervision_edge_type (EdgeType): The edge type that contains the supervision edges.
        data (HeteroData): Heterogeneous graph with the supervision edge type
    Returns:
        data (Data): Homogeneous graph with the labeled edge type removed
    """
    homogeneous_data = data.edge_type_subgraph([supervision_edge_type]).to_homogeneous(
        add_edge_type=False, add_node_type=False
    )
    # Since this is "homogeneous", supervision_edge_type[0] and supervision_edge_type[2] are the same.
    sample_node_type = supervision_edge_type[0]
    homogeneous_data.num_sampled_nodes = data.num_sampled_nodes[sample_node_type]
    homogeneous_data.num_sampled_edges = data.num_sampled_edges[supervision_edge_type]
    homogeneous_data.batch_size = homogeneous_data.batch.numel()
    return homogeneous_data


def strip_label_edges(data: HeteroData) -> HeteroData:
    """
    Removes all edges of a specific type from a heterogeneous graph.

    Modifies the input in place.

    Args:
        data (HeteroData): The input heterogeneous graph.

    Returns:
        HeteroData: The graph with the label edge types removed.
    """

    label_edge_types = [
        e_type for e_type in data.edge_types if is_label_edge_type(e_type)
    ]
    for edge_type in label_edge_types:
        del data[edge_type]
        del data.num_sampled_edges[edge_type]

    return data
