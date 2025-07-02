"""Utils for Neighbor loaders."""
from collections import abc
from copy import deepcopy
from typing import Optional, Union

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import EdgeType

from gigl.common.logger import Logger
from gigl.src.common.types.graph_data import NodeType
from gigl.types.graph import DEFAULT_HOMOGENEOUS_NODE_TYPE, is_label_edge_type

logger = Logger()


def patch_fanout_for_sampling(
    edge_types: list[EdgeType],
    num_neighbors: Union[list[int], dict[EdgeType, list[int]]],
) -> dict[EdgeType, list[int]]:
    """
    Setups an approprirate fanout for sampling.

    Does the following:
    - For all label edge types, sets the fanout to be zero.
    - For all other edge types, if the fanout is not specified, uses the original fanout.

    We add this because the existing sampling logic (below) makes strict assumptions that we need to conform to.
    https://github.com/alibaba/graphlearn-for-pytorch/blob/26fe3d4e050b081bc51a79dc9547f244f5d314da/graphlearn_torch/python/distributed/dist_neighbor_sampler.py#L317-L318

    Args:
        edge_types (list[EdgeType]): List of all edge types in the graph.
        num_neighbors (dict[EdgeType, list[int]]): Specified fanout by the user
    Returns:
        dict[EdgeType, list[int]]: Modified fanout that is approariate for sampling.
    """
    if isinstance(num_neighbors, list):
        original_fanout = num_neighbors
        num_neighbors = {}
    else:
        original_fanout = next(iter(num_neighbors.values()))
        num_neighbors = deepcopy(num_neighbors)

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


# Allowed inputs for node samplers.
# If None is provded, then all nodes in the graph will be sampled.
# And the graph must be homogeneous.
# If a single tensor is provided, it is assumed to be a tensor of node IDs.
# And the graph must be homogeneous, or labled homogeneous.
# If a tuple is provided, the first element is the node type and the second element is the tensor of node IDs.
# If a dict is provided, the keys are node types and the values are tensors of node IDs.
# If a dict is provided, the graph must be heterogeneous, and there must be only one key/value pair in the dict.
# We allow dicts to be passed in as a convenenience for users who have a heterogeneous graph with only one supervision edge type.
NodeSamplerInput = Optional[
    Union[
        torch.Tensor, tuple[NodeType, torch.Tensor], abc.Mapping[NodeType, torch.Tensor]
    ]
]


def resolve_node_sampler_input_from_user_input(
    input_nodes: NodeSamplerInput,
    dataset_nodes: Optional[Union[torch.Tensor, dict[NodeType, torch.Tensor]]],
) -> tuple[Optional[NodeType], torch.Tensor, bool]:
    """Resolves the input nodes for a node sampler.
    This function takes the user input for input nodes and resolves it to a consistent format.

    See the comment above NodeSamplerInput for the allowed inputs.

    Args:
        input_nodes (NodeSamplerInput): The input nodes provided by the user.
        dataset_nodes (Optional[Union[torch.Tensor, dict[NodeType, torch.Tensor]]): The nodes in the dataset.

    Returns:
        tuple[NodeType, torch.Tensor, bool]: A tuple containing:
            - node_type (NodeType): The type of the nodes.
            - node_ids (torch.Tensor): The tensor of node IDs.
            - is_labeled_homogeneous (bool): Whether the dataset is a labeled homogeneous graph.
    """
    is_labeled_homoogeneous = False
    if isinstance(input_nodes, torch.Tensor):
        node_ids = input_nodes

        # If the dataset is heterogeneous, we may be in the "labeled homogeneous" setting,
        # if so, then we should use DEFAULT_HOMOGENEOUS_NODE_TYPE.
        if isinstance(dataset_nodes, dict):
            if (
                len(dataset_nodes) == 1
                and DEFAULT_HOMOGENEOUS_NODE_TYPE in dataset_nodes
            ):
                node_type = DEFAULT_HOMOGENEOUS_NODE_TYPE
                is_labeled_homoogeneous = True
            else:
                raise ValueError(
                    f"For heterogeneous datasets, input_nodes must be a tuple of (node_type, node_ids) OR if it is a labeled homogeneous dataset, input_nodes may be a torch.Tensor. Received node types: {dataset_nodes.keys()}"
                )
        else:
            node_type = None
    elif isinstance(input_nodes, abc.Mapping):
        if len(input_nodes) != 1:
            raise ValueError(
                f"If input_nodes is provided as a mapping, it must contain exactly one key/value pair. Received: {input_nodes}. This may happen if you call Loader(node_ids=dataset.node_ids) with a heterogeneous dataset."
            )
        node_type, node_ids = next(iter(input_nodes.items()))
        is_labeled_homoogeneous = node_type == DEFAULT_HOMOGENEOUS_NODE_TYPE
    elif isinstance(input_nodes, tuple):
        node_type, node_ids = input_nodes
    elif input_nodes is None:
        if dataset_nodes is None:
            raise ValueError("If input_nodes is None, the dataset must have node ids.")
        if isinstance(dataset_nodes, torch.Tensor):
            node_type = None
            node_ids = dataset_nodes
        elif isinstance(dataset_nodes, dict):
            raise ValueError(
                f"Input nodes must be provided for a heterogeneous graph. Received: {dataset_nodes}"
            )

    return (
        node_type,
        node_ids,
        is_labeled_homoogeneous,
    )
