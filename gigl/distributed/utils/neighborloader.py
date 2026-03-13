"""Utils for Neighbor loaders."""
from collections import abc
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, TypeVar, Union

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import EdgeType, NodeType

from gigl.common.logger import Logger
from gigl.types.graph import FeatureInfo, is_label_edge_type

logger = Logger()

_GraphType = TypeVar("_GraphType", Data, HeteroData)


class SamplingClusterSetup(Enum):
    """
    The setup of the sampling cluster.
    """

    COLOCATED = "colocated"
    GRAPH_STORE = "graph_store"


class ShardStrategy(Enum):
    """Strategy for sharding node IDs across compute nodes.

    Controls how data from storage servers is distributed to compute nodes.
    Both strategies produce the same total coverage (every node appears on
    exactly one compute node), but differ in which servers each compute node
    communicates with.

    Attributes:
        ROUND_ROBIN: Each compute node gets a slice of nodes from every server.
            Server-side sharding via rank/world_size. This is the current default.
        CONTIGUOUS: Assign entire servers to compute nodes. Each compute node
            only gets nodes from its assigned servers, with empty tensors for
            the rest. Boundary servers are split fractionally when servers
            don't divide evenly across compute nodes.

    Examples:
        **2 storage nodes, 2 compute nodes** (even split):

        Suppose each server holds 10 node IDs::

            Server 0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            Server 1: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

        ``ROUND_ROBIN`` — every compute node gets a slice from *every* server::

            Compute 0 (rank=0, world_size=2):
                {0: [0,1,2,3,4],  1: [10,11,12,13,14]}
            Compute 1 (rank=1, world_size=2):
                {0: [5,6,7,8,9],  1: [15,16,17,18,19]}

        ``CONTIGUOUS`` — each compute node gets *entire* servers::

            Compute 0 (rank=0, world_size=2):
                {0: [0,1,2,3,4,5,6,7,8,9],  1: []}        # all of server 0
            Compute 1 (rank=1, world_size=2):
                {0: [],  1: [10,11,12,13,14,15,16,17,18,19]}  # all of server 1

        **3 storage nodes, 2 compute nodes** (fractional boundary):

        Server 1 is split at the boundary — compute 0 gets the first half,
        compute 1 gets the second half::

            Server 0: [0..9],  Server 1: [10..19],  Server 2: [20..29]

            Compute 0 (rank=0): {0: [0..9], 1: [10..14], 2: []}
            Compute 1 (rank=1): {0: [],     1: [15..19], 2: [20..29]}

    See Also:
        :func:`compute_server_assignments` for the assignment algorithm.
    """

    ROUND_ROBIN = "round_robin"
    CONTIGUOUS = "contiguous"


@dataclass(frozen=True)
class ServerSlice:
    """A compute node's ownership of a single server's nodes.

    Fractions are represented as exact rationals (numerator, denominator)
    to avoid floating-point boundary errors. For a server with N nodes,
    the slice is ``tensor[N * start_num // start_den : N * end_num // end_den]``.

    Args:
        server_rank: The rank of the storage server.
        start_num: Numerator of the start fraction.
        start_den: Denominator of the start fraction.
        end_num: Numerator of the end fraction.
        end_den: Denominator of the end fraction.
    """

    server_rank: int
    start_num: int
    start_den: int
    end_num: int
    end_den: int

    def slice_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Slice a 1D tensor according to this assignment's rational bounds.

        Uses integer division (N * num // den) for exact, deterministic
        index computation. Returns a ``.clone()`` for partial slices to avoid
        retaining full backing storage when used with ``share_memory_()``.

        Args:
            tensor: A 1D tensor of node IDs from the server.

        Returns:
            The sliced portion of the tensor.
        """
        total = len(tensor)
        start_idx = total * self.start_num // self.start_den
        end_idx = total * self.end_num // self.end_den
        if start_idx == 0 and end_idx == total:
            return tensor
        return tensor[start_idx:end_idx].clone()


def compute_server_assignments(
    num_servers: int,
    num_compute_nodes: int,
    compute_rank: int,
) -> dict[int, ServerSlice]:
    """Compute which servers (and what fraction) a compute node owns.

    Uses integer arithmetic throughout. Compute rank R owns the server
    range ``[R * S / C, (R+1) * S / C)`` where boundaries are rational
    numbers with denominator C. For each server s in ``[0, S)``, the overlap
    with this range determines the ServerSlice fractions.

    Only servers with non-zero overlap are included in the returned dict.

    Args:
        num_servers: Total number of storage servers (S).
        num_compute_nodes: Total number of compute nodes (C).
        compute_rank: Rank of the current compute node (R).

    Returns:
        A dict mapping server rank to the ``ServerSlice`` describing the
        fraction of that server owned by this compute node.

    Raises:
        ValueError: If any argument is invalid (negative values,
            rank >= num_compute_nodes, or zero servers/compute nodes).

    Examples:
        >>> compute_server_assignments(num_servers=4, num_compute_nodes=2, compute_rank=0)
        {0: ServerSlice(server_rank=0, ...), 1: ServerSlice(server_rank=1, ...)}

        >>> compute_server_assignments(num_servers=3, num_compute_nodes=2, compute_rank=1)
        {1: ServerSlice(server_rank=1, ...), 2: ServerSlice(server_rank=2, ...)}
    """
    if num_servers <= 0:
        raise ValueError(f"num_servers must be positive, got {num_servers}")
    if num_compute_nodes <= 0:
        raise ValueError(f"num_compute_nodes must be positive, got {num_compute_nodes}")
    if compute_rank < 0 or compute_rank >= num_compute_nodes:
        raise ValueError(
            f"compute_rank must be in [0, {num_compute_nodes}), got {compute_rank}"
        )

    S = num_servers
    C = num_compute_nodes
    R = compute_rank

    # Segment boundaries (as numerators with denominator C):
    # start = R * S, end = (R + 1) * S
    seg_start = R * S
    seg_end = (R + 1) * S

    assignments: dict[int, ServerSlice] = {}
    for s in range(S):
        # Server s spans [s * C, (s + 1) * C) in numerator-space with denominator C
        server_start = s * C
        server_end = (s + 1) * C

        overlap_start = max(seg_start, server_start)
        overlap_end = min(seg_end, server_end)

        if overlap_start >= overlap_end:
            continue

        # Fraction of server s: [(overlap_start - s*C) / C, (overlap_end - s*C) / C)
        start_num = overlap_start - server_start
        end_num = overlap_end - server_start

        assignments[s] = ServerSlice(
            server_rank=s,
            start_num=start_num,
            start_den=C,
            end_num=end_num,
            end_den=C,
        )

    return assignments


@dataclass(frozen=True)
class DatasetSchema:
    """
    Shared metadata between the local and remote datasets.
    """

    # If the dataset is homogeneous with labeled edge type. E.g. one node type, one edge type, and "label" edges.
    # This happens in an otherwise homogeneous dataset when doing ABLP and when we split the dataset.
    is_homogeneous_with_labeled_edge_type: bool
    # List of all edge types in the graph.
    edge_types: Optional[list[EdgeType]]
    # Node feature info.
    node_feature_info: Optional[Union[FeatureInfo, dict[NodeType, FeatureInfo]]]
    # Edge feature info.
    edge_feature_info: Optional[Union[FeatureInfo, dict[EdgeType, FeatureInfo]]]
    # Edge direction.
    edge_dir: Union[str, Literal["in", "out"]]


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


def set_missing_features(
    data: _GraphType,
    node_feature_info: Optional[Union[FeatureInfo, dict[NodeType, FeatureInfo]]],
    edge_feature_info: Optional[Union[FeatureInfo, dict[EdgeType, FeatureInfo]]],
    device: torch.device,
) -> _GraphType:
    """
    If a feature is missing from a produced Data or HeteroData object due to not fanning out to it, populates it in-place with an empty tensor
    with the appropriate feature dim.
    Note that PyG natively does this with their DistNeighborLoader for missing edge features + edge indices and missing node features:
    https://pytorch-geometric.readthedocs.io/en/2.4.0/_modules/torch_geometric/sampler/neighbor_sampler.html#NeighborSampler

    However, native Graphlearn-for-PyTorch only does this for edge indices:
    https://github.com/alibaba/graphlearn-for-pytorch/blob/main/graphlearn_torch/python/sampler/base.py#L294-L301

    so we should do this our sampled node/edge features as well

    # TODO (mkolodner-sc): Migrate this utility to GLT once we fork their repo

    Args:
        data (_GraphType): Data or HeteroData object which we are setting the missing features for
        node_feature_info (Optional[Union[FeatureInfo, dict[NodeType, FeatureInfo]]]): Node feature dimension and data type.
            Note that if heterogeneous, only node types with features should be provided. Can be None in the homogeneous case if there are no node features
        edge_feature_info (Optional[Union[FeatureInfo, dict[EdgeType, FeatureInfo]]]): Edge feature dimension and data type.
            Note that if heterogeneous, only edge types with features should be provided. Can be None in the homogeneous case if there are no edge features
        device (torch.device): Device to move the empty features to
    Returns:
        _GraphType: Data or HeteroData type with the updated feature fields
    """
    if isinstance(data, Data):
        if isinstance(node_feature_info, dict):
            raise ValueError(
                f"Expected node feature dimension to be a FeatureInfo or None for homogeneous data, got {node_feature_info} of type {type(node_feature_info)}"
            )
        if isinstance(edge_feature_info, dict):
            raise ValueError(
                f"Expected edge feature dimension to be an int or None for homogeneous data, got {edge_feature_info} of type {type(edge_feature_info)}"
            )
        # For homogeneous case, the Data object will always have the x or edge_attr fields -- we should check if it is None to see if it set
        if node_feature_info and data.x is None:
            data.x = torch.empty(
                (0, node_feature_info.dim), dtype=node_feature_info.dtype, device=device
            )
        if edge_feature_info and data.edge_attr is None:
            data.edge_attr = torch.empty(
                (0, edge_feature_info.dim), dtype=edge_feature_info.dtype, device=device
            )

    elif isinstance(data, HeteroData):
        if isinstance(node_feature_info, FeatureInfo):
            raise ValueError(
                f"Expected node feature dimension to be an dict or None for heterogeneous data, got {node_feature_info} of type {type(node_feature_info)}"
            )
        if isinstance(edge_feature_info, FeatureInfo):
            raise ValueError(
                f"Expected edge feature dimension to be an dict or None for heterogeneous data, got {edge_feature_info} of type {type(edge_feature_info)}"
            )
        # For heterogeneous case, the HeteroData object will never have the x or edge attr for a given entity type if doesn't exist, even if we set it to None,
        # thus we should check if it hasattr to see they are present
        if node_feature_info:
            for node_type, feature_info in node_feature_info.items():
                if not hasattr(data[node_type], "x"):
                    data[node_type].x = torch.empty(
                        (0, feature_info.dim), dtype=feature_info.dtype, device=device
                    )
        if edge_feature_info:
            for edge_type, feature_info in edge_feature_info.items():
                if not hasattr(data[edge_type], "edge_attr"):
                    data[edge_type].edge_attr = torch.empty(
                        (0, feature_info.dim), dtype=feature_info.dtype, device=device
                    )
    else:
        raise ValueError(
            f"Expected provided data object to be of type `Data` or `HeteroData`, got {type(data)}"
        )

    return data
