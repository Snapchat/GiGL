"""RPC request messages for graph-store operations."""

from dataclasses import dataclass
from typing import Literal, Optional, Union

from graphlearn_torch.distributed import RemoteDistSamplingWorkerOptions
from graphlearn_torch.sampler import (
    EdgeSamplerInput,
    NodeSamplerInput,
    RemoteSamplerInput,
    SamplingConfig,
)

from gigl.distributed.sampler import ABLPNodeSamplerInput
from gigl.distributed.sampler_options import SamplerOptions
from gigl.distributed.graph_store.sharding import ServerSlice
from gigl.src.common.types.graph_data import EdgeType, NodeType


@dataclass(frozen=True)
class InitSamplingBackendRequest:
    """Request to initialize a shared sampling backend on a storage server.

    Args:
        backend_key: A unique key identifying the backend (e.g. ``"dist_neighbor_loader_0"``).
        worker_options: Options for launching remote sampling workers.
        sampler_options: Controls which sampler class is instantiated.
        sampling_config: Configuration for sampling behavior.
    """

    backend_key: str
    worker_options: RemoteDistSamplingWorkerOptions
    sampler_options: SamplerOptions
    sampling_config: SamplingConfig


@dataclass(frozen=True)
class RegisterBackendRequest:
    """Request to register one compute-rank input channel on a backend.

    Args:
        backend_id: The ID of the backend to register on.
        worker_key: A unique key identifying this compute-rank channel.
        sampler_input: The input data for sampling.
        sampling_config: Configuration for sampling behavior.
        buffer_capacity: Number of shared-memory buffer slots.
        buffer_size: Size of each buffer slot (int bytes or string like ``"1MB"``).
    """

    backend_id: int
    worker_key: str
    sampler_input: Union[
        NodeSamplerInput, EdgeSamplerInput, RemoteSamplerInput, ABLPNodeSamplerInput
    ]
    sampling_config: SamplingConfig
    buffer_capacity: int
    buffer_size: Union[int, str]


@dataclass(frozen=True)
class FetchNodesRequest:
    """Request for fetching node IDs from a storage server.

    Args:
        split: The split of the dataset to get node ids from.
        node_type: The type of nodes to get node ids for.
        server_slice: An optional :class:`~gigl.distributed.graph_store.sharding.ServerSlice`
            describing the fraction of this server's data to return.
            When ``None``, all of the server's data is returned.

    Examples:
        Fetch all nodes without sharding:

        >>> FetchNodesRequest()

        Fetch nodes of a specific type:

        >>> FetchNodesRequest(node_type="user")

        Fetch the first half of a server's training nodes:

        >>> FetchNodesRequest(split="train", server_slice=ServerSlice(0, 0, 1, 2))
    """

    split: Optional[Union[Literal["train", "val", "test"], str]] = None
    node_type: Optional[NodeType] = None
    server_slice: Optional[ServerSlice] = None


@dataclass(frozen=True)
class FetchABLPInputRequest:
    """Request for fetching ABLP input from a storage server.

    Args:
        split: The split of the dataset to get ABLP input from.
        node_type: The type of anchor nodes to retrieve.
        supervision_edge_type: The edge type used for supervision.
        server_slice: An optional :class:`~gigl.distributed.graph_store.sharding.ServerSlice`
            describing the fraction of this server's data to return.
            When ``None``, all of the server's data is returned.

    Examples:
        Fetch training ABLP input without sharding:

        >>> FetchABLPInputRequest(split="train", node_type="user", supervision_edge_type=("user", "to", "item"))

        Fetch with a server slice:

        >>> FetchABLPInputRequest(split="train", node_type="user",
        ...     supervision_edge_type=("user", "to", "item"),
        ...     server_slice=ServerSlice(0, 0, 1, 2))
    """

    split: Union[Literal["train", "val", "test"], str]
    node_type: NodeType
    supervision_edge_type: EdgeType
    server_slice: Optional[ServerSlice] = None
