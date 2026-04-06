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
        rank: The rank of the process requesting node ids.
            Must be provided together with ``world_size``.
        world_size: The total number of processes in the distributed setup.
            Must be provided together with ``rank``.
        split: The split of the dataset to get node ids from.
        node_type: The type of nodes to get node ids for.

    Examples:
        Fetch all nodes without sharding:

        >>> FetchNodesRequest()

        Fetch training nodes for rank 0 of 4:

        >>> FetchNodesRequest(rank=0, world_size=4, split="train")

        Fetch nodes of a specific type:

        >>> FetchNodesRequest(node_type="user")
    """

    rank: Optional[int] = None
    world_size: Optional[int] = None
    split: Optional[Union[Literal["train", "val", "test"], str]] = None
    node_type: Optional[NodeType] = None

    def validate(self) -> None:
        """Validate that the request has consistent rank/world_size.

        Raises:
            ValueError: If only one of ``rank`` or ``world_size`` is provided.
        """
        if (self.rank is None) ^ (self.world_size is None):
            raise ValueError(
                "rank and world_size must be provided together. "
                f"Received rank={self.rank}, world_size={self.world_size}"
            )


@dataclass(frozen=True)
class FetchABLPInputRequest:
    """Request for fetching ABLP input from a storage server.

    Args:
        split: The split of the dataset to get ABLP input from.
        node_type: The type of anchor nodes to retrieve.
        supervision_edge_type: The edge type used for supervision.
        rank: The rank of the process requesting ABLP input.
            Must be provided together with ``world_size``.
        world_size: The total number of processes in the distributed setup.
            Must be provided together with ``rank``.

    Examples:
        Fetch training ABLP input without sharding:

        >>> FetchABLPRequest(split="train", node_type="user", supervision_edge_type=("user", "to", "item"))

        Fetch training ABLP input for rank 0 of 4:

        >>> FetchABLPRequest(split="train", node_type="user", supervision_edge_type=("user", "to", "item"), rank=0, world_size=4)
    """

    split: Union[Literal["train", "val", "test"], str]
    node_type: NodeType
    supervision_edge_type: EdgeType
    rank: Optional[int] = None
    world_size: Optional[int] = None

    def validate(self) -> None:
        """Validate that the request has consistent rank/world_size.

        Raises:
            ValueError: If only one of ``rank`` or ``world_size`` is provided.
        """
        if (self.rank is None) ^ (self.world_size is None):
            raise ValueError(
                "rank and world_size must be provided together. "
                f"Received rank={self.rank}, world_size={self.world_size}"
            )
