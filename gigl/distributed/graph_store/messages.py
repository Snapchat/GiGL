"""RPC request messages for graph-store fetch operations."""

from dataclasses import dataclass
from typing import Literal, Optional, Union

from gigl.distributed.graph_store.sharding import ServerSlice
from gigl.src.common.types.graph_data import EdgeType, NodeType


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
    server_slice: Optional[ServerSlice] = None

    def validate(self) -> None:
        """Validate that the request does not mix sharding modes."""
        if (self.rank is None) ^ (self.world_size is None):
            raise ValueError(
                "rank and world_size must be provided together. "
                f"Received rank={self.rank}, world_size={self.world_size}"
            )
        if self.server_slice is not None and (
            self.rank is not None or self.world_size is not None
        ):
            raise ValueError("server_slice cannot be combined with rank/world_size.")


@dataclass(frozen=True)
class FetchABLPInputRequest:
    """Request for fetching ABLP input from a storage server."""

    split: Union[Literal["train", "val", "test"], str]
    node_type: NodeType
    supervision_edge_type: EdgeType
    rank: Optional[int] = None
    world_size: Optional[int] = None
    server_slice: Optional[ServerSlice] = None

    def validate(self) -> None:
        """Validate that the request does not mix sharding modes."""
        if (self.rank is None) ^ (self.world_size is None):
            raise ValueError(
                "rank and world_size must be provided together. "
                f"Received rank={self.rank}, world_size={self.world_size}"
            )
        if self.server_slice is not None and (
            self.rank is not None or self.world_size is not None
        ):
            raise ValueError("server_slice cannot be combined with rank/world_size.")
