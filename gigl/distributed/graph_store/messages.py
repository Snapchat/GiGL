"""RPC request messages for graph-store fetch operations."""

from dataclasses import dataclass
from typing import Literal, Optional, Union

from gigl.src.common.types.graph_data import EdgeType, NodeType


def _validate_sharding_params(
    rank: Optional[int],
    world_size: Optional[int],
) -> None:
    """Validate that sharding parameters are consistent.

    Args:
        rank: Which partition to select (0-indexed).
        world_size: Total number of partitions.

    Raises:
        ValueError: If only one of ``rank``/``world_size`` is provided,
            or if the values are out of range.
    """
    if (rank is None) ^ (world_size is None):
        raise ValueError(
            "rank and world_size must be provided together. "
            f"Received rank={rank}, world_size={world_size}"
        )
    if rank is not None and world_size is not None:
        if world_size <= 0:
            raise ValueError(f"world_size must be > 0, received {world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(
                "rank must be in [0, world_size). "
                f"Received rank={rank}, world_size={world_size}"
            )


@dataclass(frozen=True)
class FetchNodesRequest:
    """Request for fetching node IDs from a storage server.

    Args:
        rank: Which partition of the node IDs to return (0-indexed).
            Must be provided together with ``world_size``.
        world_size: Total number of partitions.
            Must be provided together with ``rank``.
        split: The split of the dataset to get node ids from.
        node_type: The type of nodes to get node ids for.

    Examples:
        Fetch all nodes without splitting:

        >>> FetchNodesRequest()

        Fetch partition 0 of 4 from training nodes:

        >>> FetchNodesRequest(rank=0, world_size=4, split="train")

        Fetch nodes of a specific type:

        >>> FetchNodesRequest(node_type="user")
    """

    rank: Optional[int] = None
    world_size: Optional[int] = None
    split: Optional[Union[Literal["train", "val", "test"], str]] = None
    node_type: Optional[NodeType] = None

    def validate(self) -> None:
        """Validate that the request has consistent sharding parameters.

        Raises:
            ValueError: If sharding parameters are partially specified or out of range.
        """
        _validate_sharding_params(
            rank=self.rank,
            world_size=self.world_size,
        )


@dataclass(frozen=True)
class FetchABLPInputRequest:
    """Request for fetching ABLP input from a storage server.

    Args:
        split: The split of the dataset to get ABLP input from.
        node_type: The type of anchor nodes to retrieve.
        supervision_edge_type: The edge type used for supervision.
        rank: Which partition of the anchor nodes to return (0-indexed).
            Must be provided together with ``world_size``.
        world_size: Total number of partitions.
            Must be provided together with ``rank``.

    Examples:
        Fetch training ABLP input without splitting:

        >>> FetchABLPInputRequest(split="train", node_type="user", supervision_edge_type=("user", "to", "item"))

        Fetch partition 0 of 4 from training ABLP input:

        >>> FetchABLPInputRequest(split="train", node_type="user", supervision_edge_type=("user", "to", "item"), rank=0, world_size=4)
    """

    split: Union[Literal["train", "val", "test"], str]
    node_type: NodeType
    supervision_edge_type: EdgeType
    rank: Optional[int] = None
    world_size: Optional[int] = None

    def validate(self) -> None:
        """Validate that the request has consistent sharding parameters.

        Raises:
            ValueError: If sharding parameters are partially specified or out of range.
        """
        _validate_sharding_params(
            rank=self.rank,
            world_size=self.world_size,
        )
