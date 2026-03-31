"""RPC request messages for graph-store fetch operations."""

from dataclasses import dataclass
from typing import Literal, Optional, Union

from gigl.src.common.types.graph_data import EdgeType, NodeType


def _validate_sharding_mode(
    rank: Optional[int],
    world_size: Optional[int],
    shard_index: Optional[int],
    num_shards: Optional[int],
) -> None:
    """Validate that requests use at most one sharding mode."""
    if (rank is None) ^ (world_size is None):
        raise ValueError(
            "rank and world_size must be provided together. "
            f"Received rank={rank}, world_size={world_size}"
        )
    if (shard_index is None) ^ (num_shards is None):
        raise ValueError(
            "shard_index and num_shards must be provided together. "
            f"Received shard_index={shard_index}, num_shards={num_shards}"
        )
    if rank is not None and shard_index is not None:
        raise ValueError(
            "rank/world_size and shard_index/num_shards are mutually exclusive. "
            f"Received rank={rank}, world_size={world_size}, shard_index={shard_index}, num_shards={num_shards}"
        )
    if shard_index is not None and num_shards is not None:
        if num_shards <= 0:
            raise ValueError(f"num_shards must be > 0, received {num_shards}")
        if shard_index < 0 or shard_index >= num_shards:
            raise ValueError(
                "shard_index must be in [0, num_shards). "
                f"Received shard_index={shard_index}, num_shards={num_shards}"
            )


@dataclass(frozen=True)
class FetchNodesRequest:
    """Request for fetching node IDs from a storage server.

    Args:
        rank: The rank of the process requesting node ids.
            Must be provided together with ``world_size``.
        world_size: The total number of processes in the distributed setup.
            Must be provided together with ``rank``.
        shard_index: The local shard index to fetch from this storage rank.
            Must be provided together with ``num_shards``.
        num_shards: The total number of local shards for this storage rank.
            Must be provided together with ``shard_index``.
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
    shard_index: Optional[int] = None
    num_shards: Optional[int] = None
    split: Optional[Union[Literal["train", "val", "test"], str]] = None
    node_type: Optional[NodeType] = None

    def validate(self) -> None:
        """Validate that the request has a consistent sharding mode.

        Raises:
            ValueError: If the request mixes or partially specifies sharding modes.
        """
        _validate_sharding_mode(
            rank=self.rank,
            world_size=self.world_size,
            shard_index=self.shard_index,
            num_shards=self.num_shards,
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
        shard_index: The local shard index to fetch from this storage rank.
            Must be provided together with ``num_shards``.
        num_shards: The total number of local shards for this storage rank.
            Must be provided together with ``shard_index``.

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
    shard_index: Optional[int] = None
    num_shards: Optional[int] = None

    def validate(self) -> None:
        """Validate that the request has a consistent sharding mode.

        Raises:
            ValueError: If the request mixes or partially specifies sharding modes.
        """
        _validate_sharding_mode(
            rank=self.rank,
            world_size=self.world_size,
            shard_index=self.shard_index,
            num_shards=self.num_shards,
        )
