"""RPC request messages for graph-store fetch operations."""

from dataclasses import dataclass
from typing import Literal, Optional, Union

from gigl.distributed.graph_store.sharding import ServerSlice
from gigl.src.common.types.graph_data import EdgeType, NodeType


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
