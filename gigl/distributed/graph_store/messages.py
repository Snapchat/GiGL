"""RPC request messages for graph-store fetch operations."""

from dataclasses import dataclass
from typing import Literal, Optional, Union

from gigl.src.common.types.graph_data import EdgeType, NodeType


def _validate_split_params(
    split_idx: Optional[int],
    num_splits: Optional[int],
) -> None:
    """Validate that split parameters are consistent.

    Args:
        split_idx: The index of the split to select.
        num_splits: The total number of splits.

    Raises:
        ValueError: If only one of ``split_idx``/``num_splits`` is provided,
            or if the values are out of range.
    """
    if (split_idx is None) ^ (num_splits is None):
        raise ValueError(
            "split_idx and num_splits must be provided together. "
            f"Received split_idx={split_idx}, num_splits={num_splits}"
        )
    if split_idx is not None and num_splits is not None:
        if num_splits <= 0:
            raise ValueError(f"num_splits must be > 0, received {num_splits}")
        if split_idx < 0 or split_idx >= num_splits:
            raise ValueError(
                "split_idx must be in [0, num_splits). "
                f"Received split_idx={split_idx}, num_splits={num_splits}"
            )


@dataclass(frozen=True)
class FetchNodesRequest:
    """Request for fetching node IDs from a storage server.

    Args:
        split_idx: Which partition of the node IDs to return (0-indexed).
            Must be provided together with ``num_splits``.
        num_splits: Total number of partitions.
            Must be provided together with ``split_idx``.
        split: The split of the dataset to get node ids from.
        node_type: The type of nodes to get node ids for.

    Examples:
        Fetch all nodes without splitting:

        >>> FetchNodesRequest()

        Fetch partition 0 of 4 from training nodes:

        >>> FetchNodesRequest(split_idx=0, num_splits=4, split="train")

        Fetch nodes of a specific type:

        >>> FetchNodesRequest(node_type="user")
    """

    split_idx: Optional[int] = None
    num_splits: Optional[int] = None
    split: Optional[Union[Literal["train", "val", "test"], str]] = None
    node_type: Optional[NodeType] = None

    def validate(self) -> None:
        """Validate that the request has consistent split parameters.

        Raises:
            ValueError: If split parameters are partially specified or out of range.
        """
        _validate_split_params(
            split_idx=self.split_idx,
            num_splits=self.num_splits,
        )


@dataclass(frozen=True)
class FetchABLPInputRequest:
    """Request for fetching ABLP input from a storage server.

    Args:
        split: The split of the dataset to get ABLP input from.
        node_type: The type of anchor nodes to retrieve.
        supervision_edge_type: The edge type used for supervision.
        split_idx: Which partition of the anchor nodes to return (0-indexed).
            Must be provided together with ``num_splits``.
        num_splits: Total number of partitions.
            Must be provided together with ``split_idx``.

    Examples:
        Fetch training ABLP input without splitting:

        >>> FetchABLPInputRequest(split="train", node_type="user", supervision_edge_type=("user", "to", "item"))

        Fetch partition 0 of 4 from training ABLP input:

        >>> FetchABLPInputRequest(split="train", node_type="user", supervision_edge_type=("user", "to", "item"), split_idx=0, num_splits=4)
    """

    split: Union[Literal["train", "val", "test"], str]
    node_type: NodeType
    supervision_edge_type: EdgeType
    split_idx: Optional[int] = None
    num_splits: Optional[int] = None

    def validate(self) -> None:
        """Validate that the request has consistent split parameters.

        Raises:
            ValueError: If split parameters are partially specified or out of range.
        """
        _validate_split_params(
            split_idx=self.split_idx,
            num_splits=self.num_splits,
        )
