"""Typed-PPR helpers for distributed sampler implementations."""

from typing import Optional, Sequence, Union, cast

import torch
from graphlearn_torch.typing import EdgeType, NodeType

# C++ PPR extraction output: flat node IDs, flat weights, and per-seed valid
# counts. Homogeneous extraction uses tensors directly; heterogeneous extraction
# uses dictionaries keyed by node type.
PPRResult = tuple[
    Union[torch.Tensor, dict[NodeType, torch.Tensor]],
    Union[torch.Tensor, dict[NodeType, torch.Tensor]],
    Union[torch.Tensor, dict[NodeType, torch.Tensor]],
]
# Heterogeneous-only view of PPRResult after typed PPR extraction.
HeteroPPRResult = tuple[
    dict[NodeType, torch.Tensor],
    dict[NodeType, torch.Tensor],
    dict[NodeType, torch.Tensor],
]
# Public typed_channel_quotas keys can be a single edge type or a grouped
# channel containing multiple edge types.
TypedPPRChannelKey = Union[EdgeType, tuple[EdgeType, ...]]
# Parsed typed-channel edge-type allowlists, ordered to match quota order.
TypedPPRChannelEdgeTypeGroups = list[tuple[EdgeType, ...]]


def parse_typed_channel_quota_groups(
    typed_channel_quotas: Optional[dict[TypedPPRChannelKey, int]],
) -> tuple[Optional[TypedPPRChannelEdgeTypeGroups], Optional[list[int]]]:
    """Validate typed-PPR channel keys and split keys from quotas.

    Public options allow each channel key to be either one canonical edge type
    or a non-empty tuple of canonical edge types. Internally, traversal setup
    needs only the edge-type groups while merge selection needs the aligned
    quota values, so this helper returns those two parallel lists.

    Args:
        typed_channel_quotas: User-provided channel mapping from edge-type
            allowlist to candidate quota.

    Returns:
        ``(None, None)`` when typed PPR is disabled. Otherwise returns
        ``(typed_channel_groups, typed_channel_quota_list)``, both ordered by
        the input mapping insertion order.

    Raises:
        ValueError: If a quota is not a positive integer, or if a channel key
            is not a canonical edge type or non-empty tuple of canonical edge
            types.
    """
    if not typed_channel_quotas:
        return None, None

    typed_channel_groups: TypedPPRChannelEdgeTypeGroups = []
    typed_channel_quota_list: list[int] = []
    invalid_quotas: dict[TypedPPRChannelKey, object] = {}

    def is_canonical_edge_type(value: object) -> bool:
        """Return whether ``value`` has GraphLearn's canonical EdgeType shape."""
        return (
            isinstance(value, tuple)
            and len(value) == 3
            and all(isinstance(part, str) for part in value)
        )

    for edge_type_key, quota in typed_channel_quotas.items():
        if not isinstance(quota, int) or isinstance(quota, bool) or quota <= 0:
            invalid_quotas[edge_type_key] = quota
            continue
        if is_canonical_edge_type(edge_type_key):
            edge_types = (cast(EdgeType, edge_type_key),)
        elif (
            isinstance(edge_type_key, tuple)
            and edge_type_key
            and all(is_canonical_edge_type(edge_type) for edge_type in edge_type_key)
        ):
            edge_types = cast(tuple[EdgeType, ...], edge_type_key)
        else:
            raise ValueError(
                "typed_channel_quotas keys must be a canonical edge type "
                "(src_type, relation, dst_type) or a non-empty tuple of "
                f"canonical edge types, got {edge_type_key!r}."
            )
        typed_channel_groups.append(edge_types)
        typed_channel_quota_list.append(quota)

    if invalid_quotas:
        raise ValueError(
            "typed_channel_quotas must contain only positive integer quotas, "
            f"got {invalid_quotas}."
        )
    return typed_channel_groups, typed_channel_quota_list


def build_edge_type_channel_group_edge_type_ids(
    edge_type_groups: TypedPPRChannelEdgeTypeGroups,
    edge_type_to_edge_type_id: dict[EdgeType, int],
    node_type_to_edge_types: dict[NodeType, list[EdgeType]],
    node_types: Sequence[NodeType],
) -> list[list[list[int]]]:
    """Convert typed-channel edge-type allowlists to PPRForwardPush IDs.

    Returns one traversal map per typed channel. Each traversal map is indexed
    as ``[node_type_id][allowed_edge_type_ids]`` and tells the C++ forward push
    state which edge-type IDs may be traversed from each node type for that
    channel.

    Args:
        edge_type_groups: Ordered typed channels, where each channel is the
            canonical edge types that its PPR state may traverse.
        edge_type_to_edge_type_id: Mapping from canonical edge type to the
            compact integer ID used by the C++ forward-push kernel.
        node_type_to_edge_types: Traversable edge types keyed by anchor node
            type, after label-edge filtering and edge-direction handling.
        node_types: Ordered node types whose positions match the kernel's
            integer node-type IDs.

    Returns:
        A list of traversal maps aligned with ``edge_type_groups``. Each
        traversal map is indexed by integer node-type ID and stores the allowed
        integer edge-type IDs for that node type.

    Raises:
        ValueError: If a configured edge type is unknown, excluded from PPR
            traversal, or cannot be traversed from any node type.
    """
    known_edge_types = set(edge_type_to_edge_type_id.keys())
    channel_edge_type_ids_by_node_type: list[list[list[int]]] = []
    for channel_edge_types in edge_type_groups:
        unknown_edge_types = set(channel_edge_types) - known_edge_types
        if unknown_edge_types:
            raise ValueError(
                "typed_channel_quotas includes non-traversable edge types "
                f"{sorted(unknown_edge_types)!r}. Edge types must exist in the "
                "graph and must not be label edge types."
            )

        channel_edge_type_set = set(channel_edge_types)
        node_type_id_to_channel_edge_type_ids = [
            [
                edge_type_to_edge_type_id[edge_type]
                for edge_type in node_type_to_edge_types.get(node_type, [])
                if edge_type in channel_edge_type_set
            ]
            for node_type in node_types
        ]
        if not any(node_type_id_to_channel_edge_type_ids):
            raise ValueError(
                "typed_channel_quotas includes edge-type "
                f"channel={channel_edge_types!r}, "
                "but no traversable edge types exist for that channel."
            )
        channel_edge_type_ids_by_node_type.append(node_type_id_to_channel_edge_type_ids)
    return channel_edge_type_ids_by_node_type
