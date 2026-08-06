"""Construction-time typed-PPR option parsing for distributed samplers.

The helpers in this module validate typed-channel edge-type keys, convert
public edge-type keys into the compact integer traversal maps consumed by the
C++ forward-push kernel, and parse channel quotas. They run once during
``DistPPRNeighborSampler`` initialization, not in the per-batch PPR sampling hot
loop.
"""

from collections.abc import Sequence
from typing import Optional, Union, cast

from graphlearn_torch.typing import EdgeType, NodeType

# Public typed_channel_quotas keys can be a single edge type or a grouped
# channel containing multiple edge types.
TypedPPRChannelKey = Union[EdgeType, tuple[EdgeType, ...]]

"""TypedPPRChannelKey describes one public typed-PPR traversal channel key.

A single canonical edge type creates one channel restricted to that edge type.
A tuple of canonical edge types creates one channel whose forward-push state may
traverse any edge type in the group. When typed PPR emits multi-column
``edge_attr`` tensors, channel columns follow the insertion order of the
``typed_channel_quotas`` mapping.
"""
# Parsed typed-channel edge-type allowlists, ordered to match the insertion
# order of typed_channel_quotas.
TypedPPRChannelEdgeTypeGroups = list[tuple[EdgeType, ...]]
# One channel's traversal map. The outer list is indexed by integer node-type
# ID; each inner list contains the integer edge-type IDs that channel may
# traverse from that node type.
TypedPPRChannelTraversalMap = list[list[int]]
# All typed-channel traversal maps, ordered to match typed-channel order.
TypedPPRChannelTraversalMaps = list[TypedPPRChannelTraversalMap]


def parse_typed_channel_quota_groups(
    typed_channel_quotas: Optional[dict[TypedPPRChannelKey, int]],
) -> tuple[Optional[TypedPPRChannelEdgeTypeGroups], Optional[list[int]]]:
    """Parse typed-PPR channel keys and split keys from integer quotas.

    Public options allow each channel key to be either one canonical edge type
    or a non-empty tuple of canonical edge types. Internally, traversal setup
    needs only the edge-type groups while merge selection needs the aligned
    quota values, so this helper returns those two parallel lists.

    This is construction-time option parsing and is not part of the per-batch
    PPR sampling hot loop.

    Args:
        typed_channel_quotas: User-provided channel mapping from edge-type
            allowlist to candidate quota.

    Returns:
        ``(None, None)`` when typed PPR is disabled. Otherwise returns
        ``(typed_channel_groups, typed_channel_quota_list)``, both ordered by
        the input mapping insertion order.

    Raises:
        ValueError: If a channel key is not a canonical edge type or non-empty
            tuple of canonical edge types, or if quotas are not positive
            integers.
    """
    if not typed_channel_quotas:
        return None, None

    typed_channel_groups: TypedPPRChannelEdgeTypeGroups = []
    typed_channel_quota_list: list[int] = []

    def is_canonical_edge_type(value: object) -> bool:
        """Return whether ``value`` has PyG's canonical edge-type shape."""
        return (
            isinstance(value, tuple)
            and len(value) == 3
            and all(isinstance(part, str) for part in value)
        )

    for edge_type_key, quota in typed_channel_quotas.items():
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
        if not isinstance(quota, int) or isinstance(quota, bool) or quota <= 0:
            raise ValueError(
                "typed_channel_quotas values must be positive integer quotas, "
                f"got {quota!r} for channel {edge_type_key!r}."
            )
        typed_channel_groups.append(edge_types)
        typed_channel_quota_list.append(quota)

    return typed_channel_groups, typed_channel_quota_list


def build_edge_type_channel_group_edge_type_ids(
    edge_type_groups: TypedPPRChannelEdgeTypeGroups,
    edge_type_to_edge_type_id: dict[EdgeType, int],
    node_type_to_edge_types: dict[NodeType, list[EdgeType]],
    node_types: Sequence[NodeType],
) -> TypedPPRChannelTraversalMaps:
    """Convert typed-channel edge-type allowlists to PPRForwardPush IDs.

    Returns one traversal map per typed channel, ordered to match
    ``edge_type_groups``. A single traversal map has shape
    ``list[list[int]]``: the outer index is ``node_type_id``, and the inner list
    contains allowed ``edge_type_id`` values for that node type in that channel.

    This conversion runs once during sampler construction; the resulting integer
    maps are reused by the per-batch C++ PPR states.

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
        ``channel_traversal_maps[channel_id][node_type_id]`` gives the allowed
        integer edge-type IDs that channel may traverse from that node type.

    Raises:
        ValueError: If a configured edge type is unknown, excluded from PPR
            traversal, or cannot be traversed from any node type.
    """
    known_edge_types = set(edge_type_to_edge_type_id.keys())
    channel_edge_type_ids_by_node_type: TypedPPRChannelTraversalMaps = []
    for channel_edge_types in edge_type_groups:
        unknown_edge_types = set(channel_edge_types) - known_edge_types
        if unknown_edge_types:
            raise ValueError(
                "typed_channel_quotas includes non-traversable edge types "
                f"{sorted(unknown_edge_types)!r}. Known traversable edge "
                f"types are {sorted(known_edge_types)!r}."
            )

        channel_edge_type_set = set(channel_edge_types)
        node_type_id_to_channel_edge_type_ids: TypedPPRChannelTraversalMap = []
        for node_type in node_types:
            channel_edge_type_ids_for_node_type: list[int] = []
            for edge_type in node_type_to_edge_types.get(node_type, []):
                if edge_type in channel_edge_type_set:
                    channel_edge_type_ids_for_node_type.append(
                        edge_type_to_edge_type_id[edge_type]
                    )
            node_type_id_to_channel_edge_type_ids.append(
                channel_edge_type_ids_for_node_type
            )
        if not any(node_type_id_to_channel_edge_type_ids):
            raise ValueError(
                "typed_channel_quotas includes edge-type "
                f"channel={channel_edge_types!r}, "
                "but no traversable edge types exist for that channel."
            )
        channel_edge_type_ids_by_node_type.append(node_type_id_to_channel_edge_type_ids)
    return channel_edge_type_ids_by_node_type
