"""Typed-PPR parsing and merge helpers for distributed PPR sampling."""

import math
from collections import defaultdict
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
# Per node type and seed, stores node_id -> typed edge_attr feature vector.
_TypedPPRScoreMap = dict[NodeType, list[dict[int, list[float]]]]
# Per node type, seed, and channel, stores candidate node IDs with their
# calibrated score for quota-aware selection.
_TypedPPRCandidates = dict[NodeType, list[list[list[tuple[int, float]]]]]
# Pair of score features and channel candidates used during typed-PPR merging.
_TypedPPRMergeState = tuple[_TypedPPRScoreMap, _TypedPPRCandidates]


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


def merge_typed_ppr_results(
    channel_results: Sequence[PPRResult],
    channel_quotas: Sequence[int],
    max_ppr_nodes: int,
    device: torch.device,
) -> HeteroPPRResult:
    """Merge typed-PPR channel results into heterogeneous PPR output.

    Each channel is extracted with finalized-PPR candidates first and residual
    top-up candidates after them. The base merge honors the configured channel
    quotas. If that result is short, remaining candidates from all channels
    compete globally by calibrated score until ``max_ppr_nodes`` is reached.

    Args:
        channel_results: Per-channel extracted PPR results, ordered to match
            ``channel_quotas``.
        channel_quotas: Per-channel finalized-PPR candidate limits for the
            first merge pass.
        max_ppr_nodes: Final per-seed cap across all typed channels.
        device: Device for returned tensors.

    Returns:
        Heterogeneous PPR result dictionaries keyed by destination node type.
    """
    num_channels = len(channel_quotas)
    num_edge_attr_features = 1 + (2 * num_channels)
    num_seeds = 0
    for _flat_ids, _flat_weights, node_type_to_valid_counts in channel_results:
        assert isinstance(node_type_to_valid_counts, dict)
        for valid_counts in node_type_to_valid_counts.values():
            num_seeds = valid_counts.numel()
            break
        if num_seeds > 0:
            break
    if num_seeds == 0:
        return {}, {}, {}

    base_scores, base_candidates = _build_typed_ppr_merge_state(
        num_seeds=num_seeds,
        num_channels=num_channels,
    )
    extended_scores, extended_candidates = _build_typed_ppr_merge_state(
        num_seeds=num_seeds,
        num_channels=num_channels,
    )
    _populate_typed_ppr_topup_merge_states(
        channel_results=channel_results,
        channel_quotas=channel_quotas,
        base_scores=base_scores,
        base_candidates=base_candidates,
        extended_scores=extended_scores,
        extended_candidates=extended_candidates,
        num_edge_attr_features=num_edge_attr_features,
        num_channels=num_channels,
    )

    node_type_to_flat_ids_out: dict[NodeType, torch.Tensor] = {}
    node_type_to_flat_weights_out: dict[NodeType, torch.Tensor] = {}
    node_type_to_valid_counts_out: dict[NodeType, torch.Tensor] = {}
    node_types = set(base_scores.keys()) | set(extended_scores.keys())
    topup_channel_quotas = [max_ppr_nodes for _ in range(num_channels)]

    for node_type in node_types:
        flat_ids: list[int] = []
        flat_weights: list[list[float]] = []
        valid_counts: list[int] = []
        base_seed_scores_by_node_type = base_scores.get(node_type)
        base_candidates_by_node_type = base_candidates.get(node_type)
        extended_seed_scores_by_node_type = extended_scores.get(node_type)
        extended_candidates_by_node_type = extended_candidates.get(node_type)

        for seed_index in range(num_seeds):
            selected_nodes: list[int] = []
            selected_node_ids: set[int] = set()

            if (
                base_seed_scores_by_node_type is not None
                and base_candidates_by_node_type is not None
            ):
                base_selected_nodes = _select_typed_ppr_node_ids(
                    seed_scores=base_seed_scores_by_node_type[seed_index],
                    candidates_by_channel=base_candidates_by_node_type[seed_index],
                    channel_quotas=channel_quotas,
                    max_ppr_nodes=max_ppr_nodes,
                )
                selected_nodes.extend(base_selected_nodes)
                selected_node_ids.update(base_selected_nodes)
                flat_weights.extend(
                    base_seed_scores_by_node_type[seed_index][node_id]
                    for node_id in base_selected_nodes
                )

            if (
                len(selected_nodes) < max_ppr_nodes
                and extended_seed_scores_by_node_type is not None
                and extended_candidates_by_node_type is not None
            ):
                extended_selected_nodes = _select_typed_ppr_node_ids(
                    seed_scores=extended_seed_scores_by_node_type[seed_index],
                    candidates_by_channel=extended_candidates_by_node_type[seed_index],
                    channel_quotas=topup_channel_quotas,
                    max_ppr_nodes=max_ppr_nodes,
                )
                for node_id in extended_selected_nodes:
                    if len(selected_nodes) >= max_ppr_nodes:
                        break
                    if node_id in selected_node_ids:
                        continue
                    selected_nodes.append(node_id)
                    selected_node_ids.add(node_id)
                    flat_weights.append(
                        extended_seed_scores_by_node_type[seed_index][node_id]
                    )

            valid_counts.append(len(selected_nodes))
            flat_ids.extend(selected_nodes)

        node_type_to_flat_ids_out[node_type] = torch.tensor(
            flat_ids,
            dtype=torch.long,
            device=device,
        )
        node_type_to_flat_weights_out[node_type] = torch.tensor(
            flat_weights,
            dtype=torch.double,
            device=device,
        ).reshape(
            len(flat_weights),
            num_edge_attr_features,
        )
        node_type_to_valid_counts_out[node_type] = torch.tensor(
            valid_counts,
            dtype=torch.long,
            device=device,
        )

    return (
        node_type_to_flat_ids_out,
        node_type_to_flat_weights_out,
        node_type_to_valid_counts_out,
    )


def _build_typed_ppr_merge_state(
    num_seeds: int,
    num_channels: int,
) -> _TypedPPRMergeState:
    """Create empty typed-PPR merge containers.

    The score map stores one feature vector per selected node. The candidate
    map stores per-channel candidate ordering so selection can first honor
    channel quotas and then perform a global calibrated-score ranking.

    Args:
        num_seeds: Number of seed nodes in the batch.
        num_channels: Number of typed-PPR traversal channels.

    Returns:
        Empty score and candidate containers keyed lazily by node type.
    """
    merged_scores: _TypedPPRScoreMap = defaultdict(
        lambda: [dict() for _ in range(num_seeds)]
    )
    channel_candidates: _TypedPPRCandidates = defaultdict(
        lambda: [[[] for _ in range(num_channels)] for _ in range(num_seeds)]
    )
    return merged_scores, channel_candidates


def _populate_typed_ppr_topup_merge_states(
    channel_results: Sequence[PPRResult],
    channel_quotas: Sequence[int],
    base_scores: _TypedPPRScoreMap,
    base_candidates: _TypedPPRCandidates,
    extended_scores: _TypedPPRScoreMap,
    extended_candidates: _TypedPPRCandidates,
    num_edge_attr_features: int,
    num_channels: int,
) -> None:
    """Populate base and top-up candidate pools for typed-PPR merging.

    For each channel and seed, the C++ extraction output is already ordered
    with finalized-PPR candidates first and residual top-up candidates after
    them. This helper builds two merge states: one constrained to the base
    channel quota and one containing every candidate returned by extraction.

    Both states use the full returned candidate pool's max score for
    calibration so finalized and residual-backed candidates are comparable
    within a channel.

    Args:
        channel_results: Per-channel extracted PPR results.
        channel_quotas: Per-channel finalized-PPR candidate limits.
        base_scores: Output score-feature map for the base merge.
        base_candidates: Output per-channel candidates for the base merge.
        extended_scores: Output score-feature map for residual top-up.
        extended_candidates: Output per-channel candidates for residual top-up.
        num_edge_attr_features: Width of the typed-PPR edge-attribute vector.
        num_channels: Number of typed-PPR channels.
    """
    for channel_index, (
        node_type_to_flat_ids,
        node_type_to_flat_weights,
        node_type_to_valid_counts,
    ) in enumerate(channel_results):
        assert isinstance(node_type_to_flat_ids, dict)
        assert isinstance(node_type_to_flat_weights, dict)
        assert isinstance(node_type_to_valid_counts, dict)

        channel_quota = channel_quotas[channel_index]

        for node_type, flat_ids in node_type_to_flat_ids.items():
            flat_ids_cpu = flat_ids.detach().cpu().tolist()
            flat_weights_cpu = (
                node_type_to_flat_weights[node_type].detach().cpu().tolist()
            )
            valid_counts_cpu = (
                node_type_to_valid_counts[node_type].detach().cpu().tolist()
            )
            flat_offset = 0

            for seed_index, valid_count in enumerate(valid_counts_cpu):
                base_nodes_and_scores: list[tuple[int, float]] = []
                extended_nodes_and_scores: list[tuple[int, float]] = []
                extended_max_score = 0.0

                for candidate_index, (node_id, raw_score) in enumerate(
                    zip(
                        flat_ids_cpu[flat_offset : flat_offset + valid_count],
                        flat_weights_cpu[flat_offset : flat_offset + valid_count],
                    )
                ):
                    if not math.isfinite(raw_score):
                        continue
                    score = min(max(raw_score, 0.0), 1.0)
                    extended_nodes_and_scores.append((node_id, score))
                    extended_max_score = max(extended_max_score, score)
                    if candidate_index < channel_quota:
                        base_nodes_and_scores.append((node_id, score))

                flat_offset += valid_count

                # Both base and residual top-up candidates are calibrated using
                # the extended candidate pool's max score. This keeps finalized
                # PPR nodes and residual top-up nodes on the same per-channel
                # scale before the global typed-PPR merge.
                _add_typed_ppr_seed_candidates(
                    seed_scores=base_scores[node_type][seed_index],
                    seed_channel_candidates=base_candidates[node_type][seed_index][
                        channel_index
                    ],
                    seed_nodes_and_scores=base_nodes_and_scores,
                    max_score=extended_max_score,
                    channel_index=channel_index,
                    num_edge_attr_features=num_edge_attr_features,
                    num_channels=num_channels,
                )
                _add_typed_ppr_seed_candidates(
                    seed_scores=extended_scores[node_type][seed_index],
                    seed_channel_candidates=extended_candidates[node_type][seed_index][
                        channel_index
                    ],
                    seed_nodes_and_scores=extended_nodes_and_scores,
                    max_score=extended_max_score,
                    channel_index=channel_index,
                    num_edge_attr_features=num_edge_attr_features,
                    num_channels=num_channels,
                )


def _add_typed_ppr_seed_candidates(
    seed_scores: dict[int, list[float]],
    seed_channel_candidates: list[tuple[int, float]],
    seed_nodes_and_scores: Sequence[tuple[int, float]],
    max_score: float,
    channel_index: int,
    num_edge_attr_features: int,
    num_channels: int,
) -> None:
    """Add one seed's candidates from one typed channel into merge state.

    This updates each node's edge-attribute feature vector with the best
    calibrated score, the channel-specific calibrated score, and the channel
    presence bit. It also appends the node to the channel candidate list used
    for quota-aware selection.

    Args:
        seed_scores: Node feature vectors for one node type and seed.
        seed_channel_candidates: Candidate list for this seed and channel.
        seed_nodes_and_scores: Raw candidate node IDs and channel scores.
        max_score: Max score in the channel's extended candidate pool.
        channel_index: Typed-channel index.
        num_edge_attr_features: Width of the typed-PPR edge-attribute vector.
        num_channels: Number of typed-PPR channels.
    """
    for node_id, score in seed_nodes_and_scores:
        calibrated_score = score / max_score if max_score > 0 else 0.0
        score_features = seed_scores.get(node_id)
        if score_features is None:
            score_features = [0.0] * num_edge_attr_features
            seed_scores[node_id] = score_features
        score_features[0] = max(score_features[0], calibrated_score)
        channel_score_index = 1 + channel_index
        channel_presence_index = 1 + num_channels + channel_index
        score_features[channel_score_index] = max(
            score_features[channel_score_index], calibrated_score
        )
        score_features[channel_presence_index] = 1.0
        seed_channel_candidates.append((node_id, calibrated_score))


def _select_typed_ppr_node_ids(
    seed_scores: dict[int, list[float]],
    candidates_by_channel: Sequence[Sequence[tuple[int, float]]],
    channel_quotas: Sequence[int],
    max_ppr_nodes: int,
) -> list[int]:
    """Select typed-PPR nodes by channel quota, then global calibrated rank.

    Each channel first contributes its own top candidates by that channel's
    calibrated score. The pooled candidates are then globally ranked by the
    best calibrated score across all channels for the seed.

    Args:
        seed_scores: Node feature vectors for one node type and seed.
        candidates_by_channel: Candidate node IDs and calibrated scores for
            each typed channel.
        channel_quotas: Per-channel candidate caps.
        max_ppr_nodes: Final per-seed cap across all typed channels.

    Returns:
        Selected node IDs in the final typed-PPR order for one seed and node
        type.
    """
    selected_node_ids: set[int] = set()
    selected_nodes: list[int] = []

    sorted_candidates_by_channel = [
        sorted(
            candidates,
            key=lambda item: (-item[1], item[0]),
        )
        for candidates in candidates_by_channel
    ]
    global_candidates: list[tuple[float, float, int, int]] = []
    for channel_index, candidates in enumerate(sorted_candidates_by_channel):
        channel_quota = channel_quotas[channel_index]
        for node_id, calibrated_score in candidates[:channel_quota]:
            global_candidates.append(
                (
                    seed_scores[node_id][0],
                    calibrated_score,
                    channel_index,
                    node_id,
                )
            )

    for (
        _best_calibrated_score,
        _channel_calibrated_score,
        _channel_index,
        node_id,
    ) in sorted(
        global_candidates,
        key=lambda item: (-item[0], -item[1], item[2], item[3]),
    ):
        if len(selected_nodes) >= max_ppr_nodes:
            break
        if node_id in selected_node_ids:
            continue
        selected_node_ids.add(node_id)
        selected_nodes.append(node_id)

    return selected_nodes
