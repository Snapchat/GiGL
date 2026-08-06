"""Construction-time typed-PPR option parsing for distributed samplers."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Union, cast

from graphlearn_torch.typing import EdgeType, NodeType

TypedPPRMetaPathStep = Union[EdgeType, tuple[EdgeType, ...]]


@dataclass(frozen=True)
class PPRMetaPath:
    """Ordered typed-PPR channel pattern using PyG's metapath convention.

    This follows the existing PyG API shape: ``AddMetaPaths`` accepts
    ``metapaths`` as a list of lists of ``(src_type, rel_type, dst_type)``
    tuples, and ``MetaPath2Vec`` accepts one ``metapath`` as an ordered list of
    those same canonical edge-type tuples. ``PPRMetaPath`` keeps that ordered
    edge-type-sequence model for typed PPR channels: the path is traversed in
    order, and each output channel column corresponds to one user-provided key
    in ``typed_channel_ratios``.

    A plain PyG metapath has one canonical edge type at each step. GiGL extends
    that only where typed channels already support grouping: one step may be a
    tuple of canonical edge types, meaning the PPR walk may choose any of those
    relations at that exact position in the ordered path. Grouped alternatives
    must share the same traversal source and destination node types after
    applying the sampler's ``edge_dir``.

    ``cyclic_from`` makes the suffix beginning at that step repeat. This mirrors
    the intuitive cyclic metapath behavior used by random-walk APIs: a complete
    repeated segment can run for as long as PPR residual mass remains. For
    example, ``PPRMetaPath((A_TO_B, B_TO_A), cyclic_from=0)`` represents
    ``(A_TO_B, B_TO_A)*`` and emits only after at least one full cycle.

    Args:
        path: Ordered metapath steps. Each step is either one canonical edge
            type or a non-empty tuple of canonical edge types that are
            alternatives for that step.
        cyclic_from: Optional step index where the repeated suffix begins.
            ``None`` means the metapath is finite and emits only after the full
            path. ``0`` repeats the entire path. Values greater than ``0`` run
            the prefix once and then repeat the suffix.
    """

    path: tuple[TypedPPRMetaPathStep, ...]
    cyclic_from: Optional[int] = None


TypedPPRChannelKey = Union[EdgeType, tuple[EdgeType, ...], PPRMetaPath]
TypedPPRChannelSpec = Union[tuple[EdgeType, ...], PPRMetaPath]
TypedPPRChannelSpecs = list[TypedPPRChannelSpec]
TypedPPRChannelTraversalMap = list[list[int]]
TypedPPRChannelTraversalMaps = list[TypedPPRChannelTraversalMap]
TypedPPRTraversalTransition = tuple[int, int]
TypedPPRChannelTraversalProgram = list[list[list[TypedPPRTraversalTransition]]]
TypedPPRChannelTraversalPrograms = list[TypedPPRChannelTraversalProgram]
TypedPPRChannelEmittingStateIds = list[list[int]]


def _is_canonical_edge_type(value: object) -> bool:
    """Return whether ``value`` has PyG's canonical edge-type shape."""
    return (
        isinstance(value, tuple)
        and len(value) == 3
        and all(isinstance(part, str) for part in value)
    )


def _normalize_metapath_step(step: TypedPPRMetaPathStep) -> tuple[EdgeType, ...]:
    """Normalize a metapath step to a non-empty tuple of edge-type alternatives."""
    if _is_canonical_edge_type(step):
        return (cast(EdgeType, step),)
    if (
        isinstance(step, tuple)
        and step
        and all(_is_canonical_edge_type(edge_type) for edge_type in step)
    ):
        return cast(tuple[EdgeType, ...], step)
    raise ValueError(
        "PPRMetaPath path steps must be canonical edge types or non-empty "
        f"tuples of canonical edge types, got {step!r}."
    )


def parse_typed_channel_ratio_specs(
    typed_channel_ratios: Optional[dict[TypedPPRChannelKey, float]],
) -> tuple[Optional[TypedPPRChannelSpecs], Optional[list[float]]]:
    """Parse typed-PPR channel keys and split keys from ratios.

    Public options allow each channel key to be either one canonical edge type
    or a non-empty tuple of canonical edge types. These existing key shapes keep
    their allowlist semantics: a channel may use any edge type in the group at
    each step.

    ``PPRMetaPath`` keys opt into ordered metapath semantics.

    Internally, traversal setup needs the ordered channel specs while merge
    selection needs the aligned ratio values, so this helper returns those two
    parallel lists.

    This is construction-time option parsing and is not part of the per-batch
    PPR sampling hot loop.

    Args:
        typed_channel_ratios: User-provided channel mapping from edge-type
            allowlist to target output ratio.

    Returns:
        ``(None, None)`` when typed PPR is disabled. Otherwise returns
        ``(typed_channel_specs, typed_channel_ratio_list)``, both ordered by
        the input mapping insertion order.

    Raises:
        ValueError: If a channel key is invalid, or if ratios are not positive
            and summing to 1.0.
    """
    if not typed_channel_ratios:
        return None, None

    typed_channel_specs: TypedPPRChannelSpecs = []
    typed_channel_ratio_list: list[float] = []

    for edge_type_key, ratio in typed_channel_ratios.items():
        if _is_canonical_edge_type(edge_type_key):
            channel_spec: TypedPPRChannelSpec = (cast(EdgeType, edge_type_key),)
        elif (
            isinstance(edge_type_key, tuple)
            and edge_type_key
            and all(_is_canonical_edge_type(edge_type) for edge_type in edge_type_key)
        ):
            channel_spec = cast(tuple[EdgeType, ...], edge_type_key)
        elif isinstance(edge_type_key, PPRMetaPath):
            if not edge_type_key.path:
                raise ValueError("PPRMetaPath path must be non-empty.")
            normalized_path = tuple(
                _normalize_metapath_step(step) for step in edge_type_key.path
            )
            cyclic_from = edge_type_key.cyclic_from
            if cyclic_from is not None and (
                isinstance(cyclic_from, bool)
                or cyclic_from < 0
                or cyclic_from >= len(normalized_path)
            ):
                raise ValueError(
                    "PPRMetaPath cyclic_from must be None or a valid path step "
                    f"index, got {cyclic_from!r} for path {edge_type_key.path!r}."
                )
            channel_spec = PPRMetaPath(
                path=normalized_path,
                cyclic_from=cyclic_from,
            )
        else:
            raise ValueError(
                "typed_channel_ratios keys must be a canonical edge type "
                "(src_type, relation, dst_type) or a non-empty tuple of "
                "canonical edge types, or a PPRMetaPath, "
                f"got {edge_type_key!r}."
            )
        if (
            not isinstance(ratio, (int, float))
            or isinstance(ratio, bool)
            or ratio <= 0.0
            or ratio > 1.0
        ):
            raise ValueError(
                "typed_channel_ratios values must be positive ratios in (0, 1], "
                f"got {ratio!r} for channel {edge_type_key!r}."
            )
        typed_channel_specs.append(channel_spec)
        typed_channel_ratio_list.append(float(ratio))

    ratio_sum = sum(typed_channel_ratio_list)
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            "typed_channel_ratios values must sum to 1.0, "
            f"got {ratio_sum} from ratios {typed_channel_ratio_list}."
        )

    return typed_channel_specs, typed_channel_ratio_list


def parse_typed_channel_ratio_groups(
    typed_channel_ratios: Optional[dict[TypedPPRChannelKey, float]],
) -> tuple[Optional[list[tuple[EdgeType, ...]]], Optional[list[float]]]:
    """Parse legacy allowlist typed-channel keys.

    This compatibility wrapper is kept for callers that expect only the
    pre-metapath edge-type group representation.

    Raises:
        ValueError: If ``typed_channel_ratios`` includes a ``PPRMetaPath`` key.
    """
    typed_channel_specs, typed_channel_ratio_list = parse_typed_channel_ratio_specs(
        typed_channel_ratios
    )
    if typed_channel_specs is None:
        return None, None
    typed_channel_groups: list[tuple[EdgeType, ...]] = []
    for typed_channel_spec in typed_channel_specs:
        if isinstance(typed_channel_spec, PPRMetaPath):
            raise ValueError(
                "parse_typed_channel_ratio_groups cannot return PPRMetaPath "
                "channels; use parse_typed_channel_ratio_specs instead."
            )
        typed_channel_groups.append(typed_channel_spec)
    return typed_channel_groups, typed_channel_ratio_list


def compute_typed_channel_target_counts(
    typed_channel_ratios: list[float],
    max_ppr_nodes: int,
) -> list[int]:
    """Convert typed-channel ratios to integer per-channel target counts.

    Ratios describe the desired attribution mix in the returned PPR sequence.
    This helper converts them to integer counts whose sum is ``max_ppr_nodes``.
    Fractional remainders are assigned from largest to smallest, with channel
    order as the deterministic tie-breaker.

    This is construction-time option parsing and is not part of the per-batch
    PPR sampling hot loop.

    Args:
        typed_channel_ratios: Per-channel ratios, ordered by typed-channel
            insertion order and summing to 1.0.
        max_ppr_nodes: Maximum PPR sequence length per seed.

    Returns:
        Integer target counts aligned with ``typed_channel_ratios``.
    """
    raw_target_counts = [ratio * max_ppr_nodes for ratio in typed_channel_ratios]
    target_counts = [math.floor(raw_count) for raw_count in raw_target_counts]
    remaining_count = max_ppr_nodes - sum(target_counts)
    channels_by_fractional_remainder = sorted(
        range(len(raw_target_counts)),
        key=lambda channel_index: (
            raw_target_counts[channel_index] - target_counts[channel_index],
            -channel_index,
        ),
        reverse=True,
    )
    for channel_index in channels_by_fractional_remainder[:remaining_count]:
        target_counts[channel_index] += 1
    return target_counts


def _get_traversal_source_and_destination(
    edge_type: EdgeType,
    edge_dir: str,
) -> tuple[NodeType, NodeType]:
    """Return the traversal source and destination node types for an edge type."""
    if edge_dir == "in":
        return edge_type[-1], edge_type[0]
    if edge_dir == "out":
        return edge_type[0], edge_type[-1]
    raise ValueError(f"Expected edge_dir to be 'in' or 'out', got {edge_dir!r}.")


def _validate_metapath_steps_chain(
    metapath: PPRMetaPath,
    edge_dir: str,
) -> list[tuple[NodeType, NodeType]]:
    """Validate a metapath and return each step's traversal endpoint types."""
    step_endpoint_types: list[tuple[NodeType, NodeType]] = []
    for step in metapath.path:
        step_edge_types = _normalize_metapath_step(step)
        first_endpoint_types = _get_traversal_source_and_destination(
            step_edge_types[0],
            edge_dir,
        )
        for edge_type in step_edge_types[1:]:
            endpoint_types = _get_traversal_source_and_destination(
                edge_type,
                edge_dir,
            )
            if endpoint_types != first_endpoint_types:
                raise ValueError(
                    "PPRMetaPath grouped-step alternatives must share the same "
                    "traversal source and destination node types, got "
                    f"{step_edge_types!r}."
                )
        step_endpoint_types.append(first_endpoint_types)

    for step_index in range(1, len(step_endpoint_types)):
        previous_destination_type = step_endpoint_types[step_index - 1][1]
        current_source_type = step_endpoint_types[step_index][0]
        if previous_destination_type != current_source_type:
            raise ValueError(
                "PPRMetaPath steps must chain by traversal node type, got "
                f"step {step_index - 1} ending at {previous_destination_type!r} "
                f"and step {step_index} starting at {current_source_type!r}."
            )

    if metapath.cyclic_from is not None:
        repeat_source_type = step_endpoint_types[metapath.cyclic_from][0]
        final_destination_type = step_endpoint_types[-1][1]
        if final_destination_type != repeat_source_type:
            raise ValueError(
                "PPRMetaPath cyclic suffix must return to the repeat start "
                "node type, got final destination "
                f"{final_destination_type!r} and cyclic_from source "
                f"{repeat_source_type!r}."
            )

    return step_endpoint_types


def _add_transition(
    program: TypedPPRChannelTraversalProgram,
    state_id: int,
    source_node_type_id: int,
    edge_type_id: int,
    destination_state_id: int,
) -> None:
    """Add one transition unless the same edge transition already exists."""
    transition = (edge_type_id, destination_state_id)
    transitions = program[state_id][source_node_type_id]
    if transition not in transitions:
        transitions.append(transition)


def _build_allowlist_channel_program(
    edge_types: tuple[EdgeType, ...],
    edge_type_to_edge_type_id: dict[EdgeType, int],
    node_type_to_edge_types: dict[NodeType, list[EdgeType]],
    node_types: Sequence[NodeType],
) -> tuple[TypedPPRChannelTraversalProgram, list[int]]:
    """Build a one-state self-loop program for an edge-type allowlist channel."""
    known_edge_types = set(edge_type_to_edge_type_id.keys())
    unknown_edge_types = set(edge_types) - known_edge_types
    if unknown_edge_types:
        raise ValueError(
            "Typed PPR channels include non-traversable edge types "
            f"{sorted(unknown_edge_types)!r}. Known traversable edge "
            f"types are {sorted(known_edge_types)!r}."
        )

    channel_edge_type_set = set(edge_types)
    program: TypedPPRChannelTraversalProgram = [[[] for _node_type in node_types]]
    for node_type_id, node_type in enumerate(node_types):
        for edge_type in node_type_to_edge_types.get(node_type, []):
            if edge_type in channel_edge_type_set:
                _add_transition(
                    program=program,
                    state_id=0,
                    source_node_type_id=node_type_id,
                    edge_type_id=edge_type_to_edge_type_id[edge_type],
                    destination_state_id=0,
                )
    if not any(program[0]):
        raise ValueError(
            "Typed PPR channels include edge-type "
            f"channel={edge_types!r}, "
            "but no traversable edge types exist for that channel."
        )
    return program, [0]


def _build_metapath_channel_program(
    metapath: PPRMetaPath,
    edge_type_to_edge_type_id: dict[EdgeType, int],
    node_type_to_node_type_id: dict[NodeType, int],
    edge_dir: str,
) -> tuple[TypedPPRChannelTraversalProgram, list[int]]:
    """Build a traversal-state program for one ordered metapath channel."""
    known_edge_types = set(edge_type_to_edge_type_id.keys())
    normalized_steps = [_normalize_metapath_step(step) for step in metapath.path]
    unknown_edge_types = {
        edge_type
        for step_edge_types in normalized_steps
        for edge_type in step_edge_types
        if edge_type not in known_edge_types
    }
    if unknown_edge_types:
        raise ValueError(
            "PPRMetaPath includes non-traversable edge types "
            f"{sorted(unknown_edge_types)!r}. Known traversable edge "
            f"types are {sorted(known_edge_types)!r}."
        )

    step_endpoint_types = _validate_metapath_steps_chain(metapath, edge_dir)
    for source_type, destination_type in step_endpoint_types:
        if source_type not in node_type_to_node_type_id:
            raise ValueError(
                f"PPRMetaPath source node type {source_type!r} is not traversable."
            )
        if destination_type not in node_type_to_node_type_id:
            raise ValueError(
                f"PPRMetaPath destination node type {destination_type!r} is unknown."
            )

    num_steps = len(normalized_steps)
    if metapath.cyclic_from is None:
        num_states = num_steps + 1
        emitting_state_ids = [num_steps]
    elif metapath.cyclic_from == 0:
        num_states = num_steps + 1
        emitting_state_ids = [num_steps]
    else:
        num_states = num_steps
        emitting_state_ids = [metapath.cyclic_from]

    program: TypedPPRChannelTraversalProgram = [
        [[] for _node_type in node_type_to_node_type_id] for _state in range(num_states)
    ]

    for step_index, step_edge_types in enumerate(normalized_steps):
        source_type = step_endpoint_types[step_index][0]
        source_node_type_id = node_type_to_node_type_id[source_type]
        if metapath.cyclic_from is not None and step_index == num_steps - 1:
            destination_state_id = (
                num_steps if metapath.cyclic_from == 0 else metapath.cyclic_from
            )
        else:
            destination_state_id = step_index + 1

        for edge_type in step_edge_types:
            _add_transition(
                program=program,
                state_id=step_index,
                source_node_type_id=source_node_type_id,
                edge_type_id=edge_type_to_edge_type_id[edge_type],
                destination_state_id=destination_state_id,
            )

    if metapath.cyclic_from == 0:
        source_type = step_endpoint_types[0][0]
        source_node_type_id = node_type_to_node_type_id[source_type]
        destination_state_id = 1 if num_steps > 1 else num_steps
        for edge_type in normalized_steps[0]:
            _add_transition(
                program=program,
                state_id=num_steps,
                source_node_type_id=source_node_type_id,
                edge_type_id=edge_type_to_edge_type_id[edge_type],
                destination_state_id=destination_state_id,
            )

    return program, emitting_state_ids


def build_typed_ppr_channel_traversal_programs(
    channel_specs: TypedPPRChannelSpecs,
    edge_type_to_edge_type_id: dict[EdgeType, int],
    node_type_to_edge_types: dict[NodeType, list[EdgeType]],
    node_types: Sequence[NodeType],
    edge_dir: str,
) -> tuple[TypedPPRChannelTraversalPrograms, TypedPPRChannelEmittingStateIds]:
    """Convert typed-channel specs to C++ traversal programs.

    Existing allowlist channels compile to a single emitting self-loop state.
    ``PPRMetaPath`` channels compile to ordered traversal states and emit only
    from the state(s) that satisfy the metapath pattern.
    """
    node_type_to_node_type_id = {
        node_type: node_type_id for node_type_id, node_type in enumerate(node_types)
    }

    traversal_programs: TypedPPRChannelTraversalPrograms = []
    emitting_state_ids_by_channel: TypedPPRChannelEmittingStateIds = []
    for channel_spec in channel_specs:
        if isinstance(channel_spec, PPRMetaPath):
            traversal_program, emitting_state_ids = _build_metapath_channel_program(
                metapath=channel_spec,
                edge_type_to_edge_type_id=edge_type_to_edge_type_id,
                node_type_to_node_type_id=node_type_to_node_type_id,
                edge_dir=edge_dir,
            )
        else:
            traversal_program, emitting_state_ids = _build_allowlist_channel_program(
                edge_types=channel_spec,
                edge_type_to_edge_type_id=edge_type_to_edge_type_id,
                node_type_to_edge_types=node_type_to_edge_types,
                node_types=node_types,
            )
        traversal_programs.append(traversal_program)
        emitting_state_ids_by_channel.append(emitting_state_ids)

    return traversal_programs, emitting_state_ids_by_channel


def build_edge_type_channel_group_edge_type_ids(
    edge_type_groups: list[tuple[EdgeType, ...]],
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
            canonical edge types that its PPR traversal may use.
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
                "Typed PPR channels include non-traversable edge types "
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
                "Typed PPR channels include edge-type "
                f"channel={channel_edge_types!r}, "
                "but no traversable edge types exist for that channel."
            )
        channel_edge_type_ids_by_node_type.append(node_type_id_to_channel_edge_type_ids)
    return channel_edge_type_ids_by_node_type
