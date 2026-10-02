from typing import Sequence

import torch

class NeighborFetchTensors:
    """One edge type's fetched neighbor payload passed from Python into ``push_residuals``.

    At the call boundary these are collected into a ``dict`` keyed by integer edge type ID.

    Attributes:
        node_ids: int64 source node IDs, shape ``[N]``.
        flat_neighbor_ids: int64 neighbor IDs concatenated across all source nodes,
            shape ``[sum(counts)]``.
        counts: int64 per-source-node neighbor count, shape ``[N]``; segments
            ``flat_neighbor_ids`` back into per-node adjacency.
        edge_ids: optional int64 edge IDs aligned with ``flat_neighbor_ids``. ``None``
            when edge features are not requested.
    """

    node_ids: torch.Tensor
    flat_neighbor_ids: torch.Tensor
    counts: torch.Tensor
    edge_ids: torch.Tensor | None
    def __init__(
        self,
        node_ids: torch.Tensor,
        flat_neighbor_ids: torch.Tensor,
        counts: torch.Tensor,
        edge_ids: torch.Tensor | None = None,
    ) -> None: ...

class PPRExtractTensors:
    """One node type's PPR extraction output.

    Returned inside a ``dict`` keyed by integer node type ID from the extract functions.

    Attributes:
        ids: int64 selected node IDs, flattened across seeds.
        weights: double feature matrix (``edge_attr``). For untyped PPR the columns are
            ``[ppr_score, hop_proximity]``; for typed PPR they are
            ``[best_score, hop_proximity, (channel_score, channel_hop_proximity, channel_presence), ...]``.
            Hop proximity is ``1 / (1 + hop)`` (anchor ``1.0``, 1-hop ``0.5``, ...).
        valid_counts: int64 count of selected nodes per seed.
    """

    ids: torch.Tensor
    weights: torch.Tensor
    valid_counts: torch.Tensor

class OriginalEdgeExtractTensors:
    """One edge type's extracted original graph edges.

    Returned inside a ``dict`` keyed by integer edge type ID.

    Attributes:
        rows: int64 source-side local node indices.
        cols: int64 destination-side local node indices.
        edge_ids: optional int64 edge IDs aligned with ``rows``/``cols``. ``None`` when
            edge IDs were not requested.
    """

    rows: torch.Tensor
    cols: torch.Tensor
    edge_ids: torch.Tensor | None

class TypedPPRQueueDrainResult:
    """Batched drain result for one typed-PPR iteration across channels.

    Typed PPR keeps one ``PPRForwardPush`` state per channel. Each channel drains its own
    queue, but Python issues at most one shared neighbor fetch per edge type. This result
    carries both which channel states still need ``push_residuals`` and the unioned frontier
    to fetch once for all channels that requested it.

    Attributes:
        drained_channel_indices: Channels whose ``drain_queue`` returned a value this
            iteration; these need ``push_residuals`` even when no fetch budget remains.
            Channel IDs are positional indices, appended in ascending order.
        fetch_channel_indices: Subset of ``drained_channel_indices`` that still have fetch
            budget and at least one non-empty uncached frontier.
        edge_type_ids_by_fetch_channel: Edge types requested by each fetch channel, aligned
            with ``fetch_channel_indices``.
        unioned_node_ids_by_edge_type_id: Unioned node frontier for one shared distributed
            neighbor fetch, keyed by integer edge type ID (edge-type scoped, not node-type
            scoped). Tensor values are int64 source node IDs to fetch.
    """

    drained_channel_indices: list[int]
    fetch_channel_indices: list[int]
    edge_type_ids_by_fetch_channel: list[list[int]]
    unioned_node_ids_by_edge_type_id: dict[int, torch.Tensor]

class PPRForwardPush:
    """C++ kernel for PPR Forward Push (Andersen et al., 2006).

    Hot-loop PPR state lives in C++; distributed neighbor fetches are driven from Python.

    The per-batch call sequence is::

        state = PPRForwardPush(seed_nodes, ...)
        while True:
            frontier = state.drain_queue()          # nodes needing neighbor lookup
            # <Python: fetch neighbors for `frontier`>
            state.push_residuals(fetched_by_etype_id)
        results = state.extract_top_k_with_residual_top_up(max_ppr_nodes, enable_residual_topup)
    """

    def __init__(
        self,
        seed_nodes: torch.Tensor,
        seed_node_type_id: int,
        alpha: float,
        requeue_threshold_factor: float,
        node_type_to_edge_type_ids: list[list[int]],
        edge_type_to_dst_ntype_id: list[int],
        degree_tensors: list[torch.Tensor],
    ) -> None:
        """Initialize PPR state for one batch of seed nodes.

        Args:
            seed_nodes: int tensor of seed node IDs for this batch.
            seed_node_type_id: Node type ID shared by all seeds.
            alpha: PPR teleport probability.
            requeue_threshold_factor: ``alpha * eps``; the per-node requeue threshold is
                ``factor * degree``.
            node_type_to_edge_type_ids: For each node type ID, the edge type IDs that
                originate from that node type.
            edge_type_to_dst_ntype_id: For each edge type ID, its destination node type ID.
            degree_tensors: Per node type, an int32 tensor of total out-degrees indexed by
                node ID.
        """
        ...

    def drain_queue(self) -> dict[int, torch.Tensor] | None:
        """Drain queued nodes needing a neighbor lookup.

        Returns:
            ``{edge_type_id: int64 node tensor}`` of nodes to look up, or ``None`` when the
            queue is empty (convergence). An empty (non-``None``) map means every queued
            node was a cache hit; call ``push_residuals({})`` to continue.
        """
        ...

    def push_residuals(
        self,
        fetched_by_etype_id: dict[int, NeighborFetchTensors],
    ) -> None:
        """Push residual mass using freshly fetched neighbor data.

        Args:
            fetched_by_etype_id: Fetched neighbor payloads keyed by integer edge type ID.
                Pass an empty map to continue on cache hits alone (see ``drain_queue``).
        """
        ...

    def extract_top_k_with_residual_top_up(
        self,
        max_ppr_nodes: int,
        enable_residual_topup: bool,
    ) -> dict[int, PPRExtractTensors]:
        """Return top-k PPR nodes plus residual-mass top-up nodes, sorted by score.

        Residual top-up issues no new neighbor fetches; it only reads the residual table
        already built by Forward Push, letting callers fill short sequences with discovered
        nodes that never crossed the requeue threshold. Top-up scores use the same mass
        scale as PPR scores (``ppr_score(node) + residual(node)``). Residual candidates only
        fill the requested top-up budget and never displace finalized-PPR nodes.

        Args:
            max_ppr_nodes: Final per-seed cap across finalized PPR and residual top-up
                candidates.
            enable_residual_topup: Whether residual candidates may fill remaining budget.

        Returns:
            Per-node-type ``PPRExtractTensors`` keyed by integer node type ID.
        """
        ...

def drain_typed_ppr_channel_queues(
    states: Sequence[PPRForwardPush],
    fetch_iteration_counts: Sequence[int],
    max_fetch_iterations: int = -1,
) -> TypedPPRQueueDrainResult:
    """Drain several independent channel states for one typed-PPR iteration.

    Typed wrapper around ``PPRForwardPush.drain_queue``: it drains each channel state,
    records every channel that still needs ``push_residuals``, and unions fetchable frontier
    nodes by edge type so Python can issue one shared distributed fetch for duplicate channel
    requests. Channel drains may run concurrently; the result merge happens in channel order
    so the returned channel-index lists are deterministic.

    Args:
        states: One ``PPRForwardPush`` per typed channel; each is mutated by its
            ``drain_queue`` call.
        fetch_iteration_counts: Number of distributed fetches already issued per channel,
            aligned with ``states``.
        max_fetch_iterations: ``-1`` means unbounded; otherwise channels at this count still
            need ``push_residuals`` but contribute no new fetch frontier.

    Returns:
        A ``TypedPPRQueueDrainResult`` describing the channels to push and the shared fetch
        request.
    """
    ...

def extract_typed_top_k_with_residual_top_up(
    states: Sequence[PPRForwardPush],
    channel_target_counts: Sequence[int],
    enable_residual_topup: bool,
) -> dict[int, PPRExtractTensors]:
    """Extract and merge completed typed-PPR channel states in one C++ step.

    For each seed/node-type, one candidate view is built per channel (including residual
    candidates when enabled). The merge selects by emitted PPR score, deduplicates nodes seen
    through multiple channels by attributing each to the channel where it scores highest,
    fills each channel target, redistributes unused slots globally by score, and emits
    per-node-type tensors.

    Args:
        states: Completed ``PPRForwardPush`` states, one per typed channel.
        channel_target_counts: Per-channel target output counts, aligned with ``states``;
            their sum is the maximum number of deduplicated nodes returned per seed.
        enable_residual_topup: Whether residual candidates may participate in target filling
            alongside finalized PPR candidates.

    Returns:
        Per-node-type ``PPRExtractTensors`` keyed by integer node type ID, matching the
        ``extract_top_k_with_residual_top_up`` contract.
    """
    ...

def extract_original_edges_from_ppr_caches(
    states: Sequence[PPRForwardPush],
    selected_node_ids_by_node_type_id: dict[int, torch.Tensor],
    include_edge_ids: bool,
) -> dict[int, OriginalEdgeExtractTensors]:
    """Extract original graph edges from one or more completed PPR states.

    Multiple states are used for typed PPR, where each channel owns its own cache. Duplicate
    emitted edges shared by channels are emitted once.

    Args:
        states: Completed ``PPRForwardPush`` states whose neighbor caches supply the edges.
        selected_node_ids_by_node_type_id: Selected node IDs keyed by integer node type ID;
            only edges among these nodes are emitted.
        include_edge_ids: Whether to populate ``OriginalEdgeExtractTensors.edge_ids``.

    Returns:
        Per-edge-type ``OriginalEdgeExtractTensors`` keyed by integer edge type ID.
    """
    ...
