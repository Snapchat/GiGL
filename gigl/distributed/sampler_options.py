"""Sampler option types for configuring which BaseGiGLSampler subclass to use in distributed loading.

Provides ``KHopNeighborSamplerOptions`` for k-hop sampling via ``DistNeighborSampler``,
and ``PPRSamplerOptions`` for PPR-based sampling via ``DistPPRNeighborSampler``.

Frozen dataclasses so they are safe to pickle across RPC boundaries
(required for Graph Store mode).
"""

from dataclasses import dataclass
from typing import Optional, Union

from graphlearn_torch.typing import EdgeType

from gigl.common.logger import Logger
from gigl.distributed.utils.dist_typed_sampler import MaxPPRNodes

logger = Logger()


@dataclass(frozen=True)
class KHopNeighborSamplerOptions:
    """Sampler options for k-hop neighbor sampling via DistNeighborSampler.

    Attributes:
        num_neighbors: Fanout per hop, either a flat list (homogeneous) or a
            dict mapping edge types to per-hop fanout lists (heterogeneous).
    """

    num_neighbors: Union[list[int], dict[EdgeType, list[int]]]


@dataclass(frozen=True)
class PPRSamplerOptions:
    """Sampler options for PPR-based neighbor sampling using DistPPRNeighborSampler.

    **Output format:** When this sampler is active, each output Data/HeteroData batch
    contains *only* PPR edges — no message-passing edges from the original graph are
    included.  For each ``(seed_type, neighbor_type)`` pair reachable via PPR walks,
    the batch will have an edge type ``(seed_type, "ppr", neighbor_type)`` with:

    - ``edge_index``: ``[2, N]`` int64 — row 0 is local seed indices, row 1 is local
      neighbor indices.
    - ``edge_attr``: ``[N]`` float — PPR score for each (seed, neighbor) pair.
      Typed PPR emits multi-column edge attrs:
      ``[best_score, channel_scores..., channel_presence_bits...]``.
      Column 0 is the scalar best score for consumers that need a single PPR
      weight.

    For homogeneous graphs these live directly on ``data.edge_index`` / ``data.edge_attr``.

    Enable residual top-up when you want longer returned sequences without
    paying the throughput cost of lowering ``eps``.  Lowering ``eps``
    re-enqueues more low-residual nodes but increases push iterations and
    neighbor-fetch work; top-up instead uses positive-residual nodes already
    discovered during Forward Push.

    Attributes:
        alpha: Restart probability (teleport probability back to seed). Higher
            values keep samples closer to seeds. Typical values: 0.15-0.25.
        eps: Convergence threshold for the Forward Push algorithm. Smaller
            values give more accurate PPR scores but require more computation.
            Typical values: 1e-4 to 1e-6.
        max_ppr_nodes: PPR output sequence specification. Pass an integer to
            run regular untyped PPR and return up to that many nodes per seed.
            Pass a mapping from typed channel key to integer target count to
            run typed PPR; the sum of target counts is the per-seed output cap.

            Example::

                # Regular untyped PPR with up to 200 nodes per seed.
                max_ppr_nodes = 200

                # Typed PPR with two traversal channels and up to 200 nodes per seed.
                max_ppr_nodes = {
                    ("user", "views", "item"): 120,
                    (
                        ("user", "likes", "item"),
                        ("user", "shares", "item"),
                    ): 80,
                }

            The typed example creates two traversal channels. The first channel
            can traverse only ``("user", "views", "item")`` edges and targets
            120 output nodes. The second channel groups
            ``("user", "likes", "item")`` and ``("user", "shares", "item")``
            into one PPR state and targets 80 output nodes. These targets are
            best-effort rather than strict per-seed guarantees because channels
            may be sparse or overlapping.

            If residual top-up is enabled, discovered-but-unpushed residual
            candidates from the same completed PPR states are included on the
            same mass scale as finalized PPR scores: ``ppr_score + residual``.
            Residual candidates follow the same typed channel targets as
            finalized PPR candidates.
        enable_residual_topup: Whether to append discovered-but-unpushed
            residual candidates when finalized PPR scores produce fewer than
            the requested ``max_ppr_nodes`` output slots. Residual top-up
            candidates are scored on the same mass scale as PPR scores:
            ``ppr_score + residual``. They fill only unused output slots and do
            not displace finalized PPR nodes when those already fill the
            sequence.
        num_neighbors_per_hop: Maximum number of neighbors fetched per node per edge
            type during PPR traversal. 1000 is sufficient in practice — high-degree
            hub nodes receive diminishing residual per neighbor, so capping the fetch
            has little effect on PPR accuracy while keeping per-hop RPC cost bounded.
            Set large to approximate fetching all neighbors.
        max_fetch_iterations: Maximum number of iterations that issue RPC neighbor
            fetches. After this many fetch iterations, subsequent iterations push
            residuals using only already-cached neighbor lists (no new RPCs).
            The algorithm still runs to convergence — re-enqueued nodes propagate
            through cached neighbors at negligible cost. ``None`` (default) means
            no fetch limit.
    """

    alpha: float = 0.5
    eps: float = 1e-4
    max_ppr_nodes: MaxPPRNodes = 50
    enable_residual_topup: bool = True
    num_neighbors_per_hop: int = 1_000
    max_fetch_iterations: Optional[int] = None


SamplerOptions = Union[KHopNeighborSamplerOptions, PPRSamplerOptions]


def resolve_sampler_options(
    num_neighbors: Union[list[int], dict[EdgeType, list[int]]],
    sampler_options: Optional[SamplerOptions],
) -> SamplerOptions:
    """Resolve sampler_options from user-provided values.

    If ``sampler_options`` is a ``PPRSamplerOptions``, returns it directly (``num_neighbors`` is unused for PPR).
    If ``sampler_options`` is ``None``, wraps ``num_neighbors`` in a ``KHopNeighborSamplerOptions``.
    If ``KHopNeighborSamplerOptions`` is provided, validates that its ``num_neighbors`` matches the explicit value.

    Args:
        num_neighbors: Fanout per hop (required for KHop; ignored for PPR).
        sampler_options: Sampler configuration, or None.

    Returns:
        The resolved SamplerOptions.

    Raises:
        ValueError: If ``KHopNeighborSamplerOptions.num_neighbors`` conflicts
            with the explicit ``num_neighbors``.
    """
    if isinstance(sampler_options, PPRSamplerOptions):
        return sampler_options

    if sampler_options is None:
        return KHopNeighborSamplerOptions(num_neighbors)

    if num_neighbors != sampler_options.num_neighbors:
        raise ValueError(
            f"num_neighbors ({num_neighbors}) does not match "
            f"sampler_options.num_neighbors ({sampler_options.num_neighbors})."
        )
    logger.info(f"Using sampler options: {sampler_options}")

    return sampler_options
