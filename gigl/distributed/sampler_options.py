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
from gigl.distributed.utils.dist_typed_sampler import TypedPPRChannelKey

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

    **Output format:** By default, each output Data/HeteroData batch contains
    *only* virtual PPR edges — no message-passing edges from the original graph
    are included. For each ``(seed_type, neighbor_type)`` pair reachable via PPR
    walks, the batch will have an edge type ``(seed_type, "ppr", neighbor_type)``
    with:

    - ``edge_index``: ``[2, N]`` int64 — row 0 is local seed indices, row 1 is local
      neighbor indices.
    - ``edge_attr``: ``[N, 2]`` float — PPR score and hop proximity for each
      (seed, neighbor) pair: ``[ppr_score, hop_proximity]``.
      ``hop_proximity`` is ``1 / (1 + hop)``: ``1.0`` for the anchor, ``0.5``
      for 1-hop, and so on.
      Typed PPR emits additional channel columns:
      ``[best_score, hop_proximity, (channel_score, channel_hop_proximity,
      channel_presence), ...]``.
      Column 0 is the scalar best score for consumers that need a single PPR
      weight, and column 1 is always the global hop proximity.
      Per-channel hop proximity is ``1 / (1 + hop)`` when that channel
      reached the node, and ``0`` when it did not.
      For present channels, the original hop count can be recovered as
      ``(1 - proximity) / proximity``; use the presence bit before applying
      this inverse because missing channels have proximity ``0``.

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
        max_ppr_nodes: Maximum number of nodes to return per seed based on PPR
            scores.
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
        typed_channel_ratios: Optional target proportions for typed PPR
            traversal channels defined by canonical edge-type allowlists. Keys
            may be either a single canonical edge type
            ``(src_type, relation, dst_type)`` or a tuple of canonical edge
            types. Each key defines one traversal channel that may use only
            those exact edge types. Edge types may appear in multiple channels
            when those channels intentionally overlap.
            If not provided, PPR treats all eligible edge types as one shared
            traversal space and emits a single scalar PPR score per output row.
            Channel order follows the insertion order of this mapping, and
            typed ``edge_attr`` channel columns use that same order. If the
            mapping is produced from an unordered config source, construct it
            deterministically before passing it to the sampler. Values are
            positive ratios that must sum to ``1.0``. The sampler converts
            ratios to per-channel target counts from ``max_ppr_nodes``.
            Finalized PPR candidates and residual top-up candidates both obey
            these target counts. If the same node appears in multiple channels,
            it is attributed to the channel where it has the highest emitted
            PPR score for that seed. If sparse channels or duplicate nodes
            leave unused target slots, the remaining slots are redistributed
            globally by score so the returned sequence can still fill up to,
            but never exceed, ``max_ppr_nodes``.
            Example::

                typed_channel_ratios = {
                    ("user", "views", "item"): 0.6,
                    (
                        ("user", "likes", "item"),
                        ("user", "shares", "item"),
                    ): 0.4,
                }

            This example creates two traversal channels. The first channel can
            traverse only ``("user", "views", "item")`` edges. With
            ``max_ppr_nodes=200``, the ``0.6`` ratio targets 120 nodes
            attributed to this channel. The second channel groups
            ``("user", "likes", "item")`` and ``("user", "shares", "item")``
            into one traversal channel; the ``0.4`` ratio targets 80 nodes
            attributed to that combined likes/shares channel. These targets are
            best-effort rather than strict per-seed guarantees because channels
            may be sparse or overlapping.

            If residual top-up is enabled, discovered-but-unpushed residual
            candidates from the same completed PPR traversals are included on the
            same mass scale as finalized PPR scores: ``ppr_score + residual``.
            Residual candidates follow the same channel targets as finalized
            PPR candidates.
        include_sampled_edges: Whether heterogeneous PPR output batches should
            also include original graph edge types alongside virtual PPR edges.
            The sampler emits original edges from adjacency rows fetched while
            running PPR and whose source and destination are both in the final
            PPR-selected node set; an emitted edge does not have to be the
            relation that uniquely caused the destination's PPR score. Original
            edges are emitted through GLT's
            regular sampled-edge channel, so their final HeteroData edge
            orientation follows the same ``edge_dir`` convention as k-hop
            sampling. Homogeneous PPR keeps the default PPR-only output because
            ``Data`` cannot represent virtual PPR and original edges as separate
            edge types. This also applies to labeled-homogeneous ABLP loaders,
            where label edges make the sampler graph heterogeneous internally but
            the output is still converted back to homogeneous ``Data``. The
            default ``False`` path is also more faithful to PyG's ``get_ppr`` API,
            which returns virtual
            seed-to-PPR-neighbor ``edge_index`` rows with PPR weights rather than
            an induced message-passing subgraph:
            https://pytorch-geometric.readthedocs.io/en/2.5.3/_modules/torch_geometric/utils/ppr.html
    """

    alpha: float = 0.5
    eps: float = 1e-4
    max_ppr_nodes: int = 50
    enable_residual_topup: bool = True
    num_neighbors_per_hop: int = 1_000
    max_fetch_iterations: Optional[int] = None
    typed_channel_ratios: Optional[dict[TypedPPRChannelKey, float]] = None
    include_sampled_edges: bool = False


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
