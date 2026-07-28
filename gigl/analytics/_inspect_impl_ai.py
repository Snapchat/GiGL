# AI-OWNED FILE
# spec: ./SPEC.md
# last-generated: 2026-05-22T00:00:00Z
# ---
"""Implementation of the HeteroData batch inspector.

Contract is defined in ``gigl/analytics/SPEC.md``. The public surface lives in
``gigl.analytics.inspect``; this module is regenerable and not intended for
direct import by application code.
"""

from dataclasses import dataclass

import torch
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType


@dataclass(frozen=True)
class HeteroDataSummary:
    """Diagnostic summary of a sampled HeteroData batch.

    Attributes:
        seeds: Number of seed nodes in the batch.
        per_hop: ``HopStats`` per hop, in order from hop 1 to hop K.
    """

    @dataclass(frozen=True)
    class HopStats:
        """Per-seed neighbor count distribution at one hop."""

        min: int
        med: int
        avg: float
        max: int

    seeds: int
    per_hop: list[HopStats]

    def __str__(self) -> str:
        parts = [f"seeds={self.seeds}"]
        for k, s in enumerate(self.per_hop, 1):
            parts.append(f"hop{k}(min={s.min} med={s.med} avg={s.avg:.1f} max={s.max})")
        return " ".join(parts)


def _summary_impl(data: HeteroData) -> HeteroDataSummary:
    """Implementation. See ``gigl.analytics.inspect.summary`` for the public contract."""
    seed_type = _detect_seed_type(data)
    batch_size = int(data[seed_type].batch_size)
    num_hops = _detect_num_hops(data, seed_type)
    edge_hop_bounds = _hop_boundaries(data)

    device = _pick_device(data)
    seeds = torch.arange(batch_size, dtype=torch.long, device=device)
    frontier: dict[NodeType, tuple[torch.Tensor, torch.Tensor]] = {
        seed_type: (seeds, seeds)
    }

    per_hop: list[HeteroDataSummary.HopStats] = []
    for hop in range(1, num_hops + 1):
        hop_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
        new_parts: dict[NodeType, list[tuple[torch.Tensor, torch.Tensor]]] = {}

        for edge_type in data.edge_types:
            src_type, dst_type = edge_type[0], edge_type[2]
            # Walk from whichever side of the edge a frontier already covers.
            # Edge_dir="out" loaders reverse the stored edges, so the seed
            # often sits on the dst side at hop 1.
            if src_type in frontier:
                f_nodes, f_seeds = frontier[src_type]
                walk_ei_src = 0
                other_type = dst_type
            elif dst_type in frontier:
                f_nodes, f_seeds = frontier[dst_type]
                walk_ei_src = 1
                other_type = src_type
            else:
                continue
            if f_nodes.numel() == 0:
                continue

            start, end = edge_hop_bounds[edge_type][hop - 1 : hop + 1]
            if start == end:
                continue
            hop_ei = data[edge_type].edge_index[:, start:end]
            walk_src = hop_ei[walk_ei_src]
            walk_dst = hop_ei[1 - walk_ei_src]

            new_nodes, new_seeds = _expand_frontier(
                walk_src, walk_dst, f_nodes, f_seeds
            )
            if new_seeds.numel() == 0:
                continue

            hop_counts.scatter_add_(
                0,
                new_seeds,
                torch.ones(new_seeds.numel(), dtype=torch.long, device=device),
            )
            if hop < num_hops:
                new_parts.setdefault(other_type, []).append((new_nodes, new_seeds))

        per_hop.append(_hop_stats(hop_counts))

        if hop < num_hops:
            frontier = {
                nt: (
                    torch.cat([n for n, _ in parts]),
                    torch.cat([s for _, s in parts]),
                )
                for nt, parts in new_parts.items()
            }

    return HeteroDataSummary(seeds=batch_size, per_hop=per_hop)


def _detect_seed_type(data: HeteroData) -> NodeType:
    seed_types = [
        nt
        for nt in data.node_types
        if getattr(data[nt], "batch_size", None) is not None and data[nt].batch_size > 0
    ]
    if len(seed_types) != 1:
        raise ValueError(
            f"Expected exactly one node type with batch_size > 0; found {seed_types}"
        )
    return seed_types[0]


def _detect_num_hops(data: HeteroData, seed_type: NodeType) -> int:
    num_sampled = _get_root_dict(data, "num_sampled_nodes")
    if seed_type not in num_sampled:
        raise ValueError(
            f"data.num_sampled_nodes[{seed_type}] is missing — required to "
            "infer the number of hops. Was this batch produced by a GiGL "
            "sampler?"
        )
    series = num_sampled[seed_type]
    num_hops = len(series) - 1
    if num_hops < 1:
        raise ValueError(
            f"data.num_sampled_nodes[{seed_type}] implies {num_hops} hops; "
            "expected at least 1."
        )
    return num_hops


def _hop_boundaries(data: HeteroData) -> dict[EdgeType, list[int]]:
    """Return prefix sums of ``data.num_sampled_edges[edge_type]`` per edge type.

    For edge type E with sampled-edge series ``[n1, n2, ...]``, the result
    ``[0, n1, n1+n2, ...]`` lets us slice ``edge_index[:, bounds[K-1]:bounds[K]]``
    to get hop-K edges.
    """
    num_sampled_edges = _get_root_dict(data, "num_sampled_edges")
    result: dict[EdgeType, list[int]] = {}
    for edge_type in data.edge_types:
        if edge_type not in num_sampled_edges:
            raise ValueError(
                f"data.num_sampled_edges[{edge_type}] is missing — required to "
                "slice edges per hop. Was this batch produced by a GiGL sampler?"
            )
        series = num_sampled_edges[edge_type]
        prefix = [0]
        running = 0
        for n in series.tolist() if torch.is_tensor(series) else series:
            running += int(n)
            prefix.append(running)
        result[edge_type] = prefix
    return result


def _get_root_dict(data: HeteroData, attr: str) -> dict:
    """Read a sampler-set dict (``num_sampled_nodes`` / ``num_sampled_edges``)
    from the HeteroData root; raise ``ValueError`` if absent."""
    try:
        value = getattr(data, attr)
    except (AttributeError, KeyError) as e:
        raise ValueError(
            f"data.{attr} is missing — required for hop accounting. "
            "Was this batch produced by a GiGL sampler?"
        ) from e
    if not isinstance(value, dict):
        raise ValueError(
            f"Expected data.{attr} to be a dict keyed by type; got "
            f"{type(value).__name__}."
        )
    return value


def _expand_frontier(
    walk_src: torch.Tensor,
    walk_dst: torch.Tensor,
    frontier_nodes: torch.Tensor,
    frontier_seeds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand the ``(node, seed)`` frontier by one hop.

    For each ``(node, seed)`` pair, enumerates every edge where ``walk_src ==
    node`` and emits ``(walk_dst, seed)`` pairs. ``walk_src`` / ``walk_dst``
    let the caller choose which side of the edge corresponds to the frontier
    (so the inspector works under either edge orientation).
    """
    device = walk_src.device
    sort_idx = torch.argsort(walk_src, stable=True)
    sorted_src = walk_src[sort_idx]
    sorted_dst = walk_dst[sort_idx]

    range_start = torch.searchsorted(sorted_src, frontier_nodes, right=False)
    range_end = torch.searchsorted(sorted_src, frontier_nodes, right=True)
    lengths = range_end - range_start

    total = int(lengths.sum().item())
    if total == 0:
        return frontier_nodes[:0], frontier_seeds[:0]

    new_seeds = frontier_seeds.repeat_interleave(lengths)
    cumlen = torch.cat(
        [torch.zeros(1, dtype=lengths.dtype, device=device), lengths.cumsum(0)]
    )
    positions = torch.arange(total, device=device)
    entry_idx = torch.searchsorted(cumlen[1:], positions, right=True)
    within = positions - cumlen[entry_idx]
    return sorted_dst[range_start[entry_idx] + within], new_seeds


def _pick_device(data: HeteroData) -> torch.device:
    """Pick a device from any populated tensor in ``data`` (falls back to CPU)."""
    for nt in data.node_types:
        x = getattr(data[nt], "x", None)
        if x is not None:
            return x.device
    for et in data.edge_types:
        ei = data[et].edge_index
        if ei.numel() > 0:
            return ei.device
    return torch.device("cpu")


def _hop_stats(counts: torch.Tensor) -> HeteroDataSummary.HopStats:
    min_val, max_val = torch.aminmax(counts)
    return HeteroDataSummary.HopStats(
        min=int(min_val.item()),
        med=int(counts.median().item()),
        avg=float(counts.float().mean().item()),
        max=int(max_val.item()),
    )
