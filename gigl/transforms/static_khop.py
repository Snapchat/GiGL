"""Fixed-shape k-hop sequence layout for sampled subgraphs.

The sparse k-hop path in ``gigl.transforms.graph_transformer`` sizes its tensors
from the data (sparse matmul, ``coalesce``, ``unique``, ``nonzero``, bool-mask
indexing), so every batch costs host syncs and graph breaks under
``torch.compile``. The functions here build the same layout as its default
(node-id-ordered, anchor-first) k-hop sequences with shapes fixed by the input
shapes and a few host ints, so they run without host syncs and compile with
``fullgraph=True``.

They rely on a bounded in-degree: a sampled subgraph gives each expanded node at
most ``max(fanout)`` sampled neighbours, so a ``[num_nodes, width]`` neighbour
table holds the whole adjacency. ``build_neighbor_table`` returns a device flag
for batches that break that bound; callers must not trust the layout when it is
set.

Example:
    >>> table, overflow = build_neighbor_table(edge_index, num_nodes, width=15, sampling_direction="in")
    >>> node_index, hops, lengths = khop_sequence_layout(
    ...     table, torch.arange(batch_size), hop_distance=2, max_seq_len=256
    ... )
"""

from typing import Literal

import torch
from torch import Tensor


def build_neighbor_table(
    edge_index: Tensor,
    num_nodes: int,
    width: int,
    sampling_direction: Literal["in", "out"] = "out",
) -> tuple[Tensor, Tensor]:
    """Builds a padded adjacency table for k-hop reachability.

    Row ``i`` lists the nodes one hop from node ``i``, padded with -1. With
    ``sampling_direction="in"`` (GLT in-edge sampling, edges point neighbour to
    target) reachability follows the reversed edges, as in
    ``heterodata_to_graph_transformer_input``.

    The table has an extra all -1 row at index ``num_nodes``, so a lookup of -1
    (padding) stays -1.

    Edges with a negative endpoint are padding and are ignored.

    Duplicate edges take one slot each and count toward ``width``.

    Args:
        edge_index: ``[2, num_edges]`` local node ids.
        num_nodes: Number of nodes; ids are in ``[0, num_nodes)``.
        width: Slots per row, normally the largest sampling fanout.
        sampling_direction: ``"in"`` reverses the edges, ``"out"`` keeps them.

    Returns:
        - ``[num_nodes + 1, width]`` long neighbour table.
        - Scalar bool device tensor, True if some node has more than ``width``
          edges. The table then misses edges and the layout built from it is wrong.
    """
    if sampling_direction == "in":
        source, target = edge_index[1], edge_index[0]
    else:
        source, target = edge_index[0], edge_index[1]
    device = edge_index.device
    dump_row = num_nodes
    is_padding = (source < 0) | (target < 0)
    source = torch.where(is_padding, dump_row, source)
    target = torch.where(is_padding, -1, target)

    # A stable sort groups each node's edges; an edge's slot is its rank in the group.
    order = torch.argsort(source, stable=True)
    sorted_source = source[order]
    counts = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
    counts.index_add_(0, source, torch.ones_like(source))
    group_starts = torch.cumsum(counts, 0) - counts
    slot = torch.arange(source.numel(), device=device) - group_starts[sorted_source]
    # Slots past ``width`` land in a spare column that is dropped below.
    table = torch.full((num_nodes + 1, width + 1), -1, dtype=torch.long, device=device)
    table.index_put_((sorted_source, slot.clamp(max=width)), target[order])
    table = table[:, :width]
    # Padding edges wrote into the dump row; reset it so -1 lookups stay -1.
    is_dump_row = torch.arange(num_nodes + 1, device=device).unsqueeze(1) == dump_row
    table = torch.where(is_dump_row, -1, table)
    overflow = (counts[:num_nodes] > width).any()
    return table, overflow


def khop_sequence_layout(
    neighbor_table: Tensor,
    anchor_indices: Tensor,
    hop_distance: int,
    max_seq_len: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Lays out each anchor's k-hop neighbourhood as a fixed-length token sequence.

    Each sequence is the anchor, then every other node within ``hop_distance``
    hops in ascending node id, truncated to ``max_seq_len``. That is the order
    the sparse path gets from ``coalesce``, so the layout equals
    ``_get_k_hop_neighbors_with_hops`` followed by
    ``_build_sequence_layout_from_sparse_neighbors`` with the anchor first.

    A token's hop is the hop at which it is first reached; the anchor's is 0.

    Memory is ``O(num_anchors * width ** hop_distance)``: the candidate set is
    every walk of up to ``hop_distance`` steps, deduplicated by a per-row sort.

    Args:
        neighbor_table: ``[num_nodes + 1, width]`` table from ``build_neighbor_table``.
        anchor_indices: ``[num_anchors]`` local node ids of the anchors.
        hop_distance: Number of hops to expand.
        max_seq_len: Tokens per sequence, anchor included.

    Returns:
        - ``[num_anchors, max_seq_len]`` long node ids, -1 at padding.
        - ``[num_anchors, max_seq_len]`` long first-reach hops, 0 at the anchor
          and at padding.
        - ``[num_anchors]`` long sequence lengths (at least 1).
    """
    dump_row = neighbor_table.size(0) - 1
    num_anchors = anchor_indices.numel()
    device = anchor_indices.device

    frontier = anchor_indices.unsqueeze(1)
    candidate_parts = [frontier]
    hop_parts = [torch.zeros_like(frontier)]
    for hop in range(1, hop_distance + 1):
        frontier = neighbor_table[torch.where(frontier >= 0, frontier, dump_row)]
        frontier = frontier.flatten(1)
        candidate_parts.append(frontier)
        hop_parts.append(torch.full_like(frontier, hop))
    candidates = torch.cat(candidate_parts, dim=1)
    candidate_hops = torch.cat(hop_parts, dim=1)

    # Sorting node * (k + 1) + hop puts each node's copies together, smallest hop
    # first, so the first copy carries the first-reach hop. Padding sorts last.
    hop_radix = hop_distance + 1
    padding_key = torch.iinfo(torch.long).max
    keys = torch.where(
        candidates >= 0, candidates * hop_radix + candidate_hops, padding_key
    )
    keys, _ = torch.sort(keys, dim=1)
    nodes = torch.where(keys == padding_key, -1, keys // hop_radix)
    hops = keys % hop_radix
    is_first_copy = torch.cat(
        [
            torch.ones_like(nodes[:, :1], dtype=torch.bool),
            nodes[:, 1:] != nodes[:, :-1],
        ],
        dim=1,
    )
    keep = is_first_copy & (nodes >= 0) & (nodes != anchor_indices.unsqueeze(1))
    # Position 0 is the anchor's, so neighbours count up from 1.
    positions = torch.cumsum(keep.long(), dim=1)
    keep = keep & (positions < max_seq_len)
    # Dropped candidates write to a spare last column.
    positions = torch.where(keep, positions, max_seq_len)

    node_index = torch.full(
        (num_anchors, max_seq_len + 1), -1, dtype=torch.long, device=device
    )
    node_index.scatter_(1, positions, torch.where(keep, nodes, -1))
    hop_index = torch.zeros(
        (num_anchors, max_seq_len + 1), dtype=torch.long, device=device
    )
    hop_index.scatter_(1, positions, torch.where(keep, hops, 0))
    node_index = torch.cat(
        [anchor_indices.unsqueeze(1), node_index[:, 1:max_seq_len]], dim=1
    )
    hop_index = torch.cat(
        [torch.zeros_like(hop_index[:, :1]), hop_index[:, 1:max_seq_len]], dim=1
    )
    lengths = 1 + keep.sum(dim=1)
    return node_index, hop_index, lengths


def gather_rows(features: Tensor, index: Tensor, valid: Tensor) -> Tensor:
    """Returns ``features[index]`` with zero rows where ``valid`` is False.

    ``index`` may hold any value (e.g. -1) where ``valid`` is False.
    ``features`` must have at least one row.

    Args:
        features: ``[num_rows, dim]`` rows to gather.
        index: Row ids of any shape.
        valid: Bool tensor shaped like ``index``.

    Returns:
        ``[*index.shape, dim]`` gathered rows.
    """
    # Invalid positions read rows spread over the table rather than one row,
    # so the gather's backward does not pile their (zero) gradients onto a
    # single row.
    spread_index = (
        torch.arange(index.numel(), device=index.device)
        .view(index.shape)
        .remainder(features.size(0))
    )
    gathered = features[torch.where(valid, index, spread_index)]
    return gathered.masked_fill(~valid.unsqueeze(-1), 0.0)
