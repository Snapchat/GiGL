"""Fixed-shape k-hop sequence layout for sampled subgraphs.

The sparse k-hop path in ``gigl.transforms.graph_transformer`` sizes its tensors
from the data (sparse matmul, ``coalesce``, ``unique``, ``nonzero``, bool-mask
indexing). The functions here build the same layout as its default
(node-id-ordered, anchor-first) k-hop sequences from a dense ``[num_nodes, width]``
neighbour table, using only sort, gather and scatter ops. Their only host read
is the table width, the largest per-node degree over all edge types combined.

Memory is ``O(num_anchors * width ** hop_distance)``, so this suits sampled
subgraphs and small hop counts; ``khop_layout`` raises ``KhopLayoutTooLargeError``
before building a layout whose candidate set exceeds a budget.

Example:
    >>> node_index, hop_index, lengths = khop_layout(
    ...     edge_index,
    ...     num_nodes=5,
    ...     anchor_indices=torch.arange(2),
    ...     hop_distance=2,
    ...     max_seq_len=8,
    ...     sampling_direction="in",
    ... )
"""

from typing import Literal, NamedTuple

import torch
from torch import Tensor

# Candidate cells (``num_anchors * (1 + width + ... + width ** hop_distance)``)
# above which ``khop_layout`` refuses to build. The layout keeps about ten int64
# tensors of this size alive, so this is about 2.5 GiB.
MAX_LAYOUT_CANDIDATES = 2**25


class KhopLayoutTooLargeError(ValueError):
    """The candidate set of a k-hop layout exceeds its budget."""


class PackedSequences(NamedTuple):
    """Where each sequence lives in the packed token buffer.

    Attributes:
        cu_seqlens: ``[num_sequences + 1]`` int32 cumulative sequence lengths on
            the tokens' device (the attention kernel requires int32, so the
            total token count must fit in int32); sequence ``i`` occupies
            ``[cu_seqlens[i], cu_seqlens[i + 1])``.
        max_seq_len: Host upper bound on any sequence length. It must not be
            smaller than the longest sequence (the attention kernel would skip
            rows) and the attention clamps it to the token count.
    """

    cu_seqlens: Tensor
    max_seq_len: int


def pack_sequences(lengths: Tensor, max_seq_len: int) -> tuple[Tensor, PackedSequences]:
    """Locates the first ``lengths[i]`` positions of every sequence in a padded grid.

    Costs one host read, of the total token count. Gather any
    ``[num_sequences, max_seq_len]`` per-position tensor into packed order with
    ``grid.flatten()[flat_index]``.

    Args:
        lengths: ``[num_sequences]`` long valid length of each sequence.
        max_seq_len: Width of the padded grid.

    Returns:
        - ``[num_tokens]`` long indices of the valid positions in the flattened
          grid, sequence by sequence.
        - The packed layout.
    """
    valid = torch.arange(max_seq_len, device=lengths.device) < lengths.unsqueeze(1)
    cu_seqlens = torch.cat([lengths.new_zeros(1), torch.cumsum(lengths, 0)])
    flat_index = valid.flatten().nonzero().squeeze(1)
    return flat_index, PackedSequences(cu_seqlens.to(torch.int32), max_seq_len)


def _build_neighbor_table(
    edge_index: Tensor,
    num_nodes: int,
    sampling_direction: Literal["in", "out"] = "out",
) -> Tensor:
    """Builds a padded adjacency table for k-hop reachability.

    Row ``i`` lists the nodes one hop from node ``i``, padded with -1. With
    ``sampling_direction="in"`` (GLT in-edge sampling, edges point neighbour to
    target) reachability follows the reversed edges, as in
    ``heterodata_to_graph_transformer_input``.

    The table is as wide as the largest row (at least 1), which costs one host
    read. Duplicate edges take one slot each.

    The table has an extra all -1 row at index ``num_nodes``, so a lookup of -1
    (padding) stays -1.

    Args:
        edge_index: ``[2, num_edges]`` int32 or int64 local node ids in
            ``[0, num_nodes)``.
        num_nodes: Number of nodes.
        sampling_direction: ``"in"`` reverses the edges, ``"out"`` keeps them.

    Returns:
        ``[num_nodes + 1, width]`` long neighbour table.
    """
    if sampling_direction == "in":
        source, target = edge_index[1], edge_index[0]
    else:
        source, target = edge_index[0], edge_index[1]
    # Scatter indices must be int64.
    source, target = source.long(), target.long()
    device = edge_index.device
    # The table is built with sort/scatter, not ``to_dense_batch``, whose
    # ``fill_value`` host-to-device copy is an extra sync on CUDA.
    # A stable sort groups each node's edges; an edge's slot is its rank in the
    # group. The degrees are scattered because ``bincount`` would sync the host
    # on CUDA.
    order = torch.argsort(source, stable=True)
    sorted_source = source[order]
    degree = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
    degree.scatter_add_(0, source, torch.ones_like(source))
    width = max(int(degree.max()), 1)
    group_starts = torch.cumsum(degree, 0) - degree
    slot = torch.arange(source.numel(), device=device) - group_starts[sorted_source]
    table = torch.full((num_nodes + 1, width), -1, dtype=torch.long, device=device)
    table.index_put_((sorted_source, slot), target[order])
    return table


def _khop_sequence_layout(
    neighbor_table: Tensor,
    anchor_indices: Tensor,
    hop_distance: int,
    max_seq_len: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Lays out each anchor's k-hop neighbourhood as a fixed-length token sequence.

    Each sequence is the anchor, then every other node within ``hop_distance``
    hops in ascending node id, truncated to ``max_seq_len``. That is the order
    the sparse path gets from ``coalesce``, so the layout equals
    ``_get_k_hop_neighbors_sparse`` followed by
    ``_build_sequence_layout_from_sparse_neighbors`` with the anchor first.

    A token's hop is the hop at which it is first reached; the anchor's is 0.

    Memory is ``O(num_anchors * width ** hop_distance)``: the candidate set is
    every walk of up to ``hop_distance`` steps, deduplicated by a per-row sort.

    Args:
        neighbor_table: ``[num_nodes + 1, width]`` table from ``_build_neighbor_table``.
        anchor_indices: ``[num_anchors]`` int32 or int64 local node ids of the
            anchors.
        hop_distance: Number of hops to expand.
        max_seq_len: Tokens per sequence, anchor included.

    Returns:
        - ``[num_anchors, max_seq_len]`` long node ids, -1 at padding.
        - ``[num_anchors, max_seq_len]`` long first-reach hops, 0 at the anchor
          and at padding.
        - ``[num_anchors]`` long sequence lengths (at least 1).
    """
    anchor_indices = anchor_indices.long()
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


def khop_layout(
    edge_index: Tensor,
    num_nodes: int,
    anchor_indices: Tensor,
    hop_distance: int,
    max_seq_len: int,
    sampling_direction: Literal["in", "out"] = "out",
    max_candidates: int = MAX_LAYOUT_CANDIDATES,
) -> tuple[Tensor, Tensor, Tensor]:
    """Builds the neighbour table and the k-hop layout of every anchor.

    Args:
        edge_index: ``[2, num_edges]`` int32 or int64 local node ids in
            ``[0, num_nodes)``.
        num_nodes: Number of nodes.
        anchor_indices: ``[num_anchors]`` int32 or int64 local node ids of the
            anchors.
        hop_distance: Number of hops to expand.
        max_seq_len: Tokens per sequence, anchor included.
        sampling_direction: ``"in"`` reverses the edges, ``"out"`` keeps them.
        max_candidates: Budget on the candidate set, see ``MAX_LAYOUT_CANDIDATES``.

    Returns:
        The outputs of ``_khop_sequence_layout``.

    Raises:
        KhopLayoutTooLargeError: If the candidate set exceeds ``max_candidates``.
    """
    neighbor_table = _build_neighbor_table(edge_index, num_nodes, sampling_direction)
    width = neighbor_table.size(1)
    num_candidates = anchor_indices.numel() * sum(
        width**hop for hop in range(hop_distance + 1)
    )
    if num_candidates > max_candidates:
        raise KhopLayoutTooLargeError(
            f"{num_candidates} candidates for {anchor_indices.numel()} anchors "
            f"at width {width} and {hop_distance} hops exceed {max_candidates}."
        )
    return _khop_sequence_layout(
        neighbor_table,
        anchor_indices,
        hop_distance,
        max_seq_len,
    )
