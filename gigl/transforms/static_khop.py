"""Fixed-shape k-hop sequence layout for sampled subgraphs.

The sparse k-hop path in ``gigl.transforms.graph_transformer`` creates tensors
whose sizes depend on the data (sparse matmul, ``coalesce``, ``unique``,
``nonzero``, bool-mask indexing). That compiles poorly and forces host syncs.
This module builds the same layout as that path's default (anchor first, then
neighbours in ascending node id) from a dense neighbour table. It uses only sort,
gather and scatter ops, whose output shapes are known up front.

How a layout is built::

    edge_index [2, num_edges]
        |  sort edges by source; an edge's slot is its rank in its source's group
        v
    neighbour table [num_nodes + 1, width]    width = largest degree (one host read)
        |                                     row i = nodes one hop from i, -1 pads;
        |                                     the extra last row is all -1
        |  hop h: gather the rows of the hop h - 1 frontier -> [num_anchors, width ** h]
        v
    candidates [num_anchors, 1 + width + ... + width ** hop_distance]
        |  sort each row by (node id, hop); the first copy of a node is its
        |  first-reach hop, so drop repeats, padding and the anchor itself
        v
    anchor first, then neighbours in ascending node id, cut at max_seq_len
        |
        v
    node_index, hop_index [num_anchors, max_seq_len] (-1 / 0 at padding), lengths
        |
        |  pack_sequences(lengths, max_seq_len)       (one host read: total tokens)
        v
    flat_index + PackedSequences(cu_seqlens, max_seq_len)  for gigl.nn.packed_attention

Memory is ``O(num_anchors * width ** hop_distance)``. Every walk of up to
``hop_distance`` steps is a candidate before deduplication, so a high-degree hub
node makes this grow quickly. ``khop_layout`` therefore counts the candidates
before building anything and raises ``KhopLayoutTooLargeError`` if they exceed
``max_candidates``. ``GraphTransformerEncoder`` catches the error and runs that
batch on the padded path instead.

Host reads (each is a GPU sync):
  - the table width: the largest per-node degree, with all edge types combined
  - the total token count in ``pack_sequences``

Example:
    >>> edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    >>> node_index, hop_index, lengths = khop_layout(
    ...     edge_index,
    ...     num_nodes=5,
    ...     anchor_indices=torch.arange(2),
    ...     hop_distance=2,
    ...     max_seq_len=4,
    ...     sampling_direction="out",
    ... )
    >>> node_index.tolist()
    [[0, 1, 2, -1], [1, 2, 3, -1]]
    >>> hop_index.tolist()
    [[0, 1, 2, 0], [0, 1, 2, 0]]
    >>> lengths.tolist()
    [3, 3]
    >>> flat_index, packed = pack_sequences(lengths, max_seq_len=4)
    >>> node_index.flatten()[flat_index].tolist()
    [0, 1, 2, 1, 2, 3]
    >>> packed.cu_seqlens.tolist()
    [0, 3, 6]
"""

from typing import Literal, NamedTuple

import torch
from jaxtyping import Int, Int32, Int64
from torch import Tensor

# Maximum number of candidate cells, ``num_anchors * (1 + width + ... + width **
# hop_distance)``, that ``khop_layout`` builds. The layout keeps about ten int64
# tensors of this size alive at once, so the limit is about 2.5 GiB.
MAX_LAYOUT_CANDIDATES = 2**25


class KhopLayoutTooLargeError(ValueError):
    """The k-hop layout would have more candidates than the allowed maximum."""


class PackedSequences(NamedTuple):
    """Where each sequence lives in the packed token buffer.

    Attributes:
        cu_seqlens: ``[num_sequences + 1]`` int32 cumulative sequence lengths,
            on the same device as the tokens. Sequence ``i`` occupies
            ``[cu_seqlens[i], cu_seqlens[i + 1])``. The attention kernel
            requires int32, so the total token count must fit in int32.
        max_seq_len: Python int upper bound on any sequence length. It must be at
            least the longest sequence, or the attention kernel skips rows. The
            attention clamps it to the token count.
    """

    cu_seqlens: Int32[Tensor, "num_sequences_plus_one"]
    max_seq_len: int


def pack_sequences(
    lengths: Int64[Tensor, "num_sequences"], max_seq_len: int
) -> tuple[Int64[Tensor, "num_tokens"], PackedSequences]:
    """Finds the valid positions of every sequence in a padded grid.

    Sequence ``i`` is valid in its first ``lengths[i]`` positions. This costs one
    host read, of the total token count. To gather any
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
    edge_index: Int[Tensor, "2 num_edges"],
    num_nodes: int,
    sampling_direction: Literal["in", "out"] = "out",
) -> Int64[Tensor, "num_nodes_plus_one width"]:
    """Builds a padded adjacency table for k-hop reachability.

    Row ``i`` lists the nodes one hop from node ``i``, padded with -1. With
    ``sampling_direction="in"`` (GLT in-edge sampling, where edges point from
    neighbour to target) reachability follows the reversed edges, as in
    ``heterodata_to_graph_transformer_input``.

    The table is as wide as the largest row (at least 1), which costs one host
    read. Duplicate edges take one slot each.

    The table has an extra all -1 row at index ``num_nodes``. Looking up a -1
    (padding) id is redirected to that row, so it yields -1 again.

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
    # Build the table with sort and scatter rather than ``to_dense_batch``, whose
    # ``fill_value`` host-to-device copy is an extra sync on CUDA.
    # A stable sort groups each node's edges, and an edge's slot is its rank in
    # that group. Degrees are counted with ``scatter_add_`` because ``bincount``
    # would sync with the host on CUDA.
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
    neighbor_table: Int64[Tensor, "num_nodes_plus_one width"],
    anchor_indices: Int[Tensor, "num_anchors"],
    hop_distance: int,
    max_seq_len: int,
) -> tuple[
    Int64[Tensor, "num_anchors max_seq_len"],
    Int64[Tensor, "num_anchors max_seq_len"],
    Int64[Tensor, "num_anchors"],
]:
    """Lays out each anchor's k-hop neighbourhood as a fixed-length token sequence.

    Each sequence is the anchor, then every other node within ``hop_distance``
    hops in ascending node id, truncated to ``max_seq_len``. The sparse path gets
    the same order from ``coalesce``, so this layout equals
    ``_get_k_hop_neighbors_sparse`` followed by
    ``_build_sequence_layout_from_sparse_neighbors`` with the anchor moved first.

    A token's hop is the hop at which it is first reached; the anchor's is 0.

    Memory is ``O(num_anchors * width ** hop_distance)``. The candidates are every
    walk of up to ``hop_distance`` steps, which a per-row sort then deduplicates.

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

    # Sort on ``node * (hop_distance + 1) + hop``. This groups each node's copies
    # with the smallest hop first, so the first copy carries the first-reach hop.
    # Padding gets the largest key and sorts last.
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
    # Position 0 belongs to the anchor, so neighbours are numbered from 1.
    positions = torch.cumsum(keep.long(), dim=1)
    keep = keep & (positions < max_seq_len)
    # Dropped candidates are scattered into a spare last column, which is cut off
    # below.
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
    edge_index: Int[Tensor, "2 num_edges"],
    num_nodes: int,
    anchor_indices: Int[Tensor, "num_anchors"],
    hop_distance: int,
    max_seq_len: int,
    sampling_direction: Literal["in", "out"] = "out",
    max_candidates: int = MAX_LAYOUT_CANDIDATES,
) -> tuple[
    Int64[Tensor, "num_anchors max_seq_len"],
    Int64[Tensor, "num_anchors max_seq_len"],
    Int64[Tensor, "num_anchors"],
]:
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
        max_candidates: Maximum number of candidates; see
            ``MAX_LAYOUT_CANDIDATES``.

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
            f"The k-hop layout needs {num_candidates} candidates for "
            f"{anchor_indices.numel()} anchors at width {width} and "
            f"{hop_distance} hops, more than the maximum of {max_candidates}."
        )
    return _khop_sequence_layout(
        neighbor_table,
        anchor_indices,
        hop_distance,
        max_seq_len,
    )
