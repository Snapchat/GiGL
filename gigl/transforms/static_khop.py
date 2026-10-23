"""Builds each anchor's k-hop token sequence with fixed-shape tensor ops.

This is the layout half of the packed backend of ``GraphTransformerEncoder``. For
every anchor it picks which nodes become its tokens (the anchor, then its
neighbours within ``hop_distance`` hops), and ``pack_sequences`` turns the padded
result into the offsets that ``gigl.nn.packed_attention`` needs to run the
transformer on real tokens only.

Why this exists:
  - Sequences are short and uneven (about 90 real tokens out of a 256-token
    budget in our models), so a padded ``[num_sequences, max_seq_len]`` batch
    spends most of its compute and memory on padding. Packing the real tokens
    into one buffer, with ``cu_seqlens`` marking the boundaries, avoids that.
  - The sparse k-hop path in ``gigl.transforms.graph_transformer`` (sparse
    matmul, ``coalesce``, ``unique``, ``nonzero``, bool-mask indexing) produces
    tensors whose sizes depend on the data. It needs many host syncs and blocks
    ``torch.compile``. This module instead uses a dense neighbour table and only
    sort, gather and scatter, whose output shapes are known up front. It
    produces the same layout as that path's default: anchor first, then
    neighbours in ascending node id.
  - A dense table has one cost: the candidates before deduplication grow as
    ``num_anchors * width ** hop_distance``, and one hub node sets ``width`` for
    the whole batch. ``khop_layout`` counts the candidates before it builds
    anything and raises ``KhopLayoutTooLargeError`` past ``max_candidates``.
    ``GraphTransformerEncoder`` catches that and runs the batch on the padded
    path.
  - The whole path makes exactly two host reads (each is a GPU sync): the table
    width (the largest per-node degree, all edge types combined) and, in
    ``pack_sequences``, the total token count.

How a layout is built::

    edge_index [2, num_edges]
        |  sort edges by source; an edge's slot is its rank in its source's group
        v
    neighbour table [num_nodes + 1, width]    width = largest degree (host read 1)
        |                                     row i = nodes one hop from i, -1 pads;
        |                                     the extra last row is all -1
        |  hop h: gather the rows of the hop h - 1 frontier -> [num_anchors, width ** h]
        v
    candidates [num_anchors, 1 + width + ... + width ** hop_distance]
        |  sort each row by (node id, hop); the first copy of a node is its
        |  first-reach hop, so drop repeats, padding and the anchor itself
        v
    anchor first, then neighbours in ascending node id (or by hop, then node id,
    with ``prioritize_hop_order``), cut at max_seq_len
        |
        v
    layout [num_anchors, max_seq_len] of (node << hop_bits) | hop (-1 at padding), lengths
        |
        |  pack_sequences(lengths, max_seq_len)       (host read 2: total tokens)
        v
    flat_index + PackedSequences(cu_seqlens, max_seq_len)  for gigl.nn.packed_attention

Constraints:
  - Memory is ``O(num_anchors * width ** hop_distance)``; see
    ``MAX_LAYOUT_CANDIDATES`` for the limit.
  - ``cu_seqlens`` is int32, so the total token count must fit in int32.

Example:
    >>> edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    >>> layout, lengths, hop_bits = khop_layout(
    ...     edge_index,
    ...     num_nodes=5,
    ...     anchor_indices=torch.arange(2),
    ...     hop_distance=2,
    ...     max_seq_len=4,
    ...     sampling_direction="out",
    ... )
    >>> hop_bits
    2
    >>> (layout >> hop_bits).tolist()
    [[0, 1, 2, -1], [1, 2, 3, -1]]
    >>> (layout & 3).tolist()
    [[0, 1, 2, 3], [0, 1, 2, 3]]
    >>> lengths.tolist()
    [3, 3]
    >>> flat_index, packed = pack_sequences(lengths, max_seq_len=4)
    >>> (layout.flatten()[flat_index] >> hop_bits).tolist()
    [0, 1, 2, 1, 2, 3]
    >>> packed.cu_seqlens.tolist()
    [0, 3, 6]
"""

from typing import Literal, NamedTuple

import torch
import torch.nn.functional as F
from jaxtyping import Int, Int32, Int64
from torch import Tensor

# Maximum number of int64 cells ``khop_layout`` builds: the candidates,
# ``num_anchors * (1 + width + ... + width ** hop_distance)``, plus the neighbour
# table, ``(num_nodes + 1) * width``. The layout keeps about ten int64 tensors of
# the candidate size alive at once, so the limit is about 2.5 GiB. The width is
# the largest degree in the whole batch, so one hub node can push every anchor
# over the limit and send the batch to the padded path.
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
    """Computes what is needed to drop the padding from a padded grid of sequences.

    Sequence ``i`` is valid in its first ``lengths[i]`` positions. Any
    ``[num_sequences, max_seq_len]`` per-position tensor gathers into packed
    order with ``grid.flatten()[flat_index]``, and ``cu_seqlens`` marks where
    each packed sequence starts. This costs one host read, of the total token
    count; it is why the packed buffer holds exactly the real tokens.

    A length above ``max_seq_len`` is clamped to it, because the grid has no
    positions beyond that. Checking for it would cost another host read.

    Args:
        max_seq_len: Width of the padded grid.

    Returns:
        - ``[num_tokens]`` long indices of the valid positions in the flattened
          grid, sequence by sequence.
        - The packed layout.
    """
    lengths = lengths.clamp(max=max_seq_len)
    valid = torch.arange(max_seq_len, device=lengths.device) < lengths.unsqueeze(1)
    cu_seqlens = torch.cat([lengths.new_zeros(1), torch.cumsum(lengths, 0)])
    flat_index = valid.flatten().nonzero().squeeze(1)
    return flat_index, PackedSequences(cu_seqlens.to(torch.int32), max_seq_len)


def _build_neighbor_table(
    edge_index: Int[Tensor, "2 num_edges"],
    num_nodes: int,
    num_anchors: int,
    hop_distance: int,
    max_candidates: int,
    sampling_direction: Literal["in", "out"] = "out",
) -> Int64[Tensor, "num_nodes_plus_one width"]:
    """Builds the dense neighbour table that lets k-hop expansion run as gathers.

    Row ``i`` lists the nodes one hop from node ``i``, padded with -1. With
    ``sampling_direction="in"`` (GLT in-edge sampling, where edges point from
    neighbour to target) reachability follows the reversed edges, as in
    ``heterodata_to_graph_transformer_input``.

    The table is as wide as the largest row (at least 1), which costs one host
    read. The layout's size is checked right after that read and before the table
    is allocated, because one hub node makes the table itself large. Duplicate
    edges take one slot each.

    The table has an extra all -1 row at index ``num_nodes``. Looking up a -1
    (padding) id selects that row, so it yields -1 again.

    Args:
        num_anchors: Number of anchors the layout will be built for.
        hop_distance: Number of hops the layout will expand.
        max_candidates: Maximum of the table cells plus the layout's candidates; see
            ``MAX_LAYOUT_CANDIDATES``.
        sampling_direction: ``"in"`` reverses the edges, ``"out"`` keeps them.

    Returns:
        ``[num_nodes + 1, width]`` long neighbour table.

    Raises:
        KhopLayoutTooLargeError: If the table cells plus candidates exceed ``max_candidates``.
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
    num_candidates = num_anchors * sum(width**hop for hop in range(hop_distance + 1))
    num_cells = num_candidates + (num_nodes + 1) * width
    if num_cells > max_candidates:
        raise KhopLayoutTooLargeError(
            f"The k-hop layout needs {num_candidates} candidates for {num_anchors} "
            f"anchors at width {width} and {hop_distance} hops, plus a "
            f"{num_nodes + 1} x {width} neighbour table, more than the maximum of "
            f"{max_candidates}."
        )
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
    prioritize_hop_order: bool = False,
) -> tuple[Int64[Tensor, "num_anchors max_seq_len"], Int64[Tensor, "num_anchors"], int]:
    """Picks the tokens of every anchor's k-hop neighbourhood, with fixed shapes.

    Gathering through the neighbour table, rather than sparse matrix ops, keeps
    every tensor's shape independent of the data. Each sequence is the anchor, then every other node within ``hop_distance``
    hops in ascending node id, truncated to ``max_seq_len``. The sparse path gets
    the same order from ``coalesce``, so this layout equals
    ``_get_k_hop_neighbors_sparse`` followed by
    ``_build_sequence_layout_from_sparse_neighbors`` with the anchor moved first.
    With ``prioritize_hop_order`` the neighbours are ordered by first-reach hop,
    then node id, so truncation drops the farthest hops first, as in
    ``_build_hop_ordered_sequence_layout_from_sparse_edges``.

    A token's hop is the hop at which it is first reached; the anchor's is 0.

    Args and returns are as in ``khop_layout``, with the neighbour table from
    ``_build_neighbor_table`` in place of the edges.
    """
    anchor_indices = anchor_indices.long()
    hop_bits = hop_distance.bit_length()
    padding_key = torch.iinfo(torch.long).max

    # A candidate's sort key is ``(node << hop_bits) | hop``. It groups each node's
    # copies with the smallest hop first, so the first copy carries the
    # first-reach hop. Padding gets the largest key and sorts last.
    frontier = anchor_indices.unsqueeze(1)
    key_parts = [frontier << hop_bits]
    for hop in range(1, hop_distance + 1):
        # A -1 id selects the all -1 last row of the table.
        frontier = neighbor_table[frontier].flatten(1)
        key_parts.append(
            torch.where(frontier >= 0, (frontier << hop_bits) | hop, padding_key)
        )
    keys, _ = torch.sort(torch.cat(key_parts, dim=1), dim=1)
    nodes = keys >> hop_bits
    hops = keys & ((1 << hop_bits) - 1)
    keep = (keys != padding_key) & F.pad(nodes.diff(dim=1) != 0, (1, 0), value=True)
    if prioritize_hop_order:
        # Re-sort the kept candidates by (hop, node id) so truncation below drops
        # the farthest hops first.
        node_radix = neighbor_table.size(0)
        order_keys, _ = torch.sort(
            torch.where(keep, hops * node_radix + nodes, padding_key), dim=1
        )
        keep = order_keys != padding_key
        nodes = order_keys % node_radix
        keys = (nodes << hop_bits) | (order_keys // node_radix)
    # The anchor goes to column 0 below; the other kept candidates are numbered
    # from 1 and cut at ``max_seq_len``.
    keep = keep & (nodes != anchor_indices.unsqueeze(1))
    positions = torch.cumsum(keep, dim=1)
    keep = keep & (positions < max_seq_len)
    # Dropped candidates scatter into column 0, which the anchor overwrites.
    layout = torch.full(
        (anchor_indices.numel(), max_seq_len), -1, dtype=torch.long, device=keys.device
    )
    layout.scatter_(1, torch.where(keep, positions, 0), keys)
    layout[:, 0] = anchor_indices << hop_bits
    return layout, 1 + keep.sum(dim=1), hop_bits


def khop_layout(
    edge_index: Int[Tensor, "2 num_edges"],
    num_nodes: int,
    anchor_indices: Int[Tensor, "num_anchors"],
    hop_distance: int,
    max_seq_len: int,
    sampling_direction: Literal["in", "out"] = "out",
    max_candidates: int = MAX_LAYOUT_CANDIDATES,
    prioritize_hop_order: bool = False,
) -> tuple[Int64[Tensor, "num_anchors max_seq_len"], Int64[Tensor, "num_anchors"], int]:
    """Builds the k-hop token layout of every anchor; the entry point of this module.

    Builds the neighbour table, checks the candidate budget, then expands and
    deduplicates each anchor's neighbourhood into a padded
    ``[num_anchors, max_seq_len]`` layout. Pass ``lengths`` to ``pack_sequences``
    to drop the padding. A host read of the table width is the only sync here.

    Args:
        edge_index: ``[2, num_edges]`` int32 or int64 local node ids in
            ``[0, num_nodes)``.
        num_nodes: Number of nodes.
        anchor_indices: ``[num_anchors]`` int32 or int64 local node ids of the
            anchors.
        hop_distance: Number of hops to expand.
        max_seq_len: Tokens per sequence, anchor included.
        sampling_direction: ``"in"`` reverses the edges, ``"out"`` keeps them.
        max_candidates: Maximum number of candidates plus neighbour-table cells; see
            ``MAX_LAYOUT_CANDIDATES``.
        prioritize_hop_order: Order neighbours by hop before node id.

    Returns:
        - ``layout``: ``[num_anchors, max_seq_len]`` long. Each valid cell packs
          a token's local node id and first-reach hop as
          ``(node << hop_bits) | hop``; decode with ``cell >> hop_bits`` and
          ``cell & ((1 << hop_bits) - 1)``. Column 0 is the anchor (hop 0),
          followed by its neighbours within ``hop_distance`` hops, in ascending
          node id (or by hop, then node id, with ``prioritize_hop_order``). The
          hop is the fewest hops from the anchor to the node. Cells at and after
          the sequence length are -1, so only decode the valid ones.
        - ``lengths``: ``[num_anchors]`` long count of valid tokens per anchor,
          anchor included, so at least 1 and at most ``max_seq_len``.
        - ``hop_bits``: Python int, ``hop_distance.bit_length()``.

    Raises:
        KhopLayoutTooLargeError: If the table cells plus candidates exceed ``max_candidates``.
    """
    neighbor_table = _build_neighbor_table(
        edge_index,
        num_nodes,
        anchor_indices.numel(),
        hop_distance,
        max_candidates,
        sampling_direction,
    )
    return _khop_sequence_layout(
        neighbor_table,
        anchor_indices,
        hop_distance,
        max_seq_len,
        prioritize_hop_order,
    )
