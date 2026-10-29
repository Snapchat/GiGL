"""Shared pieces for building per-anchor token sequences.

``select_sequences`` picks each anchor's tokens from its candidates into a padded
``[num_sequences, max_seq_len]`` grid, whatever generated the candidates (k-hop
expansion, PPR edges). ``pack_sequences`` turns that grid into a ``PackedLayout``:
the offsets and per-token indices that ``gigl.nn.packed_attention`` needs to run
on real tokens only. ``PackedRelationEdges`` holds the relation edges between
packed tokens.
"""

from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F
from jaxtyping import Int, Int32, Int64
from torch import Tensor


class PackedRelationEdges(NamedTuple):
    """Relation edges between packed tokens, grouped by relation.

    Attributes:
        target: ``[num_edges]`` long row of each edge's target in the layer's
            queries and outputs.
        source: ``[num_edges]`` long packed token row of each edge's source.
        relation: ``[num_edges]`` long relation of each edge.
        segments: ``(relation, start, end)`` edge range of every relation, in
            order, covering ``[0, num_edges)``; empty if there are no edges. They
            are host values, so the per-relation transforms never read the device.
        all_filtered: The batch has relation edges but none reach these outputs.
    """

    target: Tensor
    source: Tensor
    relation: Tensor
    segments: tuple[tuple[int, int, int], ...]
    all_filtered: bool = False


class PackedLayout(NamedTuple):
    """Where each sequence lives in the packed token buffer, and what each token is.

    Attributes:
        cu_seqlens: ``[num_sequences + 1]`` int32 cumulative sequence lengths,
            on the same device as the tokens. Sequence ``i`` occupies
            ``[cu_seqlens[i], cu_seqlens[i + 1])``. The attention kernel
            requires int32, so the total token count must fit in int32.
        max_seq_len: Python int upper bound on any sequence length. It must be at
            least the longest sequence, or the attention kernel skips rows. The
            attention clamps it to the token count.
        flat_index: ``[num_tokens]`` long indices of the valid cells of the
            flattened ``[num_sequences, max_seq_len]`` grid, sequence by sequence.
        sequence_id: ``[num_tokens]`` long sequence of each token,
            ``flat_index // max_seq_len``.
        position: ``[num_tokens]`` long position of each token within its
            sequence, ``flat_index % max_seq_len``. Position 0 is the anchor.
        node_index: ``[num_tokens]`` long row of each token in the concatenated
            node features.
        token_attrs: Per-token values the builder knows, each
            ``[num_tokens, ...]``.
        relation_edges: Relation edges in token coordinates, when the builder
            makes them.
    """

    cu_seqlens: Int32[Tensor, "num_sequences_plus_one"]
    max_seq_len: int
    flat_index: Int64[Tensor, "num_tokens"]
    sequence_id: Int64[Tensor, "num_tokens"]
    position: Int64[Tensor, "num_tokens"]
    node_index: Int64[Tensor, "num_tokens"]
    token_attrs: dict[str, Tensor]
    relation_edges: Optional[PackedRelationEdges] = None


def pack_sequences(
    lengths: Int64[Tensor, "num_sequences"],
    max_seq_len: int,
    node_grid: Optional[Int64[Tensor, "num_sequences max_seq_len"]] = None,
    attr_grids: Optional[dict[str, Tensor]] = None,
) -> PackedLayout:
    """Computes what is needed to drop the padding from a padded grid of sequences.

    Sequence ``i`` is valid in its first ``lengths[i]`` positions. Any
    ``[num_sequences, max_seq_len]`` per-position tensor gathers into packed
    order with ``grid.flatten()[flat_index]``, and ``cu_seqlens`` marks where
    each packed sequence starts. This costs one host read, of the total token
    count; it is why the packed buffer holds exactly the real tokens.

    A length above ``max_seq_len`` is clamped to it, because the grid has no
    positions beyond that. Checking for it would cost another host read.

    Args:
        lengths: ``[num_sequences]`` number of valid positions of each sequence.
        max_seq_len: Width of the padded grid.
        node_grid: ``[num_sequences, max_seq_len]`` node row of each cell. If
            None, ``node_index`` is the flat index.
        attr_grids: ``[num_sequences, max_seq_len, ...]`` grids to gather into
            ``token_attrs``, by name.

    Returns:
        The packed layout, without relation edges.
    """
    lengths = lengths.clamp(max=max_seq_len)
    valid = torch.arange(max_seq_len, device=lengths.device) < lengths.unsqueeze(1)
    cu_seqlens = torch.cat([lengths.new_zeros(1), torch.cumsum(lengths, 0)])
    flat_index = valid.flatten().nonzero().squeeze(1)
    if node_grid is None:
        node_index = flat_index
    else:
        node_index = node_grid.flatten()[flat_index]
    token_attrs = {}
    if attr_grids is not None:
        for name, grid in attr_grids.items():
            token_attrs[name] = grid.flatten(0, 1)[flat_index]
    return PackedLayout(
        cu_seqlens=cu_seqlens.to(torch.int32),
        max_seq_len=max_seq_len,
        flat_index=flat_index,
        sequence_id=flat_index // max_seq_len,
        position=flat_index % max_seq_len,
        node_index=node_index,
        token_attrs=token_attrs,
    )


# Bits of a sort key that hold the candidate's column; a node id takes the bits
# above. A fixed width keeps shapes, not the column count, out of the arithmetic, so
# ``torch.compile`` does not recompile when the candidate count changes.
_COLUMN_BITS = 26
_COLUMN_MASK = (1 << _COLUMN_BITS) - 1


def select_sequences(
    anchor_indices: Int[Tensor, "num_anchors"],
    candidates: Int64[Tensor, "num_anchors num_candidates"],
    max_seq_len: int,
    priority: Optional[Tensor] = None,
    ties_by_column: bool = False,
    include_anchor: bool = True,
) -> tuple[
    Int64[Tensor, "num_anchors max_seq_len"],
    Int64[Tensor, "num_anchors max_seq_len"],
    Int64[Tensor, "num_anchors"],
]:
    """Picks each anchor's tokens from its candidates, with fixed shapes.

    Every anchor's sequence is the anchor, then its distinct candidate nodes other
    than the anchor, ordered by ``priority`` (ties by ascending node id, or by
    column with ``ties_by_column``) and cut at ``max_seq_len``. A node that
    appears in several columns keeps the lowest column, so put the preferred copy
    of a node first. This is the selection step shared by the k-hop and PPR
    sequence builders: they differ in how they generate the candidates and in
    what ``priority`` ranks.

    Args:
        anchor_indices: ``[num_anchors]`` node ids of the anchors.
        candidates: ``[num_anchors, num_candidates]`` long candidate node ids,
            -1 at padding. The anchor itself may appear. There can be at most
            ``2 ** 26`` columns.
        max_seq_len: Tokens per sequence, anchor included.
        priority: ``[num_anchors, num_candidates]`` integer or float rank of each
            candidate; smaller comes first. If None, candidates are ordered by
            ascending node id, which costs one sort instead of two. Pass the
            column index to keep the candidates' own order.
        ties_by_column: Whether candidates with equal ``priority`` stay in column
            order instead of ascending node id. It has no effect without
            ``priority``.
        include_anchor: Whether column 0 holds the anchor, which is then dropped
            from the candidates. If False, the sequence is only candidates, and
            the anchor stays in it if it is one of them.

    Returns:
        - ``nodes``: ``[num_anchors, max_seq_len]`` long node ids, the anchor in
          column 0 if ``include_anchor``, -1 from the sequence length on.
        - ``source``: ``[num_anchors, max_seq_len]`` long column of ``candidates``
          each token came from, so a per-candidate payload follows with
          ``payload.gather(1, source.clamp(min=0))``. It is -1 at the anchor and
          at padding.
        - ``lengths``: ``[num_anchors]`` long count of valid tokens, anchor
          included. It is at most ``max_seq_len``, except that it counts the
          anchor even when ``max_seq_len`` is 0.
    """
    anchor_indices = anchor_indices.long()
    num_candidates = candidates.size(1)
    padding_key = torch.iinfo(torch.long).max

    # Sort ``(node << _COLUMN_BITS) | column``: each node's copies are adjacent
    # with the lowest column first, and padding sorts last.
    columns = torch.arange(num_candidates, device=candidates.device)
    keys = torch.where(
        candidates >= 0, (candidates << _COLUMN_BITS) | columns, padding_key
    )
    keys = torch.sort(keys, dim=1).values
    nodes = keys >> _COLUMN_BITS
    keep = (keys != padding_key) & F.pad(nodes.diff(dim=1) != 0, (1, 0), value=True)
    if include_anchor:
        keep = keep & (nodes != anchor_indices.unsqueeze(1))
    if priority is not None:
        if ties_by_column:
            # Put the kept candidates back in column order, so the stable sort
            # below leaves priority ties there.
            by_column = torch.sort(
                torch.where(keep, keys & _COLUMN_MASK, num_candidates), dim=1
            ).indices
            keys = keys.gather(1, by_column)
            keep = keep.gather(1, by_column)
        # Dropped candidates rank last, so the kept ones come first, in
        # ``priority`` order and then ascending node id. Padding decodes to an
        # out-of-range column; its priority is never used.
        columns = (keys & _COLUMN_MASK).clamp(max=num_candidates - 1)
        if priority.is_floating_point():
            worst = torch.inf
        else:
            worst = torch.iinfo(priority.dtype).max
        order = torch.sort(
            torch.where(keep, priority.gather(1, columns), worst),
            dim=1,
            stable=True,
        ).indices
        keys = keys.gather(1, order)
        keep = keep.gather(1, order)
    # With the anchor, it takes column 0 and the kept candidates are numbered from
    # 1. The kept candidates are cut at ``max_seq_len``.
    first_position = int(include_anchor)
    positions = torch.cumsum(keep, dim=1) - 1 + first_position
    keep = keep & (positions < max_seq_len)
    # Dropped candidates scatter into an extra last column, which is sliced off.
    packed = torch.full(
        (anchor_indices.numel(), max_seq_len + 1),
        -1,
        dtype=torch.long,
        device=keys.device,
    )
    packed.scatter_(1, torch.where(keep, positions, max_seq_len), keys)
    packed = packed[:, :max_seq_len]
    if include_anchor and max_seq_len > 0:
        packed[:, 0] = anchor_indices << _COLUMN_BITS
    source = torch.where(packed >= 0, packed & _COLUMN_MASK, -1)
    if include_anchor and max_seq_len > 0:
        source[:, 0] = -1
    return packed >> _COLUMN_BITS, source, first_position + keep.sum(dim=1)
