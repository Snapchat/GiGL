"""Packed token layout: where each sequence lives in a buffer without padding.

``pack_sequences`` turns a padded ``[num_sequences, max_seq_len]`` grid of
sequences into the offsets (``PackedLayout``) that ``gigl.nn.packed_attention``
needs to run on real tokens only. It does not depend on how the sequences were
built. ``select_sequences`` is the shared step that builds the padded grid from
per-anchor candidates.
"""

from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F
from jaxtyping import Int, Int32, Int64
from torch import Tensor


class PackedLayout(NamedTuple):
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
) -> tuple[Int64[Tensor, "num_tokens"], PackedLayout]:
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
    return flat_index, PackedLayout(cu_seqlens.to(torch.int32), max_seq_len)


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
) -> tuple[
    Int64[Tensor, "num_anchors max_seq_len"],
    Int64[Tensor, "num_anchors max_seq_len"],
    Int64[Tensor, "num_anchors"],
]:
    """Picks each anchor's tokens from its candidates, with fixed shapes.

    Every anchor's sequence is the anchor, then its distinct candidate nodes other
    than the anchor, ordered by ``priority`` (ties by ascending node id) and cut
    at ``max_seq_len``. A node that appears in several columns keeps the lowest
    column, so put the preferred copy of a node first. This is the selection step
    shared by the k-hop and PPR sequence builders: they differ in how they
    generate the candidates and in what ``priority`` ranks.

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

    Returns:
        - ``nodes``: ``[num_anchors, max_seq_len]`` long node ids, the anchor in
          column 0, -1 from the sequence length on.
        - ``source``: ``[num_anchors, max_seq_len]`` long column of ``candidates``
          each token came from, so a per-candidate payload follows with
          ``payload.gather(1, source.clamp(min=0))``. It is -1 at the anchor and
          at padding.
        - ``lengths``: ``[num_anchors]`` long count of valid tokens, anchor
          included, so at least 1 and at most ``max_seq_len``.
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
    keep = (
        (keys != padding_key)
        & F.pad(nodes.diff(dim=1) != 0, (1, 0), value=True)
        & (nodes != anchor_indices.unsqueeze(1))
    )
    if priority is not None:
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
    # The anchor goes to column 0 below; the other kept candidates are numbered
    # from 1 and cut at ``max_seq_len``.
    positions = torch.cumsum(keep, dim=1)
    keep = keep & (positions < max_seq_len)
    # Dropped candidates scatter into column 0, which the anchor overwrites.
    packed = torch.full(
        (anchor_indices.numel(), max_seq_len), -1, dtype=torch.long, device=keys.device
    )
    packed.scatter_(1, torch.where(keep, positions, 0), keys)
    packed[:, 0] = anchor_indices << _COLUMN_BITS
    source = torch.where(packed >= 0, packed & _COLUMN_MASK, -1)
    source[:, 0] = -1
    return packed >> _COLUMN_BITS, source, 1 + keep.sum(dim=1)
