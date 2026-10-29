"""Packed token layout: where each sequence lives in a buffer without padding.

``pack_sequences`` turns a padded ``[num_sequences, max_seq_len]`` grid of
sequences into a ``PackedLayout``: the offsets and per-token indices that
``gigl.nn.packed_attention`` needs to run on real tokens only. It does not depend
on how the sequences were built.
"""

from typing import NamedTuple, Optional

import torch
from jaxtyping import Int32, Int64
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
