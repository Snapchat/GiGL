"""Packed token layout: where each sequence lives in a buffer without padding.

``pack_sequences`` turns a padded ``[num_sequences, max_seq_len]`` grid of
sequences into the offsets (``PackedLayout``) that ``gigl.nn.packed_attention``
needs to run on real tokens only. It does not depend on how the sequences were
built.
"""

from typing import NamedTuple

import torch
from jaxtyping import Int32, Int64
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
