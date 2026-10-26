"""Attention over packed variable-length sequences: the compute half of the packed backend.

This is the attention side of the packed backend of ``GraphTransformerEncoder``.
The encoder turns each anchor's k-hop neighbourhood into a token sequence
(``gigl.transforms.static_khop``), copies the real tokens of all sequences into
one ``[num_tokens, dim]`` buffer, and runs the transformer on that buffer.
``packed_attention`` is the attention step: each sequence attends only to itself,
with boundaries given by ``cu_seqlens``.

Why this exists:
  - With uneven sequence lengths (about 90 real tokens out of 256 in our models),
    most of the linear-layer, FFN and attention work on a padded
    ``[num_sequences, max_seq_len, dim]`` batch is spent on padding. Packed
    tokens spend it only on real ones.
  - On CUDA this calls FlexAttention (``torch.nn.attention.flex_attention``, a
    public API) with a block mask that keeps each sequence to itself. The mask is
    built from ``cu_seqlens`` with tensor ops only, so there is no host sync and
    the layers compile once for any token count. It runs in fp32, which
    inference needs for accuracy, and FlexAttention supplies the backward.
    FlexAttention can also add per-score terms through ``score_mod``, which is
    where attention biases such as PPR scores would go.
  - On CPU the same inputs go through SDPA on a padded grid. That is the
    reference the CUDA path is tested against, and it lets the packed backend run
    anywhere.
  - Packing needs one host read (the total token count, in ``pack_sequences``)
    so the buffer holds exactly the real tokens. Everything else has fixed
    shapes and no syncs, so the layers can be compiled with ``torch.compile``.

Data flow of a packed forward::

    padded grid [num_sequences, max_seq_len, ...]      lengths [num_sequences]
        |                                                   |
        |      pack_sequences(lengths, max_seq_len)  <------+
        |          -> flat_index, PackedSequences(cu_seqlens, max_seq_len)
        |  grid.flatten()[flat_index]
        v
    packed buffer [num_tokens, dim]
        |
        |  per layer, on real tokens only:
        |    norm -> q / k / v linears -> [num_tokens, num_heads, head_dim]
        |    packed_attention(q, k, v, packed, ...)
        |        CUDA: FlexAttention with a block mask built from cu_seqlens
        |        CPU:  SDPA on a padded grid (reference)
        |    output linear + residual -> FFN + residual
        v
    anchor readout: rows cu_seqlens[:-1], the first token of every sequence

Constraints:
  - FlexAttention runs on CUDA only; other devices use the padded SDPA path.
  - Attention dropout is not supported.
  - ``cu_seqlens`` is int32, so the token count must fit in int32.
  - Each mask block covers 128 tokens, so on average a query block also visits
    the neighbouring sequences' tokens in its blocks and masks them out. With
    about 90 real tokens per sequence this costs little next to the linears.
  - Torch 2.8 builds the mask tensors in the compiled region with a flexible
    layout that FlexAttention rejects, so they go through ``_fixed_layout``.
  - Fewer than 128 queries (the anchor-only layer on a small batch) take
    FlexAttention's decoding kernel, which needs the 128-token mask blocks.

Example:
    >>> from gigl.transforms.static_khop import pack_sequences
    >>> lengths = torch.tensor([3, 1, 2])
    >>> flat_index, packed = pack_sequences(lengths, max_seq_len=3)
    >>> flat_index.tolist()
    [0, 1, 2, 3, 6, 7]
    >>> num_tokens = int(lengths.sum())
    >>> query = key = value = torch.randn(num_tokens, 2, 8)
    >>> output = packed_attention(
    ...     query, key, value, packed, query_is_first_token=False
    ... )
    >>> output.shape
    torch.Size([6, 2, 8])
    >>> first_token_only = packed_attention(
    ...     query[packed.cu_seqlens[:-1].long()],
    ...     key,
    ...     value,
    ...     packed,
    ...     query_is_first_token=True,
    ... )
    >>> first_token_only.shape
    torch.Size([3, 2, 8])

Results:
  We have seen the following with this path. These are observations on one
  production model, not benchmarks of this module alone.

  One L4 GPU, 3-layer graph transformer (hidden 512, 8 heads), sequences of up
  to 256 tokens holding about 90 real tokens on average, realistic batches,
  steady state. The first three figures were measured with aten's memory-
  efficient varlen kernel in place of FlexAttention; against it, FlexAttention
  ran 0.1% slower to 3.7% faster in forward and 2% to 3% slower in a training
  step (see the FlexAttention comparison below):
    - Forward, no compile: 225 -> 76 ms per batch (~3.0x).
    - Forward, per-layer ``torch.compile``: 53 ms per batch (~4.3x vs the
      padded path without compile).
    - Training step: 944 -> 364 ms (~2.6x).
    - Peak GPU memory: 15.4 -> 6.8 GiB training, 6.9 -> 3.8 GiB forward.

  120 L4 GPUs. These runs also include ``torch.compile`` and loader changes,
  so the gain is not from packing alone:
    - Training step: 6.62 -> 2.66 s (2.5x), peak memory 9.8 -> 4.3 GiB.
    - Full-graph inference over ~230M nodes: 1884 -> 1014 s end to end (1.9x).
      The model stopped being the bottleneck; sampling was.
"""

from typing import Callable

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int32, Int64
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from torch_geometric.utils import to_dense_batch

from gigl.transforms.static_khop import PackedSequences

_BLOCK_SIZE = 128
_compiled_flex_attention = torch.compile(flex_attention)


@torch.library.custom_op("gigl::fixed_layout", mutates_args=())
def _fixed_layout(tensor: Tensor) -> Tensor:
    """Returns a copy of ``tensor`` whose memory layout Inductor treats as fixed.

    Inductor lowers a custom op as an extern call, and an extern call's output
    has a fixed layout. FlexAttention needs that for every tensor that its masks
    read or that make up the BlockMask. Tensors computed inside the same
    compiled region have a flexible layout, which fails with "convert
    FlexibleLayout to FixedLayout first" on torch 2.8.
    """
    return tensor.clone()


@_fixed_layout.register_fake
def _(tensor: Tensor) -> Tensor:
    return torch.empty_like(tensor)


class _DocumentMask:
    """``mask_mod`` letting each token attend to the tokens of its own sequence.

    The sequence ids are a tensor attribute, so every batch runs the same
    function code and ``torch.compile`` has nothing to guard on per batch.
    """

    def __init__(self, sequence: Int64[Tensor, "num_tokens"]) -> None:
        self._sequence = sequence

    def __call__(
        self, batch: Tensor, head: Tensor, query_index: Tensor, key_index: Tensor
    ) -> Tensor:
        return self._sequence[query_index] == self._sequence[key_index]


class _FirstTokenMask:
    """``mask_mod`` for one query per sequence: query ``i`` attends to the tokens of sequence ``i``."""

    def __init__(self, sequence: Int64[Tensor, "num_tokens"]) -> None:
        self._sequence = sequence

    def __call__(
        self, batch: Tensor, head: Tensor, query_index: Tensor, key_index: Tensor
    ) -> Tensor:
        return self._sequence[key_index] == query_index


def _block_mask(
    mask_mod: Callable[..., Tensor],
    num_queries: int,
    num_tokens: int,
    first_key: Int64[Tensor, "num_query_blocks"],
    end_key: Int64[Tensor, "num_query_blocks"],
) -> BlockMask:
    """Builds a BlockMask whose query block ``i`` visits the key blocks covering ``[first_key[i], end_key[i])``.

    It uses ``BlockMask.from_kv_blocks`` instead of ``create_block_mask``, which
    would evaluate the mask on a dense ``num_queries x num_tokens`` grid (2.4 GB
    for 49k tokens). Only tensor ops on the sequence offsets run, so there is no
    host sync, and with symbolic sizes the whole thing compiles once. Every
    visited block applies ``mask_mod``.
    """
    num_key_blocks = (num_tokens + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    first_block = first_key // _BLOCK_SIZE
    num_visited = ((end_key + _BLOCK_SIZE - 1) // _BLOCK_SIZE - first_block).to(
        torch.int32
    )
    # The visited blocks come first in each row; the remaining entries only keep
    # the row a permutation and sit past ``num_visited``.
    key_blocks = torch.arange(num_key_blocks, device=first_key.device)
    indices = ((first_block[:, None] + key_blocks[None, :]) % num_key_blocks).to(
        torch.int32
    )
    mask = BlockMask.from_kv_blocks(
        num_visited[None, None],
        indices[None, None],
        BLOCK_SIZE=_BLOCK_SIZE,
        mask_mod=mask_mod,
        seq_lengths=(num_queries, num_tokens),
    )
    return BlockMask(
        mask.seq_lengths,
        _fixed_layout(mask.kv_num_blocks),
        _fixed_layout(mask.kv_indices),
        None,
        None,
        _fixed_layout(mask.q_num_blocks),
        _fixed_layout(mask.q_indices),
        None,
        None,
        mask.BLOCK_SIZE,
        mask.mask_mod,
    )


def _flex_packed_attention(
    query: Float[Tensor, "num_query_tokens num_heads head_dim"],
    key: Float[Tensor, "num_key_tokens num_heads head_dim"],
    value: Float[Tensor, "num_key_tokens num_heads head_dim"],
    cu_seqlens: Int32[Tensor, "num_sequences_plus_one"],
    query_is_first_token: bool,
) -> Float[Tensor, "num_query_tokens num_heads head_dim"]:
    """Runs attention on packed sequences without padding, with FlexAttention on CUDA.

    The BlockMask is built here from ``cu_seqlens``, so inside a compiled
    region it follows the traced sizes: a layer compiled with the token
    dimension marked dynamic compiles once. Differentiable in ``query``,
    ``key`` and ``value``.

    Args:
        cu_seqlens: Sequence offsets; the keys are all the tokens they cover.
        query_is_first_token: ``query`` holds one query per sequence.
    """
    cu_seqlens = cu_seqlens.long()
    num_tokens = key.size(0)
    # Counting the offsets at or below a token gives its sequence plus one.
    # Slicing ``cu_seqlens`` here would leave Inductor a view it cannot lower.
    sequence = _fixed_layout(
        torch.searchsorted(
            cu_seqlens, torch.arange(num_tokens, device=key.device), right=True
        )
        - 1
    )
    if query_is_first_token:
        num_queries = cu_seqlens.size(0) - 1
        first_query = torch.arange(0, num_queries, _BLOCK_SIZE, device=key.device)
        last_query = (first_query + _BLOCK_SIZE).clamp(max=num_queries) - 1
        mask = _block_mask(
            _FirstTokenMask(sequence),
            num_queries,
            num_tokens,
            cu_seqlens[first_query],
            cu_seqlens[last_query + 1],
        )
    else:
        first_query = torch.arange(0, num_tokens, _BLOCK_SIZE, device=key.device)
        last_query = (first_query + _BLOCK_SIZE).clamp(max=num_tokens) - 1
        mask = _block_mask(
            _DocumentMask(sequence),
            num_tokens,
            num_tokens,
            cu_seqlens[sequence[first_query]],
            cu_seqlens[sequence[last_query] + 1],
        )
    output = _compiled_flex_attention(
        query.transpose(0, 1).unsqueeze(0),
        key.transpose(0, 1).unsqueeze(0),
        value.transpose(0, 1).unsqueeze(0),
        block_mask=mask,
    )
    return output.squeeze(0).transpose(0, 1)


def _padded_packed_attention(
    query: Float[Tensor, "num_queries num_heads head_dim"],
    key: Float[Tensor, "num_tokens num_heads head_dim"],
    value: Float[Tensor, "num_tokens num_heads head_dim"],
    packed: PackedSequences,
    query_is_first_token: bool,
) -> Float[Tensor, "num_queries num_heads head_dim"]:
    """Runs packed attention on any device by padding, as the CPU path and the test reference.

    Unpacks the tokens to ``[num_sequences, max_seq_len]``, masks padded keys
    with the same additive ``finfo.min / 2`` bias as ``GraphTransformerEncoder``,
    and unpacks the outputs. ``_varlen_attention`` is checked against this.

    Args:
        query: Packed queries, or one query per sequence when
            ``query_is_first_token``.
        packed: Layout from ``pack_sequences``.
        query_is_first_token: ``query`` holds one query per sequence, the
            sequence's first token.

    Returns:
        Attention output shaped like ``query``.
    """
    first_token = packed.cu_seqlens[:-1].long()
    lengths = packed.cu_seqlens[1:] - packed.cu_seqlens[:-1]
    num_sequences = lengths.numel()
    token_sequence = torch.repeat_interleave(
        torch.arange(num_sequences, device=lengths.device),
        lengths,
        output_size=key.size(0),
    )
    if query_is_first_token:
        # Run full attention with each sequence's query in its first row, then
        # keep those rows.
        query = torch.zeros_like(key).index_copy(0, first_token, query)
    qkv_grid, mask = to_dense_batch(
        torch.cat([query, key, value], dim=1),
        token_sequence,
        batch_size=num_sequences,
        max_num_nodes=packed.max_seq_len,
    )
    query_grid, key_grid, value_grid = (
        grid.transpose(1, 2) for grid in qkv_grid.chunk(3, dim=2)
    )
    attn_bias = torch.zeros(mask.shape, dtype=query.dtype, device=query.device)
    attn_bias = attn_bias.masked_fill(~mask, torch.finfo(query.dtype).min / 2.0)
    output = F.scaled_dot_product_attention(
        query_grid,
        key_grid,
        value_grid,
        attn_mask=attn_bias.view(num_sequences, 1, 1, -1),
    )
    output = output.transpose(1, 2)[mask]
    if query_is_first_token:
        return output[first_token]
    return output


def packed_attention(
    query: Float[Tensor, "num_queries num_heads head_dim"],
    key: Float[Tensor, "num_tokens num_heads head_dim"],
    value: Float[Tensor, "num_tokens num_heads head_dim"],
    packed: PackedSequences,
    query_is_first_token: bool,
) -> Float[Tensor, "num_queries num_heads head_dim"]:
    """Attention within each packed sequence; the entry point of this module.

    Uses FlexAttention with a document mask on GPU and padded SDPA elsewhere. With
    ``query_is_first_token``, only each sequence's first token is queried, which
    is all the anchor readout of the last layer needs.

    Args and returns are as in ``_padded_packed_attention``.
    """
    if not query.is_cuda:
        return _padded_packed_attention(query, key, value, packed, query_is_first_token)
    return _flex_packed_attention(
        query, key, value, packed.cu_seqlens, query_is_first_token
    )
