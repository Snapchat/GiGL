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
  - On CUDA this calls aten's memory-efficient varlen attention kernel directly.
    That op is private because torch 2.8 has no public attention API that takes
    ``cu_seqlens``; ``_varlen_attention`` explains why the public options do not
    fit. The kernel runs in fp32, which inference needs for accuracy, and aten's
    own autograd formula supplies the backward.
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
        |          -> PackedLayout(cu_seqlens, max_seq_len, flat_index, ...)
        |  grid.flatten()[flat_index]
        v
    packed buffer [num_tokens, dim]
        |
        |  per layer, on real tokens only:
        |    norm -> q / k / v linears -> [num_tokens, num_heads, head_dim]
        |    packed_attention(q, k, v, packed, ...)
        |        CUDA: aten memory-efficient attention with cu_seqlens
        |              (SDPA on a padded grid for per-pair biases or dropout)
        |        CPU:  SDPA on a padded grid (reference)
        |    output linear + residual -> FFN + residual
        v
    anchor readout: rows cu_seqlens[:-1], the first token of every sequence

Constraints:
  - The varlen kernel is CUDA only; other devices use the padded SDPA path.
    On CUDA the kernel runs when there is no ``per_pair`` bias and no dropout.
    With a ``per_pair`` bias or ``dropout_p > 0``, CUDA runs the padded SDPA
    path on the GPU instead, which costs ``num_sequences * max_seq_len`` rows of
    attention work rather than only the real tokens, and holds a dense
    ``num_sequences x num_heads x query_len x max_seq_len`` bias plus the same-sized
    scores.
  - ``head_dim`` need not be a multiple of 8. The aligned kernels need it, so
    ``packed_attention`` zero-pads q, k and v up to a multiple of 8 and rescales
    q so the kernel's scores match the unpadded head dim.
  - A ``PackedAttentionBias`` is part of the call. The padded SDPA path applies
    it. On CUDA a ``per_key`` bias runs on the kernel by folding it into q and k
    as an extra head-dim column; a ``per_pair`` bias goes to the padded SDPA
    path on the GPU.
  - Attention dropout (``dropout_p``) is applied by SDPA on the padded path. The
    varlen kernel is never called with dropout, so on CUDA ``dropout_p > 0``
    also selects the padded path. The padded path draws one mask per head and
    sequence.
  - ``cu_seqlens`` is int32, so the token count must fit in int32.

Example:
    >>> from gigl.transforms.static_khop import pack_sequences
    >>> lengths = torch.tensor([3, 1, 2])
    >>> packed = pack_sequences(lengths, max_seq_len=3)
    >>> packed.flat_index.tolist()
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
  steady state:
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

import math
from typing import Callable, NamedTuple, Optional, cast

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int32
from torch import Tensor

from gigl.transforms.static_khop import PackedLayout


def _pad_head_dim(tensor: Tensor, multiple: int = 8) -> Tensor:
    """Zero-pads the last dim of ``tensor`` up to a multiple of ``multiple``."""
    remainder = tensor.size(-1) % multiple
    if remainder == 0:
        return tensor
    return F.pad(tensor, (0, multiple - remainder))


def _rescale_query(query: Tensor, padded_dim: int) -> Tensor:
    """Scales q by ``sqrt(padded_dim / head_dim)`` so the kernel's scores use the unpadded head dim.

    The kernel scales scores by ``1 / sqrt(padded_dim)``. An explicit ``scale`` is
    not passed because torch 2.8's meta backward rejects it, and compiled training
    traces that meta backward.
    """
    return query * math.sqrt(padded_dim / query.size(-1))


def _pad_query_key(query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
    """Zero-pads q and k to a multiple of 8 and rescales q for the padded head dim.

    Multiplying q by ``sqrt(d_pad / d)`` makes
    ``q' . k' / sqrt(d_pad) == q . k / sqrt(d)``.
    """
    key = _pad_head_dim(key)
    query = _pad_head_dim(_rescale_query(query, key.size(-1)))
    return query, key


def _fold_per_key_bias(
    query: Float[Tensor, "num_queries num_heads head_dim"],
    key: Float[Tensor, "num_tokens num_heads head_dim"],
    per_key: Float[Tensor, "num_tokens num_heads"],
) -> tuple[Tensor, Tensor]:
    """Folds a per-key score bias into q and k as one extra head-dim column.

    With ``d`` the head dim and ``d_pad`` ``d + 1`` rounded up to a multiple of 8,
    ``query' = [query * sqrt(d_pad / d), 1, 0...]`` and
    ``key' = [key, per_key * sqrt(d_pad), 0...]``. The kernel's default scale
    ``1 / sqrt(d_pad)`` then gives
    ``q' . k' / sqrt(d_pad) == q . k / sqrt(d) + per_key``. The bias is cast to
    the key dtype.
    """
    head_dim = query.size(-1)
    padded_dim = (head_dim + 1 + 7) // 8 * 8
    query = _rescale_query(query, padded_dim)
    ones = torch.ones_like(query[..., :1])
    key_bias = (per_key.to(key.dtype) * math.sqrt(padded_dim)).unsqueeze(-1)
    query = _pad_head_dim(torch.cat([query, ones], dim=-1))
    key = _pad_head_dim(torch.cat([key, key_bias], dim=-1))
    return query, key


def _varlen_attention(
    query: Float[Tensor, "num_query_tokens num_heads head_dim"],
    key: Float[Tensor, "num_key_tokens num_heads head_dim"],
    value: Float[Tensor, "num_key_tokens num_heads value_dim"],
    cu_seqlens_q: Int32[Tensor, "num_sequences_plus_one"],
    cu_seqlens_k: Int32[Tensor, "num_sequences_plus_one"],
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> Float[Tensor, "num_query_tokens num_heads value_dim"]:
    """Runs attention on packed sequences without padding, on the CUDA varlen kernel.

    Query sequence ``i`` attends to key sequence ``i``, so padded positions cost
    nothing. The kernel reads the sequence offsets on the device and takes the
    length bounds as host ints, so the call does not sync with the host (it can
    be compiled). Differentiable in ``query``, ``key`` and ``value`` through the
    autograd formula that aten registers for the kernel.

    CUDA only; see the module docstring for the dropout limit. The head dims of
    ``query`` and ``key`` must be a multiple of 8; ``value`` may differ. The
    kernel scales scores by ``1 / sqrt(head_dim)`` of the head dim it is given.
    It picks its tile size from the head dim, so a padded head dim can move
    the call to the next size tier (64 padded to 72 lands in the tier for dims up
    to 128). The backward is non-deterministic unless
    ``torch.use_deterministic_algorithms(True)``, and it needs
    ``max_seqlen_q <= num_query_tokens`` and ``max_seqlen_k <= num_key_tokens``.

    This calls the private ``aten._efficient_attention_forward`` op because torch
    2.8 has no public attention API that takes ``cu_seqlens``:

    - ``F.scaled_dot_product_attention`` only takes padded
      ``[batch, heads, seq, dim]`` inputs.
    - Jagged nested tensors reach the same kernel through SDPA, but do not trace
      under ``torch.compile`` and sync with the host to find the min and max
      sequence lengths.
    - FlexAttention compiles, but its 128-token blocks waste more work than
      varlen on short sequences.
    - The memory-efficient kernel runs in fp32, which inference needs for
      accuracy. Flash Attention kernels support fp16 and bf16 only.

    Because the op is private, its arguments can change between torch releases.
    The CUDA tests compare this call with the padded path, so such a change fails
    tests rather than silently changing outputs.

    TODO(kmonte): when GiGL moves to torch >= 2.10, try the public
    ``torch.nn.attention.varlen.varlen_attn``. It takes the same packed inputs
    but wraps Flash Attention, so check fp32 support, backward and
    ``torch.compile`` before switching. If it lacks fp32, use it for bf16 only
    and keep this call for fp32.

    Args:
        cu_seqlens_q: Query sequence offsets.
        cu_seqlens_k: Key sequence offsets.
        max_seqlen_q: Upper bound on any query sequence length.
        max_seqlen_k: Upper bound on any key sequence length.
    """
    # The backward needs the logsumexp; skip it when nothing will differentiate.
    needs_grad = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (query, key, value)
    )
    # The op's type stub declares every argument as ``...``. It returns
    # (output, logsumexp, philox_seed, philox_offset, max_q, max_k); only the
    # output is used. The kernel takes ``[batch, tokens, heads, head_dim]``, and
    # ``cu_seqlens`` splits the single batch row into sequences.
    efficient_attention_forward = cast(
        Callable[..., tuple[Tensor, ...]],
        torch.ops.aten._efficient_attention_forward,
    )
    output = efficient_attention_forward(
        query.unsqueeze(0),
        key.unsqueeze(0),
        value.unsqueeze(0),
        None,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        0.0,
        0,
        needs_grad,
    )[0]
    return output.squeeze(0)


class PackedAttentionBias(NamedTuple):
    """Additive attention-score bias for ``packed_attention``.

    Both fields are optional and added to the scaled scores before the softmax.

    Attributes:
        per_key: One bias per key token and head. Every query in a sequence sees
            the same value for a given key.
        per_pair: One bias per sequence, head, query position and key position,
            on the padded grid, shaped
            ``[num_sequences, num_heads, query_len, max_seq_len]``. ``query_len``
            is ``max_seq_len``, or 1 when the queries are the first tokens
            (``query_is_first_token``); a ``max_seq_len``-row grid is also
            accepted then, and only its first row is used.
        grid: The full additive bias on the padded grid, as ``dense_attention_bias``
            builds it: padded-key mask, ``per_key`` and ``per_pair`` already
            combined, shaped ``[num_sequences, num_heads or 1, query_len or 1,
            max_seq_len]`` with the same first-row rule as ``per_pair``. When set,
            the padded SDPA path uses it as is and ignores ``per_key`` and
            ``per_pair``, so a caller that runs several layers builds it once.
    """

    per_key: Optional[Float[Tensor, "num_tokens num_heads"]] = None
    per_pair: Optional[
        Float[Tensor, "num_sequences num_heads max_seq_len max_seq_len"]
    ] = None
    grid: Optional[Tensor] = None


def dense_attention_bias(
    layout: PackedLayout,
    bias: Optional[PackedAttentionBias],
    dtype: torch.dtype,
    query_len: int,
) -> Tensor:
    """Builds the additive score bias on the padded grid.

    Returns ``bias.grid`` sliced to its first ``query_len`` rows when that is set.
    Otherwise starts from the padding mask, ``finfo.min / 2`` on invalid keys, shaped
    ``[num_sequences, 1, 1, max_seq_len]``. Adds ``bias.per_key`` scattered onto
    the grid as ``[num_sequences, num_heads, 1, max_seq_len]`` and
    ``bias.per_pair`` as ``[num_sequences, num_heads, query_len, max_seq_len]``,
    each only when present. ``bias.per_pair`` is sliced to its first
    ``query_len`` rows when it has more. The result broadcasts against the
    scores; a dense grid is only built when ``per_pair`` is present.
    """
    if bias is not None and bias.grid is not None:
        return bias.grid[:, :, :query_len, :].to(dtype)
    lengths = layout.cu_seqlens[1:] - layout.cu_seqlens[:-1]
    valid = torch.arange(layout.max_seq_len, device=lengths.device) < lengths.unsqueeze(
        1
    )
    attn_bias = torch.zeros(valid.shape, dtype=dtype, device=valid.device)
    attn_bias = attn_bias.masked_fill(~valid, torch.finfo(dtype).min / 2.0)
    attn_bias = attn_bias.view(valid.size(0), 1, 1, valid.size(1))
    if bias is None:
        return attn_bias
    if bias.per_key is not None:
        num_sequences, max_seq_len = valid.shape
        num_heads = bias.per_key.size(1)
        per_key_grid = torch.zeros(
            num_sequences * max_seq_len,
            num_heads,
            dtype=dtype,
            device=valid.device,
        ).index_copy(0, layout.flat_index, bias.per_key.to(dtype))
        attn_bias = attn_bias + per_key_grid.view(
            num_sequences, max_seq_len, num_heads
        ).permute(0, 2, 1).unsqueeze(2)
    if bias.per_pair is not None:
        attn_bias = attn_bias + bias.per_pair[:, :, :query_len, :].to(dtype)
    return attn_bias


def _padded_packed_attention(
    query: Float[Tensor, "num_queries num_heads head_dim"],
    key: Float[Tensor, "num_tokens num_heads head_dim"],
    value: Float[Tensor, "num_tokens num_heads head_dim"],
    layout: PackedLayout,
    query_is_first_token: bool,
    bias: Optional[PackedAttentionBias] = None,
    dropout_p: float = 0.0,
) -> Float[Tensor, "num_queries num_heads head_dim"]:
    """Runs packed attention on any device by padding, as the CPU path and the test reference.

    Scatters the tokens to the ``[num_sequences, max_seq_len]`` grid through
    ``layout.flat_index``, masks padded keys with the same additive
    ``finfo.min / 2`` bias as ``GraphTransformerEncoder``, adds ``bias`` when
    given, and gathers the outputs back. Every shape is known from the layout,
    so the call does not sync with the host and compiles without graph breaks.
    ``_varlen_attention`` is checked against this. With ``query_is_first_token``,
    the query grid holds one row per sequence, so only the first token's scores
    are computed.

    Args:
        query: Packed queries, or one query per sequence when
            ``query_is_first_token``.
        layout: Layout from ``pack_sequences``.
        query_is_first_token: ``query`` holds one query per sequence, the
            sequence's first token.
        bias: Optional additive score bias.
        dropout_p: Dropout probability on the attention weights, passed to SDPA.

    Returns:
        Attention output shaped like ``query``.
    """
    lengths = layout.cu_seqlens[1:] - layout.cu_seqlens[:-1]
    num_sequences = lengths.numel()
    max_seq_len = layout.max_seq_len

    def to_grid(tokens: Tensor) -> Tensor:
        """``[tokens, heads, dim]`` to ``[num_sequences, heads, max_seq_len, dim]``, zeros in padding."""
        grid = tokens.new_zeros(num_sequences * max_seq_len, *tokens.shape[1:])
        grid = grid.index_copy(0, layout.flat_index, tokens)
        return grid.view(num_sequences, max_seq_len, *tokens.shape[1:]).transpose(1, 2)

    if query_is_first_token:
        query_grid = query.unsqueeze(2)
        query_len = 1
    else:
        query_grid = to_grid(query)
        query_len = max_seq_len
    output = F.scaled_dot_product_attention(
        query_grid,
        to_grid(key),
        to_grid(value),
        attn_mask=dense_attention_bias(layout, bias, query.dtype, query_len),
        dropout_p=dropout_p,
    )
    if query_is_first_token:
        return output.squeeze(2)
    return output.transpose(1, 2).flatten(0, 1)[layout.flat_index]


def packed_attention(
    query: Float[Tensor, "num_queries num_heads head_dim"],
    key: Float[Tensor, "num_tokens num_heads head_dim"],
    value: Float[Tensor, "num_tokens num_heads head_dim"],
    layout: PackedLayout,
    query_is_first_token: bool,
    bias: Optional[PackedAttentionBias] = None,
    dropout_p: float = 0.0,
) -> Float[Tensor, "num_queries num_heads head_dim"]:
    """Attention within each packed sequence; the entry point of this module.

    Uses the CUDA varlen kernel on GPU and padded SDPA elsewhere. With
    ``query_is_first_token``, only each sequence's first token is queried, which
    is all the anchor readout of the last layer needs. An empty
    ``PackedAttentionBias`` behaves like ``bias=None``.

    On CUDA, a ``per_key`` bias runs on the varlen kernel through
    ``_fold_per_key_bias``. A ``per_pair`` bias or ``dropout_p > 0`` runs the
    padded SDPA path on the GPU. Off CUDA, the padded SDPA path always runs.

    Args and returns are as in ``_padded_packed_attention``.
    """
    has_grid = bias is not None and (bias.per_pair is not None or bias.grid is not None)
    if not query.is_cuda or has_grid or dropout_p > 0:
        return _padded_packed_attention(
            query, key, value, layout, query_is_first_token, bias, dropout_p
        )
    head_dim = query.size(-1)
    if bias is not None and bias.per_key is not None:
        query, key = _fold_per_key_bias(query, key, bias.per_key)
    elif head_dim % 8 != 0:
        query, key = _pad_query_key(query, key)
    if head_dim % 8 != 0:
        value = _pad_head_dim(value)
    if query_is_first_token:
        cu_seqlens_q = torch.arange(
            layout.cu_seqlens.size(0), dtype=torch.int32, device=query.device
        )
        max_seqlen_q = 1
    else:
        cu_seqlens_q = layout.cu_seqlens
        max_seqlen_q = min(layout.max_seq_len, query.size(0))
    output = _varlen_attention(
        query,
        key,
        value,
        cu_seqlens_q,
        layout.cu_seqlens,
        max_seqlen_q,
        min(layout.max_seq_len, key.size(0)),
    )
    return output[..., :head_dim]
