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
        |        CPU:  SDPA on a padded grid (reference)
        |    output linear + residual -> FFN + residual
        v
    anchor readout: rows cu_seqlens[:-1], the first token of every sequence

Constraints:
  - The varlen kernel is CUDA only; other devices use the padded SDPA path.
  - ``head_dim`` need not be a multiple of 8. The aligned kernels need it, so
    ``packed_attention`` zero-pads q, k and v up to a multiple of 8 and rescales
    q so the kernel's scores match the unpadded head dim.
  - A ``PackedAttentionBias`` is part of the call. The CPU path applies it. On
    CUDA a ``per_key`` bias runs on the kernel by folding it into q and k as an
    extra head-dim column; a ``per_pair`` bias raises ``NotImplementedError``.
  - Attention dropout is not supported; the kernel's varlen dropout has not been
    tested against the padded path.
  - ``cu_seqlens`` is int32, so the token count must fit in int32.

Example:
    >>> from gigl.transforms.packed_layout import pack_sequences
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
from jaxtyping import Bool, Float, Int32
from torch import Tensor
from torch_geometric.utils import to_dense_batch

from gigl.transforms.packed_layout import PackedLayout


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
            on the padded ``[max_seq_len, max_seq_len]`` grid.
    """

    per_key: Optional[Float[Tensor, "num_tokens num_heads"]] = None
    per_pair: Optional[
        Float[Tensor, "num_sequences num_heads max_seq_len max_seq_len"]
    ] = None


def _dense_attention_bias(
    layout: PackedLayout,
    bias: Optional[PackedAttentionBias],
    valid: Bool[Tensor, "num_sequences max_seq_len"],
    dtype: torch.dtype,
) -> Tensor:
    """Builds the additive score bias on the padded grid.

    Starts from the padding mask, ``finfo.min / 2`` on invalid keys, shaped
    ``[num_sequences, 1, 1, max_seq_len]``. Adds ``bias.per_key`` scattered onto
    the grid as ``[num_sequences, num_heads, 1, max_seq_len]`` and
    ``bias.per_pair`` as ``[num_sequences, num_heads, max_seq_len, max_seq_len]``,
    each only when present. The result broadcasts against the scores; a dense
    grid is only built when ``per_pair`` is present.
    """
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
        attn_bias = attn_bias + bias.per_pair.to(dtype)
    return attn_bias


def _padded_packed_attention(
    query: Float[Tensor, "num_queries num_heads head_dim"],
    key: Float[Tensor, "num_tokens num_heads head_dim"],
    value: Float[Tensor, "num_tokens num_heads head_dim"],
    layout: PackedLayout,
    query_is_first_token: bool,
    bias: Optional[PackedAttentionBias] = None,
) -> Float[Tensor, "num_queries num_heads head_dim"]:
    """Runs packed attention on any device by padding, as the CPU path and the test reference.

    Unpacks the tokens to ``[num_sequences, max_seq_len]``, masks padded keys
    with the same additive ``finfo.min / 2`` bias as ``GraphTransformerEncoder``,
    adds ``bias`` when given, and unpacks the outputs. ``_varlen_attention`` is
    checked against this.

    Args:
        query: Packed queries, or one query per sequence when
            ``query_is_first_token``.
        layout: Layout from ``pack_sequences``.
        query_is_first_token: ``query`` holds one query per sequence, the
            sequence's first token.
        bias: Optional additive score bias.

    Returns:
        Attention output shaped like ``query``.
    """
    first_token = layout.cu_seqlens[:-1].long()
    lengths = layout.cu_seqlens[1:] - layout.cu_seqlens[:-1]
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
        max_num_nodes=layout.max_seq_len,
    )
    query_grid, key_grid, value_grid = (
        grid.transpose(1, 2) for grid in qkv_grid.chunk(3, dim=2)
    )
    output = F.scaled_dot_product_attention(
        query_grid,
        key_grid,
        value_grid,
        attn_mask=_dense_attention_bias(layout, bias, mask, query.dtype),
    )
    output = output.transpose(1, 2)[mask]
    if query_is_first_token:
        return output[first_token]
    return output


def packed_attention(
    query: Float[Tensor, "num_queries num_heads head_dim"],
    key: Float[Tensor, "num_tokens num_heads head_dim"],
    value: Float[Tensor, "num_tokens num_heads head_dim"],
    layout: PackedLayout,
    query_is_first_token: bool,
    bias: Optional[PackedAttentionBias] = None,
) -> Float[Tensor, "num_queries num_heads head_dim"]:
    """Attention within each packed sequence; the entry point of this module.

    Uses the CUDA varlen kernel on GPU and padded SDPA elsewhere. With
    ``query_is_first_token``, only each sequence's first token is queried, which
    is all the anchor readout of the last layer needs. An empty
    ``PackedAttentionBias`` behaves like ``bias=None``.

    On CUDA, ``per_key`` runs on the varlen kernel through ``_fold_per_key_bias``;
    ``per_pair`` raises ``NotImplementedError`` (no FlexAttention backend).

    Args and returns are as in ``_padded_packed_attention``.
    """
    if not query.is_cuda:
        return _padded_packed_attention(
            query, key, value, layout, query_is_first_token, bias
        )
    if bias is not None and bias.per_pair is not None:
        raise NotImplementedError("the FlexAttention backend is not in yet")
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
