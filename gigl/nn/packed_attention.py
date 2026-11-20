"""Attention over packed variable-length sequences: the compute half of the packed backend.

This is the attention side of the packed backend of ``GraphTransformerEncoder``.
The encoder builds one token sequence per anchor (for example, from its k-hop
neighborhood with ``gigl.transforms.static_khop``), copies the real tokens of all
sequences into one ``[num_tokens, dim]`` buffer, and runs the transformer on that
buffer. ``packed_attention`` is the attention step and does not depend on how the
sequences were built: each sequence attends only to itself, with boundaries given
by ``cu_seqlens``.

Why this exists:
  - With uneven sequence lengths (about 90 real tokens out of 256 in our models),
    most of the linear-layer, FFN and attention work on a padded
    ``[num_sequences, max_seq_len, dim]`` batch is spent on padding. Packed
    tokens spend it only on real ones.
  - On CUDA this calls aten's memory-efficient varlen attention kernel, wrapped
    in a custom op. The kernel is private because torch 2.8 has no public attention API that takes
    ``cu_seqlens``; ``_varlen_attention`` explains why the public options do not
    fit. The kernel runs in fp32, which inference needs for accuracy, and aten's
    own backward kernel supplies the gradients.
  - On CPU the same inputs go through SDPA on padded grids, one per length
    bucket of the layout. That is the reference the CUDA path is tested against,
    and it lets the packed backend run anywhere.
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
        |    packed_attention(q, k, v, layout, ...)
        |        CUDA: aten memory-efficient attention with cu_seqlens
        |        CUDA, per-pair bias on every token or dropout:
        |             SDPA per length bucket
        |        CPU:  SDPA per length bucket (reference)
        |    output linear + residual -> FFN + residual
        v
    anchor readout: rows cu_seqlens[:-1], the first token of every sequence

Constraints:
  - The varlen kernel is CUDA only; other devices use SDPA per length bucket.
  - ``head_dim`` need not be a multiple of 8. The aligned kernels need it, so
    ``packed_attention`` zero-pads q, k and v up to a multiple of 8 and rescales
    q so the kernel's scores match the unpadded head dim.
  - A ``PackedAttentionBias`` is part of the call. The CPU path applies it. On
    CUDA a ``per_key`` bias runs on the kernel by folding it into q and k as an
    extra head-dim column. A ``per_pair`` bias on every token runs SDPA per
    bucket. With ``query_is_first_token``, row 0 of each sequence's ``per_pair``
    block is a per-key bias, so it runs on the kernel too.
  - Attention dropout always runs SDPA per bucket, never the varlen kernel: in
    torch 2.8 the kernel's varlen dropout uses different masks in the forward
    and the backward pass when ``cu_seqlens`` is given, so its gradients are
    wrong.
  - ``cu_seqlens`` is int32, so the token count must fit in int32.

Example:
    >>> from gigl.transforms.sequence_layout import pack_sequences
    >>> lengths = torch.tensor([3, 1, 2])
    >>> layout = pack_sequences(lengths, max_seq_len=3)
    >>> layout.flat_index.tolist()
    [0, 1, 2, 3, 6, 7]
    >>> num_tokens = int(lengths.sum())
    >>> query = key = value = torch.randn(num_tokens, 2, 8)
    >>> output = packed_attention(
    ...     query, key, value, layout, query_is_first_token=False
    ... )
    >>> output.shape
    torch.Size([6, 2, 8])
    >>> first_token_only = packed_attention(
    ...     query[layout.cu_seqlens[:-1].long()],
    ...     key,
    ...     value,
    ...     layout,
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
from jaxtyping import Float, Int32, Int64
from torch import Tensor

from gigl.transforms.sequence_layout import PackedBucket, PackedLayout

# The kernel returns its philox seed and offset on the CPU when dropout is off,
# but its meta registration returns them as CUDA tensors. ``torch.compile`` then
# treats them as CUDA buffers and can reuse their storage for another small int64
# buffer, such as the sequence-start index of a one-sequence batch, so a kernel
# writes through a host pointer. These ops hide the kernel's seed and offset: they
# return only the output and logsumexp, and the backward supplies its own.


@torch.library.custom_op("gigl::varlen_attention", mutates_args=())
def _varlen_attention_op(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    compute_log_sumexp: bool,
) -> tuple[Tensor, Tensor]:
    """Runs the varlen kernel and returns ``(output, logsumexp)``.

    The type stub of the aten op declares every argument as ``...``. It returns
    (output, logsumexp, philox_seed, philox_offset, max_q, max_k). The kernel takes
    ``[batch, tokens, heads, head_dim]``, and ``cu_seqlens`` splits the single
    batch row into sequences.
    """
    efficient_attention_forward = cast(
        Callable[..., tuple[Tensor, ...]],
        torch.ops.aten._efficient_attention_forward,
    )
    output, logsumexp = efficient_attention_forward(
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
        compute_log_sumexp,
    )[:2]
    return output.squeeze(0), logsumexp


@_varlen_attention_op.register_fake
def _(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    compute_log_sumexp: bool,
) -> tuple[Tensor, Tensor]:
    # The per-key fold widens q and k but not v, so the output takes v's head dim.
    output = query.new_empty((*query.shape[:-1], value.size(-1)))
    logsumexp_size = 0
    if compute_log_sumexp:
        # The kernel pads the logsumexp rows to a multiple of 32.
        logsumexp_size = (max_seqlen_q + 31) // 32 * 32
    logsumexp = query.new_empty(
        (cu_seqlens_q.size(0) - 1, query.size(1), logsumexp_size), dtype=torch.float32
    )
    return output, logsumexp


@torch.library.custom_op("gigl::varlen_attention_backward", mutates_args=())
def _varlen_attention_backward_op(
    grad_output: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    output: Tensor,
    logsumexp: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Returns the gradients of ``_varlen_attention_op`` in query, key and value."""
    efficient_attention_backward = cast(
        Callable[..., tuple[Tensor, ...]],
        torch.ops.aten._efficient_attention_backward,
    )
    # Without dropout the kernel does not read the seed and offset.
    philox = torch.zeros((), dtype=torch.long)
    grad_query, grad_key, grad_value = efficient_attention_backward(
        grad_output.unsqueeze(0),
        query.unsqueeze(0),
        key.unsqueeze(0),
        value.unsqueeze(0),
        None,
        output.unsqueeze(0),
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        logsumexp,
        0.0,
        philox,
        philox,
        0,
        False,
    )[:3]
    return grad_query.squeeze(0), grad_key.squeeze(0), grad_value.squeeze(0)


@_varlen_attention_backward_op.register_fake
def _(
    grad_output: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    output: Tensor,
    logsumexp: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> tuple[Tensor, Tensor, Tensor]:
    return (
        torch.empty_like(query),
        torch.empty_like(key),
        torch.empty_like(value),
    )


def _setup_varlen_attention_context(ctx, inputs, output) -> None:
    query, key, value, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, _ = (
        inputs
    )
    ctx.save_for_backward(
        query, key, value, output[0], output[1], cu_seqlens_q, cu_seqlens_k
    )
    ctx.max_seqlen_q = max_seqlen_q
    ctx.max_seqlen_k = max_seqlen_k


def _varlen_attention_backward(ctx, grad_output, grad_logsumexp):
    query, key, value, output, logsumexp, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
    grad_query, grad_key, grad_value = _varlen_attention_backward_op(
        grad_output,
        query,
        key,
        value,
        output,
        logsumexp,
        cu_seqlens_q,
        cu_seqlens_k,
        ctx.max_seqlen_q,
        ctx.max_seqlen_k,
    )
    return grad_query, grad_key, grad_value, None, None, None, None, None


torch.library.register_autograd(
    "gigl::varlen_attention",
    _varlen_attention_backward,
    setup_context=_setup_varlen_attention_context,
)


def _pad_head_dim(tensor: Tensor, multiple: int = 8) -> Tensor:
    """Zero-pads the last dim of ``tensor`` up to a multiple of ``multiple``."""
    remainder = tensor.size(-1) % multiple
    if remainder == 0:
        return tensor
    return F.pad(tensor, (0, multiple - remainder))


def _rescale_query(query: Tensor, padded_dim: int) -> Tensor:
    """Scales q so the kernel's scores use the unpadded head dim.

    The factor is ``sqrt(padded_dim / head_dim)``. The kernel scales scores by
    ``1 / sqrt(padded_dim)``. An explicit ``scale`` is not passed because torch
    2.8's meta backward rejects it, and compiled training traces that meta
    backward.
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
    backward registered on ``gigl::varlen_attention``.

    CUDA only; see the module docstring for the dropout limit. The head dims of
    ``query`` and ``key`` must be a multiple of 8; ``value`` may differ. The
    kernel scales scores by ``1 / sqrt(head_dim)`` of the head dim it is given.
    It picks its tile size from the head dim, so a padded head dim can move
    the call to the next size tier (64 padded to 72 lands in the tier for dims up
    to 128). The backward is non-deterministic unless
    ``torch.use_deterministic_algorithms(True)``, and it needs
    ``max_seqlen_q <= num_query_tokens`` and ``max_seqlen_k <= num_key_tokens``.

    This calls the private ``aten._efficient_attention_forward`` op, wrapped in
    ``gigl::varlen_attention``, because torch
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
    return _varlen_attention_op(
        query,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        needs_grad,
    )[0]


class PackedAttentionBias(NamedTuple):
    """Additive attention-score bias for ``packed_attention``.

    Both fields are optional and added to the scaled scores before the softmax.

    Attributes:
        per_key: One bias per key token and head. Every query in a sequence sees
            the same value for a given key.
        per_pair: One bias per sequence, head, query position and key position,
            as one flat tensor in packed sequence order. It is the concatenation,
            over ``layout_buckets(layout)`` in order, of the row-major
            ``[bucket_sequences, num_heads, bucket_len, bucket_len]`` block of
            each bucket, where ``bucket_sequences = end_sequence -
            first_sequence``. Without buckets it is one
            ``[num_sequences, num_heads, max_seq_len, max_seq_len]`` block. The
            cell of packed sequence ``s``, head ``h``, query position ``q`` and
            key position ``k`` is at
            ``sequence_offset[s] + (h * L + q) * L + k``, where ``L`` is the
            bucket length of ``s`` and ``sequence_offset`` comes from
            ``per_pair_sequence_offsets``; ``per_pair_linear_index`` computes it.
            Cells beyond a sequence's length are ignored: padded keys are masked
            and padded query rows are dropped.
    """

    per_key: Optional[Float[Tensor, "num_tokens num_heads"]] = None
    per_pair: Optional[Float[Tensor, "num_pair_cells"]] = None


def layout_buckets(layout: PackedLayout) -> tuple[PackedBucket, ...]:
    """Returns the buckets of ``layout``, or the one bucket that covers everything.

    Reads shapes only, never the device.

    Args:
        layout: Layout from ``pack_sequences``.

    Returns:
        ``layout.buckets``, or one bucket of ``max_seq_len`` over all sequences and
        tokens when it is None.
    """
    if layout.buckets is not None:
        return layout.buckets
    return (
        PackedBucket(
            0,
            layout.cu_seqlens.numel() - 1,
            0,
            layout.flat_index.numel(),
            layout.max_seq_len,
        ),
    )


def per_pair_bias_num_cells(layout: PackedLayout, num_heads: int) -> int:
    """Returns the length of the flat ``PackedAttentionBias.per_pair`` for ``layout``.

    Args:
        layout: Layout from ``pack_sequences``.
        num_heads: Number of attention heads.

    Returns:
        The sum over buckets of ``bucket_sequences * num_heads * bucket_len ** 2``.
    """
    return sum(
        (bucket.end_sequence - bucket.first_sequence) * num_heads * bucket.bucket_len**2
        for bucket in layout_buckets(layout)
    )


def per_pair_bias_views(
    per_pair: Float[Tensor, "num_pair_cells"], layout: PackedLayout, num_heads: int
) -> list[Tensor]:
    """Splits a flat per-pair bias into one 4-D view per bucket.

    The views share memory with ``per_pair``; nothing is copied.

    Args:
        per_pair: Flat bias, laid out as described in ``PackedAttentionBias``.
        layout: Layout from ``pack_sequences``.
        num_heads: Number of attention heads.

    Returns:
        For each bucket, its ``[bucket_sequences, num_heads, bucket_len,
        bucket_len]`` block, in bucket order.

    Raises:
        ValueError: If ``per_pair`` does not have ``per_pair_bias_num_cells``
            elements.
    """
    expected_cells = per_pair_bias_num_cells(layout, num_heads)
    if per_pair.numel() != expected_cells:
        raise ValueError(
            f"per_pair has {per_pair.numel()} cells but the layout needs "
            f"{expected_cells} for {num_heads} heads."
        )
    views: list[Tensor] = []
    first_cell = 0
    for bucket in layout_buckets(layout):
        shape = (
            bucket.end_sequence - bucket.first_sequence,
            num_heads,
            bucket.bucket_len,
            bucket.bucket_len,
        )
        num_cells = shape[0] * shape[1] * shape[2] * shape[3]
        views.append(per_pair[first_cell : first_cell + num_cells].view(shape))
        first_cell += num_cells
    return views


def per_pair_sequence_offsets(
    layout: PackedLayout, num_heads: int
) -> tuple[Int64[Tensor, "num_sequences"], Int64[Tensor, "num_sequences"]]:
    """Returns where each packed sequence's block starts in the flat per-pair bias.

    Reads shapes only, never the device.

    Args:
        layout: Layout from ``pack_sequences``.
        num_heads: Number of attention heads.

    Returns:
        ``(sequence_offset, sequence_bucket_len)``, both int64 on the layout's
        device. ``sequence_bucket_len[s]`` is the bucket length of packed
        sequence ``s``. ``sequence_offset[s]`` is the first cell of its
        ``[num_heads, bucket_len, bucket_len]`` block.
    """
    device = layout.cu_seqlens.device
    sequence_bucket_len = torch.cat(
        [
            torch.full(
                (bucket.end_sequence - bucket.first_sequence,),
                bucket.bucket_len,
                dtype=torch.long,
                device=device,
            )
            for bucket in layout_buckets(layout)
        ]
    )
    block_cells = num_heads * sequence_bucket_len**2
    sequence_offset = torch.cumsum(block_cells, 0) - block_cells
    return sequence_offset, sequence_bucket_len


def per_pair_linear_index(
    sequence_offset: Int64[Tensor, "num_sequences"],
    sequence_bucket_len: Int64[Tensor, "num_sequences"],
    sequence: Tensor,
    head: Tensor | int,
    query_position: Tensor | int,
    key_position: Tensor | int,
) -> Tensor:
    """Returns the cells of a flat per-pair bias for the given pairs.

    Computes ``sequence_offset[sequence] + (head * L + query_position) * L +
    key_position`` with ``L = sequence_bucket_len[sequence]``. The arguments
    broadcast against each other.

    Args:
        sequence_offset: First cell of each packed sequence's block, from
            ``per_pair_sequence_offsets``.
        sequence_bucket_len: Bucket length of each packed sequence, from
            ``per_pair_sequence_offsets``.
        sequence: Packed sequence of each pair.
        head: Head of each pair.
        query_position: Position of the query within its sequence.
        key_position: Position of the key within its sequence.

    Returns:
        Long indices into the flat per-pair bias.
    """
    bucket_len = sequence_bucket_len[sequence]
    return (
        sequence_offset[sequence]
        + (head * bucket_len + query_position) * bucket_len
        + key_position
    )


def _first_row_per_key_bias(
    per_pair: Float[Tensor, "num_pair_cells"], layout: PackedLayout, num_heads: int
) -> Float[Tensor, "num_tokens num_heads"]:
    """Gathers query row 0 of every sequence's per-pair block as a per-key bias.

    A first-token query only reads row 0 of its sequence's block, and every key
    sees it as one value per head, which is a per-key bias.

    Args:
        per_pair: Flat per-pair bias, laid out as described in
            ``PackedAttentionBias``.
        layout: Layout from ``pack_sequences``.
        num_heads: Number of attention heads.

    Returns:
        For each token ``t``, ``per_pair`` at (its sequence, every head, query
        position 0, its position).
    """
    sequence_offset, sequence_bucket_len = per_pair_sequence_offsets(layout, num_heads)
    cells = per_pair_linear_index(
        sequence_offset,
        sequence_bucket_len,
        layout.sequence_id.unsqueeze(1),
        torch.arange(num_heads, device=per_pair.device).unsqueeze(0),
        0,
        layout.position.unsqueeze(1),
    )
    return per_pair[cells]


def _bucket_cell_index(layout: PackedLayout, bucket: PackedBucket) -> Tensor:
    """Returns each bucket token's cell in the bucket's flattened ``[sequences * bucket_len]`` grid."""
    token_range = slice(bucket.first_token, bucket.end_token)
    return (
        layout.sequence_id[token_range] - bucket.first_sequence
    ) * bucket.bucket_len + layout.position[token_range]


def _dense_attention_bias(
    layout: PackedLayout,
    bucket: PackedBucket,
    bias: Optional[PackedAttentionBias],
    per_pair_block: Optional[Tensor],
    dtype: torch.dtype,
) -> Tensor:
    """Builds the additive score bias on one bucket's padded grid.

    Starts from the padding mask, ``finfo.min / 2`` on invalid keys, shaped
    ``[bucket_sequences, 1, 1, bucket_len]``. Adds the bucket's tokens of
    ``bias.per_key`` scattered onto the grid as ``[bucket_sequences, num_heads,
    1, bucket_len]`` and ``per_pair_block`` (the bucket's view of
    ``bias.per_pair``), each only when present. The result broadcasts against
    the scores; a dense grid is only built when ``per_pair_block`` is present.
    """
    num_sequences = bucket.end_sequence - bucket.first_sequence
    lengths = (
        layout.cu_seqlens[bucket.first_sequence + 1 : bucket.end_sequence + 1]
        - layout.cu_seqlens[bucket.first_sequence : bucket.end_sequence]
    )
    positions = torch.arange(bucket.bucket_len, device=lengths.device)
    valid = positions < lengths.unsqueeze(1)
    attn_bias = torch.zeros(valid.shape, dtype=dtype, device=valid.device)
    attn_bias = attn_bias.masked_fill(~valid, torch.finfo(dtype).min / 2.0)
    attn_bias = attn_bias.view(num_sequences, 1, 1, bucket.bucket_len)
    if bias is not None and bias.per_key is not None:
        per_key = bias.per_key[bucket.first_token : bucket.end_token]
        num_heads = per_key.size(1)
        per_key_grid = torch.zeros(
            num_sequences * bucket.bucket_len,
            num_heads,
            dtype=dtype,
            device=valid.device,
        ).index_copy(0, _bucket_cell_index(layout, bucket), per_key.to(dtype))
        attn_bias = attn_bias + per_key_grid.view(
            num_sequences, bucket.bucket_len, num_heads
        ).permute(0, 2, 1).unsqueeze(2)
    if per_pair_block is not None:
        attn_bias = attn_bias + per_pair_block.to(dtype)
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
    """Runs packed attention on any device with SDPA per length bucket.

    This is the CPU path and the reference the CUDA path is tested against. It
    also runs on CUDA for a per-pair bias on every token and for dropout.

    For each bucket of ``layout_buckets(layout)``, unpacks the bucket's tokens to
    a ``[bucket_sequences, bucket_len]`` grid, masks padded keys with the same
    additive ``finfo.min / 2`` bias as ``GraphTransformerEncoder``, adds ``bias``
    when given, runs ``F.scaled_dot_product_attention`` and unpacks the outputs.
    The grids are per bucket so the padding work of a sequence is bounded by its
    bucket length, not by ``max_seq_len`` for every sequence. The bucket outputs
    are concatenated in bucket order, which is packed order. Without buckets
    there is one grid of ``max_seq_len``.

    Applies attention dropout with probability ``dropout_p`` to the weights.
    ``_varlen_attention`` is checked against this. The function reads sizes from
    the host bucket tuples and tensor shapes only, never from the device.

    Args:
        query: Packed queries, or one query per sequence when
            ``query_is_first_token``.
        key: Packed keys.
        value: Packed values.
        layout: Layout from ``pack_sequences``.
        query_is_first_token: ``query`` holds one query per sequence, the
            sequence's first token. Each such query is one row of its grid, and
            reads row 0 of its ``per_pair`` block.
        bias: Optional additive score bias.
        dropout_p: Probability of dropping an attention weight.

    Returns:
        Attention output shaped like ``query``.
    """
    num_heads = query.size(1)
    per_pair_blocks: list[Optional[Tensor]] = [None] * len(layout_buckets(layout))
    if bias is not None and bias.per_pair is not None:
        per_pair_blocks = list(per_pair_bias_views(bias.per_pair, layout, num_heads))
    outputs: list[Tensor] = []
    for bucket, per_pair_block in zip(layout_buckets(layout), per_pair_blocks):
        num_sequences = bucket.end_sequence - bucket.first_sequence
        grid_cells = num_sequences * bucket.bucket_len
        token_range = slice(bucket.first_token, bucket.end_token)
        cell_index = _bucket_cell_index(layout, bucket)

        def to_grid(tokens: Tensor, token_cells: Tensor) -> Tensor:
            grid = tokens.new_zeros(grid_cells, *tokens.shape[1:])
            grid = grid.index_copy(0, token_cells, tokens)
            return grid.view(num_sequences, bucket.bucket_len, *tokens.shape[1:])

        key_grid = to_grid(key[token_range], cell_index).transpose(1, 2)
        value_grid = to_grid(value[token_range], cell_index).transpose(1, 2)
        if query_is_first_token:
            query_grid = query[bucket.first_sequence : bucket.end_sequence].unsqueeze(2)
            if per_pair_block is not None:
                per_pair_block = per_pair_block[:, :, :1, :]
        else:
            query_grid = to_grid(query[token_range], cell_index).transpose(1, 2)
        output = F.scaled_dot_product_attention(
            query_grid,
            key_grid,
            value_grid,
            attn_mask=_dense_attention_bias(
                layout, bucket, bias, per_pair_block, query.dtype
            ),
            dropout_p=dropout_p,
        )
        if query_is_first_token:
            outputs.append(output.squeeze(2))
        else:
            output = output.transpose(1, 2).flatten(0, 1)
            outputs.append(output.index_select(0, cell_index))
    return torch.cat(outputs)


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

    Uses the CUDA varlen kernel on GPU and SDPA per length bucket elsewhere. With
    ``query_is_first_token``, only each sequence's first token is queried, which
    is all the anchor readout of the last layer needs. An empty
    ``PackedAttentionBias`` behaves like ``bias=None``.

    The route is chosen in this order:

    1. Not CUDA, or ``dropout_p > 0``: SDPA per bucket
       (``_padded_packed_attention``). Dropout never uses the varlen kernel,
       whose dropout is wrong in the backward pass in torch 2.8.
    2. A ``per_pair`` bias with ``query_is_first_token``: row 0 of each
       sequence's block becomes a per-key bias (``_first_row_per_key_bias``)
       and the call continues on the varlen kernel, with or without gradient.
    3. A ``per_pair`` bias on every token, with or without gradient: SDPA per
       bucket.
    4. Otherwise the varlen kernel, with a ``per_key`` bias folded into q and k
       by ``_fold_per_key_bias``.

    Every sequence must contain at least one token (``lengths >= 1``). An empty
    sequence gives wrong output on the padded path and is not supported.

    Args and returns are as in ``_padded_packed_attention``.
    """
    if not query.is_cuda or dropout_p > 0:
        return _padded_packed_attention(
            query, key, value, layout, query_is_first_token, bias, dropout_p
        )
    if bias is not None and bias.per_pair is not None:
        # The one place to route a per-pair bias, with or without gradient.
        if not query_is_first_token:
            return _padded_packed_attention(
                query, key, value, layout, query_is_first_token, bias
            )
        per_key = _first_row_per_key_bias(bias.per_pair, layout, query.size(1))
        if bias.per_key is not None:
            per_key = bias.per_key + per_key
        bias = PackedAttentionBias(per_key=per_key)
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
