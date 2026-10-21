"""Token packing and attention over packed variable-length sequences.

When sequence lengths vary, most of the linear-layer FLOPs on a padded
``[num_sequences, max_seq_len, dim]`` batch go to padding. Packing copies the
real tokens of all sequences into one ``[num_tokens, dim]`` buffer. Linear layers
and FFNs then run on real tokens only, and attention runs within each sequence,
with sequence boundaries given by ``cu_seqlens``.

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
        |        CUDA: aten memory-efficient attention with cu_seqlens
        |        CPU:  SDPA on a padded grid (reference)
        |    output linear + residual -> FFN + residual
        v
    anchor readout: rows cu_seqlens[:-1], the first token of every sequence

``packed_attention`` is the entry point. On CUDA it calls aten's memory-efficient
attention kernel with ``cu_seqlens``, and aten's own autograd formula supplies the
backward.

Constraints:
  - The varlen kernel is CUDA only; other devices use the padded SDPA path.
  - ``head_dim`` must be a multiple of 8 for the aligned kernels.
  - Attention dropout is not supported; the kernel's varlen dropout has not been
    tested against the padded path.
  - ``cu_seqlens`` is int32, so the token count must fit in int32.
  - Packing copies the total token count to the host once (a GPU sync), so the
    buffer holds exactly the real tokens.

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
"""

from typing import Callable, cast

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int32
from torch import Tensor
from torch_geometric.utils import to_dense_batch

from gigl.transforms.static_khop import PackedSequences


def _varlen_attention(
    query: Float[Tensor, "num_query_tokens num_heads head_dim"],
    key: Float[Tensor, "num_key_tokens num_heads head_dim"],
    value: Float[Tensor, "num_key_tokens num_heads head_dim"],
    cu_seqlens_q: Int32[Tensor, "num_sequences_plus_one"],
    cu_seqlens_k: Int32[Tensor, "num_sequences_plus_one"],
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> Float[Tensor, "num_query_tokens num_heads head_dim"]:
    """Attention over packed sequences with the CUDA memory-efficient varlen kernel.

    Query sequence ``i`` attends to key sequence ``i``. The kernel reads the
    sequence offsets on the device and takes the length bounds as host ints, so
    the call does not sync with the host. Differentiable in ``query``, ``key``
    and ``value`` through the autograd formula that aten registers for the
    kernel.

    CUDA only. ``head_dim`` must suit the memory-efficient kernel (64 works in
    fp32, fp16 and bf16). The backward is non-deterministic unless
    ``torch.use_deterministic_algorithms(True)``, and it needs
    ``max_seqlen_q <= num_query_tokens`` and ``max_seqlen_k <= num_key_tokens``.
    Attention dropout is not supported; the kernel's varlen dropout has not been
    tested against the padded path.

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

    TODO: when GiGL moves to torch >= 2.10, try the public
    ``torch.nn.attention.varlen.varlen_attn``. It takes the same packed inputs
    but wraps Flash Attention, so check fp32 support, backward and
    ``torch.compile`` before switching. If it lacks fp32, use it for bf16 only
    and keep this call for fp32.

    Args:
        query: ``[num_query_tokens, num_heads, head_dim]``.
        key: ``[num_key_tokens, num_heads, head_dim]``.
        value: ``[num_key_tokens, num_heads, head_dim]``.
        cu_seqlens_q: ``[num_sequences + 1]`` int32 query offsets.
        cu_seqlens_k: ``[num_sequences + 1]`` int32 key offsets.
        max_seqlen_q: Upper bound on any query sequence length.
        max_seqlen_k: Upper bound on any key sequence length.

    Returns:
        ``[num_query_tokens, num_heads, head_dim]`` attention output.
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


def _padded_packed_attention(
    query: Float[Tensor, "num_queries num_heads head_dim"],
    key: Float[Tensor, "num_tokens num_heads head_dim"],
    value: Float[Tensor, "num_tokens num_heads head_dim"],
    packed: PackedSequences,
    query_is_first_token: bool,
) -> Float[Tensor, "num_queries num_heads head_dim"]:
    """Attention over packed sequences through SDPA on a padded grid.

    Unpacks the tokens to ``[num_sequences, max_seq_len]``, masks padded keys
    with the same additive ``finfo.min / 2`` bias as ``GraphTransformerEncoder``,
    and unpacks the outputs. It runs on any device. It is the CPU path and the
    reference that ``_varlen_attention`` is checked against.

    Args:
        query: ``[num_tokens, num_heads, head_dim]`` packed queries, or
            ``[num_sequences, num_heads, head_dim]`` when ``query_is_first_token``.
        key: ``[num_tokens, num_heads, head_dim]`` packed keys.
        value: ``[num_tokens, num_heads, head_dim]`` packed values.
        packed: Layout from ``pack_sequences``.
        query_is_first_token: ``query`` holds one query per sequence, the
            sequence's first token.

    Returns:
        Attention output shaped like ``query``.
    """
    lengths = packed.cu_seqlens[1:] - packed.cu_seqlens[:-1]
    num_sequences = lengths.numel()
    token_sequence = torch.repeat_interleave(
        torch.arange(num_sequences, device=lengths.device),
        lengths,
        output_size=key.size(0),
    )

    key_grid, mask = to_dense_batch(
        key, token_sequence, batch_size=num_sequences, max_num_nodes=packed.max_seq_len
    )
    value_grid, _ = to_dense_batch(
        value,
        token_sequence,
        batch_size=num_sequences,
        max_num_nodes=packed.max_seq_len,
    )
    key_grid, value_grid = key_grid.transpose(1, 2), value_grid.transpose(1, 2)
    attn_bias = torch.zeros(mask.shape, dtype=query.dtype, device=query.device)
    attn_bias = attn_bias.masked_fill(~mask, torch.finfo(query.dtype).min / 2.0)
    attn_bias = attn_bias.view(num_sequences, 1, 1, -1)
    if query_is_first_token:
        return F.scaled_dot_product_attention(
            query.unsqueeze(2), key_grid, value_grid, attn_mask=attn_bias
        ).squeeze(2)
    query_grid, _ = to_dense_batch(
        query,
        token_sequence,
        batch_size=num_sequences,
        max_num_nodes=packed.max_seq_len,
    )
    query_grid = query_grid.transpose(1, 2)
    output = F.scaled_dot_product_attention(
        query_grid, key_grid, value_grid, attn_mask=attn_bias
    )
    return output.transpose(1, 2)[mask]


def packed_attention(
    query: Float[Tensor, "num_queries num_heads head_dim"],
    key: Float[Tensor, "num_tokens num_heads head_dim"],
    value: Float[Tensor, "num_tokens num_heads head_dim"],
    packed: PackedSequences,
    query_is_first_token: bool,
) -> Float[Tensor, "num_queries num_heads head_dim"]:
    """Attention over packed sequences: the CUDA varlen kernel, or SDPA on CPU.

    Args and returns are as in ``_padded_packed_attention``.
    """
    if not query.is_cuda:
        return _padded_packed_attention(query, key, value, packed, query_is_first_token)
    if query_is_first_token:
        cu_seqlens_q = torch.arange(
            packed.cu_seqlens.size(0), dtype=torch.int32, device=query.device
        )
        max_seqlen_q = 1
    else:
        cu_seqlens_q = packed.cu_seqlens
        max_seqlen_q = min(packed.max_seq_len, query.size(0))
    return _varlen_attention(
        query,
        key,
        value,
        cu_seqlens_q,
        packed.cu_seqlens,
        max_seqlen_q,
        min(packed.max_seq_len, key.size(0)),
    )
