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
        |          -> flat_index, PackedLayout(cu_seqlens, max_seq_len)
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
  - ``head_dim`` must be a multiple of 8 for the aligned kernels.
  - Attention dropout is not supported; the kernel's varlen dropout has not been
    tested against the padded path.
  - ``cu_seqlens`` is int32, so the token count must fit in int32.

Example:
    >>> from gigl.transforms.packed_layout import pack_sequences
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

from typing import Callable, cast

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int32
from torch import Tensor
from torch_geometric.utils import to_dense_batch

from gigl.transforms.packed_layout import PackedLayout


def _varlen_attention(
    query: Float[Tensor, "num_query_tokens num_heads head_dim"],
    key: Float[Tensor, "num_key_tokens num_heads head_dim"],
    value: Float[Tensor, "num_key_tokens num_heads head_dim"],
    cu_seqlens_q: Int32[Tensor, "num_sequences_plus_one"],
    cu_seqlens_k: Int32[Tensor, "num_sequences_plus_one"],
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> Float[Tensor, "num_query_tokens num_heads head_dim"]:
    """Runs attention on packed sequences without padding, on the CUDA varlen kernel.

    Query sequence ``i`` attends to key sequence ``i``, so padded positions cost
    nothing. The kernel reads the sequence offsets on the device and takes the
    length bounds as host ints, so the call does not sync with the host (it can
    be compiled). Differentiable in ``query``, ``key`` and ``value`` through the
    autograd formula that aten registers for the kernel.

    CUDA only; see the module docstring for the head dim and dropout limits. The
    backward is non-deterministic unless
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


def _padded_packed_attention(
    query: Float[Tensor, "num_queries num_heads head_dim"],
    key: Float[Tensor, "num_tokens num_heads head_dim"],
    value: Float[Tensor, "num_tokens num_heads head_dim"],
    packed: PackedLayout,
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
    packed: PackedLayout,
    query_is_first_token: bool,
) -> Float[Tensor, "num_queries num_heads head_dim"]:
    """Attention within each packed sequence; the entry point of this module.

    Uses the CUDA varlen kernel on GPU and padded SDPA elsewhere. With
    ``query_is_first_token``, only each sequence's first token is queried, which
    is all the anchor readout of the last layer needs.

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
