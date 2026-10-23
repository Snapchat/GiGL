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
  - On CUDA the packed buffers are wrapped as jagged nested tensors and go
    through ``F.scaled_dot_product_attention``, which runs the memory-efficient
    varlen kernel. That kernel runs in fp32, which inference needs for accuracy.
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
        |        CUDA: SDPA on jagged nested tensors
        |        CPU:  SDPA on a padded grid (reference)
        |    output linear + residual -> FFN + residual
        v
    anchor readout: rows cu_seqlens[:-1], the first token of every sequence

Constraints:
  - The nested-tensor path is CUDA only; other devices use the padded SDPA path.
  - ``head_dim`` must be a multiple of 8 for the aligned kernels.
  - Attention dropout is not supported; it has not been tested against the padded
    path.
  - ``cu_seqlens`` is int32, so the token count must fit in int32.

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

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from torch_geometric.utils import to_dense_batch

from gigl.transforms.static_khop import PackedSequences


def _njt_forward(
    query: Float[Tensor, "num_queries num_heads head_dim"],
    key: Float[Tensor, "num_tokens num_heads head_dim"],
    value: Float[Tensor, "num_tokens num_heads head_dim"],
    packed: PackedSequences,
    query_is_first_token: bool,
) -> Float[Tensor, "num_queries num_heads head_dim"]:
    """Wraps the flat buffers as jagged nested tensors and runs SDPA on them.

    Each buffer becomes a jagged nested tensor with the sequence offsets, and
    query sequence ``i`` attends to key sequence ``i``. Passing the length bounds
    as host ints keeps the call from syncing with the host.
    ``F.scaled_dot_product_attention`` dispatches jagged inputs to the
    memory-efficient kernel on CUDA.
    """
    max_seqlen_k = min(packed.max_seq_len, key.size(0))
    if query_is_first_token:
        query_offsets = torch.arange(
            packed.cu_seqlens.size(0), dtype=torch.int32, device=query.device
        )
        max_seqlen_q = 1
    else:
        query_offsets = packed.cu_seqlens
        max_seqlen_q = min(packed.max_seq_len, query.size(0))

    def jagged(tokens: Tensor, offsets: Tensor, max_seqlen: int) -> Tensor:
        return torch.nested.nested_tensor_from_jagged(
            tokens, offsets, min_seqlen=1, max_seqlen=max_seqlen
        ).transpose(1, 2)

    output = F.scaled_dot_product_attention(
        jagged(query, query_offsets, max_seqlen_q),
        jagged(key, packed.cu_seqlens, max_seqlen_k),
        jagged(value, packed.cu_seqlens, max_seqlen_k),
    )
    return output.transpose(1, 2).values()


class _NjtAttention(torch.autograd.Function):
    """Jagged nested attention that saves only dense tensors for the backward.

    A jagged nested tensor carries a fresh nested-int size on every construction,
    so saving one for backward makes a checkpointed layer's recompute disagree
    with its forward (``CheckpointError``). This function saves the dense inputs
    and rebuilds the nested tensors in the backward.
    """

    @staticmethod
    def forward(ctx, query, key, value, packed, query_is_first_token):
        ctx.save_for_backward(query, key, value)
        ctx.packed = packed
        ctx.query_is_first_token = query_is_first_token
        with torch.no_grad():
            return _njt_forward(query, key, value, packed, query_is_first_token)

    @staticmethod
    def backward(ctx, grad_output):
        query, key, value = (
            tensor.detach().requires_grad_() for tensor in ctx.saved_tensors
        )
        with torch.enable_grad():
            output = _njt_forward(
                query, key, value, ctx.packed, ctx.query_is_first_token
            )
        grads = torch.autograd.grad(output, (query, key, value), grad_output)
        return (*grads, None, None)


@torch.compiler.disable
def _njt_attention(
    query: Float[Tensor, "num_queries num_heads head_dim"],
    key: Float[Tensor, "num_tokens num_heads head_dim"],
    value: Float[Tensor, "num_tokens num_heads head_dim"],
    packed: PackedSequences,
    query_is_first_token: bool,
) -> Float[Tensor, "num_queries num_heads head_dim"]:
    """Runs attention on packed sequences without padding, through jagged nested tensors.

    Jagged nested tensors do not compile under dynamic shapes, so this runs
    eagerly: a compiled layer splits into a graph before and a graph after this
    call. It takes the layout rather than sizes so no symbolic size crosses the
    graph break.

    Args and returns are as in ``_padded_packed_attention``.
    """
    return _NjtAttention.apply(query, key, value, packed, query_is_first_token)


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
    and unpacks the outputs. ``_njt_attention`` is checked against this.

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

    Uses jagged nested tensors on GPU and padded SDPA elsewhere. With
    ``query_is_first_token``, only each sequence's first token is queried, which
    is all the anchor readout of the last layer needs.

    Args and returns are as in ``_padded_packed_attention``.
    """
    if not query.is_cuda:
        return _padded_packed_attention(query, key, value, packed, query_is_first_token)
    return _njt_attention(query, key, value, packed, query_is_first_token)
