"""Token packing and attention over packed variable-length sequences.

A padded ``[num_sequences, max_seq_len, dim]`` batch spends most of its linear
FLOPs on padding when lengths vary. Packing puts every valid token of every
sequence into one ``[num_tokens, dim]`` buffer, so linears and FFNs run on real
tokens only and attention runs per sequence.

Packing reads the token count back to the host once, so the buffer is exactly
as large as the real tokens and no slot is wasted.
"""

from typing import Any, Callable, NamedTuple, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import to_dense_batch


class PackedSequences(NamedTuple):
    """Where each sequence lives in the packed token buffer.

    Attributes:
        cu_seqlens: ``[num_sequences + 1]`` int32 cumulative sequence lengths on
            the tokens' device; sequence ``i`` occupies
            ``[cu_seqlens[i], cu_seqlens[i + 1])``.
        max_seq_len: Host upper bound on any sequence length.
    """

    cu_seqlens: Tensor
    max_seq_len: int


def pack_sequences(
    node_index: Tensor, hop_index: Tensor, lengths: Tensor
) -> tuple[Tensor, Tensor, PackedSequences]:
    """Packs the first ``lengths[i]`` tokens of every sequence back to back.

    Costs one host read, of the total token count.

    Args:
        node_index: ``[num_sequences, max_seq_len]`` long node id per position.
        hop_index: ``[num_sequences, max_seq_len]`` long hop per position.
        lengths: ``[num_sequences]`` long valid length of each sequence.

    Returns:
        - ``[num_tokens]`` long node id per packed token.
        - ``[num_tokens]`` long hop per packed token.
        - The packed layout.
    """
    valid = torch.arange(node_index.size(1), device=lengths.device) < lengths.unsqueeze(
        1
    )
    cu_seqlens = torch.cat([lengths.new_zeros(1), torch.cumsum(lengths, 0)])
    # One ``nonzero`` (one host sync) serves both gathers.
    flat = valid.flatten().nonzero().squeeze(1)
    return (
        node_index.flatten()[flat],
        hop_index.flatten()[flat],
        PackedSequences(cu_seqlens.to(torch.int32), node_index.size(1)),
    )


class _VarlenAttention(torch.autograd.Function):
    """Memory-efficient varlen attention with the explicit backward kernel."""

    @staticmethod
    def forward(
        ctx: Any,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        cu_seqlens_q: Tensor,
        cu_seqlens_k: Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> Tensor:
        # The op's stub types every argument as ``...``; it returns
        # (output, logsumexp, philox_seed, philox_offset, max_q, max_k).
        efficient_attention_forward = cast(
            Callable[..., tuple[Tensor, ...]],
            torch.ops.aten._efficient_attention_forward,
        )
        needs_grad = any(ctx.needs_input_grad[:3])
        output, logsumexp, philox_seed, philox_offset = efficient_attention_forward(
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
        )[:4]
        output = output.squeeze(0)
        if needs_grad:
            ctx.save_for_backward(
                query,
                key,
                value,
                output,
                logsumexp,
                cu_seqlens_q,
                cu_seqlens_k,
                philox_seed,
                philox_offset,
            )
            ctx.max_seqlens = (max_seqlen_q, max_seqlen_k)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        (grad_output,) = grad_outputs
        (
            query,
            key,
            value,
            output,
            logsumexp,
            cu_seqlens_q,
            cu_seqlens_k,
            philox_seed,
            philox_offset,
        ) = ctx.saved_tensors
        max_seqlen_q, max_seqlen_k = ctx.max_seqlens
        efficient_attention_backward = cast(
            Callable[..., tuple[Tensor, ...]],
            torch.ops.aten._efficient_attention_backward,
        )
        grad_query, grad_key, grad_value, _ = efficient_attention_backward(
            grad_output.unsqueeze(0).contiguous(),
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
            philox_seed,
            philox_offset,
            0,
            False,
        )
        return (
            grad_query.squeeze(0),
            grad_key.squeeze(0),
            grad_value.squeeze(0),
            None,
            None,
            None,
            None,
        )


def varlen_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> Tensor:
    """Attention over packed sequences with the CUDA memory-efficient varlen kernel.

    Query sequence ``i`` attends to key sequence ``i``. The kernel reads the
    sequence offsets on the device and takes the length bounds from the host,
    so it never syncs. Differentiable in ``query``, ``key`` and ``value``.

    CUDA only. ``head_dim`` must suit the memory-efficient kernel (64 works in
    fp32, fp16 and bf16).

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
    return cast(
        Tensor,
        _VarlenAttention.apply(
            query, key, value, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k
        ),
    )


def padded_packed_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    packed: PackedSequences,
    query_is_first_token: bool,
) -> Tensor:
    """Attention over packed sequences through SDPA on a padded grid.

    Unpacks the tokens to ``[num_sequences, max_seq_len]``, masks padded keys
    with the same additive ``finfo.min / 2`` bias as ``GraphTransformerEncoder``,
    and unpacks the outputs. It runs on any device; it is the CPU path and
    the reference for ``varlen_attention``.

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
    query: Tensor,
    key: Tensor,
    value: Tensor,
    packed: PackedSequences,
    query_is_first_token: bool,
) -> Tensor:
    """Attention over packed sequences: ``varlen_attention`` on CUDA, else ``padded_packed_attention``.

    Args and returns are as in ``padded_packed_attention``.
    """
    if not query.is_cuda:
        return padded_packed_attention(query, key, value, packed, query_is_first_token)
    if query_is_first_token:
        cu_seqlens_q = torch.arange(
            packed.cu_seqlens.size(0), dtype=torch.int32, device=query.device
        )
        max_seqlen_q = 1
    else:
        cu_seqlens_q = packed.cu_seqlens
        max_seqlen_q = packed.max_seq_len
    return varlen_attention(
        query,
        key,
        value,
        cu_seqlens_q,
        packed.cu_seqlens,
        max_seqlen_q,
        packed.max_seq_len,
    )
