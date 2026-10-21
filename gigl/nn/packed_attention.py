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
from torch.autograd.function import once_differentiable
from torch_geometric.utils import to_dense_batch

from gigl.transforms.static_khop import PackedSequences

# The memory-efficient attention kernels sit behind custom ops, so Inductor treats
# them as opaque instead of applying its layout rules for aten attention ops, which
# need concrete strides and fail on the ``[1, tokens, ...]`` views of a token count
# that is only known at run time.
_LSE_ALIGNMENT = 32


def _varlen_attention_forward_kernel(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    compute_logsumexp: bool,
) -> tuple[Tensor, Tensor]:
    # The op's stub types every argument as ``...``; it returns
    # (output, logsumexp, philox_seed, philox_offset, max_q, max_k).
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
        compute_logsumexp,
    )[:2]
    return output.squeeze(0).contiguous(), logsumexp


def _varlen_attention_forward_fake(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    compute_logsumexp: bool,
) -> tuple[Tensor, Tensor]:
    padded_q = -(-max_seqlen_q // _LSE_ALIGNMENT) * _LSE_ALIGNMENT
    return (
        torch.empty_like(query),
        query.new_empty(
            (
                cu_seqlens_q.size(0) - 1,
                query.size(1),
                padded_q if compute_logsumexp else 0,
            ),
            dtype=torch.float32,
        ),
    )


def _varlen_attention_backward_kernel(
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
    # Dropout is off, so the kernel never reads the Philox state.
    philox_seed = torch.zeros((), dtype=torch.long)
    philox_offset = torch.zeros((), dtype=torch.long)
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
        grad_query.squeeze(0).contiguous(),
        grad_key.squeeze(0).contiguous(),
        grad_value.squeeze(0).contiguous(),
    )


def _varlen_attention_backward_fake(
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


def _register_custom_op(
    name: str, kernel: Callable[..., Any], fake: Callable[..., Any]
) -> Any:
    """Registers ``gigl::<name>``, or returns the op if it is already registered.

    A reload of this module in the same process must not register the op twice.
    """
    if hasattr(torch.ops.gigl, name):
        return getattr(torch.ops.gigl, name)
    op = torch.library.custom_op(f"gigl::{name}", mutates_args=())(kernel)
    op.register_fake(fake)
    return op


_varlen_attention_forward = _register_custom_op(
    "varlen_attention_forward",
    _varlen_attention_forward_kernel,
    _varlen_attention_forward_fake,
)
_varlen_attention_backward = _register_custom_op(
    "varlen_attention_backward",
    _varlen_attention_backward_kernel,
    _varlen_attention_backward_fake,
)


class VarlenAttentionOutput(NamedTuple):
    """Result of the varlen attention forward op.

    Attributes:
        output: Attention output, shaped like the query.
        logsumexp: Per-row softmax normalizer the backward kernel needs; its last
            dim is zero when the forward ran without ``compute_logsumexp``.
    """

    output: Tensor
    logsumexp: Tensor


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
        needs_grad = any(ctx.needs_input_grad[:3])
        attention = VarlenAttentionOutput(
            *_varlen_attention_forward(
                query,
                key,
                value,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                needs_grad,
            )
        )
        if needs_grad:
            ctx.save_for_backward(
                query,
                key,
                value,
                attention.output,
                attention.logsumexp,
                cu_seqlens_q,
                cu_seqlens_k,
            )
            ctx.max_seqlens = (max_seqlen_q, max_seqlen_k)
        return attention.output

    @staticmethod
    @once_differentiable
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
        ) = ctx.saved_tensors
        max_seqlen_q, max_seqlen_k = ctx.max_seqlens
        grad_query, grad_key, grad_value = _varlen_attention_backward(
            grad_output,
            query,
            key,
            value,
            output,
            logsumexp,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
        )
        return grad_query, grad_key, grad_value, None, None, None, None


def _varlen_attention(
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
    fp32, fp16 and bf16). The backward is non-deterministic unless
    ``torch.use_deterministic_algorithms(True)``, and it needs
    ``max_seqlen_q <= num_query_tokens`` and ``max_seqlen_k <= num_key_tokens``.
    Attention dropout is not supported: in torch 2.8 the varlen kernel draws one
    dropout mask shared across heads and sequences.

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
    if not torch.is_grad_enabled():
        # ``needs_input_grad`` ignores grad mode; detached inputs skip the
        # logsumexp and the saved tensors.
        query, key, value = query.detach(), key.detach(), value.detach()
    return cast(
        Tensor,
        _VarlenAttention.apply(
            query, key, value, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k
        ),
    )


def _padded_packed_attention(
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
    the reference for ``_varlen_attention``.

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
