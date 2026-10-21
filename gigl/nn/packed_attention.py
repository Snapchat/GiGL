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
        |        CUDA: gigl::varlen_attention_{forward,backward} custom ops
        |        CPU:  SDPA on a padded grid (reference)
        |    output linear + residual -> FFN + residual
        v
    anchor readout: rows cu_seqlens[:-1], the first token of every sequence

``packed_attention`` is the entry point. On CUDA it calls the memory-efficient
varlen kernel through two custom ops, ``gigl::varlen_attention_forward`` and
``gigl::varlen_attention_backward``, which an autograd function connects. The
kernels are wrapped in custom ops so Inductor treats them as opaque. Inductor's
layout rules for the aten attention ops need concrete strides, and they fail on
the ``[1, tokens, ...]`` views here because the token count is only known at run
time.

Constraints:
  - The varlen kernel is CUDA only; other devices use the padded SDPA path.
  - ``head_dim`` must be a multiple of 8 for the aligned kernels.
  - Attention dropout is not supported: in torch 2.8 the varlen kernel draws one
    dropout mask shared across heads and sequences.
  - ``cu_seqlens`` is int32, so the token count must fit in int32.
  - Packing copies the total token count to the host once (a GPU sync), so the
    buffer holds exactly the real tokens.

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

from typing import Any, Callable, NamedTuple, cast

import torch
import torch.nn.functional as F
from jaxtyping import Float, Float32, Int32
from torch import Tensor
from torch.autograd.function import once_differentiable
from torch_geometric.utils import to_dense_batch

from gigl.transforms.static_khop import PackedSequences

# The kernel pads the logsumexp's last dim to a multiple of this.
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
    """Forward kernel of ``gigl::varlen_attention_forward``.

    The types stay plain ``Tensor``: ``torch.library.custom_op`` infers the op
    schema from the annotations and rejects jaxtyping types.

    Args:
        query: ``[num_query_tokens, num_heads, head_dim]``.
        key: ``[num_key_tokens, num_heads, head_dim]``.
        value: ``[num_key_tokens, num_heads, head_dim]``.
        cu_seqlens_q: ``[num_sequences + 1]`` int32.
        cu_seqlens_k: ``[num_sequences + 1]`` int32.
        max_seqlen_q: Upper bound on any query sequence length.
        max_seqlen_k: Upper bound on any key sequence length.
        compute_logsumexp: Whether to compute the logsumexp the backward needs.

    Returns:
        - ``[num_query_tokens, num_heads, head_dim]`` attention output.
        - ``[num_sequences, num_heads, padded_max_seqlen]`` float32 logsumexp,
          with ``padded_max_seqlen`` zero unless ``compute_logsumexp``.
    """
    # The op's type stub declares every argument as ``...``. It returns
    # (output, logsumexp, philox_seed, philox_offset, max_q, max_k); only the
    # first two are used.
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
    """Shape-only fake of the forward kernel, with the same plain-``Tensor`` types."""
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
    """Backward kernel of ``gigl::varlen_attention_backward``.

    Plain ``Tensor`` types for the same reason as the forward kernel.

    Args:
        grad_output: ``[num_query_tokens, num_heads, head_dim]``.
        query: ``[num_query_tokens, num_heads, head_dim]``.
        key: ``[num_key_tokens, num_heads, head_dim]``.
        value: ``[num_key_tokens, num_heads, head_dim]``.
        output: ``[num_query_tokens, num_heads, head_dim]`` forward output.
        logsumexp: ``[num_sequences, num_heads, padded_max_seqlen]`` forward logsumexp.
        cu_seqlens_q: ``[num_sequences + 1]`` int32.
        cu_seqlens_k: ``[num_sequences + 1]`` int32.
        max_seqlen_q: Upper bound on any query sequence length.
        max_seqlen_k: Upper bound on any key sequence length.

    Returns:
        The gradients of ``query``, ``key`` and ``value``, shaped like them.
    """
    # Dropout is off, so the kernel never reads these random-number-generator
    # (Philox) placeholders.
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
    """Shape-only fake of the backward kernel, with the same plain-``Tensor`` types."""
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

    output: Float[Tensor, "num_query_tokens num_heads head_dim"]
    logsumexp: Float32[Tensor, "num_sequences num_heads padded_max_seqlen"]


class _VarlenAttention(torch.autograd.Function):
    """Memory-efficient varlen attention with the explicit backward kernel."""

    @staticmethod
    def forward(
        ctx: Any,
        query: Float[Tensor, "num_query_tokens num_heads head_dim"],
        key: Float[Tensor, "num_key_tokens num_heads head_dim"],
        value: Float[Tensor, "num_key_tokens num_heads head_dim"],
        cu_seqlens_q: Int32[Tensor, "num_sequences_plus_one"],
        cu_seqlens_k: Int32[Tensor, "num_sequences_plus_one"],
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> Float[Tensor, "num_query_tokens num_heads head_dim"]:
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
    and ``value``.

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
        # ``needs_input_grad`` does not reflect grad mode. Detaching the inputs
        # makes it false, so the forward skips the logsumexp and saving tensors.
        query, key, value = query.detach(), key.detach(), value.detach()
    return cast(
        Tensor,
        _VarlenAttention.apply(
            query, key, value, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k
        ),
    )


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
