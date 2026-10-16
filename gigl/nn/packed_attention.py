"""Token packing and attention over packed variable-length sequences.

A padded ``[num_sequences, max_seq_len, dim]`` batch spends most of its linear
FLOPs on padding when lengths vary. Packing puts every valid token of every
sequence into one ``[token_capacity, dim]`` buffer, so linears and FFNs run on
real tokens only and attention runs per sequence.

``token_capacity`` is a host int, so all shapes stay fixed and the code runs
without host syncs. A batch with more tokens than the capacity sets a device
overflow flag; its outputs are wrong and the caller must recompute it another way.
"""

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, cast

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class PackedForwardConfig:
    """Opts a ``GraphTransformerEncoder`` into packed-token attention.

    Attributes:
        tokens_per_anchor: Packed token slots per anchor. A batch whose
            sequences hold more than ``num_anchors * tokens_per_anchor`` tokens
            overflows and is recomputed on the padded path. The mean sequence
            length plus some headroom is a good start; slack slots still cost
            linear FLOPs.
        neighbor_table_width: Edges kept per node when building the k-hop
            layout. Must be at least the number of sampled edges any node
            expands along (in the ``sampling_direction``), i.e. the largest
            sampling fanout, summed over edge types into one node type. A
            batch with a wider node overflows and is recomputed on the padded
            path.
    """

    tokens_per_anchor: int
    neighbor_table_width: int

    def __post_init__(self) -> None:
        if self.tokens_per_anchor < 1 or self.neighbor_table_width < 1:
            raise ValueError(
                "tokens_per_anchor and neighbor_table_width must be positive, got "
                f"{self.tokens_per_anchor} and {self.neighbor_table_width}."
            )


class PackedTokens(NamedTuple):
    """Where each packed token comes from; slots past the real tokens are slack.

    Attributes:
        token_node: ``[token_capacity]`` long node id per token, -1 on slack.
        token_hop: ``[token_capacity]`` long hop per token, 0 on slack.
        token_valid: ``[token_capacity]`` bool, False on slack.
        token_sequence: ``[token_capacity]`` long sequence of each token.
        token_position: ``[token_capacity]`` long position of each token in its sequence.
        cu_seqlens: ``[num_sequences + 1]`` int32 cumulative sequence lengths,
            clamped to ``token_capacity``.
        sequence_starts: ``[num_sequences]`` long packed index of each sequence's
            first token, clamped to ``token_capacity - 1``.
        overflow: Scalar bool, True if the sequences hold more than
            ``token_capacity`` tokens.
        max_seq_len: Upper bound on any sequence length.
    """

    token_node: Tensor
    token_hop: Tensor
    token_valid: Tensor
    token_sequence: Tensor
    token_position: Tensor
    cu_seqlens: Tensor
    sequence_starts: Tensor
    overflow: Tensor
    max_seq_len: int


def pack_tokens(
    node_index: Tensor, hop_index: Tensor, lengths: Tensor, token_capacity: int
) -> PackedTokens:
    """Packs the first ``lengths[i]`` tokens of every sequence into ``token_capacity`` slots.

    Sequences keep their order and their tokens stay contiguous, so sequence
    ``i`` occupies ``[cu_seqlens[i], cu_seqlens[i + 1])``.

    Args:
        node_index: ``[num_sequences, max_seq_len]`` long node id per position.
        hop_index: ``[num_sequences, max_seq_len]`` long hop per position.
        lengths: ``[num_sequences]`` long valid length of each sequence (at least 1).
        token_capacity: Number of packed slots.

    Returns:
        The packed layout; see ``PackedTokens``.
    """
    num_sequences, max_seq_len = node_index.shape
    ends = torch.cumsum(lengths, dim=0)
    starts = ends - lengths
    total_tokens = ends[-1]
    slot = torch.arange(token_capacity, device=lengths.device)
    token_sequence = torch.searchsorted(ends, slot, right=True).clamp(
        max=num_sequences - 1
    )
    token_position = (slot - starts[token_sequence]).clamp(min=0, max=max_seq_len - 1)
    token_valid = slot < total_tokens
    flat_index = token_sequence * max_seq_len + token_position
    return PackedTokens(
        token_node=torch.where(token_valid, node_index.reshape(-1)[flat_index], -1),
        token_hop=torch.where(token_valid, hop_index.reshape(-1)[flat_index], 0),
        token_valid=token_valid,
        token_sequence=token_sequence,
        token_position=token_position,
        cu_seqlens=torch.cat([ends.new_zeros(1), ends])
        .clamp(max=token_capacity)
        .to(torch.int32),
        sequence_starts=starts.clamp(max=token_capacity - 1),
        overflow=total_tokens > token_capacity,
        max_seq_len=max_seq_len,
    )


class _VarlenAttention(torch.autograd.Function):
    """Memory-efficient varlen attention with the explicit backward kernel.

    Rows outside every sequence (packing slack) are zeroed in the output and in
    the input gradients: the kernels leave them unwritten, and a NaN there would
    reach the weight gradients through ``0 * NaN``.
    """

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
        output = _zero_rows_past(output.squeeze(0), cu_seqlens_q[-1])
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
            _zero_rows_past(grad_query.squeeze(0), cu_seqlens_q[-1]),
            _zero_rows_past(grad_key.squeeze(0), cu_seqlens_k[-1]),
            _zero_rows_past(grad_value.squeeze(0), cu_seqlens_k[-1]),
            None,
            None,
            None,
            None,
        )


def _zero_rows_past(rows: Tensor, num_valid_rows: Tensor) -> Tensor:
    """Zeroes ``rows[num_valid_rows:]`` without a host read of ``num_valid_rows``."""
    valid = torch.arange(rows.size(0), device=rows.device) < num_valid_rows
    return torch.where(valid.view(-1, *([1] * (rows.dim() - 1))), rows, 0.0)


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
    so it never syncs. Rows outside every query sequence are zero.
    Differentiable in ``query``, ``key`` and ``value``; slack rows get zero
    gradients.

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


def _scatter_to_grid(
    tokens: Tensor, destination: Tensor, num_sequences: int, max_seq_len: int
) -> Tensor:
    """Scatters ``[token_capacity, heads, dim]`` tokens to ``[num_sequences, heads, max_seq_len, dim]``.

    ``destination`` holds each token's flat grid slot; slack tokens point one
    past the grid and are dropped.
    """
    grid_size = num_sequences * max_seq_len
    grid = tokens.new_zeros((grid_size + 1,) + tokens.shape[1:])
    grid = grid.index_copy(0, destination, tokens)[:grid_size]
    return grid.view(num_sequences, max_seq_len, *tokens.shape[1:]).transpose(1, 2)


def padded_packed_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    packed: PackedTokens,
    query_is_first_token: bool,
) -> Tensor:
    """Attention over packed sequences through SDPA on a padded grid.

    Scatters the packed tokens to ``[num_sequences, max_seq_len]``, masks padded
    keys with the same additive ``finfo.min / 2`` bias as
    ``GraphTransformerEncoder``, and gathers the outputs back. It runs on any
    device; it is the CPU path and the reference for ``varlen_attention``.

    Args:
        query: ``[token_capacity, num_heads, head_dim]`` packed queries, or
            ``[num_sequences, num_heads, head_dim]`` when ``query_is_first_token``.
        key: ``[token_capacity, num_heads, head_dim]`` packed keys.
        value: ``[token_capacity, num_heads, head_dim]`` packed values.
        packed: Layout from ``pack_tokens``.
        query_is_first_token: ``query`` holds one query per sequence, the
            sequence's first token.

    Returns:
        Attention output shaped like ``query``. Slack rows hold arbitrary values.
    """
    num_sequences = packed.sequence_starts.size(0)
    max_seq_len = packed.max_seq_len
    num_heads, head_dim = key.shape[1], key.shape[2]
    grid_size = num_sequences * max_seq_len
    # Slack tokens all go to one spare row past the grid.
    destination = torch.where(
        packed.token_valid,
        packed.token_sequence * max_seq_len + packed.token_position,
        grid_size,
    )

    key_grid = _scatter_to_grid(key, destination, num_sequences, max_seq_len)
    value_grid = _scatter_to_grid(value, destination, num_sequences, max_seq_len)
    key_valid = torch.zeros(grid_size + 1, dtype=torch.bool, device=key.device)
    key_valid = key_valid.index_copy(0, destination, packed.token_valid)[:grid_size]
    key_valid = key_valid.view(num_sequences, 1, 1, max_seq_len)
    attn_bias = torch.zeros(
        (num_sequences, 1, 1, max_seq_len), dtype=query.dtype, device=query.device
    ).masked_fill(~key_valid, torch.finfo(query.dtype).min / 2.0)

    if query_is_first_token:
        output = F.scaled_dot_product_attention(
            query.unsqueeze(2), key_grid, value_grid, attn_mask=attn_bias
        )
        return output.squeeze(2)
    output = F.scaled_dot_product_attention(
        _scatter_to_grid(query, destination, num_sequences, max_seq_len),
        key_grid,
        value_grid,
        attn_mask=attn_bias,
    )
    output = output.transpose(1, 2).reshape(grid_size, num_heads, head_dim)
    return output[destination.clamp(max=grid_size - 1)]


def packed_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    packed: PackedTokens,
    query_is_first_token: bool,
) -> Tensor:
    """Attention over packed sequences: ``varlen_attention`` on CUDA, else ``padded_packed_attention``.

    Args and returns are as in ``padded_packed_attention``.
    """
    if not query.is_cuda:
        return padded_packed_attention(query, key, value, packed, query_is_first_token)
    if query_is_first_token:
        cu_seqlens_q = torch.arange(
            packed.sequence_starts.size(0) + 1, dtype=torch.int32, device=query.device
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
