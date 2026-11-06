"""Slow reference for attention over packed variable-length sequences.

The reference loops over sequences and writes out ``softmax(q @ k.T * scale +
bias) @ v`` with explicit matmuls. It calls nothing from ``gigl``, so a parity
test that compares against it does not share code with the module under test.
"""

import math
from typing import Optional

import torch
from jaxtyping import Float, Int32
from torch import Tensor


def reference_packed_attention(
    query: Float[Tensor, "num_queries num_heads head_dim"],
    key: Float[Tensor, "num_tokens num_heads head_dim"],
    value: Float[Tensor, "num_tokens num_heads head_dim"],
    cu_seqlens: Int32[Tensor, "num_sequences_plus_one"],
    bias: Optional[Float[Tensor, "num_sequences num_heads query_len key_len"]] = None,
    query_is_first_token: bool = False,
    scale: Optional[float] = None,
) -> Float[Tensor, "num_queries num_heads head_dim"]:
    """Attends within each packed sequence with explicit matmul and softmax.

    The math runs in float64 and the result is cast back to the dtype of
    ``query``. The function is differentiable through autograd.

    Sequence ``i`` owns the key rows ``cu_seqlens[i]:cu_seqlens[i + 1]``. Its
    queries are the same rows of ``query``, or one row ``i`` when
    ``query_is_first_token``.

    Example:
        >>> cu_seqlens = torch.tensor([0, 2, 3], dtype=torch.int32)
        >>> value = torch.tensor([[[1.0]], [[3.0]], [[5.0]]])
        >>> zeros = torch.zeros(3, 1, 1)
        >>> reference_packed_attention(zeros, zeros, value, cu_seqlens).flatten().tolist()
        [2.0, 2.0, 5.0]

    Args:
        query: Packed queries, or one query per sequence when
            ``query_is_first_token``.
        key: Packed keys.
        value: Packed values.
        cu_seqlens: Sequence offsets into the packed rows.
        bias: Optional additive score bias. Sequence ``i`` reads
            ``bias[i, :, :query_len, :length]``, where ``query_len`` is 1 when
            ``query_is_first_token``.
        query_is_first_token: ``query`` holds one query per sequence, the
            sequence's first token.
        scale: Score scale. Defaults to ``1 / sqrt(head_dim)``.

    Returns:
        Attention output shaped like ``query``.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(query.size(-1))
    outputs: list[Tensor] = []
    for sequence in range(cu_seqlens.numel() - 1):
        start = int(cu_seqlens[sequence])
        end = int(cu_seqlens[sequence + 1])
        if query_is_first_token:
            sequence_query = query[sequence : sequence + 1]
        else:
            sequence_query = query[start:end]
        # [heads, queries, head_dim] and [heads, keys, head_dim]
        query_heads = sequence_query.double().transpose(0, 1)
        key_heads = key[start:end].double().transpose(0, 1)
        value_heads = value[start:end].double().transpose(0, 1)
        scores = query_heads @ key_heads.transpose(1, 2) * scale
        if bias is not None:
            scores = (
                scores + bias[sequence, :, : scores.size(1), : end - start].double()
            )
        weights = torch.softmax(scores, dim=-1)
        outputs.append((weights @ value_heads).transpose(0, 1))
    return torch.cat(outputs, dim=0).to(query.dtype)
