"""FlexAttention over packed sequences matches padded attention."""

import itertools
import unittest
from unittest import mock

import torch
from absl.testing import absltest
from torch.nn.attention.flex_attention import flex_attention

from gigl.nn import packed_attention as packed_attention_module
from gigl.nn.packed_attention import _padded_packed_attention, packed_attention
from gigl.transforms.static_khop import pack_sequences
from tests.test_assets.test_case import TestCase

_MAX_SEQ_LEN = 24
_NUM_HEADS = 2
_HEAD_DIM = 64


def _packed_inputs(
    device: torch.device,
    dtype: torch.dtype,
    num_sequences: int,
    query_is_first_token: bool,
):
    generator = torch.Generator().manual_seed(0)
    lengths = torch.randint(1, _MAX_SEQ_LEN + 1, (num_sequences,), generator=generator)
    num_tokens = int(lengths.sum())
    _, packed = pack_sequences(lengths.to(device), _MAX_SEQ_LEN)
    num_queries = num_sequences if query_is_first_token else num_tokens
    query, key, value = (
        torch.randn(rows, _NUM_HEADS, _HEAD_DIM, generator=generator)
        .to(device, dtype)
        .requires_grad_()
        for rows in (num_queries, num_tokens, num_tokens)
    )
    return packed, query, key, value


class PackedAttentionTest(TestCase):
    def test_flex_attention_matches_padded_attention_on_cpu(self) -> None:
        # 150 sequences hold about 1900 tokens, so queries and keys span several
        # 128-token blocks. Eager flex_attention stands in for the compiled one,
        # which needs a GPU.
        for query_is_first_token in (False, True):
            packed, query, key, value = _packed_inputs(
                torch.device("cpu"), torch.float32, 150, query_is_first_token
            )
            expected = _padded_packed_attention(
                query, key, value, packed, query_is_first_token
            )
            with (
                self.subTest(first_token=query_is_first_token),
                mock.patch.object(
                    packed_attention_module, "_compiled_flex_attention", flex_attention
                ),
            ):
                actual = packed_attention_module._flex_packed_attention(
                    query, key, value, packed.cu_seqlens, query_is_first_token
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5)
                grad_output = torch.randn_like(expected)
                (expected_grads, actual_grads) = (
                    torch.autograd.grad(output, (query, key, value), grad_output)
                    for output in (expected, actual)
                )
                for expected_grad, actual_grad in zip(expected_grads, actual_grads):
                    torch.testing.assert_close(
                        actual_grad, expected_grad, rtol=0, atol=1e-5
                    )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class PackedAttentionCudaTest(TestCase):
    def test_packed_attention_matches_padded_packed_attention(self) -> None:
        device = torch.device("cuda")
        for dtype, atol in ((torch.float32, 1e-4), (torch.bfloat16, 5e-2)):
            for query_is_first_token in (False, True):
                packed, query, key, value = _packed_inputs(
                    device, dtype, 150, query_is_first_token
                )
                expected = _padded_packed_attention(
                    query, key, value, packed, query_is_first_token
                )
                actual = packed_attention(
                    query, key, value, packed, query_is_first_token
                )
                grad_output = torch.randn_like(expected)
                (expected_grads, actual_grads) = (
                    torch.autograd.grad(output, (query, key, value), grad_output)
                    for output in (expected, actual)
                )
                with self.subTest(dtype=dtype, first_token=query_is_first_token):
                    torch.testing.assert_close(
                        actual.float(), expected.float(), rtol=0, atol=atol
                    )
                    for name, expected_grad, actual_grad in zip(
                        ("query", "key", "value"), expected_grads, actual_grads
                    ):
                        with self.subTest(grad=name):
                            torch.testing.assert_close(
                                actual_grad.float(),
                                expected_grad.float(),
                                rtol=0,
                                atol=atol,
                            )

    def test_compiled_packed_attention_handles_new_token_counts(self) -> None:
        device = torch.device("cuda")
        compiled = torch.compile(
            lambda query, key, value, packed, first: packed_attention(
                query, key, value, packed, first
            ),
            dynamic=False,
        )
        for query_is_first_token, num_sequences in itertools.product(
            (False, True), (150, 170)
        ):
            packed, query, key, value = _packed_inputs(
                device, torch.float32, num_sequences, query_is_first_token
            )
            for tensor in (query, key, value, packed.cu_seqlens):
                torch._dynamo.mark_dynamic(tensor, 0)
            with self.subTest(
                first_token=query_is_first_token, num_sequences=num_sequences
            ):
                torch.testing.assert_close(
                    compiled(query, key, value, packed, query_is_first_token),
                    _padded_packed_attention(
                        query, key, value, packed, query_is_first_token
                    ),
                    rtol=0,
                    atol=1e-4,
                )


if __name__ == "__main__":
    absltest.main()
