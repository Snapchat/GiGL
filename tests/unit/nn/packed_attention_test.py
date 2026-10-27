"""Varlen attention over packed sequences matches padded attention."""

import itertools
import unittest

import torch
from absl.testing import absltest

from gigl.nn.packed_attention import _padded_packed_attention, _varlen_attention
from gigl.transforms.static_khop import pack_sequences
from tests.test_assets.test_case import TestCase

_MAX_SEQ_LEN = 24


class PackedAttentionTest(TestCase):
    def test_varlen_attention_traces_and_backpropagates_on_meta_tensors(self) -> None:
        cu_seqlens = torch.tensor([0, 5, 6, 12], dtype=torch.int32, device="meta")
        query, key, value = (
            torch.empty(12, 2, 64, device="meta", requires_grad=True) for _ in range(3)
        )
        output = _varlen_attention(query, key, value, cu_seqlens, cu_seqlens, 6, 6)
        self.assertEqual(output.shape, query.shape)
        self.assertIsNotNone(output.grad_fn)
        output.backward(torch.empty_like(output))
        for tensor in (query, key, value):
            assert tensor.grad is not None
            self.assertEqual(tensor.grad.shape, tensor.shape)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class PackedAttentionCudaTest(TestCase):
    def test_varlen_attention_matches_padded_packed_attention(self) -> None:
        device = torch.device("cuda")
        num_sequences, num_heads, head_dim = 6, 2, 64
        lengths = torch.tensor([5, 1, 24, 3, 12, 9], device=device)
        num_tokens = int(lengths.sum())
        packed = pack_sequences(lengths, _MAX_SEQ_LEN)
        generator = torch.Generator().manual_seed(0)
        for dtype, atol in ((torch.float32, 1e-4), (torch.bfloat16, 5e-2)):
            for query_is_first_token, track_grad in itertools.product(
                (False, True), (False, True)
            ):
                num_queries = num_sequences if query_is_first_token else num_tokens
                query, key, value = (
                    torch.randn(rows, num_heads, head_dim, generator=generator)
                    .to(device, dtype)
                    .requires_grad_()
                    for rows in (num_queries, num_tokens, num_tokens)
                )
                expected = _padded_packed_attention(
                    query, key, value, packed, query_is_first_token
                )
                if query_is_first_token:
                    cu_seqlens_q = torch.arange(
                        num_sequences + 1, dtype=torch.int32, device=device
                    )
                    max_seqlen_q = 1
                else:
                    cu_seqlens_q = packed.cu_seqlens
                    max_seqlen_q = _MAX_SEQ_LEN
                with (
                    self.subTest(
                        dtype=dtype, first_token=query_is_first_token, grad=track_grad
                    ),
                    torch.set_grad_enabled(track_grad),
                ):
                    actual = _varlen_attention(
                        query,
                        key,
                        value,
                        cu_seqlens_q,
                        packed.cu_seqlens,
                        max_seqlen_q,
                        _MAX_SEQ_LEN,
                    )
                    self.assertEqual(actual.requires_grad, track_grad)
                    torch.testing.assert_close(
                        actual.float().cpu(),
                        expected.float().cpu(),
                        rtol=0,
                        atol=atol,
                    )

    def test_varlen_attention_gradients_match_padded_packed_attention(self) -> None:
        device = torch.device("cuda")
        num_heads, head_dim = 2, 64
        lengths = torch.tensor([5, 1, 24, 3, 12], device=device)
        num_tokens = int(lengths.sum())
        packed = pack_sequences(lengths, _MAX_SEQ_LEN)
        generator = torch.Generator().manual_seed(0)

        def random_rows(num_rows: int) -> torch.Tensor:
            return torch.randn(num_rows, num_heads, head_dim, generator=generator).to(
                device
            )

        for dtype, atol in ((torch.float32, 1e-4), (torch.bfloat16, 5e-2)):
            for query_is_first_token in (False, True):
                num_queries = 5 if query_is_first_token else num_tokens
                inputs = [
                    random_rows(rows).to(dtype)
                    for rows in (num_queries, num_tokens, num_tokens)
                ]
                grad_output = random_rows(num_queries).to(dtype)
                varlen_inputs, padded_inputs = (
                    [tensor.clone().requires_grad_() for tensor in inputs]
                    for _ in range(2)
                )
                if query_is_first_token:
                    cu_seqlens_q = torch.arange(6, dtype=torch.int32, device=device)
                    max_seqlen_q = 1
                else:
                    cu_seqlens_q = packed.cu_seqlens
                    max_seqlen_q = _MAX_SEQ_LEN
                _varlen_attention(
                    varlen_inputs[0],
                    varlen_inputs[1],
                    varlen_inputs[2],
                    cu_seqlens_q,
                    packed.cu_seqlens,
                    max_seqlen_q,
                    _MAX_SEQ_LEN,
                ).backward(grad_output)
                _padded_packed_attention(
                    padded_inputs[0],
                    padded_inputs[1],
                    padded_inputs[2],
                    packed,
                    query_is_first_token,
                ).backward(grad_output)
                for name, actual, expected in zip(
                    ("query", "key", "value"), varlen_inputs, padded_inputs
                ):
                    with self.subTest(
                        dtype=dtype, first_token=query_is_first_token, grad=name
                    ):
                        assert actual.grad is not None and expected.grad is not None
                        torch.testing.assert_close(
                            actual.grad.float(),
                            expected.grad.float(),
                            rtol=0,
                            atol=atol,
                        )


if __name__ == "__main__":
    absltest.main()
