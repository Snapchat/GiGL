"""Varlen attention over packed sequences matches padded attention."""

import itertools
import unittest

import torch
from absl.testing import absltest

from gigl.nn.packed_attention import (
    PackedAttentionBias,
    _pad_head_dim,
    _pad_query_key,
    _padded_packed_attention,
    _varlen_attention,
    packed_attention,
)
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

    def test_pad_head_dim_zero_pads_the_last_dim_to_a_multiple_of_8(self) -> None:
        tensor = torch.randn(5, 2, 12)
        padded = _pad_head_dim(tensor)
        self.assertEqual(padded.shape, (5, 2, 16))
        torch.testing.assert_close(padded[..., :12], tensor)
        self.assertEqual(padded[..., 12:].abs().sum().item(), 0.0)
        self.assertEqual(_pad_head_dim(torch.randn(5, 2, 16)).shape, (5, 2, 16))

    def test_padded_query_key_scores_match_the_unpadded_scores(self) -> None:
        query, key = torch.randn(6, 2, 12), torch.randn(6, 2, 12)
        padded_query, padded_key = _pad_query_key(query, key)
        self.assertEqual(padded_query.shape[-1], 16)
        torch.testing.assert_close(
            torch.einsum("qhd,khd->hqk", padded_query, padded_key) / 16**0.5,
            torch.einsum("qhd,khd->hqk", query, key) / 12**0.5,
            rtol=0,
            atol=1e-5,
        )

    def _bias_inputs(self):
        lengths = [3, 1, 2]
        layout = pack_sequences(torch.tensor(lengths), 3)
        generator = torch.Generator().manual_seed(0)
        query, key, value = (
            torch.randn(6, 2, 8, generator=generator) for _ in range(3)
        )
        per_key = torch.randn(6, 2, generator=generator)
        per_pair = torch.randn(3, 2, 3, 3, generator=generator)
        return lengths, layout, query, key, value, per_key, per_pair

    def _dense_reference(self, lengths, query, key, value, per_key, per_pair):
        """Returns softmax(q k^T / sqrt(D) + bias) v per sequence, as [num_tokens, H, D]."""
        outputs = []
        start = 0
        for sequence, length in enumerate(lengths):
            rows = slice(start, start + length)
            scores = torch.einsum("qhd,khd->hqk", query[rows], key[rows]) / 8**0.5
            if per_key is not None:
                scores = scores + per_key[rows].t()[:, None, :]
            if per_pair is not None:
                scores = scores + per_pair[sequence, :, :length, :length]
            weights = scores.softmax(dim=-1)
            outputs.append(torch.einsum("hqk,khd->qhd", weights, value[rows]))
            start += length
        return torch.cat(outputs)

    def test_padded_packed_attention_with_bias_matches_dense_reference(self) -> None:
        lengths, layout, query, key, value, per_key, per_pair = self._bias_inputs()
        packed = _padded_packed_attention(
            query, key, value, layout, False, PackedAttentionBias(per_key, per_pair)
        )
        reference = self._dense_reference(lengths, query, key, value, per_key, per_pair)
        torch.testing.assert_close(packed, reference, rtol=1e-5, atol=1e-6)

    def test_padded_packed_attention_per_key_only_matches_dense_reference(self) -> None:
        lengths, layout, query, key, value, per_key, _ = self._bias_inputs()
        packed = _padded_packed_attention(
            query, key, value, layout, False, PackedAttentionBias(per_key=per_key)
        )
        reference = self._dense_reference(lengths, query, key, value, per_key, None)
        torch.testing.assert_close(packed, reference, rtol=1e-5, atol=1e-6)

    def test_padded_packed_attention_first_token_query_with_bias(self) -> None:
        lengths, layout, query, key, value, per_key, per_pair = self._bias_inputs()
        first_token = layout.cu_seqlens[:-1].long()
        packed = _padded_packed_attention(
            query[first_token],
            key,
            value,
            layout,
            True,
            PackedAttentionBias(per_key, per_pair),
        )
        reference = self._dense_reference(lengths, query, key, value, per_key, per_pair)
        torch.testing.assert_close(packed, reference[first_token], rtol=1e-5, atol=1e-6)

    def test_empty_bias_matches_no_bias(self) -> None:
        _, layout, query, key, value, _, _ = self._bias_inputs()
        without = packed_attention(query, key, value, layout, False, bias=None)
        empty = packed_attention(
            query, key, value, layout, False, bias=PackedAttentionBias()
        )
        torch.testing.assert_close(empty, without, rtol=0, atol=0)


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

    def test_varlen_head_dim_12_matches_cpu_reference(self) -> None:
        device = torch.device("cuda")
        num_heads, head_dim = 2, 12
        lengths = torch.tensor([5, 1, 24, 3, 12])
        num_tokens = int(lengths.sum())
        packed = pack_sequences(lengths, _MAX_SEQ_LEN)
        cuda_packed = pack_sequences(lengths.to(device), _MAX_SEQ_LEN)
        generator = torch.Generator().manual_seed(0)
        for query_is_first_token in (False, True):
            num_queries = 5 if query_is_first_token else num_tokens
            cpu_inputs = [
                torch.randn(rows, num_heads, head_dim, generator=generator)
                for rows in (num_queries, num_tokens, num_tokens)
            ]
            grad_output = torch.randn(
                num_queries, num_heads, head_dim, generator=generator
            )
            cpu_leaves, cuda_leaves = (
                [tensor.clone().requires_grad_() for tensor in group]
                for group in (cpu_inputs, [t.to(device) for t in cpu_inputs])
            )
            expected = _padded_packed_attention(
                cpu_leaves[0],
                cpu_leaves[1],
                cpu_leaves[2],
                packed,
                query_is_first_token,
            )
            expected.backward(grad_output)
            actual = packed_attention(
                cuda_leaves[0],
                cuda_leaves[1],
                cuda_leaves[2],
                cuda_packed,
                query_is_first_token,
            )
            actual.backward(grad_output.to(device))
            with self.subTest(first_token=query_is_first_token):
                self.assertEqual(actual.shape, expected.shape)
                torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=1e-4)
                for name, cuda_leaf, cpu_leaf in zip(
                    ("query", "key", "value"), cuda_leaves, cpu_leaves
                ):
                    assert cuda_leaf.grad is not None and cpu_leaf.grad is not None
                    with self.subTest(grad=name):
                        torch.testing.assert_close(
                            cuda_leaf.grad.cpu(), cpu_leaf.grad, rtol=0, atol=1e-4
                        )


if __name__ == "__main__":
    absltest.main()
