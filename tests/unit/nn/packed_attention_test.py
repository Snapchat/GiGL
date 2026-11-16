"""Varlen attention over packed sequences matches padded attention."""

import itertools
import unittest

import torch
from absl.testing import absltest
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from gigl.nn.packed_attention import (
    _padded_packed_attention,
    _varlen_attention,
    packed_attention,
)
from gigl.transforms.sequence_layout import pack_sequences
from tests.test_assets.packed_attention_reference import reference_packed_attention
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

    def test_packed_attention_matches_reference_on_cpu(self) -> None:
        num_heads, head_dim = 2, 4
        lengths = torch.tensor([5, 1, 7, 3])
        num_tokens = int(lengths.sum())
        _, layout = pack_sequences(lengths, 8)
        generator = torch.Generator().manual_seed(0)
        for query_is_first_token in (False, True):
            num_queries = lengths.numel()
            if not query_is_first_token:
                num_queries = num_tokens
            query, key, value = (
                torch.randn(
                    rows, num_heads, head_dim, generator=generator, dtype=torch.float64
                ).requires_grad_()
                for rows in (num_queries, num_tokens, num_tokens)
            )
            grad_output = torch.randn(
                num_queries,
                num_heads,
                head_dim,
                generator=generator,
                dtype=torch.float64,
            )
            with self.subTest(first_token=query_is_first_token):
                actual = packed_attention(
                    query, key, value, layout, query_is_first_token
                )
                expected = reference_packed_attention(
                    query,
                    key,
                    value,
                    layout.cu_seqlens,
                    query_is_first_token=query_is_first_token,
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=1e-10)
                actual_grads = torch.autograd.grad(
                    actual, (query, key, value), grad_output
                )
                expected_grads = torch.autograd.grad(
                    expected, (query, key, value), grad_output
                )
                for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(
                        actual_grad, expected_grad, rtol=0, atol=1e-10
                    )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class PackedAttentionCudaTest(TestCase):
    def test_varlen_attention_matches_padded_packed_attention(self) -> None:
        device = torch.device("cuda")
        num_sequences, num_heads, head_dim = 6, 2, 64
        lengths = torch.tensor([5, 1, 24, 3, 12, 9], device=device)
        num_tokens = int(lengths.sum())
        _, layout = pack_sequences(lengths, _MAX_SEQ_LEN)
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
                    query, key, value, layout, query_is_first_token
                )
                with sdpa_kernel(SDPBackend.MATH):
                    expected_math = _padded_packed_attention(
                        query, key, value, layout, query_is_first_token
                    )
                expected_reference = reference_packed_attention(
                    query,
                    key,
                    value,
                    layout.cu_seqlens,
                    query_is_first_token=query_is_first_token,
                )
                with (
                    self.subTest(
                        dtype=dtype, first_token=query_is_first_token, grad=track_grad
                    ),
                    torch.set_grad_enabled(track_grad),
                ):
                    actual = packed_attention(
                        query, key, value, layout, query_is_first_token
                    )
                    self.assertEqual(actual.requires_grad, track_grad)
                    for reference in (expected, expected_math, expected_reference):
                        torch.testing.assert_close(
                            actual.float().cpu(),
                            reference.float().cpu(),
                            rtol=0,
                            atol=atol,
                        )

    def test_varlen_attention_gradients_match_padded_packed_attention(self) -> None:
        device = torch.device("cuda")
        num_heads, head_dim = 2, 64
        lengths = torch.tensor([5, 1, 24, 3, 12], device=device)
        num_tokens = int(lengths.sum())
        _, layout = pack_sequences(lengths, _MAX_SEQ_LEN)
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
                varlen_inputs, padded_inputs, math_inputs, reference_inputs = (
                    [tensor.clone().requires_grad_() for tensor in inputs]
                    for _ in range(4)
                )
                packed_attention(
                    varlen_inputs[0],
                    varlen_inputs[1],
                    varlen_inputs[2],
                    layout,
                    query_is_first_token,
                ).backward(grad_output)
                _padded_packed_attention(
                    padded_inputs[0],
                    padded_inputs[1],
                    padded_inputs[2],
                    layout,
                    query_is_first_token,
                ).backward(grad_output)
                with sdpa_kernel(SDPBackend.MATH):
                    _padded_packed_attention(
                        math_inputs[0],
                        math_inputs[1],
                        math_inputs[2],
                        layout,
                        query_is_first_token,
                    ).backward(grad_output)
                reference_packed_attention(
                    reference_inputs[0],
                    reference_inputs[1],
                    reference_inputs[2],
                    layout.cu_seqlens,
                    query_is_first_token=query_is_first_token,
                ).backward(grad_output)
                for name, index in (("query", 0), ("key", 1), ("value", 2)):
                    actual = varlen_inputs[index]
                    for reference_name, references in (
                        ("padded", padded_inputs),
                        ("math", math_inputs),
                        ("reference", reference_inputs),
                    ):
                        expected = references[index]
                        with self.subTest(
                            dtype=dtype,
                            first_token=query_is_first_token,
                            grad=name,
                            against=reference_name,
                        ):
                            assert actual.grad is not None
                            assert expected.grad is not None
                            torch.testing.assert_close(
                                actual.grad.float(),
                                expected.grad.float(),
                                rtol=0,
                                atol=atol,
                            )

    def test_compiled_first_token_attention_backpropagates_for_any_sequence_count(
        self,
    ) -> None:
        # The kernel's philox seed and offset are CPU tensors without dropout, but
        # a bare aten call registers them as CUDA tensors under compile. The
        # compiled backward can then reuse the freed offset as the one-element
        # int64 sequence-start index that a single sequence produces, and the
        # launch gets a host pointer.
        device = torch.device("cuda")
        for num_sequences in (2, 1):
            with self.subTest(num_sequences=num_sequences):
                torch._dynamo.reset()
                _, layout = pack_sequences(
                    torch.full((num_sequences,), 12, device=device), _MAX_SEQ_LEN
                )
                inputs = torch.randn(
                    int(layout.cu_seqlens[-1]),
                    64,
                    device=device,
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )
                weight = torch.randn(
                    64, 512, device=device, dtype=torch.bfloat16, requires_grad=True
                )

                def attend(inputs, weight, layout):
                    hidden = inputs @ weight
                    first = layout.cu_seqlens[:-1].long()
                    heads = hidden.view(-1, 8, 64)
                    output = packed_attention(
                        hidden[first].view(-1, 8, 64),
                        heads,
                        heads,
                        layout,
                        query_is_first_token=True,
                    )
                    return output.sum() + hidden[first.clamp(min=0)].sum()

                def checkpointed(inputs, weight, layout):
                    return checkpoint(
                        attend, inputs, weight, layout, use_reentrant=False
                    )

                torch.compile(checkpointed)(inputs, weight, layout).backward()
                torch.cuda.synchronize()
                assert inputs.grad is not None
                self.assertTrue(torch.isfinite(inputs.grad).all())


if __name__ == "__main__":
    absltest.main()
