"""Varlen attention over packed sequences matches padded attention."""

import itertools
import unittest

import torch
from absl.testing import absltest
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import checkpoint

from gigl.nn.packed_attention import (
    PackedAttentionBias,
    _fold_per_key_bias,
    _pad_head_dim,
    _pad_query_key,
    _padded_packed_attention,
    _varlen_attention,
    packed_attention,
)
from gigl.transforms.sequence_layout import pack_sequences
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

    def test_fold_per_key_bias_reproduces_the_scores(self) -> None:
        query, key = torch.randn(6, 2, 8), torch.randn(6, 2, 8)
        per_key = torch.randn(6, 2)
        folded_query, folded_key = _fold_per_key_bias(query, key, per_key)
        self.assertEqual(folded_query.shape[-1], 16)
        self.assertEqual(folded_key.shape[-1], 16)
        # The kernel scales by 1 / sqrt(d_pad), the folded head dim.
        torch.testing.assert_close(
            torch.einsum("qhd,khd->hqk", folded_query, folded_key) / 16**0.5,
            torch.einsum("qhd,khd->hqk", query, key) / 8**0.5 + per_key.t()[:, None, :],
            rtol=0,
            atol=1e-5,
        )

    def test_fold_per_key_bias_with_an_odd_head_dim_reproduces_the_scores(self) -> None:
        query, key = torch.randn(6, 2, 12), torch.randn(6, 2, 12)
        per_key = torch.randn(6, 2)
        folded_query, folded_key = _fold_per_key_bias(query, key, per_key)
        self.assertEqual(folded_query.shape[-1], 16)
        torch.testing.assert_close(
            torch.einsum("qhd,khd->hqk", folded_query, folded_key) / 16**0.5,
            torch.einsum("qhd,khd->hqk", query, key) / 12**0.5
            + per_key.t()[:, None, :],
            rtol=0,
            atol=1e-5,
        )

    def test_fold_per_key_bias_casts_the_bias_to_the_key_dtype(self) -> None:
        query = torch.randn(6, 2, 8, dtype=torch.bfloat16)
        key = torch.randn(6, 2, 8, dtype=torch.bfloat16)
        _, folded_key = _fold_per_key_bias(query, key, torch.randn(6, 2))
        self.assertEqual(folded_key.dtype, torch.bfloat16)

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

    def test_first_token_query_with_full_per_pair_grid_uses_row_zero(self) -> None:
        lengths, layout, query, key, value, _, per_pair = self._bias_inputs()
        first_token = layout.cu_seqlens[:-1].long()
        for bias_rows in (1, 3):
            with self.subTest(per_pair_rows=bias_rows):
                actual = packed_attention(
                    query[first_token],
                    key,
                    value,
                    layout,
                    True,
                    PackedAttentionBias(per_pair=per_pair[:, :, :bias_rows, :]),
                )
                reference = self._dense_reference(
                    lengths, query, key, value, None, per_pair
                )
                torch.testing.assert_close(
                    actual, reference[first_token], rtol=1e-5, atol=1e-6
                )

    def test_zero_dropout_matches_no_dropout(self) -> None:
        _, layout, query, key, value, _, _ = self._bias_inputs()
        without = packed_attention(query, key, value, layout, False)
        zero = packed_attention(query, key, value, layout, False, dropout_p=0.0)
        torch.testing.assert_close(zero, without, rtol=0, atol=0)

    def test_dropout_changes_output_and_backpropagates(self) -> None:
        _, layout, query, key, value, _, _ = self._bias_inputs()
        without = packed_attention(query, key, value, layout, False)
        for query_is_first_token in (False, True):
            with self.subTest(first_token=query_is_first_token):
                leaves = [
                    tensor.clone().requires_grad_() for tensor in (query, key, value)
                ]
                if query_is_first_token:
                    leaves[0] = (
                        query[layout.cu_seqlens[:-1].long()].clone().requires_grad_()
                    )
                torch.manual_seed(0)
                output = packed_attention(
                    leaves[0],
                    leaves[1],
                    leaves[2],
                    layout,
                    query_is_first_token,
                    dropout_p=0.5,
                )
                self.assertEqual(output.shape, leaves[0].shape)
                if not query_is_first_token:
                    self.assertFalse(torch.allclose(output, without))
                output.sum().backward()
                for leaf in leaves:
                    assert leaf.grad is not None
                    self.assertGreater(float(leaf.grad.abs().sum()), 0.0)

    def test_full_dropout_gives_zeros(self) -> None:
        _, layout, query, key, value, _, _ = self._bias_inputs()
        output = packed_attention(query, key, value, layout, False, dropout_p=1.0)
        torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)

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

    def _assert_matches_cpu_reference(
        self,
        head_dim: int,
        query_is_first_token: bool,
        dtype: torch.dtype = torch.float32,
        atol: float = 1e-4,
        check_gradients: bool = True,
        with_per_key: bool = True,
        per_pair_query_len: int = 0,
        max_seq_len: int = _MAX_SEQ_LEN,
    ) -> None:
        """Compares ``packed_attention`` on CUDA to the CPU padded reference.

        Optionally adds a per-key bias and a per-pair bias with
        ``per_pair_query_len`` query rows (0 for none).
        """
        device = torch.device("cuda")
        num_heads = 2
        lengths = torch.tensor([5, 1, 24, 3, 12]).clamp(max=max_seq_len)
        num_tokens = int(lengths.sum())
        packed = pack_sequences(lengths, max_seq_len)
        cuda_packed = pack_sequences(lengths.to(device), max_seq_len)
        generator = torch.Generator().manual_seed(0)
        if query_is_first_token:
            num_queries = lengths.numel()
        else:
            num_queries = num_tokens
        cpu_inputs = [
            torch.randn(rows, num_heads, head_dim, generator=generator)
            for rows in (num_queries, num_tokens, num_tokens)
        ]
        if with_per_key:
            cpu_inputs.append(torch.randn(num_tokens, num_heads, generator=generator))
        if per_pair_query_len > 0:
            cpu_inputs.append(
                torch.randn(
                    lengths.numel(),
                    num_heads,
                    per_pair_query_len,
                    max_seq_len,
                    generator=generator,
                )
            )
        grad_output = torch.randn(num_queries, num_heads, head_dim, generator=generator)
        cpu_leaves = [tensor.clone().requires_grad_() for tensor in cpu_inputs]
        cuda_leaves = [
            tensor.to(device, dtype).requires_grad_() for tensor in cpu_inputs[:3]
        ]
        # The biases stay fp32 so the casts to the query dtype are exercised.
        cuda_leaves.extend(
            tensor.to(device).requires_grad_() for tensor in cpu_inputs[3:]
        )
        bias_names = []
        if with_per_key:
            bias_names.append("per_key")
        if per_pair_query_len > 0:
            bias_names.append("per_pair")
        cpu_bias = None
        cuda_bias = None
        if bias_names:
            cpu_bias = PackedAttentionBias(**dict(zip(bias_names, cpu_leaves[3:])))
            cuda_bias = PackedAttentionBias(**dict(zip(bias_names, cuda_leaves[3:])))
        expected = _padded_packed_attention(
            cpu_leaves[0],
            cpu_leaves[1],
            cpu_leaves[2],
            packed,
            query_is_first_token,
            cpu_bias,
        )
        actual = packed_attention(
            cuda_leaves[0],
            cuda_leaves[1],
            cuda_leaves[2],
            cuda_packed,
            query_is_first_token,
            cuda_bias,
        )
        self.assertEqual(actual.shape, expected.shape)
        torch.testing.assert_close(
            actual.float().cpu(), expected.detach(), rtol=0, atol=atol
        )
        if not check_gradients:
            return
        expected.backward(grad_output)
        actual.backward(grad_output.to(device, dtype))
        for name, cuda_leaf, cpu_leaf in zip(
            ("query", "key", "value", *bias_names),
            cuda_leaves,
            cpu_leaves,
        ):
            assert cuda_leaf.grad is not None and cpu_leaf.grad is not None
            with self.subTest(grad=name):
                torch.testing.assert_close(
                    cuda_leaf.grad.float().cpu(), cpu_leaf.grad, rtol=0, atol=atol
                )

    def test_varlen_head_dim_12_matches_cpu_reference(self) -> None:
        for query_is_first_token in (False, True):
            with self.subTest(first_token=query_is_first_token):
                self._assert_matches_cpu_reference(
                    head_dim=12,
                    query_is_first_token=query_is_first_token,
                    with_per_key=False,
                )

    def test_varlen_per_key_bias_matches_cpu_reference(self) -> None:
        self._assert_matches_cpu_reference(head_dim=64, query_is_first_token=False)

    def test_varlen_per_key_bias_first_token_query_matches_cpu_reference(self) -> None:
        self._assert_matches_cpu_reference(head_dim=64, query_is_first_token=True)

    def test_varlen_per_key_bias_with_head_dim_12_matches_cpu_reference(self) -> None:
        for query_is_first_token in (False, True):
            with self.subTest(first_token=query_is_first_token):
                self._assert_matches_cpu_reference(
                    head_dim=12, query_is_first_token=query_is_first_token
                )

    def test_per_pair_bias_matches_cpu_reference(self) -> None:
        for query_is_first_token, per_pair_query_len in itertools.product(
            (False, True), (1, _MAX_SEQ_LEN)
        ):
            if not query_is_first_token and per_pair_query_len == 1:
                continue
            for dtype, atol in ((torch.float32, 1e-4), (torch.bfloat16, 1e-1)):
                with self.subTest(
                    first_token=query_is_first_token,
                    query_len=per_pair_query_len,
                    dtype=dtype,
                ):
                    self._assert_matches_cpu_reference(
                        head_dim=64,
                        query_is_first_token=query_is_first_token,
                        dtype=dtype,
                        atol=atol,
                        per_pair_query_len=per_pair_query_len,
                    )

    def test_per_pair_bias_without_gradient_matches_cpu_reference(self) -> None:
        # 20 is not a multiple of 8, so the kernel's bias is padded to 24 keys.
        for max_seq_len, with_per_key, (dtype, atol) in itertools.product(
            (_MAX_SEQ_LEN, 20),
            (False, True),
            ((torch.float32, 1e-4), (torch.bfloat16, 1e-1)),
        ):
            with (
                self.subTest(
                    max_seq_len=max_seq_len, per_key=with_per_key, dtype=dtype
                ),
                torch.no_grad(),
            ):
                self._assert_matches_cpu_reference(
                    head_dim=64,
                    query_is_first_token=False,
                    dtype=dtype,
                    atol=atol,
                    check_gradients=False,
                    with_per_key=with_per_key,
                    per_pair_query_len=max_seq_len,
                    max_seq_len=max_seq_len,
                )

    def test_per_pair_bias_that_is_a_permuted_view_matches_the_padded_path(
        self,
    ) -> None:
        # The pairwise bias comes from a linear layer over ``[batch, seq, seq,
        # heads]``, so it reaches the kernel with a last-dim stride above 1.
        device = torch.device("cuda")
        lengths = torch.tensor([5, 1, 24, 3, 12], device=device)
        layout = pack_sequences(lengths, _MAX_SEQ_LEN)
        num_tokens = int(lengths.sum())
        query, key, value = (
            torch.randn(num_tokens, 2, 64, device=device) for _ in range(3)
        )
        per_pair = torch.randn(5, _MAX_SEQ_LEN, _MAX_SEQ_LEN, 2, device=device).permute(
            0, 3, 1, 2
        )
        bias = PackedAttentionBias(per_pair=per_pair)
        with torch.no_grad():
            torch.testing.assert_close(
                packed_attention(query, key, value, layout, False, bias),
                _padded_packed_attention(query, key, value, layout, False, bias),
                rtol=0,
                atol=1e-4,
            )

    def _dispatched_kernel(
        self, query_is_first_token: bool, requires_grad: bool
    ) -> bool:
        """Returns whether ``packed_attention`` with a per-pair bias ran the varlen kernel."""
        device = torch.device("cuda")
        lengths = torch.tensor([5, 1, 24, 3, 12], device=device)
        layout = pack_sequences(lengths, _MAX_SEQ_LEN)
        num_tokens = int(lengths.sum())
        num_queries = lengths.numel() if query_is_first_token else num_tokens
        query = torch.randn(num_queries, 2, 64, device=device)
        key, value = (torch.randn(num_tokens, 2, 64, device=device) for _ in range(2))
        query_len = 1 if query_is_first_token else _MAX_SEQ_LEN
        per_pair = torch.randn(5, 2, query_len, _MAX_SEQ_LEN, device=device)
        for tensor in (query, key, value, per_pair):
            tensor.requires_grad_(requires_grad)
        names: list[str] = []

        class _Recorder(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                names.append(str(func))
                return func(*args, **(kwargs or {}))

        with _Recorder():
            packed_attention(
                query,
                key,
                value,
                layout,
                query_is_first_token,
                PackedAttentionBias(per_pair=per_pair),
            )
        return any("varlen_attention" in name for name in names)

    def test_per_pair_bias_dispatch(self) -> None:
        for query_is_first_token, requires_grad, expected_kernel in (
            # A first-token row 0 is a per-key bias, so it runs the kernel with a gradient.
            (True, True, True),
            (True, False, True),
            (False, False, True),
            # The kernel has no backward for a bias, so training uses the padded path.
            (False, True, False),
        ):
            with self.subTest(
                first_token=query_is_first_token, requires_grad=requires_grad
            ):
                self.assertEqual(
                    self._dispatched_kernel(query_is_first_token, requires_grad),
                    expected_kernel,
                )

    def test_varlen_attention_raises_when_the_bias_requires_grad(self) -> None:
        lengths = torch.tensor([3, 2], device="cuda")
        layout = pack_sequences(lengths, 8)
        query, key, value = (
            torch.randn(5, 2, 8, device="cuda", requires_grad=True) for _ in range(3)
        )
        attn_bias = torch.randn(2, 2, 8, 8, device="cuda", requires_grad=True)
        with self.assertRaises(ValueError):
            _varlen_attention(
                query, key, value, layout.cu_seqlens, layout.cu_seqlens, 8, 5, attn_bias
            )

    def test_dropout_runs_and_backpropagates_on_cuda(self) -> None:
        device = torch.device("cuda")
        lengths = torch.tensor([5, 1, 24, 3, 12], device=device)
        layout = pack_sequences(lengths, _MAX_SEQ_LEN)
        num_tokens = int(lengths.sum())
        query, key, value = (
            torch.randn(num_tokens, 2, 64, device=device, requires_grad=True)
            for _ in range(3)
        )
        output = packed_attention(query, key, value, layout, False, dropout_p=0.3)
        self.assertEqual(output.shape, query.shape)
        output.sum().backward()
        for leaf in (query, key, value):
            assert leaf.grad is not None
            self.assertTrue(torch.isfinite(leaf.grad).all())

    def test_varlen_per_key_bias_under_bf16_autocast_matches_fp32_reference(
        self,
    ) -> None:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self._assert_matches_cpu_reference(
                head_dim=64,
                query_is_first_token=False,
                dtype=torch.bfloat16,
                atol=5e-2,
                check_gradients=False,
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
                layout = pack_sequences(
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
