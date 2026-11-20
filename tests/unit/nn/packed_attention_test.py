"""Varlen attention over packed sequences matches padded attention."""

import itertools
import unittest
from typing import Callable, Optional

import torch
from absl.testing import absltest
from torch._dynamo.testing import CompileCounter
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import checkpoint

from gigl.nn.packed_attention import (
    PackedAttentionBias,
    _first_row_per_key_bias,
    _fold_per_key_bias,
    _kernel_bias_view,
    _pad_head_dim,
    _pad_query_key,
    _padded_packed_attention,
    _stride_zero_key_bias,
    _varlen_attention,
    layout_buckets,
    packed_attention,
    per_pair_bias_num_cells,
    per_pair_bias_views,
    per_pair_linear_index,
    per_pair_sequence_offsets,
)
from gigl.transforms.sequence_layout import PackedLayout, pack_sequences
from tests.test_assets.packed_attention_reference import reference_packed_attention
from tests.test_assets.test_case import TestCase

_MAX_SEQ_LEN = 24
_BUCKETED_MAX_SEQ_LEN = 256
# Two or more sequences in each of (0, 32], (32, 64], (64, 128] and (128, 256].
_BUCKETED_LENGTHS = [5, 20, 40, 60, 70, 100, 120, 130, 200, 256, 150]


def _bucketed_layout(
    lengths: Optional[list[int]] = None, device: Optional[torch.device] = None
) -> PackedLayout:
    layout = pack_sequences(
        torch.tensor(lengths or _BUCKETED_LENGTHS, device=device),
        _BUCKETED_MAX_SEQ_LEN,
        sort_by_length=True,
    )
    assert layout.buckets is not None
    return layout


def _dense_reference_bias(
    layout: PackedLayout,
    num_heads: int,
    per_key: Optional[torch.Tensor],
    per_pair: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Builds the reference's ``[sequences, heads, max_seq_len, max_seq_len]`` bias.

    Copies each bucket's block of the flat ``per_pair`` into the top-left corner
    of its sequences' grids, and broadcasts ``per_key`` over the query rows.
    """
    if per_key is None and per_pair is None:
        return None
    num_sequences = layout.cu_seqlens.numel() - 1
    max_seq_len = layout.max_seq_len
    dtype = torch.float64
    dense = torch.zeros(num_sequences, num_heads, max_seq_len, max_seq_len, dtype=dtype)
    if per_pair is not None:
        views = per_pair_bias_views(per_pair, layout, num_heads)
        for bucket, block in zip(layout_buckets(layout), views):
            length = bucket.bucket_len
            dense[bucket.first_sequence : bucket.end_sequence, :, :length, :length] = (
                block
            )
    if per_key is not None:
        grid = torch.zeros(num_sequences * max_seq_len, num_heads, dtype=dtype)
        grid = grid.index_copy(0, layout.flat_index, per_key)
        dense = dense + grid.view(num_sequences, max_seq_len, num_heads).permute(
            0, 2, 1
        ).unsqueeze(2)
    return dense


class PackedAttentionTest(TestCase):
    def test_varlen_attention_with_a_bias_raises_when_an_input_requires_grad(
        self,
    ) -> None:
        cu_seqlens = torch.tensor([0, 5, 6, 12], dtype=torch.int32, device="meta")
        attn_bias = torch.empty(3, 2, 8, 8, device="meta")
        for name in ("query", "key", "value"):
            with self.subTest(requires_grad=name):
                tensors = {
                    tensor_name: torch.empty(
                        12, 2, 64, device="meta", requires_grad=tensor_name == name
                    )
                    for tensor_name in ("query", "key", "value")
                }
                with self.assertRaises(ValueError):
                    _varlen_attention(
                        tensors["query"],
                        tensors["key"],
                        tensors["value"],
                        cu_seqlens,
                        cu_seqlens,
                        6,
                        6,
                        attn_bias,
                    )

    def test_varlen_attention_with_a_bias_traces_on_meta_tensors(self) -> None:
        cu_seqlens = torch.tensor([0, 5, 6, 12], dtype=torch.int32, device="meta")
        query, key, value = (torch.empty(12, 2, 64, device="meta") for _ in range(3))
        attn_bias = torch.empty(3, 2, 8, 8, device="meta")
        output = _varlen_attention(
            query, key, value, cu_seqlens, cu_seqlens, 6, 6, attn_bias
        )
        self.assertEqual(output.shape, query.shape)

    def test_kernel_bias_view_without_copy_raises_when_the_view_does_not_fit(
        self,
    ) -> None:
        # One head and full sequences: the view's last query row starts past the
        # end of the grid.
        grid = torch.randn(3, 1, 8, 8)
        with self.assertRaises(RuntimeError):
            _kernel_bias_view(grid, 24, 24, allow_copy=False)

    def test_kernel_bias_view_copies_the_grid_when_the_view_does_not_fit(
        self,
    ) -> None:
        grid = torch.randn(3, 1, 8, 8)
        view = _kernel_bias_view(grid, 24, 24)
        self.assertEqual(view.shape, (1, 1, 24, 24))
        self.assertEqual(view.stride(), grid.stride())
        flat_grid = grid.flatten()
        for query_row, key_column in ((0, 0), (7, 7), (8, 0), (16, 7), (23, 0)):
            cell = query_row * 8 + key_column
            expected = flat_grid[cell] if cell < flat_grid.numel() else 0.0
            self.assertEqual(float(view[0, 0, query_row, key_column]), float(expected))

    def test_kernel_bias_view_without_copy_does_not_raise_when_the_view_fits(
        self,
    ) -> None:
        grid = torch.randn(3, 2, 8, 8)
        view = _kernel_bias_view(grid, 24, 24, allow_copy=False)
        self.assertEqual(view.shape, (1, 2, 24, 24))
        self.assertEqual(view.data_ptr(), grid.data_ptr())

    def test_stride_zero_key_bias_holds_per_key_at_every_valid_cell(self) -> None:
        for query_is_first_token, num_heads, (
            lengths,
            max_seq_len,
        ) in itertools.product(
            (False, True),
            (1, 2),
            (
                ([5, 1, 24, 3, 12], 24),
                ([8, 8, 8], 8),
                ([20, 20, 20], 20),
                ([7], 7),
                ([3, 5], 30),
            ),
        ):
            with self.subTest(
                first_token=query_is_first_token,
                num_heads=num_heads,
                max_seq_len=max_seq_len,
                lengths=lengths,
            ):
                layout = pack_sequences(torch.tensor(lengths), max_seq_len)
                num_tokens = sum(lengths)
                per_key = torch.randn(num_tokens, num_heads)
                bias = _stride_zero_key_bias(
                    per_key, layout, torch.float32, query_is_first_token
                )
                padded_len = (max_seq_len + 7) // 8 * 8
                query_len = 1 if query_is_first_token else max_seq_len
                self.assertEqual(
                    bias.shape, (len(lengths), num_heads, query_len, padded_len)
                )
                self.assertEqual(
                    bias.stride(), (num_heads * padded_len, padded_len, 0, 1)
                )
                num_query_tokens = len(lengths) if query_is_first_token else num_tokens
                _kernel_bias_view(bias, num_query_tokens, num_tokens, allow_copy=False)
                for token in range(num_tokens):
                    sequence = int(layout.sequence_id[token])
                    position = int(layout.position[token])
                    for query_row in range(query_len):
                        torch.testing.assert_close(
                            bias[sequence, :, query_row, position], per_key[token]
                        )

    def test_stride_zero_key_bias_casts_to_the_requested_dtype(self) -> None:
        layout = pack_sequences(torch.tensor([3, 2]), 4)
        bias = _stride_zero_key_bias(
            torch.randn(5, 2), layout, torch.bfloat16, query_is_first_token=False
        )
        self.assertEqual(bias.dtype, torch.bfloat16)

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

    def _bias_inputs(
        self,
    ) -> tuple[
        list[int],
        PackedLayout,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        lengths = [3, 1, 2]
        layout = pack_sequences(torch.tensor(lengths), 3)
        generator = torch.Generator().manual_seed(0)
        query, key, value = (
            torch.randn(6, 2, 8, generator=generator) for _ in range(3)
        )
        per_key = torch.randn(6, 2, generator=generator)
        per_pair = torch.randn(3, 2, 3, 3, generator=generator)
        return lengths, layout, query, key, value, per_key, per_pair

    def _dense_reference(
        self,
        lengths: list[int],
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        per_key: Optional[torch.Tensor],
        per_pair: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Returns softmax(q k^T / sqrt(D) + bias) v per sequence, as [num_tokens, H, D]."""
        outputs: list[torch.Tensor] = []
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
            query,
            key,
            value,
            layout,
            False,
            PackedAttentionBias(per_key, per_pair.flatten()),
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
            PackedAttentionBias(per_key, per_pair.flatten()),
        )
        reference = self._dense_reference(lengths, query, key, value, per_key, per_pair)
        torch.testing.assert_close(packed, reference[first_token], rtol=1e-5, atol=1e-6)

    def test_empty_bias_matches_no_bias(self) -> None:
        lengths, layout, query, key, value, per_key, per_pair = self._bias_inputs()
        without = packed_attention(query, key, value, layout, False, bias=None)
        empty = packed_attention(
            query, key, value, layout, False, bias=PackedAttentionBias()
        )
        reference = reference_packed_attention(query, key, value, layout.cu_seqlens)
        torch.testing.assert_close(empty, without, rtol=0, atol=0)
        torch.testing.assert_close(empty, reference, rtol=0, atol=1e-5)
        # Both calls take the same path on CPU, so also check that the
        # comparison can fail: any non-empty bias changes the output.
        for bias in (
            PackedAttentionBias(per_key=per_key),
            PackedAttentionBias(per_pair=per_pair.flatten()),
        ):
            changed = packed_attention(query, key, value, layout, False, bias=bias)
            self.assertGreater((changed - without).abs().max(), 1e-3)

    def test_padded_packed_attention_with_a_per_pair_bias_matches_reference(
        self,
    ) -> None:
        num_heads, head_dim = 2, 4
        lengths = torch.tensor([5, 1, 7, 3])
        num_tokens = int(lengths.sum())
        layout = pack_sequences(lengths, 8)
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
            per_pair = torch.randn(
                lengths.numel(),
                num_heads,
                8,
                8,
                generator=generator,
                dtype=torch.float64,
            ).requires_grad_()
            grad_output = torch.randn(
                num_queries,
                num_heads,
                head_dim,
                generator=generator,
                dtype=torch.float64,
            )
            with self.subTest(first_token=query_is_first_token):
                actual = _padded_packed_attention(
                    query,
                    key,
                    value,
                    layout,
                    query_is_first_token,
                    PackedAttentionBias(per_pair=per_pair.flatten()),
                )
                expected = reference_packed_attention(
                    query,
                    key,
                    value,
                    layout.cu_seqlens,
                    bias=per_pair,
                    query_is_first_token=query_is_first_token,
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=1e-10)
                inputs = (query, key, value, per_pair)
                for actual_grad, expected_grad in zip(
                    torch.autograd.grad(actual, inputs, grad_output),
                    torch.autograd.grad(expected, inputs, grad_output),
                ):
                    torch.testing.assert_close(
                        actual_grad, expected_grad, rtol=0, atol=1e-10
                    )

    def test_packed_attention_matches_reference_on_cpu(self) -> None:
        num_heads, head_dim = 2, 4
        lengths = torch.tensor([5, 1, 7, 3])
        num_tokens = int(lengths.sum())
        layout = pack_sequences(lengths, 8)
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

    def _random_inputs(
        self,
        layout: PackedLayout,
        query_is_first_token: bool,
        with_per_key: bool,
        with_per_pair: bool,
        num_heads: int = 2,
        head_dim: int = 4,
    ) -> dict[str, torch.Tensor]:
        """Returns float64 leaves ``query``, ``key``, ``value`` and the requested biases."""
        generator = torch.Generator().manual_seed(0)
        num_tokens = layout.flat_index.numel()
        num_queries = (
            layout.cu_seqlens.numel() - 1 if query_is_first_token else num_tokens
        )
        shapes: dict[str, tuple[int, ...]] = {
            "query": (num_queries, num_heads, head_dim),
            "key": (num_tokens, num_heads, head_dim),
            "value": (num_tokens, num_heads, head_dim),
        }
        if with_per_key:
            shapes["per_key"] = (num_tokens, num_heads)
        if with_per_pair:
            shapes["per_pair"] = (per_pair_bias_num_cells(layout, num_heads),)
        return {
            name: torch.randn(
                shape, generator=generator, dtype=torch.float64
            ).requires_grad_()
            for name, shape in shapes.items()
        }

    def _assert_cpu_matches_reference(
        self,
        layout: PackedLayout,
        query_is_first_token: bool,
        with_per_key: bool,
        with_per_pair: bool,
    ) -> None:
        """Checks outputs and gradients of ``packed_attention`` against the loop reference."""
        num_heads = 2
        leaves = self._random_inputs(
            layout, query_is_first_token, with_per_key, with_per_pair, num_heads
        )
        bias = None
        if with_per_key or with_per_pair:
            bias = PackedAttentionBias(leaves.get("per_key"), leaves.get("per_pair"))
        actual = packed_attention(
            leaves["query"],
            leaves["key"],
            leaves["value"],
            layout,
            query_is_first_token,
            bias,
        )
        expected = reference_packed_attention(
            leaves["query"],
            leaves["key"],
            leaves["value"],
            layout.cu_seqlens,
            bias=_dense_reference_bias(
                layout, num_heads, leaves.get("per_key"), leaves.get("per_pair")
            ),
            query_is_first_token=query_is_first_token,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-10)
        grad_output = torch.randn(
            expected.shape,
            generator=torch.Generator().manual_seed(1),
            dtype=torch.float64,
        )
        inputs = list(leaves.values())
        for name, actual_grad, expected_grad in zip(
            leaves,
            torch.autograd.grad(actual, inputs, grad_output),
            torch.autograd.grad(expected, inputs, grad_output),
        ):
            with self.subTest(grad=name):
                torch.testing.assert_close(
                    actual_grad, expected_grad, rtol=0, atol=1e-10
                )

    def test_bucketed_layout_matches_reference_on_cpu(self) -> None:
        layout = _bucketed_layout()
        assert layout.buckets is not None
        self.assertEqual(len(layout.buckets), 4)
        for with_per_key, with_per_pair, query_is_first_token in itertools.product(
            (False, True), (False, True), (False, True)
        ):
            with self.subTest(
                per_key=with_per_key,
                per_pair=with_per_pair,
                first_token=query_is_first_token,
            ):
                self._assert_cpu_matches_reference(
                    layout, query_is_first_token, with_per_key, with_per_pair
                )

    def test_unbucketed_edge_layouts_match_reference_on_cpu(self) -> None:
        layouts = {
            "max_seq_len_below_32": pack_sequences(
                torch.tensor([5, 20, 7, 13]), 20, sort_by_length=True
            ),
            "all_lengths_1": pack_sequences(
                torch.ones(5, dtype=torch.long), 40, sort_by_length=True
            ),
            "one_sequence": pack_sequences(torch.tensor([37]), 64, sort_by_length=True),
        }
        for name, layout in layouts.items():
            self.assertIsNone(layout.buckets)
            for with_per_key, with_per_pair, query_is_first_token in itertools.product(
                (False, True), (False, True), (False, True)
            ):
                with self.subTest(
                    layout=name,
                    per_key=with_per_key,
                    per_pair=with_per_pair,
                    first_token=query_is_first_token,
                ):
                    self._assert_cpu_matches_reference(
                        layout, query_is_first_token, with_per_key, with_per_pair
                    )

    def test_per_pair_linear_index_matches_the_bucket_views(self) -> None:
        layout = _bucketed_layout()
        assert layout.buckets is not None
        num_heads = 3
        num_cells = per_pair_bias_num_cells(layout, num_heads)
        sequence_offset, sequence_bucket_len = per_pair_sequence_offsets(
            layout, num_heads
        )
        sequence_bucket_len_host = sequence_bucket_len.tolist()
        generator = torch.Generator().manual_seed(0)
        pairs: dict[tuple[int, int, int, int], float] = {}
        for _ in range(300):
            sequence = int(
                torch.randint(len(sequence_bucket_len_host), (), generator=generator)
            )
            bucket_len = sequence_bucket_len_host[sequence]
            head, query_position, key_position = (
                int(torch.randint(bound, (), generator=generator))
                for bound in (num_heads, bucket_len, bucket_len)
            )
            pairs[(sequence, head, query_position, key_position)] = float(
                len(pairs) + 1
            )
        keys = list(pairs)
        columns = [torch.tensor([key[i] for key in keys]) for i in range(4)]
        flat = torch.zeros(num_cells, dtype=torch.float64)
        flat[per_pair_linear_index(sequence_offset, sequence_bucket_len, *columns)] = (
            torch.tensor([pairs[key] for key in keys], dtype=torch.float64)
        )
        views = per_pair_bias_views(flat, layout, num_heads)
        for (sequence, head, query_position, key_position), value in pairs.items():
            bucket_index = next(
                i
                for i, bucket in enumerate(layout.buckets)
                if sequence < bucket.end_sequence
            )
            first_sequence = layout.buckets[bucket_index].first_sequence
            actual = views[bucket_index][
                sequence - first_sequence, head, query_position, key_position
            ]
            self.assertEqual(float(actual), value)

    def test_per_pair_bias_num_cells(self) -> None:
        num_heads = 3
        bucketed = _bucketed_layout()
        expected = 0
        for length in _BUCKETED_LENGTHS:
            bucket_len = next(bound for bound in (32, 64, 128, 256) if length <= bound)
            expected += num_heads * bucket_len**2
        self.assertEqual(per_pair_bias_num_cells(bucketed, num_heads), expected)
        unbucketed = pack_sequences(
            torch.tensor([5, 20, 7, 13]), 20, sort_by_length=True
        )
        self.assertIsNone(unbucketed.buckets)
        self.assertEqual(
            per_pair_bias_num_cells(unbucketed, num_heads), 4 * num_heads * 20**2
        )

    def test_per_pair_bias_views_reject_a_wrong_size(self) -> None:
        layout = _bucketed_layout()
        num_cells = per_pair_bias_num_cells(layout, 2)
        for wrong_size in (num_cells - 1, num_cells + 1):
            with self.assertRaises(ValueError):
                per_pair_bias_views(torch.zeros(wrong_size), layout, 2)

    def test_per_pair_bias_views_are_views_with_the_bucket_shapes(self) -> None:
        layout = _bucketed_layout()
        assert layout.buckets is not None
        flat = torch.zeros(per_pair_bias_num_cells(layout, 2))
        views = per_pair_bias_views(flat, layout, 2)
        for bucket, view in zip(layout.buckets, views):
            self.assertEqual(
                view.shape,
                (
                    bucket.end_sequence - bucket.first_sequence,
                    2,
                    bucket.bucket_len,
                    bucket.bucket_len,
                ),
            )
            view.fill_(1.0)
        self.assertEqual(float(flat.min()), 1.0)

    def test_first_row_per_key_bias_is_row_zero_of_the_dense_bias(self) -> None:
        layout = _bucketed_layout()
        num_heads = 2
        per_pair = torch.randn(
            per_pair_bias_num_cells(layout, num_heads),
            generator=torch.Generator().manual_seed(0),
            dtype=torch.float64,
        )
        dense = _dense_reference_bias(layout, num_heads, None, per_pair)
        assert dense is not None
        expected = dense[layout.sequence_id, :, 0, layout.position]
        actual = _first_row_per_key_bias(per_pair, layout, num_heads)
        self.assertEqual(actual.shape, (layout.flat_index.numel(), num_heads))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def _dropout_layouts(self) -> dict[str, tuple[PackedLayout, int]]:
        _, unbucketed, *_ = self._bias_inputs()
        return {"unbucketed": (unbucketed, 8), "bucketed": (_bucketed_layout(), 4)}

    def _dropout_inputs(
        self, layout: PackedLayout, head_dim: int, query_is_first_token: bool
    ) -> list[torch.Tensor]:
        generator = torch.Generator().manual_seed(0)
        num_tokens = layout.flat_index.numel()
        num_queries = (
            layout.cu_seqlens.numel() - 1 if query_is_first_token else num_tokens
        )
        return [
            torch.randn(rows, 2, head_dim, generator=generator)
            for rows in (num_queries, num_tokens, num_tokens)
        ]

    def test_zero_dropout_matches_no_dropout(self) -> None:
        for name, (layout, head_dim) in self._dropout_layouts().items():
            for query_is_first_token in (False, True):
                with self.subTest(layout=name, first_token=query_is_first_token):
                    query, key, value = self._dropout_inputs(
                        layout, head_dim, query_is_first_token
                    )
                    without = packed_attention(
                        query, key, value, layout, query_is_first_token
                    )
                    zero = packed_attention(
                        query, key, value, layout, query_is_first_token, dropout_p=0.0
                    )
                    torch.testing.assert_close(zero, without, rtol=0, atol=0)

    def test_dropout_changes_output_and_backpropagates(self) -> None:
        for name, (layout, head_dim) in self._dropout_layouts().items():
            for query_is_first_token in (False, True):
                with self.subTest(layout=name, first_token=query_is_first_token):
                    leaves = [
                        tensor.requires_grad_()
                        for tensor in self._dropout_inputs(
                            layout, head_dim, query_is_first_token
                        )
                    ]
                    without = packed_attention(
                        leaves[0], leaves[1], leaves[2], layout, query_is_first_token
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
                    self.assertFalse(torch.allclose(output, without))
                    output.sum().backward()
                    for leaf in leaves:
                        assert leaf.grad is not None
                        self.assertGreater(float(leaf.grad.abs().sum()), 0.0)

    def test_full_dropout_gives_zeros(self) -> None:
        for name, (layout, head_dim) in self._dropout_layouts().items():
            for query_is_first_token in (False, True):
                with self.subTest(layout=name, first_token=query_is_first_token):
                    query, key, value = self._dropout_inputs(
                        layout, head_dim, query_is_first_token
                    )
                    output = packed_attention(
                        query, key, value, layout, query_is_first_token, dropout_p=1.0
                    )
                    torch.testing.assert_close(
                        output, torch.zeros_like(output), rtol=0, atol=0
                    )

    def test_first_token_query_with_full_per_pair_grid_uses_row_zero(self) -> None:
        lengths, layout, query, key, value, _, per_pair = self._bias_inputs()
        first_token = layout.cu_seqlens[:-1].long()
        actual = packed_attention(
            query[first_token],
            key,
            value,
            layout,
            True,
            PackedAttentionBias(per_pair=per_pair.flatten()),
        )
        reference = self._dense_reference(lengths, query, key, value, None, per_pair)
        torch.testing.assert_close(actual, reference[first_token], rtol=1e-5, atol=1e-6)

    def test_compiled_attention_does_not_recompile_for_new_bucket_sizes(self) -> None:
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        counter = CompileCounter()
        compiled = torch.compile(packed_attention, backend=counter)
        generator = torch.Generator().manual_seed(0)
        bucketed_batches = [
            [5, 20, 40, 60, 70, 100, 120, 130],
            [1, 2, 33, 34, 65, 66, 129, 130, 131],
            [3, 4, 5, 40, 41, 70, 71, 72, 200, 201],
            [32, 32, 64, 64, 128, 128, 256, 256],
            [7, 8, 50, 51, 52, 90, 91, 140, 141, 142, 143],
            [31, 32, 33, 64, 65, 128, 129],
        ]
        one_bucket_batches = [
            [130, 200, 256],
            [129, 140, 150, 160],
            [200, 250],
            [256, 256, 256],
        ]
        batches = [
            bucketed_batches[0],
            one_bucket_batches[0],
            bucketed_batches[1],
            *bucketed_batches[2:],
            *one_bucket_batches[1:],
        ]
        counts: list[int] = []
        for lengths in batches:
            layout = pack_sequences(
                torch.tensor(lengths), _BUCKETED_MAX_SEQ_LEN, sort_by_length=True
            )
            num_tokens = layout.flat_index.numel()
            query, key, value = (
                torch.randn(num_tokens, 2, 4, generator=generator) for _ in range(3)
            )
            per_key = torch.randn(num_tokens, 2, generator=generator)
            per_pair = torch.randn(
                per_pair_bias_num_cells(layout, 2), generator=generator
            )
            compiled(
                query,
                key,
                value,
                layout,
                False,
                PackedAttentionBias(per_key, per_pair),
            )
            counts.append(counter.frame_count)
        # The batches have two structures (bucketed, or one bucket). The first
        # bucketed and one-bucket calls compile, and the second bucketed call
        # recompiles with the host sizes as symbolic ints (this took five calls
        # in the order above). After that, new sizes must not compile again.
        self.assertEqual(counts[4:], [counts[4]] * (len(batches) - 4))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class PackedAttentionCudaTest(TestCase):
    def test_padded_grid_backend_with_a_per_pair_bias_matches_reference(self) -> None:
        device = torch.device("cuda")
        num_heads, head_dim = 2, 16
        lengths = torch.tensor([5, 1, 24, 3, 12], device=device)
        num_tokens = int(lengths.sum())
        layout = pack_sequences(lengths, _MAX_SEQ_LEN)
        generator = torch.Generator().manual_seed(0)
        for query_is_first_token in (False, True):
            num_queries = lengths.numel()
            if not query_is_first_token:
                num_queries = num_tokens
            query, key, value = (
                torch.randn(rows, num_heads, head_dim, generator=generator)
                .to(device)
                .requires_grad_()
                for rows in (num_queries, num_tokens, num_tokens)
            )
            per_pair = (
                torch.randn(
                    lengths.numel(),
                    num_heads,
                    _MAX_SEQ_LEN,
                    _MAX_SEQ_LEN,
                    generator=generator,
                )
                .to(device)
                .requires_grad_()
            )
            grad_output = torch.randn(
                num_queries, num_heads, head_dim, generator=generator
            ).to(device)
            with self.subTest(first_token=query_is_first_token):
                actual = _padded_packed_attention(
                    query,
                    key,
                    value,
                    layout,
                    query_is_first_token,
                    PackedAttentionBias(per_pair=per_pair.flatten()),
                )
                expected = reference_packed_attention(
                    query,
                    key,
                    value,
                    layout.cu_seqlens,
                    bias=per_pair,
                    query_is_first_token=query_is_first_token,
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=1e-4)
                inputs = (query, key, value, per_pair)
                for actual_grad, expected_grad in zip(
                    torch.autograd.grad(actual, inputs, grad_output),
                    torch.autograd.grad(expected, inputs, grad_output),
                ):
                    torch.testing.assert_close(
                        actual_grad, expected_grad, rtol=0, atol=1e-4
                    )

    def test_varlen_attention_matches_padded_packed_attention(self) -> None:
        device = torch.device("cuda")
        num_sequences, num_heads, head_dim = 6, 2, 64
        lengths = torch.tensor([5, 1, 24, 3, 12, 9], device=device)
        num_tokens = int(lengths.sum())
        layout = pack_sequences(lengths, _MAX_SEQ_LEN)
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
        layout = pack_sequences(lengths, _MAX_SEQ_LEN)
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

    def _assert_matches_cpu_reference(
        self,
        head_dim: int,
        query_is_first_token: bool,
        dtype: torch.dtype = torch.float32,
        atol: float = 1e-4,
        rtol: float = 0.0,
        check_gradients: bool = True,
        with_per_key: bool = True,
        with_per_pair: bool = False,
        bucketed: bool = False,
        gradient_relative_l2: Optional[float] = None,
        max_seq_len: int = _MAX_SEQ_LEN,
    ) -> None:
        """Compares ``packed_attention`` on CUDA to the CPU padded reference.

        Optionally adds a per-key bias and a per-pair bias. With ``bucketed`` the
        layout is a sorted one with four length buckets, otherwise an unsorted
        one without buckets, whose ``max_seq_len`` is the argument and whose
        lengths are clamped to it.

        With ``gradient_relative_l2`` each gradient is compared by relative L2
        error instead of elementwise. Outputs are always compared elementwise.
        """
        device = torch.device("cuda")
        num_heads = 2
        if bucketed:
            packed = _bucketed_layout()
            cuda_packed = _bucketed_layout(device=device)
        else:
            lengths = torch.tensor([5, 1, 24, 3, 12]).clamp(max=max_seq_len)
            packed = pack_sequences(lengths, max_seq_len)
            cuda_packed = pack_sequences(lengths.to(device), max_seq_len)
        num_tokens = packed.flat_index.numel()
        generator = torch.Generator().manual_seed(0)
        if query_is_first_token:
            num_queries = packed.cu_seqlens.numel() - 1
        else:
            num_queries = num_tokens
        shapes: dict[str, tuple[int, ...]] = {
            "query": (num_queries, num_heads, head_dim),
            "key": (num_tokens, num_heads, head_dim),
            "value": (num_tokens, num_heads, head_dim),
        }
        if with_per_key:
            shapes["per_key"] = (num_tokens, num_heads)
        if with_per_pair:
            shapes["per_pair"] = (per_pair_bias_num_cells(packed, num_heads),)
        cpu_inputs = {
            name: torch.randn(shape, generator=generator)
            for name, shape in shapes.items()
        }
        grad_output = torch.randn(num_queries, num_heads, head_dim, generator=generator)
        cpu_leaves = {
            name: tensor.clone().requires_grad_() for name, tensor in cpu_inputs.items()
        }
        # The biases stay fp32 so the casts to the query dtype are exercised.
        cuda_leaves: dict[str, torch.Tensor] = {}
        for name, tensor in cpu_inputs.items():
            leaf_dtype = dtype if name in ("query", "key", "value") else torch.float32
            cuda_leaves[name] = tensor.to(device, leaf_dtype).requires_grad_()
        cpu_bias = None
        cuda_bias = None
        if with_per_key or with_per_pair:
            cpu_bias = PackedAttentionBias(
                cpu_leaves.get("per_key"), cpu_leaves.get("per_pair")
            )
            cuda_bias = PackedAttentionBias(
                cuda_leaves.get("per_key"), cuda_leaves.get("per_pair")
            )
        expected = _padded_packed_attention(
            cpu_leaves["query"],
            cpu_leaves["key"],
            cpu_leaves["value"],
            packed,
            query_is_first_token,
            cpu_bias,
        )
        actual = packed_attention(
            cuda_leaves["query"],
            cuda_leaves["key"],
            cuda_leaves["value"],
            cuda_packed,
            query_is_first_token,
            cuda_bias,
        )
        self.assertEqual(actual.shape, expected.shape)
        torch.testing.assert_close(
            actual.float().cpu(), expected.detach(), rtol=rtol, atol=atol
        )
        if not check_gradients:
            return
        expected.backward(grad_output)
        actual.backward(grad_output.to(device, dtype))
        for name, cuda_leaf in cuda_leaves.items():
            cpu_leaf = cpu_leaves[name]
            assert cuda_leaf.grad is not None and cpu_leaf.grad is not None
            with self.subTest(grad=name):
                if gradient_relative_l2 is None:
                    torch.testing.assert_close(
                        cuda_leaf.grad.float().cpu(),
                        cpu_leaf.grad,
                        rtol=rtol,
                        atol=atol,
                    )
                else:
                    error = (cuda_leaf.grad.float().cpu() - cpu_leaf.grad).norm() / (
                        cpu_leaf.grad.norm()
                    )
                    self.assertLess(
                        error,
                        gradient_relative_l2,
                        msg=f"{name} gradient relative L2 error {error}",
                    )

    def test_per_pair_bias_matches_cpu_reference(self) -> None:
        for bucketed, query_is_first_token, head_dim, with_per_key in itertools.product(
            (False, True), (False, True), (64, 12), (False, True)
        ):
            # bf16 gradients that sum over up to ~200 queries need a relative
            # tolerance. On the bucketed layout, elementwise checks of small
            # gradient elements are dominated by bf16 rounding, so compare the
            # relative L2 error, which still catches a wrong gradient.
            for dtype, atol, rtol in (
                (torch.float32, 1e-4, 0.0),
                (torch.bfloat16, 1e-1, 2e-2),
            ):
                gradient_relative_l2 = None
                if bucketed and dtype == torch.bfloat16:
                    gradient_relative_l2 = 2e-2
                with self.subTest(
                    bucketed=bucketed,
                    first_token=query_is_first_token,
                    head_dim=head_dim,
                    per_key=with_per_key,
                    dtype=dtype,
                ):
                    self._assert_matches_cpu_reference(
                        head_dim=head_dim,
                        query_is_first_token=query_is_first_token,
                        dtype=dtype,
                        atol=atol,
                        rtol=rtol,
                        with_per_key=with_per_key,
                        with_per_pair=True,
                        bucketed=bucketed,
                        gradient_relative_l2=gradient_relative_l2,
                    )

    def _kernel_query_head_dims(
        self,
        query_is_first_token: bool,
        requires_grad: bool,
        dropout_p: float = 0.0,
        bucketed: bool = True,
        max_seq_len: int = _MAX_SEQ_LEN,
        head_dim: int = 64,
        with_per_pair: bool = True,
        with_per_key: bool = False,
    ) -> list[int]:
        """Returns the query head dim of each ``varlen_attention`` call that ``packed_attention`` made.

        An empty list means ``packed_attention`` did not run the varlen kernel.
        """
        device = torch.device("cuda")
        if bucketed:
            layout = _bucketed_layout(device=device)
        else:
            lengths = torch.tensor([5, 1, 24, 3, 12], device=device)
            layout = pack_sequences(lengths.clamp(max=max_seq_len), max_seq_len)
        num_tokens = layout.flat_index.numel()
        num_queries = (
            layout.cu_seqlens.numel() - 1 if query_is_first_token else num_tokens
        )
        query = torch.randn(num_queries, 2, head_dim, device=device)
        key, value = (
            torch.randn(num_tokens, 2, head_dim, device=device) for _ in range(2)
        )
        per_pair = None
        if with_per_pair:
            per_pair = torch.randn(per_pair_bias_num_cells(layout, 2), device=device)
        per_key = None
        if with_per_key:
            per_key = torch.randn(num_tokens, 2, device=device)
        for tensor in (query, key, value, per_pair, per_key):
            if tensor is not None:
                tensor.requires_grad_(requires_grad)
        head_dims: list[int] = []

        class _Recorder(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if func == torch.ops.gigl.varlen_attention.default:
                    head_dims.append(list(args)[0].size(-1))
                return func(*args, **(kwargs or {}))

        with _Recorder():
            packed_attention(
                query,
                key,
                value,
                layout,
                query_is_first_token,
                PackedAttentionBias(per_key=per_key, per_pair=per_pair),
                dropout_p=dropout_p,
            )
        return head_dims

    def test_per_pair_bias_dispatch(self) -> None:
        for (
            query_is_first_token,
            requires_grad,
            dropout_p,
            bucketed,
            max_seq_len,
            expected_kernel,
        ) in (
            # A first-token row 0 is a per-key bias, so it runs the kernel.
            (True, True, 0.0, True, _MAX_SEQ_LEN, True),
            (True, False, 0.0, True, _MAX_SEQ_LEN, True),
            # Every token queried without a gradient runs the kernel per bucket
            # when every bucket length is a multiple of 8.
            (False, False, 0.0, True, _MAX_SEQ_LEN, True),
            (False, False, 0.0, False, 24, True),
            # The kernel has no backward for a bias.
            (False, True, 0.0, True, _MAX_SEQ_LEN, False),
            (False, True, 0.0, False, 24, False),
            # 20 is not a multiple of 8.
            (False, False, 0.0, False, 20, False),
            # Dropout never runs the kernel.
            (True, True, 0.3, True, _MAX_SEQ_LEN, False),
            (False, False, 0.3, True, _MAX_SEQ_LEN, False),
            (False, False, 0.3, False, 24, False),
        ):
            with self.subTest(
                first_token=query_is_first_token,
                requires_grad=requires_grad,
                dropout_p=dropout_p,
                bucketed=bucketed,
                max_seq_len=max_seq_len,
            ):
                head_dims = self._kernel_query_head_dims(
                    query_is_first_token,
                    requires_grad,
                    dropout_p,
                    bucketed,
                    max_seq_len,
                )
                self.assertEqual(len(head_dims) > 0, expected_kernel)

    def test_per_pair_bias_runs_one_kernel_call_per_bucket(self) -> None:
        layout = _bucketed_layout()
        assert layout.buckets is not None
        self.assertGreater(len(layout.buckets), 1)
        for with_per_key in (False, True):
            with self.subTest(per_key=with_per_key):
                head_dims = self._kernel_query_head_dims(
                    query_is_first_token=False,
                    requires_grad=False,
                    with_per_key=with_per_key,
                )
                self.assertEqual(len(head_dims), len(layout.buckets))
        head_dims = self._kernel_query_head_dims(
            query_is_first_token=False, requires_grad=False, bucketed=False
        )
        self.assertEqual(len(head_dims), 1)

    def test_per_key_bias_dispatch_distinguishes_stride_zero_from_the_fold(
        self,
    ) -> None:
        for query_is_first_token, requires_grad, head_dim, expected_head_dims in (
            # Stride 0 keeps the head dim.
            (False, False, 64, [64]),
            (True, False, 64, [64]),
            # The fold widens 64 to 72, and 12 to 16.
            (False, True, 64, [72]),
            (True, True, 64, [72]),
            (False, False, 12, [16]),
            (True, False, 12, [16]),
        ):
            with self.subTest(
                first_token=query_is_first_token,
                requires_grad=requires_grad,
                head_dim=head_dim,
            ):
                self.assertEqual(
                    self._kernel_query_head_dims(
                        query_is_first_token,
                        requires_grad,
                        bucketed=False,
                        head_dim=head_dim,
                        with_per_pair=False,
                        with_per_key=True,
                    ),
                    expected_head_dims,
                )

    def test_first_token_per_pair_bias_uses_stride_zero_without_a_gradient(
        self,
    ) -> None:
        for requires_grad, expected_head_dims in ((False, [64]), (True, [72])):
            with self.subTest(requires_grad=requires_grad):
                self.assertEqual(
                    self._kernel_query_head_dims(
                        query_is_first_token=True, requires_grad=requires_grad
                    ),
                    expected_head_dims,
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

    def test_inference_per_pair_bias_matches_cpu_reference(self) -> None:
        # Unbucketed with max_seq_len 20 is not a multiple of 8, so the call keeps
        # SDPA per bucket and must still match.
        for bucketed, max_seq_len, head_dim, with_per_key, (
            dtype,
            atol,
        ) in itertools.product(
            (True, False),
            (_MAX_SEQ_LEN, 20),
            (64, 12),
            (False, True),
            ((torch.float32, 1e-4), (torch.bfloat16, 1e-1)),
        ):
            if bucketed and max_seq_len == 20:
                continue
            with (
                self.subTest(
                    bucketed=bucketed,
                    max_seq_len=max_seq_len,
                    head_dim=head_dim,
                    per_key=with_per_key,
                    dtype=dtype,
                ),
                torch.no_grad(),
            ):
                self._assert_matches_cpu_reference(
                    head_dim=head_dim,
                    query_is_first_token=False,
                    dtype=dtype,
                    atol=atol,
                    check_gradients=False,
                    with_per_key=with_per_key,
                    with_per_pair=True,
                    bucketed=bucketed,
                    max_seq_len=max_seq_len,
                )

    def test_inference_per_key_bias_with_stride_zero_matches_cpu_reference(
        self,
    ) -> None:
        for query_is_first_token, head_dim, (dtype, atol) in itertools.product(
            (False, True),
            (64, 12),
            ((torch.float32, 1e-4), (torch.bfloat16, 1e-1)),
        ):
            with (
                self.subTest(
                    first_token=query_is_first_token, head_dim=head_dim, dtype=dtype
                ),
                torch.no_grad(),
            ):
                self._assert_matches_cpu_reference(
                    head_dim=head_dim,
                    query_is_first_token=query_is_first_token,
                    dtype=dtype,
                    atol=atol,
                    check_gradients=False,
                    with_per_key=True,
                )

    def test_per_pair_bias_that_is_a_strided_flat_view_matches_the_padded_path(
        self,
    ) -> None:
        # The flat bias can be a strided view of a larger tensor, so its last-dim
        # stride is above 1 and the kernel's bias needs a contiguous copy.
        device = torch.device("cuda")
        lengths = torch.tensor([5, 1, 24, 3, 12], device=device)
        layout = pack_sequences(lengths, _MAX_SEQ_LEN)
        num_tokens = int(lengths.sum())
        query, key, value = (
            torch.randn(num_tokens, 2, 64, device=device) for _ in range(3)
        )
        num_cells = per_pair_bias_num_cells(layout, 2)
        per_pair = torch.randn(num_cells, 2, device=device)[:, 0]
        self.assertEqual(per_pair.stride(), (2,))
        bias = PackedAttentionBias(per_pair=per_pair)
        with torch.no_grad():
            torch.testing.assert_close(
                packed_attention(query, key, value, layout, False, bias),
                _padded_packed_attention(query, key, value, layout, False, bias),
                rtol=0,
                atol=1e-4,
            )

    def test_per_pair_bias_with_one_head_and_full_sequences_matches_the_padded_path(
        self,
    ) -> None:
        # With one head and every sequence full, the kernel's strided bias view
        # reaches past the end of the bias grid.
        device = torch.device("cuda")
        for max_seq_len in (_MAX_SEQ_LEN, 16):
            with self.subTest(max_seq_len=max_seq_len):
                lengths = torch.full((3,), max_seq_len, device=device)
                layout = pack_sequences(lengths, max_seq_len)
                num_tokens = int(lengths.sum())
                query, key, value = (
                    torch.randn(num_tokens, 1, 64, device=device) for _ in range(3)
                )
                per_pair = torch.randn(3 * max_seq_len * max_seq_len, device=device)
                bias = PackedAttentionBias(per_pair=per_pair)
                with torch.no_grad():
                    torch.testing.assert_close(
                        packed_attention(query, key, value, layout, False, bias),
                        _padded_packed_attention(
                            query, key, value, layout, False, bias
                        ),
                        rtol=0,
                        atol=1e-4,
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
        with torch.no_grad():
            without = packed_attention(query, key, value, layout, False)
        self.assertFalse(torch.allclose(output, without))
        output.sum().backward()
        for leaf in (query, key, value):
            assert leaf.grad is not None
            self.assertTrue(torch.isfinite(leaf.grad).all())

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

    def test_compiled_per_key_bias_training_matches_eager(self) -> None:
        # The per-key fold widens q and k to 72 but not v, so the varlen op's
        # fake output must take the value head dim for compiled calls.
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        device = torch.device("cuda")
        lengths = torch.tensor([5, 1, 24, 3, 12], device=device)
        layout = pack_sequences(lengths, _MAX_SEQ_LEN)
        num_tokens = int(lengths.sum())
        generator = torch.Generator(device=device).manual_seed(0)
        inputs = [
            torch.randn(num_tokens, 2, 64, device=device, generator=generator)
            for _ in range(3)
        ]
        bias = PackedAttentionBias(
            per_key=torch.randn(num_tokens, 2, device=device, generator=generator)
        )

        def run(
            attention: Callable[..., torch.Tensor],
        ) -> tuple[torch.Tensor, list[torch.Tensor]]:
            leaves = [tensor.clone().requires_grad_() for tensor in inputs]
            output = attention(*leaves, layout, False, bias)
            output.sum().backward()
            gradients: list[torch.Tensor] = []
            for leaf in leaves:
                assert leaf.grad is not None
                gradients.append(leaf.grad)
            return output.detach(), gradients

        expected_output, expected_gradients = run(packed_attention)
        actual_output, actual_gradients = run(torch.compile(packed_attention))
        torch.testing.assert_close(actual_output, expected_output, rtol=1e-4, atol=1e-4)
        for actual, expected in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

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
