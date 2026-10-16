"""Packed-token ``GraphTransformerEncoder.forward`` matches the padded forward."""

import copy
import unittest
import warnings
import weakref
from unittest.mock import patch

import jaxtyping
import torch
import torch.nn as nn
from absl.testing import absltest
from parameterized import param, parameterized
from torch_geometric.data import HeteroData

from gigl.nn.graph_transformer import (
    GraphTransformerEncoder,
    GraphTransformerEncoderLayer,
)
from gigl.nn.packed_attention import (
    PackedForwardConfig,
    pack_tokens,
    padded_packed_attention,
    varlen_attention,
)
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from tests.test_assets.test_case import TestCase

_USER = NodeType("user")
_ITEM = NodeType("item")
_FOLLOWS = EdgeType(_USER, Relation("follows"), _USER)
_BUYS = EdgeType(_USER, Relation("buys"), _ITEM)
_BOUGHT_BY = EdgeType(_ITEM, Relation("bought_by"), _USER)
_HOP = "hop_distance"
_MAX_SEQ_LEN = 24


def _random_batch(seed: int, hetero: bool = False) -> HeteroData:
    """A random batch with duplicate edges, self loops and shared neighbours."""
    generator = torch.Generator().manual_seed(seed)
    data = HeteroData()
    data[_USER].x = torch.randn(50, 8, generator=generator)
    data[_USER].batch_size = 12
    follows = torch.randint(0, 50, (2, 110), generator=generator)
    data[_FOLLOWS.tuple_repr()].edge_index = torch.cat(
        [follows, follows[:, :6], torch.tensor([[0, 9], [0, 9]])], dim=1
    )
    if hetero:
        data[_ITEM].x = torch.randn(20, 8, generator=generator)
        data[_BUYS.tuple_repr()].edge_index = torch.stack(
            [
                torch.randint(0, 50, (40,), generator=generator),
                torch.randint(0, 20, (40,), generator=generator),
            ]
        )
        data[_BOUGHT_BY.tuple_repr()].edge_index = torch.stack(
            [
                torch.randint(0, 20, (40,), generator=generator),
                torch.randint(0, 50, (40,), generator=generator),
            ]
        )
    return data


def _max_degree(data: HeteroData, sampling_direction: str) -> int:
    """The largest number of edges any node expands along."""
    num_nodes = sum(data[node_type].num_nodes for node_type in data.node_types)
    offsets: dict[str, int] = {}
    offset = 0
    for node_type in data.node_types:
        offsets[node_type] = offset
        offset += data[node_type].num_nodes
    expanded = [
        data[edge_type].edge_index[1 if sampling_direction == "in" else 0]
        + offsets[edge_type[2] if sampling_direction == "in" else edge_type[0]]
        for edge_type in data.edge_types
    ]
    return int(torch.bincount(torch.cat(expanded), minlength=num_nodes).max())


def _encoder_kwargs(hetero: bool = False, **overrides: object) -> dict:
    kwargs: dict = dict(
        node_type_to_feat_dim_map={_USER: 8, _ITEM: 8} if hetero else {_USER: 8},
        edge_type_to_feat_dim_map=(
            {_FOLLOWS: 0, _BUYS: 0, _BOUGHT_BY: 0} if hetero else {_FOLLOWS: 0}
        ),
        hid_dim=16,
        out_dim=8,
        num_layers=2,
        num_heads=2,
        max_seq_len=_MAX_SEQ_LEN,
        hop_distance=2,
        readout_mode="anchor_only",
        should_l2_normalize_embedding_layer_output=True,
        anchor_based_input_attr_names=[_HOP],
        anchor_based_input_embedding_dict=nn.ModuleDict({_HOP: nn.Embedding(3, 16)}),
        layout_hop_distance_attr_name=_HOP,
        sampling_direction="in",
        activation="swiglu",
    )
    kwargs.update(overrides)
    return kwargs


def _encoder_pair(
    packed_forward: PackedForwardConfig, data: HeteroData, kwargs: dict
) -> tuple[GraphTransformerEncoder, GraphTransformerEncoder]:
    """A padded encoder and a packed one with the same weights, in eval mode.

    Each encoder gets its own copy of ``kwargs``, so modules passed in it (such
    as the token-input embeddings) are not shared between the two.
    """
    padded = GraphTransformerEncoder(**copy.deepcopy(kwargs)).eval()
    with torch.no_grad():
        # Materializes lazy modules before the weights are copied.
        padded(data, anchor_node_type=_USER)
    packed = GraphTransformerEncoder(
        **copy.deepcopy(kwargs), packed_forward=packed_forward
    ).eval()
    packed.load_state_dict(padded.state_dict())
    return padded, packed


def _assert_training_gradients_match(
    test_case: TestCase,
    padded: GraphTransformerEncoder,
    packed: GraphTransformerEncoder,
    data: HeteroData,
    atol: float = 1e-6,
) -> None:
    """Training-mode losses and every parameter gradient agree."""
    losses = []
    for encoder in (padded, packed):
        encoder.train()
        loss = encoder(data, anchor_node_type=_USER).sin().sum()
        loss.backward()
        losses.append(loss.detach())
    torch.testing.assert_close(losses[1], losses[0], rtol=0, atol=atol)
    packed_parameters = dict(packed.named_parameters())
    for name, parameter in padded.named_parameters():
        test_case.assertIsNotNone(parameter.grad, name)
        torch.testing.assert_close(
            packed_parameters[name].grad,
            parameter.grad,
            rtol=1e-5,
            atol=atol,
            msg=lambda message: f"{name}: {message}",
        )


class PackedForwardTest(TestCase):
    @parameterized.expand(
        [
            param("one_layer", num_layers=1),
            param("three_layers", num_layers=3),
            param("no_layers", num_layers=0),
            param("out_direction", sampling_direction="out"),
            param("hetero", hetero=True),
            param(
                "continuous_hop_input_and_node_pe",
                anchor_based_input_embedding_dict=None,
                pe_attr_names=["random_walk_pe"],
            ),
            param(
                "no_token_input",
                anchor_based_input_attr_names=None,
                anchor_based_input_embedding_dict=None,
                layout_hop_distance_attr_name=None,
            ),
            # More slots per anchor than max_seq_len are clamped to it.
            param("capacity_above_max_seq_len", tokens_per_anchor=4 * _MAX_SEQ_LEN),
            param("reference_uses_length_buckets", length_buckets=[4, 8, 16]),
        ]
    )
    def test_matches_padded_forward(
        self,
        _: str,
        hetero: bool = False,
        tokens_per_anchor: int = _MAX_SEQ_LEN,
        **overrides: object,
    ) -> None:
        for seed in range(3):
            data = _random_batch(seed, hetero=hetero)
            data[_USER].random_walk_pe = torch.randn(50, 3)
            if hetero:
                data[_ITEM].random_walk_pe = torch.randn(20, 3)
            kwargs = _encoder_kwargs(hetero=hetero, **overrides)
            config = PackedForwardConfig(
                tokens_per_anchor=tokens_per_anchor,
                neighbor_table_width=_max_degree(
                    data, str(kwargs["sampling_direction"])
                ),
            )
            padded, packed = _encoder_pair(config, data, kwargs)
            with self.subTest(seed=seed), torch.no_grad():
                torch.testing.assert_close(
                    packed(data, anchor_node_type=_USER),
                    padded(data, anchor_node_type=_USER),
                    rtol=0,
                    atol=1e-6,
                )
                self.assertEqual(packed.packed_overflow_count, 0)

    def test_matches_padded_forward_with_explicit_anchors(self) -> None:
        data = _random_batch(0)
        config = PackedForwardConfig(_MAX_SEQ_LEN, _max_degree(data, "in"))
        padded, packed = _encoder_pair(config, data, _encoder_kwargs())
        # Out of order, with a duplicate.
        anchor_node_ids = torch.tensor([7, 3, 3, 40, 0])
        with torch.no_grad():
            torch.testing.assert_close(
                packed(data, anchor_node_type=_USER, anchor_node_ids=anchor_node_ids),
                padded(data, anchor_node_type=_USER, anchor_node_ids=anchor_node_ids),
                rtol=0,
                atol=1e-6,
            )

    def test_empty_anchor_batch(self) -> None:
        data = _random_batch(0)
        config = PackedForwardConfig(_MAX_SEQ_LEN, _max_degree(data, "in"))
        padded, packed = _encoder_pair(config, data, _encoder_kwargs())
        no_anchors = torch.empty(0, dtype=torch.long)
        with torch.no_grad():
            actual = packed(data, anchor_node_type=_USER, anchor_node_ids=no_anchors)
            expected = padded(data, anchor_node_type=_USER, anchor_node_ids=no_anchors)
        self.assertEqual(actual.shape, (0, 8))
        torch.testing.assert_close(actual, expected)

    def test_edgeless_batch_fills_capacity_exactly(self) -> None:
        data = HeteroData()
        data[_USER].x = torch.randn(6, 8)
        data[_FOLLOWS.tuple_repr()].edge_index = torch.empty((2, 0), dtype=torch.long)
        # Six length-1 sequences fill one token per anchor exactly.
        config = PackedForwardConfig(tokens_per_anchor=1, neighbor_table_width=1)
        padded, packed = _encoder_pair(config, data, _encoder_kwargs())
        with torch.no_grad():
            torch.testing.assert_close(
                packed(data, anchor_node_type=_USER),
                padded(data, anchor_node_type=_USER),
                rtol=0,
                atol=1e-6,
            )
        self.assertEqual(packed.packed_overflow_count, 0)

    @parameterized.expand(
        [
            param("token_capacity", tokens_per_anchor=1, width_slack=0),
            param(
                "neighbor_table_width", tokens_per_anchor=_MAX_SEQ_LEN, width_slack=-1
            ),
        ]
    )
    def test_overflow_falls_back_to_padded_forward(
        self, _: str, tokens_per_anchor: int, width_slack: int
    ) -> None:
        data = _random_batch(0)
        config = PackedForwardConfig(
            tokens_per_anchor=tokens_per_anchor,
            neighbor_table_width=_max_degree(data, "in") + width_slack,
        )
        padded, packed = _encoder_pair(config, data, _encoder_kwargs())
        with torch.no_grad():
            expected = padded(data, anchor_node_type=_USER)
            for expected_count in (1, 2):
                torch.testing.assert_close(
                    packed(data, anchor_node_type=_USER), expected, rtol=0, atol=0
                )
                self.assertEqual(packed.packed_overflow_count, expected_count)

    @parameterized.expand(
        [
            param("packed", tokens_per_anchor=_MAX_SEQ_LEN),
            param("overflow_fallback", tokens_per_anchor=1),
        ]
    )
    def test_training_gradients_match_padded_forward(
        self, _: str, tokens_per_anchor: int
    ) -> None:
        data = _random_batch(0)
        config = PackedForwardConfig(tokens_per_anchor, _max_degree(data, "in"))
        padded, packed = _encoder_pair(config, data, _encoder_kwargs(dropout_rate=0.0))
        _assert_training_gradients_match(self, padded, packed, data)
        self.assertEqual(packed.packed_overflow_count, int(tokens_per_anchor == 1))

    def test_overflow_frees_packed_graph_before_padded_recompute(self) -> None:
        data = _random_batch(0)
        config = PackedForwardConfig(1, _max_degree(data, "in"))
        _, packed = _encoder_pair(config, data, _encoder_kwargs(dropout_rate=0.0))
        packed_states: list[weakref.ref] = []
        alive_during_padded: list[bool] = []
        encode_packed = GraphTransformerEncoder._encode_packed
        encode_sequences = GraphTransformerEncoder._encode_sequences

        def spy_encode_packed(self, **kwargs):
            anchor_states, overflow = encode_packed(self, **kwargs)
            packed_states.append(weakref.ref(anchor_states))
            return anchor_states, overflow

        def spy_encode_sequences(self, **kwargs):
            alive_during_padded.append(packed_states[-1]() is not None)
            return encode_sequences(self, **kwargs)

        with (
            patch.object(GraphTransformerEncoder, "_encode_packed", spy_encode_packed),
            patch.object(
                GraphTransformerEncoder, "_encode_sequences", spy_encode_sequences
            ),
        ):
            packed.train()(data, anchor_node_type=_USER).sum().backward()
        self.assertEqual(packed.packed_overflow_count, 1)
        self.assertEqual(alive_during_padded, [False])

    def test_training_rejects_attention_dropout(self) -> None:
        data = _random_batch(0)
        config = PackedForwardConfig(_MAX_SEQ_LEN, _max_degree(data, "in"))
        _, packed = _encoder_pair(
            config, data, _encoder_kwargs(attention_dropout_rate=0.1)
        )
        with torch.no_grad():
            packed(data, anchor_node_type=_USER)
        with self.assertRaises(ValueError):
            packed.train()(data, anchor_node_type=_USER)

    def test_rejects_unsupported_configs(self) -> None:
        config = PackedForwardConfig(tokens_per_anchor=8, neighbor_table_width=4)
        for overrides in (
            dict(readout_mode="anchor_neighbor_attention"),
            dict(prioritize_hop_order=True),
            dict(relation_message_mode="edge_type_linear"),
            dict(anchor_based_attention_bias_attr_names=[_HOP]),
            dict(layout_hop_distance_attr_name=None),
        ):
            kwargs = _encoder_kwargs()
            kwargs.update(overrides)
            with self.subTest(**overrides), self.assertRaises(ValueError):
                GraphTransformerEncoder(**kwargs, packed_forward=config)
        with self.assertRaises(ValueError):
            PackedForwardConfig(tokens_per_anchor=0, neighbor_table_width=4)

    def test_pack_tokens_layout(self) -> None:
        node_index = torch.tensor([[5, 6, -1], [7, -1, -1], [8, 9, 10]])
        hops = torch.tensor([[0, 1, 0], [0, 0, 0], [0, 1, 2]])
        packed = pack_tokens(
            node_index, hops, torch.tensor([2, 1, 3]), token_capacity=8
        )
        self.assertEqual(packed.token_node.tolist(), [5, 6, 7, 8, 9, 10, -1, -1])
        self.assertEqual(packed.token_hop.tolist(), [0, 1, 0, 0, 1, 2, 0, 0])
        self.assertEqual(packed.cu_seqlens.tolist(), [0, 2, 3, 6])
        self.assertEqual(packed.sequence_starts.tolist(), [0, 2, 3])
        self.assertEqual(packed.max_seq_len, 3)
        self.assertFalse(bool(packed.overflow))
        self.assertTrue(
            bool(pack_tokens(node_index, hops, torch.tensor([2, 1, 3]), 5).overflow)
        )

    def test_compiled_layer_has_no_graph_breaks_on_packed_tokens(self) -> None:
        layer = GraphTransformerEncoderLayer(
            model_dim=16,
            num_heads=2,
            feedforward_dim=32,
            dropout_rate=0.0,
            activation="swiglu",
        ).eval()
        lengths = torch.tensor([3, 1, 5, 2])
        packed = pack_tokens(
            torch.zeros(4, 6, dtype=torch.long),
            torch.zeros(4, 6, dtype=torch.long),
            lengths,
            token_capacity=14,
        )
        x = torch.where(packed.token_valid.view(1, -1, 1), torch.randn(1, 14, 16), 0.0)
        compiled = torch.compile(layer, fullgraph=True)
        # The test-only runtime shape checks are not traceable; production code
        # has none.
        jaxtyping.config.update("jaxtyping_disable", True)
        try:
            with torch.no_grad():
                for query_seq_len in (None, 1):
                    torch.testing.assert_close(
                        compiled(x, query_seq_len=query_seq_len, packed_tokens=packed),
                        layer(x, query_seq_len=query_seq_len, packed_tokens=packed),
                    )
        finally:
            jaxtyping.config.update("jaxtyping_disable", False)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class PackedForwardCudaTest(TestCase):
    def test_varlen_attention_matches_padded_packed_attention(self) -> None:
        device = torch.device("cuda")
        num_sequences, num_heads, head_dim, token_capacity = 6, 2, 64, 100
        generator = torch.Generator().manual_seed(0)
        node_index = torch.zeros(num_sequences, _MAX_SEQ_LEN, dtype=torch.long)
        packed = pack_tokens(
            node_index.to(device),
            node_index.to(device),
            torch.tensor([5, 1, 24, 3, 12, 9], device=device),
            token_capacity,
        )
        valid = packed.token_valid.cpu()
        for dtype, atol in ((torch.float32, 1e-4), (torch.bfloat16, 5e-2)):
            for query_is_first_token in (False, True):
                num_queries = num_sequences if query_is_first_token else token_capacity
                query, key, value = (
                    torch.randn(rows, num_heads, head_dim, generator=generator).to(
                        device, dtype
                    )
                    for rows in (num_queries, token_capacity, token_capacity)
                )
                expected = padded_packed_attention(
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
                actual = varlen_attention(
                    query,
                    key,
                    value,
                    cu_seqlens_q,
                    packed.cu_seqlens,
                    max_seqlen_q,
                    _MAX_SEQ_LEN,
                )
                if not query_is_first_token:
                    self.assertEqual(actual[~valid.to(device)].abs().max().item(), 0)
                    actual, expected = actual[valid], expected[valid]
                with self.subTest(dtype=dtype, first_token=query_is_first_token):
                    torch.testing.assert_close(
                        actual.float().cpu(), expected.float().cpu(), rtol=0, atol=atol
                    )

    def test_compiled_layer_has_no_graph_breaks_on_packed_tokens(self) -> None:
        device = torch.device("cuda")
        layer = (
            GraphTransformerEncoderLayer(
                model_dim=128,
                num_heads=2,
                feedforward_dim=256,
                dropout_rate=0.0,
                activation="swiglu",
            )
            .to(device)
            .eval()
        )
        packed = pack_tokens(
            torch.zeros(4, 6, dtype=torch.long, device=device),
            torch.zeros(4, 6, dtype=torch.long, device=device),
            torch.tensor([3, 1, 5, 2], device=device),
            token_capacity=14,
        )
        x = torch.where(
            packed.token_valid.view(1, -1, 1),
            torch.randn(1, 14, 128, device=device),
            0.0,
        )
        compiled = torch.compile(layer, fullgraph=True)
        jaxtyping.config.update("jaxtyping_disable", True)
        try:
            with torch.no_grad():
                for query_seq_len in (None, 1):
                    torch.testing.assert_close(
                        compiled(x, query_seq_len=query_seq_len, packed_tokens=packed),
                        layer(x, query_seq_len=query_seq_len, packed_tokens=packed),
                        rtol=0,
                        atol=1e-4,
                    )
        finally:
            jaxtyping.config.update("jaxtyping_disable", False)

    def test_varlen_attention_gradients_match_and_ignore_slack(self) -> None:
        device = torch.device("cuda")
        num_heads, head_dim, token_capacity = 2, 64, 60
        node_index = torch.zeros(5, _MAX_SEQ_LEN, dtype=torch.long, device=device)
        packed = pack_tokens(
            node_index,
            node_index,
            torch.tensor([5, 1, 24, 3, 12], device=device),
            token_capacity,
        )
        token_valid = packed.token_valid.view(-1, 1, 1)
        generator = torch.Generator().manual_seed(0)

        def random_rows(num_rows: int) -> torch.Tensor:
            return torch.randn(num_rows, num_heads, head_dim, generator=generator).to(
                device
            )

        for query_is_first_token in (False, True):
            num_queries = 5 if query_is_first_token else token_capacity
            query, key, value = (
                random_rows(num_queries),
                random_rows(token_capacity),
                random_rows(token_capacity),
            )
            grad_output = random_rows(num_queries)
            # Slack rows of the varlen inputs hold NaN; they must not leak.
            varlen_query = (
                query.clone()
                if query_is_first_token
                else query.masked_fill(~token_valid, torch.nan)
            ).requires_grad_()
            varlen_key = key.masked_fill(~token_valid, torch.nan).requires_grad_()
            varlen_value = value.masked_fill(~token_valid, torch.nan).requires_grad_()
            padded_query, padded_key, padded_value = (
                tensor.clone().requires_grad_() for tensor in (query, key, value)
            )
            if query_is_first_token:
                cu_seqlens_q = torch.arange(6, dtype=torch.int32, device=device)
                max_seqlen_q = 1
                output_valid = torch.ones_like(token_valid[:5])
            else:
                cu_seqlens_q = packed.cu_seqlens
                max_seqlen_q = _MAX_SEQ_LEN
                output_valid = token_valid
            varlen_attention(
                varlen_query,
                varlen_key,
                varlen_value,
                cu_seqlens_q,
                packed.cu_seqlens,
                max_seqlen_q,
                _MAX_SEQ_LEN,
            ).backward(grad_output)
            expected = padded_packed_attention(
                padded_query, padded_key, padded_value, packed, query_is_first_token
            )
            torch.where(output_valid, expected, 0.0).backward(grad_output)
            for name, actual_input, expected_input in (
                ("query", varlen_query, padded_query),
                ("key", varlen_key, padded_key),
                ("value", varlen_value, padded_value),
            ):
                with self.subTest(first_token=query_is_first_token, grad=name):
                    assert actual_input.grad is not None
                    assert expected_input.grad is not None
                    torch.testing.assert_close(
                        actual_input.grad,
                        expected_input.grad,
                        rtol=0,
                        atol=1e-4,
                    )

    def test_training_gradients_match_padded_forward(self) -> None:
        device = torch.device("cuda")
        data = _random_batch(0).to(device)
        config = PackedForwardConfig(_MAX_SEQ_LEN, _max_degree(data, "in"))
        kwargs = _encoder_kwargs(
            hid_dim=128,
            dropout_rate=0.0,
            anchor_based_input_embedding_dict=nn.ModuleDict(
                {_HOP: nn.Embedding(3, 128)}
            ),
        )
        padded, packed = _encoder_pair(config, _random_batch(0), kwargs)
        _assert_training_gradients_match(
            self, padded.to(device), packed.to(device), data, atol=1e-4
        )
        self.assertEqual(packed.packed_overflow_count, 0)

    def test_matches_padded_forward_with_one_host_sync(self) -> None:
        device = torch.device("cuda")
        batches = [_random_batch(seed).to(device) for seed in range(3)]
        config = PackedForwardConfig(
            _MAX_SEQ_LEN, max(_max_degree(data, "in") for data in batches)
        )
        kwargs = _encoder_kwargs(
            hid_dim=128,
            anchor_based_input_embedding_dict=nn.ModuleDict(
                {_HOP: nn.Embedding(3, 128)}
            ),
        )
        padded, packed = _encoder_pair(config, _random_batch(0), kwargs)
        padded, packed = padded.to(device), packed.to(device)
        with torch.no_grad():
            packed(batches[0], anchor_node_type=_USER)
            for data in batches:
                torch.cuda.synchronize()
                torch.cuda.set_sync_debug_mode("warn")
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        actual = packed(data, anchor_node_type=_USER)
                finally:
                    torch.cuda.set_sync_debug_mode("default")
                syncs = [w for w in caught if "synchroniz" in str(w.message)]
                self.assertEqual(len(syncs), 1, [str(w.message) for w in syncs])
                torch.testing.assert_close(
                    actual, padded(data, anchor_node_type=_USER), rtol=0, atol=1e-4
                )
        self.assertEqual(packed.packed_overflow_count, 0)


if __name__ == "__main__":
    absltest.main()
