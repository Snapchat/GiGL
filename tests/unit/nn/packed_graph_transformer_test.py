"""The packed ``GraphTransformerEncoder`` path matches the padded path."""

import copy
import itertools
import unittest
import warnings
from typing import Any, Callable, Literal, cast
from unittest import mock

import torch
import torch.nn as nn
from absl.testing import absltest
from parameterized import param, parameterized
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from gigl.nn.graph_transformer import (
    GraphTransformerEncoder,
    GraphTransformerEncoderLayer,
    PackedRelationEdges,
)
from gigl.nn.packed_attention import (
    PackedSequences,
    _padded_packed_attention,
    _varlen_attention,
    pack_sequences,
)
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.transforms.add_positional_encodings import AddHeteroHopDistanceEncoding
from gigl.transforms.static_khop import khop_layout
from tests.test_assets.test_case import TestCase

_USER = NodeType("user")
_ITEM = NodeType("item")
_FOLLOWS = EdgeType(_USER, Relation("follows"), _USER)
_BUYS = EdgeType(_USER, Relation("buys"), _ITEM)
_BOUGHT_BY = EdgeType(_ITEM, Relation("bought_by"), _USER)
_USER_EDGE_TYPES = [
    EdgeType(_USER, Relation(name), _USER)
    for name in ("is_friends_with", "bidi_engage", "bidi_imc")
]
_HOP = "hop_distance"
_MAX_SEQ_LEN = 24


def _random_batch(
    seed: int, hetero: bool = False, direction: Literal["in", "out"] = "in"
) -> HeteroData:
    """A random batch with duplicate edges, self loops and shared neighbours.

    Carries the ``hop_distance`` attribute the padded backend reads.
    """
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
    return AddHeteroHopDistanceEncoding(h_max=2, sampling_direction=direction)(data)


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
        sampling_direction="in",
        activation="swiglu",
    )
    kwargs.update(overrides)
    return kwargs


def _materialized_encoder(data: HeteroData, kwargs: dict) -> GraphTransformerEncoder:
    """An eval-mode encoder with its lazy modules materialized on ``data``."""
    encoder = GraphTransformerEncoder(**kwargs).eval()
    with torch.no_grad():
        encoder(data, anchor_node_type=_USER)
    return encoder


def _three_relation_batch(seed: int, num_edges: int = 80) -> HeteroData:
    """A user graph with three user-to-user edge types."""
    generator = torch.Generator().manual_seed(seed)
    data = HeteroData()
    data[_USER].x = torch.randn(60, 8, generator=generator)
    data[_USER].batch_size = 12
    for edge_type in _USER_EDGE_TYPES:
        data[edge_type.tuple_repr()].edge_index = torch.randint(
            0, 60, (2, num_edges), generator=generator
        )
    return AddHeteroHopDistanceEncoding(h_max=2, sampling_direction="in")(data)


def _sequence_lengths(data: HeteroData) -> torch.Tensor:
    """Each anchor's token count in the batch's 2-hop layout."""
    edge_index = torch.cat(
        [data[edge_type.tuple_repr()].edge_index for edge_type in _USER_EDGE_TYPES],
        dim=1,
    )
    return khop_layout(
        edge_index,
        num_nodes=data[_USER].x.size(0),
        anchor_indices=torch.arange(data[_USER].batch_size),
        hop_distance=2,
        max_seq_len=_MAX_SEQ_LEN,
        sampling_direction="in",
    )[2]


def _relation_encoder_kwargs(
    mode: str = "edge_type_attention", **overrides: object
) -> dict:
    kwargs = _encoder_kwargs()
    kwargs.update(
        edge_type_to_feat_dim_map={edge_type: 0 for edge_type in _USER_EDGE_TYPES},
        relation_message_mode=mode,
        num_layers=3,
        dropout_rate=0.0,
    )
    kwargs.update(overrides)
    return kwargs


def _relation_encoder(data: HeteroData, kwargs: dict) -> GraphTransformerEncoder:
    """An eval-mode encoder whose relation parameters, which start at no-op
    values, are random."""
    encoder = _materialized_encoder(data, kwargs)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, parameter in encoder.named_parameters():
            if "_relation_message" in name:
                parameter.add_(0.3 * torch.randn(parameter.shape, generator=generator))
    return encoder


def _run_packed(
    encoder: GraphTransformerEncoder, data: HeteroData, **forward_kwargs: object
) -> torch.Tensor:
    """Runs ``encoder`` on the packed path regardless of device."""
    with mock.patch.object(encoder, "_use_packed_backend", return_value=True):
        return encoder(data, anchor_node_type=_USER, **forward_kwargs)


def _gradients(
    forward: Callable[[], torch.Tensor], encoder: GraphTransformerEncoder
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The training loss and every parameter gradient of one ``forward``."""
    encoder.zero_grad()
    loss = forward().sin().sum()
    loss.backward()
    return loss.detach(), {
        name: parameter.grad.clone()
        for name, parameter in encoder.named_parameters()
        if parameter.grad is not None
    }


def _assert_training_gradients_match(
    test_case: TestCase,
    expected: tuple[torch.Tensor, dict[str, torch.Tensor]],
    actual: tuple[torch.Tensor, dict[str, torch.Tensor]],
    atol: float = 1e-6,
) -> None:
    """Training-mode losses and every parameter gradient agree."""
    torch.testing.assert_close(actual[0].cpu(), expected[0].cpu(), rtol=0, atol=atol)
    test_case.assertEqual(actual[1].keys(), expected[1].keys())
    test_case.assertTrue(expected[1])
    for name, gradient in expected[1].items():
        torch.testing.assert_close(
            actual[1][name].cpu(),
            gradient.cpu(),
            rtol=1e-5,
            atol=atol,
            msg=lambda message: f"{name}: {message}",
        )


class PackedPathTest(TestCase):
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
            ),
        ]
    )
    def test_matches_padded_path(
        self, _: str, hetero: bool = False, **overrides: object
    ) -> None:
        for seed, hop_distance in itertools.product(range(3), (1, 2)):
            kwargs = _encoder_kwargs(
                hetero=hetero, hop_distance=hop_distance, **overrides
            )
            data = _random_batch(
                seed, hetero, cast(Literal["in", "out"], kwargs["sampling_direction"])
            )
            data[_USER].random_walk_pe = torch.randn(50, 3)
            if hetero:
                data[_ITEM].random_walk_pe = torch.randn(20, 3)
            encoder = _materialized_encoder(data, kwargs)
            with self.subTest(seed=seed, hop=hop_distance), torch.no_grad():
                torch.testing.assert_close(
                    _run_packed(encoder, data),
                    encoder(data, anchor_node_type=_USER),
                    rtol=0,
                    atol=1e-6,
                )

    def test_matches_padded_path_with_explicit_anchors(self) -> None:
        data = _random_batch(0)
        encoder = _materialized_encoder(data, _encoder_kwargs())
        # Out of order, with a duplicate.
        anchor_node_ids = torch.tensor([7, 3, 3, 40, 0])
        with torch.no_grad():
            torch.testing.assert_close(
                _run_packed(encoder, data, anchor_node_ids=anchor_node_ids),
                encoder(data, anchor_node_type=_USER, anchor_node_ids=anchor_node_ids),
                rtol=0,
                atol=1e-6,
            )

    def test_empty_anchor_batch(self) -> None:
        data = _random_batch(0)
        encoder = _materialized_encoder(data, _encoder_kwargs())
        no_anchors = torch.empty(0, dtype=torch.long)
        with torch.no_grad():
            actual = _run_packed(encoder, data, anchor_node_ids=no_anchors)
            expected = encoder(data, anchor_node_type=_USER, anchor_node_ids=no_anchors)
        self.assertEqual(actual.shape, (0, 8))
        torch.testing.assert_close(actual, expected)

    def test_edgeless_batch(self) -> None:
        data = HeteroData()
        data[_USER].x = torch.randn(6, 8)
        data[_FOLLOWS.tuple_repr()].edge_index = torch.empty((2, 0), dtype=torch.long)
        data = AddHeteroHopDistanceEncoding(h_max=2, sampling_direction="in")(data)
        encoder = _materialized_encoder(data, _encoder_kwargs())
        with torch.no_grad():
            torch.testing.assert_close(
                _run_packed(encoder, data),
                encoder(data, anchor_node_type=_USER),
                rtol=0,
                atol=1e-6,
            )

    def test_training_gradients_match_padded_path(self) -> None:
        data = _random_batch(0)
        encoder = _materialized_encoder(data, _encoder_kwargs(dropout_rate=0.0)).train()
        _assert_training_gradients_match(
            self,
            _gradients(lambda: encoder(data, anchor_node_type=_USER), encoder),
            _gradients(lambda: _run_packed(encoder, data), encoder),
        )

    def test_packed_path_ignores_the_hop_distance_attribute(self) -> None:
        data = _random_batch(0)
        encoder = _materialized_encoder(data, _encoder_kwargs())
        expected = _run_packed(encoder, data)
        del data[_HOP]
        torch.testing.assert_close(_run_packed(encoder, data), expected)

    @parameterized.expand(
        [
            param("supported", supported=True),
            param(
                "neighbor_readout",
                readout_mode="anchor_neighbor_attention",
            ),
            param("prioritize_hop_order", prioritize_hop_order=True),
            param(
                "relation_message_linear",
                supported=True,
                relation_message_mode="edge_type_linear",
            ),
            param(
                "relation_message_attention",
                supported=True,
                relation_message_mode="edge_type_attention",
            ),
            param(
                "relation_attention_with_message",
                relation_attention_mode="edge_type_bilinear",
                relation_message_mode="edge_type_attention",
            ),
            param("relation_attention", relation_attention_mode="edge_type_bilinear"),
            param("anchor_bias", anchor_based_attention_bias_attr_names=[_HOP]),
            param("head_dim_not_multiple_of_8", num_heads=4),
            param("three_hops", hop_distance=3),
            param("pairwise_bias", pairwise_attention_bias_attr_names=[_HOP]),
            param(
                "other_token_input",
                anchor_based_input_attr_names=["other"],
                anchor_based_input_embedding_dict=None,
            ),
            param(
                "ppr",
                sequence_construction_method="ppr",
                sampling_direction="out",
            ),
        ]
    )
    def test_supports_packed_backend(
        self, _: str, supported: bool = False, **overrides: Any
    ) -> None:
        encoder = GraphTransformerEncoder(**_encoder_kwargs(**overrides))
        self.assertEqual(encoder._supports_packed_backend, supported)

    def test_packed_path_is_used_on_cuda_unless_training_with_attention_dropout(
        self,
    ) -> None:
        encoder = GraphTransformerEncoder(**_encoder_kwargs(attention_dropout_rate=0.1))
        self.assertTrue(encoder.eval()._use_packed_backend(torch.device("cuda")))
        self.assertFalse(encoder.train()._use_packed_backend(torch.device("cuda")))
        self.assertFalse(encoder.eval()._use_packed_backend(torch.device("cpu")))

    def test_forward_accepts_a_string_device(self) -> None:
        data = _random_batch(0)
        encoder = _materialized_encoder(data, _encoder_kwargs())
        with torch.no_grad():
            self.assertEqual(
                encoder(data, anchor_node_type=_USER, device="cpu").shape, (12, 8)
            )

    def test_layer_rejects_attention_dropout_in_training(self) -> None:
        layer = GraphTransformerEncoderLayer(
            model_dim=16, num_heads=2, feedforward_dim=32, attention_dropout_rate=0.1
        )
        packed = PackedSequences(torch.tensor([0, 2], dtype=torch.int32), 2)
        layer.eval().forward_packed(torch.randn(2, 16), packed)
        with self.assertRaises(ValueError):
            layer.train().forward_packed(torch.randn(2, 16), packed)

    def test_pack_sequences_layout(self) -> None:
        flat_index, packed = pack_sequences(torch.tensor([2, 1, 3]), 3)
        self.assertEqual(flat_index.tolist(), [0, 1, 3, 6, 7, 8])
        self.assertEqual(packed.cu_seqlens.tolist(), [0, 2, 3, 6])
        self.assertEqual(packed.cu_seqlens.dtype, torch.int32)
        self.assertEqual(packed.max_seq_len, 3)


class PackedRelationMessageTest(TestCase):
    def test_batches_mix_truncated_and_short_sequences(self) -> None:
        for seed in range(3):
            lengths = _sequence_lengths(_three_relation_batch(seed))
            self.assertTrue((lengths == _MAX_SEQ_LEN).any(), lengths)
            self.assertTrue((lengths < _MAX_SEQ_LEN).any(), lengths)

    @parameterized.expand(
        [
            param("attention"),
            param("linear", mode="edge_type_linear"),
            param("one_layer", num_layers=1),
            param("one_hop", hop_distance=1),
            param("out_direction", sampling_direction="out"),
            param("sparse_graph", num_edges=20),
        ]
    )
    def test_matches_padded_path(
        self,
        _: str,
        mode: str = "edge_type_attention",
        num_edges: int = 80,
        **overrides: object,
    ) -> None:
        for seed in range(3):
            data = _three_relation_batch(seed, num_edges)
            if overrides.get("sampling_direction") == "out":
                data = AddHeteroHopDistanceEncoding(h_max=2, sampling_direction="out")(
                    data
                )
            encoder = _relation_encoder(
                data, _relation_encoder_kwargs(mode, **overrides)
            )
            no_messages = copy.deepcopy(encoder)
            for name, parameter in no_messages.named_parameters():
                if "_relation_message_matrices" in name:
                    parameter.data.zero_()
            with self.subTest(seed=seed), torch.no_grad():
                expected = encoder(data, anchor_node_type=_USER)
                torch.testing.assert_close(
                    _run_packed(encoder, data), expected, rtol=0, atol=1e-6
                )
                # The relation channel is what the comparison above exercises.
                self.assertGreater(
                    (expected - no_messages(data, anchor_node_type=_USER)).abs().max(),
                    1e-3,
                )

    def test_matches_padded_path_with_explicit_anchors(self) -> None:
        data = _three_relation_batch(0)
        encoder = _relation_encoder(data, _relation_encoder_kwargs())
        anchor_node_ids = torch.tensor([7, 3, 3, 40, 0])
        with torch.no_grad():
            torch.testing.assert_close(
                _run_packed(encoder, data, anchor_node_ids=anchor_node_ids),
                encoder(data, anchor_node_type=_USER, anchor_node_ids=anchor_node_ids),
                rtol=0,
                atol=1e-6,
            )

    def test_edgeless_batch(self) -> None:
        data = _three_relation_batch(0, num_edges=0)
        encoder = _relation_encoder(data, _relation_encoder_kwargs())
        with torch.no_grad():
            torch.testing.assert_close(
                _run_packed(encoder, data),
                encoder(data, anchor_node_type=_USER),
                rtol=0,
                atol=1e-6,
            )

    @parameterized.expand(
        [
            param("attention", mode="edge_type_attention"),
            param("linear", mode="edge_type_linear"),
        ]
    )
    def test_training_gradients_match_padded_path(self, _: str, mode: str) -> None:
        data = _three_relation_batch(0)
        encoder = _relation_encoder(data, _relation_encoder_kwargs(mode)).train()
        expected = _gradients(lambda: encoder(data, anchor_node_type=_USER), encoder)
        actual = _gradients(lambda: _run_packed(encoder, data), encoder)
        # The relation parameters of every layer are trained.
        self.assertTrue(
            all(
                actual[1][f"_encoder_layers.{layer}._relation_message_matrices"]
                .abs()
                .max()
                > 0
                for layer in range(3)
            )
        )
        _assert_training_gradients_match(self, expected, actual)

    def test_training_keeps_relation_parameters_in_the_graph_without_edges(
        self,
    ) -> None:
        data = _three_relation_batch(0, num_edges=0)
        encoder = _relation_encoder(data, _relation_encoder_kwargs()).train()
        _, gradients = _gradients(lambda: _run_packed(encoder, data), encoder)
        for layer in range(3):
            self.assertEqual(
                gradients[f"_encoder_layers.{layer}._relation_message_matrices"]
                .abs()
                .max(),
                0,
            )

    def test_per_layer_checkpointing_matches_plain_gradients(self) -> None:
        class CheckpointedEncoder(GraphTransformerEncoder):
            layer_calls = 0

            def _run_packed_layer(
                self,
                layer: GraphTransformerEncoderLayer,
                x: torch.Tensor,
                packed: PackedSequences,
                first_token_output: bool,
                relation_edges: PackedRelationEdges | None,
            ) -> torch.Tensor:
                def run(x: torch.Tensor) -> torch.Tensor:
                    self.layer_calls += 1
                    return super(CheckpointedEncoder, self)._run_packed_layer(
                        layer, x, packed, first_token_output, relation_edges
                    )

                return checkpoint(run, x, use_reentrant=False)

        data = _three_relation_batch(0)
        encoder = _relation_encoder(data, _relation_encoder_kwargs()).train()
        checkpointed = CheckpointedEncoder(**_relation_encoder_kwargs()).train()
        checkpointed.load_state_dict(encoder.state_dict())
        expected = _gradients(lambda: _run_packed(encoder, data), encoder)
        actual = _gradients(lambda: _run_packed(checkpointed, data), checkpointed)
        # Each of the 3 layers ran in the forward and again in the backward.
        self.assertEqual(checkpointed.layer_calls, 6)
        _assert_training_gradients_match(self, expected, actual, atol=1e-7)

    def test_layer_requires_relation_edges_for_relation_messages(self) -> None:
        layer = GraphTransformerEncoderLayer(
            model_dim=16,
            num_heads=2,
            feedforward_dim=32,
            relation_message_mode="edge_type_attention",
            num_relations=2,
        )
        packed = PackedSequences(torch.tensor([0, 2], dtype=torch.int32), 2)
        with self.assertRaises(ValueError):
            layer.eval().forward_packed(torch.randn(2, 16), packed)

    def test_layer_rejects_relation_attention_bias(self) -> None:
        layer = GraphTransformerEncoderLayer(
            model_dim=16,
            num_heads=2,
            feedforward_dim=32,
            relation_attention_mode="edge_type_bilinear",
            num_relations=2,
        )
        packed = PackedSequences(torch.tensor([0, 2], dtype=torch.int32), 2)
        with self.assertRaises(ValueError):
            layer.eval().forward_packed(torch.randn(2, 16), packed)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class PackedPathCudaTest(TestCase):
    def test_compiled_forward_packed_matches_eager(self) -> None:
        layer = (
            GraphTransformerEncoderLayer(
                model_dim=16,
                num_heads=2,
                feedforward_dim=32,
                dropout_rate=0.0,
                activation="swiglu",
            )
            .eval()
            .to("cuda")
        )
        _, packed = pack_sequences(torch.tensor([3, 1, 5, 2], device="cuda"), 6)
        x = torch.randn(11, 16, device="cuda")
        compiled = torch.compile(layer.forward_packed)
        with torch.no_grad():
            for first_token_output in (False, True):
                torch.testing.assert_close(
                    compiled(x, packed, first_token_output),
                    layer.forward_packed(x, packed, first_token_output),
                )

    def test_varlen_attention_matches_padded_packed_attention(self) -> None:
        device = torch.device("cuda")
        num_sequences, num_heads, head_dim = 6, 2, 64
        lengths = torch.tensor([5, 1, 24, 3, 12, 9], device=device)
        num_tokens = int(lengths.sum())
        _, packed = pack_sequences(lengths, _MAX_SEQ_LEN)
        generator = torch.Generator().manual_seed(0)
        for dtype, atol in ((torch.float32, 1e-4), (torch.bfloat16, 5e-2)):
            for query_is_first_token in (False, True):
                num_queries = num_sequences if query_is_first_token else num_tokens
                query, key, value = (
                    torch.randn(rows, num_heads, head_dim, generator=generator).to(
                        device, dtype
                    )
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
                actual = _varlen_attention(
                    query,
                    key,
                    value,
                    cu_seqlens_q,
                    packed.cu_seqlens,
                    max_seqlen_q,
                    _MAX_SEQ_LEN,
                )
                with self.subTest(dtype=dtype, first_token=query_is_first_token):
                    torch.testing.assert_close(
                        actual.float().cpu(), expected.float().cpu(), rtol=0, atol=atol
                    )

    def test_varlen_attention_gradients_match_padded_packed_attention(self) -> None:
        device = torch.device("cuda")
        num_heads, head_dim = 2, 64
        lengths = torch.tensor([5, 1, 24, 3, 12], device=device)
        num_tokens = int(lengths.sum())
        _, packed = pack_sequences(lengths, _MAX_SEQ_LEN)
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

    def test_cuda_training_gradients_match_cpu(self) -> None:
        data = _random_batch(0)
        cuda_data = _random_batch(0).to("cuda")
        kwargs = _encoder_kwargs(
            hid_dim=128,
            dropout_rate=0.0,
            anchor_based_input_embedding_dict=nn.ModuleDict(
                {_HOP: nn.Embedding(3, 128)}
            ),
        )
        cpu_encoder = _materialized_encoder(data, kwargs).train()
        cuda_encoder = copy.deepcopy(cpu_encoder).to("cuda")
        _assert_training_gradients_match(
            self,
            _gradients(lambda: cpu_encoder(data, anchor_node_type=_USER), cpu_encoder),
            _gradients(
                lambda: cuda_encoder(cuda_data, anchor_node_type=_USER), cuda_encoder
            ),
            atol=1e-4,
        )

    def test_cuda_forward_matches_cpu_with_two_host_syncs(self) -> None:
        device = torch.device("cuda")
        batches = [_random_batch(seed) for seed in range(3)]
        warmup_data = _random_batch(0).to(device)
        cuda_batches = [_random_batch(seed).to(device) for seed in range(3)]
        kwargs = _encoder_kwargs(
            hid_dim=128,
            anchor_based_input_embedding_dict=nn.ModuleDict(
                {_HOP: nn.Embedding(3, 128)}
            ),
        )
        cpu_encoder = _materialized_encoder(batches[0], kwargs)
        cuda_encoder = copy.deepcopy(cpu_encoder).to(device)
        with torch.no_grad():
            cuda_encoder(warmup_data, anchor_node_type=_USER)
            for data, cuda_data in zip(batches, cuda_batches):
                torch.cuda.synchronize()
                torch.cuda.set_sync_debug_mode("warn")
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        actual = cuda_encoder(cuda_data, anchor_node_type=_USER)
                finally:
                    torch.cuda.set_sync_debug_mode("default")
                syncs = [w for w in caught if "synchroniz" in str(w.message)]
                self.assertEqual(len(syncs), 2, [str(w.message) for w in syncs])
                torch.testing.assert_close(
                    actual.cpu(),
                    cpu_encoder(data, anchor_node_type=_USER),
                    rtol=0,
                    atol=1e-4,
                )


if __name__ == "__main__":
    absltest.main()
