"""The packed ``GraphTransformerEncoder`` path matches the padded path."""

import copy
import itertools
import unittest
import warnings
from typing import Any, Callable, Literal, Optional, cast
from unittest import mock

import torch
import torch.nn as nn
from absl.testing import absltest
from parameterized import param, parameterized
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

import gigl.nn.graph_transformer
from gigl.nn.graph_transformer import (
    GraphTransformerEncoder,
    GraphTransformerEncoderLayer,
    PackedRelationEdges,
)
from gigl.nn.packed_attention import PackedAttentionBias
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.transforms.add_positional_encodings import AddHeteroHopDistanceEncoding
from gigl.transforms.static_khop import (
    KhopLayoutTooLargeError,
    PackedLayout,
    khop_layout,
    pack_sequences,
)
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
_NEIGHBOR_READOUT = "anchor_neighbor_attention"
_READOUT_MODES = ["anchor_only", _NEIGHBOR_READOUT]


def _random_batch(
    seed: int,
    hetero: bool = False,
    direction: Literal["in", "out"] = "in",
    h_max: int = 2,
) -> HeteroData:
    """A random batch with duplicate edges, self loops and shared neighbours.

    Carries the ``hop_distance`` attribute the padded backend reads and
    ``hop_times_two``, a second sparse attribute with every hop value doubled.
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
    data = AddHeteroHopDistanceEncoding(h_max=h_max, sampling_direction=direction)(data)
    hop = data[_HOP]
    data["hop_times_two"] = torch.sparse_csr_tensor(
        hop.crow_indices(), hop.col_indices(), hop.values() * 2, size=hop.shape
    )
    return data


def _oversized_layout_batch() -> HeteroData:
    """One anchor behind a hub with more neighbours than the layout budget allows at two hops."""
    num_nodes = 6000
    data = HeteroData()
    data[_USER].x = torch.randn(num_nodes, 8)
    data[_USER].batch_size = 1
    data[_FOLLOWS.tuple_repr()].edge_index = torch.stack(
        [torch.arange(1, num_nodes), torch.zeros(num_nodes - 1, dtype=torch.long)]
    )
    return AddHeteroHopDistanceEncoding(h_max=2, sampling_direction="in")(data)


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
        sampling_direction="in",
        activation="swiglu",
    )
    kwargs.update(overrides)
    if "anchor_based_input_embedding_dict" not in overrides:
        kwargs["anchor_based_input_embedding_dict"] = nn.ModuleDict(
            {_HOP: nn.Embedding(kwargs["hop_distance"] + 1, kwargs["hid_dim"])}
        )
    return kwargs


def _materialized_encoder(data: HeteroData, kwargs: dict) -> GraphTransformerEncoder:
    """An eval-mode encoder with its lazy modules materialized on ``data``."""
    encoder = GraphTransformerEncoder(**kwargs).eval()
    with torch.no_grad():
        encoder(data, anchor_node_type=_USER)
    return encoder


def _three_relation_batch(
    seed: int, num_edges: int = 80, num_relations: int = 3
) -> HeteroData:
    """A user graph with ``num_relations`` of three user-to-user edge types."""
    generator = torch.Generator().manual_seed(seed)
    data = HeteroData()
    data[_USER].x = torch.randn(60, 8, generator=generator)
    data[_USER].batch_size = 12
    for edge_type in _USER_EDGE_TYPES[:num_relations]:
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
    )[1]


def _relation_encoder_kwargs(
    mode: str = "edge_type_attention", num_relations: int = 3, **overrides: object
) -> dict:
    kwargs = _encoder_kwargs()
    kwargs.update(
        edge_type_to_feat_dim_map={
            edge_type: 0 for edge_type in _USER_EDGE_TYPES[:num_relations]
        },
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
    torch.testing.assert_close(actual[0].cpu(), expected[0].cpu(), rtol=1e-6, atol=atol)
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
    def test_anchor_only_readout_shape_does_not_depend_on_layer_count(self) -> None:
        # Every sequence has length 1 on an edgeless batch.
        data = HeteroData()
        data[_USER].x = torch.randn(20, 8)
        data[_USER].batch_size = 12
        data[_FOLLOWS.tuple_repr()].edge_index = torch.empty(2, 0, dtype=torch.long)
        data = AddHeteroHopDistanceEncoding(h_max=2, sampling_direction="in")(data)
        for num_layers in (0, 2):
            with self.subTest(num_layers=num_layers):
                encoder = _materialized_encoder(
                    data, _encoder_kwargs(num_layers=num_layers)
                )
                layer_outputs = []
                run_layers = encoder._run_packed_layers

                def record(*args: Any) -> torch.Tensor:
                    layer_outputs.append(run_layers(*args))
                    return layer_outputs[-1]

                with (
                    mock.patch.object(encoder, "_run_packed_layers", record),
                    torch.no_grad(),
                ):
                    output = _run_packed(encoder, data)
                self.assertEqual([o.shape for o in layer_outputs], [(12, 16)])
                self.assertEqual(output.shape, (12, 8))

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
            param("three_hops", hop_distances=(3,)),
            param("four_hops", hop_distances=(4,)),
            param("hop_order", prioritize_hop_order=True, max_seq_len=10),
            param(
                "hop_order_three_hops",
                hop_distances=(3,),
                prioritize_hop_order=True,
                max_seq_len=10,
            ),
            param(
                "hop_order_out_direction",
                prioritize_hop_order=True,
                max_seq_len=10,
                sampling_direction="out",
            ),
            param("neighbor_readout", readout_mode=_NEIGHBOR_READOUT),
            param(
                "neighbor_readout_three_hops",
                hop_distances=(3,),
                readout_mode=_NEIGHBOR_READOUT,
            ),
            param(
                "neighbor_readout_out_direction",
                readout_mode=_NEIGHBOR_READOUT,
                sampling_direction="out",
            ),
            param(
                "neighbor_readout_hop_order",
                readout_mode=_NEIGHBOR_READOUT,
                prioritize_hop_order=True,
                max_seq_len=10,
            ),
            param(
                "neighbor_readout_truncated",
                readout_mode=_NEIGHBOR_READOUT,
                max_seq_len=6,
            ),
            param(
                "neighbor_readout_no_layers",
                readout_mode=_NEIGHBOR_READOUT,
                num_layers=0,
            ),
            param(
                "anchor_bias",
                anchor_based_attention_bias_attr_names=[_HOP],
            ),
            param(
                "two_anchor_biases",
                anchor_based_attention_bias_attr_names=[_HOP, "hop_times_two"],
            ),
            param(
                "anchor_bias_neighbor_readout",
                anchor_based_attention_bias_attr_names=[_HOP],
                readout_mode=_NEIGHBOR_READOUT,
            ),
            param(
                "two_continuous_inputs",
                anchor_based_input_attr_names=[_HOP, "hop_times_two"],
                anchor_based_input_embedding_dict=None,
            ),
            param(
                "relation_message_with_neighbor_readout",
                relation_message_mode="edge_type_attention",
                readout_mode=_NEIGHBOR_READOUT,
            ),
            param("head_dim_10", hid_dim=20, num_heads=2),
        ]
    )
    def test_matches_padded_path(
        self,
        _: str,
        hetero: bool = False,
        hop_distances: tuple[int, ...] = (1, 2),
        **overrides: Any,
    ) -> None:
        for seed, hop_distance in itertools.product(range(3), hop_distances):
            kwargs = _encoder_kwargs(
                hetero=hetero, hop_distance=hop_distance, **overrides
            )
            data = _random_batch(
                seed,
                hetero,
                cast(Literal["in", "out"], kwargs["sampling_direction"]),
                h_max=hop_distance,
            )
            data[_USER].random_walk_pe = torch.randn(50, 3)
            if hetero:
                data[_ITEM].random_walk_pe = torch.randn(20, 3)
            encoder = _materialized_encoder(data, kwargs)
            self.assertTrue(encoder._supports_packed_backend)
            with self.subTest(seed=seed, hop=hop_distance), torch.no_grad():
                torch.testing.assert_close(
                    _run_packed(encoder, data),
                    encoder(data, anchor_node_type=_USER),
                    rtol=0,
                    atol=1e-6,
                )

    @parameterized.expand(_READOUT_MODES)
    def test_matches_padded_path_with_explicit_anchors(self, readout_mode: str) -> None:
        data = _random_batch(0)
        encoder = _materialized_encoder(
            data, _encoder_kwargs(readout_mode=readout_mode)
        )
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

    @parameterized.expand(_READOUT_MODES)
    def test_edgeless_batch(self, readout_mode: str) -> None:
        data = HeteroData()
        data[_USER].x = torch.randn(6, 8)
        data[_FOLLOWS.tuple_repr()].edge_index = torch.empty((2, 0), dtype=torch.long)
        data = AddHeteroHopDistanceEncoding(h_max=2, sampling_direction="in")(data)
        encoder = _materialized_encoder(
            data, _encoder_kwargs(readout_mode=readout_mode)
        )
        with torch.no_grad():
            torch.testing.assert_close(
                _run_packed(encoder, data),
                encoder(data, anchor_node_type=_USER),
                rtol=0,
                atol=1e-6,
            )

    @parameterized.expand(_READOUT_MODES)
    def test_anchors_without_neighbours_match_padded_path(
        self, readout_mode: str
    ) -> None:
        # Nodes 3 and 4 have no edges, so their sequences hold only the anchor.
        data = HeteroData()
        data[_USER].x = torch.randn(6, 8)
        data[_FOLLOWS.tuple_repr()].edge_index = torch.tensor([[1, 2, 2], [0, 0, 1]])
        data = AddHeteroHopDistanceEncoding(h_max=2, sampling_direction="in")(data)
        encoder = _materialized_encoder(
            data, _encoder_kwargs(readout_mode=readout_mode)
        )
        anchor_node_ids = torch.tensor([0, 3, 1, 4])
        with torch.no_grad():
            actual = _run_packed(encoder, data, anchor_node_ids=anchor_node_ids)
            expected = encoder(
                data, anchor_node_type=_USER, anchor_node_ids=anchor_node_ids
            )
        self.assertTrue(actual.isfinite().all())
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)

    @parameterized.expand(
        [
            param("default"),
            param("one_anchor", anchors=[3], max_seq_len=256),
            param("two_anchors", anchors=[3, 40], max_seq_len=256),
            param("neighbor_readout", readout_mode=_NEIGHBOR_READOUT),
            param(
                "neighbor_readout_hop_order",
                readout_mode=_NEIGHBOR_READOUT,
                prioritize_hop_order=True,
                max_seq_len=10,
            ),
            param(
                "anchor_bias",
                anchor_based_attention_bias_attr_names=[_HOP],
            ),
            param(
                "two_anchor_biases",
                anchor_based_attention_bias_attr_names=[_HOP, "hop_times_two"],
            ),
            param(
                "anchor_bias_neighbor_readout",
                anchor_based_attention_bias_attr_names=[_HOP],
                readout_mode=_NEIGHBOR_READOUT,
            ),
            param(
                "two_continuous_inputs",
                anchor_based_input_attr_names=[_HOP, "hop_times_two"],
                anchor_based_input_embedding_dict=None,
            ),
        ]
    )
    def test_training_gradients_match_padded_path(
        self, _: str, anchors: Optional[list[int]] = None, **overrides: Any
    ) -> None:
        data = _random_batch(0)
        anchor_node_ids = None
        if anchors is not None:
            anchor_node_ids = torch.tensor(anchors)
        encoder = _materialized_encoder(
            data, _encoder_kwargs(dropout_rate=0.0, **overrides)
        ).train()
        _assert_training_gradients_match(
            self,
            _gradients(
                lambda: encoder(
                    data, anchor_node_type=_USER, anchor_node_ids=anchor_node_ids
                ),
                encoder,
            ),
            _gradients(
                lambda: _run_packed(encoder, data, anchor_node_ids=anchor_node_ids),
                encoder,
            ),
        )

    def test_packed_path_runs_without_the_hop_distance_attribute(self) -> None:
        data = _random_batch(0)
        encoder = _materialized_encoder(data, _encoder_kwargs())
        expected = _run_packed(encoder, data)
        del data[_HOP]
        torch.testing.assert_close(_run_packed(encoder, data), expected)
        with self.assertRaisesRegex(ValueError, "apply the corresponding transform"):
            encoder(data, anchor_node_type=_USER)

    def test_packed_path_raises_when_an_anchor_attribute_is_missing(self) -> None:
        data = _random_batch(0)
        encoder = _materialized_encoder(
            data, _encoder_kwargs(anchor_based_attention_bias_attr_names=[_HOP])
        )
        encoder._anchor_based_attention_bias_attr_names = ["absent"]
        with self.assertRaisesRegex(ValueError, "absent"):
            _run_packed(encoder, data)

    def test_fallback_batch_without_the_hop_distance_attribute_raises(self) -> None:
        data = _oversized_layout_batch()
        encoder = _materialized_encoder(data, _encoder_kwargs())
        del data[_HOP]
        with self.assertRaisesRegex(ValueError, "AddHeteroHopDistanceEncoding"):
            _run_packed(encoder, data)
        # Without the hop input the same batch falls back and runs.
        encoder = _materialized_encoder(
            data,
            _encoder_kwargs(
                anchor_based_input_attr_names=None,
                anchor_based_input_embedding_dict=None,
            ),
        )
        self.assertEqual(_run_packed(encoder, data).shape, (1, 8))

    def test_fallback_batch_without_the_hop_distance_bias_attribute_raises(
        self,
    ) -> None:
        data = _oversized_layout_batch()
        encoder = _materialized_encoder(
            data,
            _encoder_kwargs(
                anchor_based_input_attr_names=None,
                anchor_based_input_embedding_dict=None,
                anchor_based_attention_bias_attr_names=[_HOP],
            ),
        )
        del data[_HOP]
        with self.assertRaisesRegex(ValueError, "AddHeteroHopDistanceEncoding"):
            _run_packed(encoder, data)

    def test_packed_path_uses_the_layout_hop_not_the_attribute(self) -> None:
        # A transform with a smaller h_max stores different hops than the layout
        # derives; the packed path follows the layout.
        data = _random_batch(0)
        encoder = _materialized_encoder(data, _encoder_kwargs())
        expected = _run_packed(encoder, data)
        data[_HOP] = AddHeteroHopDistanceEncoding(h_max=1, sampling_direction="in")(
            data
        )[_HOP]
        torch.testing.assert_close(_run_packed(encoder, data), expected)

    def test_hop_embedding_must_cover_every_packed_hop(self) -> None:
        too_small = {_HOP: nn.Embedding(2, 16)}
        with self.assertRaisesRegex(ValueError, "hop_distance=2"):
            GraphTransformerEncoder(
                **_encoder_kwargs(
                    anchor_based_input_embedding_dict=nn.ModuleDict(too_small)
                )
            )
        # A config that always runs padded reads the batch's own hop values.
        encoder = GraphTransformerEncoder(
            **_encoder_kwargs(
                pairwise_attention_bias_attr_names=["x"],
                anchor_based_input_embedding_dict=nn.ModuleDict(too_small),
            )
        )
        self.assertFalse(encoder._supports_packed_backend)

    @parameterized.expand(
        [
            param("supported", supported=True),
            param(
                "neighbor_readout",
                readout_mode="anchor_neighbor_attention",
                supported=True,
            ),
            param("prioritize_hop_order", prioritize_hop_order=True, supported=True),
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
            param(
                "anchor_bias",
                anchor_based_attention_bias_attr_names=[_HOP],
                supported=True,
            ),
            param(
                "two_anchor_biases",
                anchor_based_attention_bias_attr_names=[_HOP, "hop_times_two"],
                supported=True,
            ),
            param("head_dim_4", num_heads=4, supported=True),
            param("head_dim_10", hid_dim=20, supported=True),
            param("five_hops", hop_distance=5),
            param("attention_dropout", attention_dropout_rate=0.1),
            param("pairwise_bias", pairwise_attention_bias_attr_names=[_HOP]),
            param(
                "other_token_input",
                anchor_based_input_attr_names=["other"],
                anchor_based_input_embedding_dict=None,
                supported=True,
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

    def test_packed_path_selection_does_not_depend_on_training_mode(self) -> None:
        encoder = GraphTransformerEncoder(**_encoder_kwargs())
        for training in (True, False):
            encoder.train(training)
            self.assertTrue(encoder._use_packed_backend(torch.device("cuda")))
            self.assertFalse(encoder._use_packed_backend(torch.device("cpu")))

    def test_constructor_rejects_invalid_sequence_bounds(self) -> None:
        with self.assertRaises(ValueError):
            GraphTransformerEncoder(**_encoder_kwargs(max_seq_len=0))
        with self.assertRaises(ValueError):
            GraphTransformerEncoder(**_encoder_kwargs(hop_distance=-1))

    def test_batch_with_an_oversized_layout_runs_on_the_padded_path(self) -> None:
        data = _oversized_layout_batch()
        encoder = _materialized_encoder(data, _encoder_kwargs(sampling_direction="in"))
        with (
            torch.no_grad(),
            self.assertLogs(gigl.nn.graph_transformer.logger.logger, "WARNING") as logs,
        ):
            outputs = [_run_packed(encoder, data) for _ in range(5)]
            expected = encoder(data, anchor_node_type=_USER)
        # The fallback is counted per batch and warned about at powers of two.
        self.assertEqual(encoder._packed_fallback_count, 5)
        self.assertEqual(len(logs.records), 3, logs.output)
        for record, count in zip(logs.records, (1, 2, 4)):
            self.assertIn(f"({count} so far)", record.getMessage())
        for output in outputs:
            torch.testing.assert_close(output, expected)

    def test_int32_edge_index(self) -> None:
        data = _random_batch(0)
        encoder = _materialized_encoder(data, _encoder_kwargs())
        with torch.no_grad():
            expected = _run_packed(encoder, data)
            edge_store = data[_FOLLOWS.tuple_repr()]
            edge_store.edge_index = edge_store.edge_index.int()
            torch.testing.assert_close(_run_packed(encoder, data), expected)

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
        packed = pack_sequences(torch.tensor([2]), 2)
        layer.eval().forward_packed(torch.randn(2, 16), packed)
        with self.assertRaises(ValueError):
            layer.train().forward_packed(torch.randn(2, 16), packed)

    @parameterized.expand(
        [
            param("without_fallback", fall_back_at=(), settled_from=2),
            param("with_fallback", fall_back_at=(3,), settled_from=5),
            param(
                "neighbor_readout",
                fall_back_at=(),
                settled_from=2,
                readout_mode=_NEIGHBOR_READOUT,
            ),
        ]
    )
    def test_whole_encoder_compile_does_not_recompile_across_batches(
        self,
        _: str,
        fall_back_at: tuple[int, ...],
        settled_from: int,
        readout_mode: str = "anchor_only",
    ) -> None:
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        encoder = _materialized_encoder(
            _random_batch(0), _encoder_kwargs(readout_mode=readout_mode)
        )
        real_pack_khop_layout = gigl.nn.graph_transformer.pack_khop_layout
        fall_back = {"now": False}

        def pack_khop_layout_or_too_large(*args: Any, **kwargs: Any) -> Any:
            if fall_back["now"]:
                raise KhopLayoutTooLargeError("too many candidates")
            return real_pack_khop_layout(*args, **kwargs)

        graph_counts = []
        with (
            mock.patch.object(encoder, "_use_packed_backend", return_value=True),
            mock.patch.object(
                gigl.nn.graph_transformer,
                "pack_khop_layout",
                side_effect=pack_khop_layout_or_too_large,
            ),
        ):
            compiled = torch.compile(encoder, backend="eager")
            for seed in range(8):
                fall_back["now"] = seed in fall_back_at
                with torch.no_grad():
                    compiled(_random_batch(seed), anchor_node_type=_USER)
                graph_counts.append(
                    torch._dynamo.utils.counters["stats"]["unique_graphs"]
                )
        self.assertEqual(encoder._packed_fallback_count, len(fall_back_at))
        # The first batches (and a fallback batch) may compile new graphs; the
        # batches after them reuse the graphs.
        settled = graph_counts[settled_from:]
        self.assertEqual(settled, [settled[0]] * len(settled), graph_counts)


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
        **overrides: Any,
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

    @parameterized.expand([param("first", 0), param("middle", 1), param("last", 2)])
    def test_matches_padded_path_when_a_relation_has_no_edges(
        self, _: str, empty_relation: int
    ) -> None:
        data = _three_relation_batch(0)
        data[_USER_EDGE_TYPES[empty_relation].tuple_repr()].edge_index = torch.empty(
            (2, 0), dtype=torch.long
        )
        data = AddHeteroHopDistanceEncoding(h_max=2, sampling_direction="in")(data)
        encoder = _relation_encoder(data, _relation_encoder_kwargs())
        with torch.no_grad():
            torch.testing.assert_close(
                _run_packed(encoder, data),
                encoder(data, anchor_node_type=_USER),
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
        _assert_training_gradients_match(self, expected, actual, atol=1e-5)

    @parameterized.expand(
        [
            param("no_edges", num_edges=0),
            # Every edge into an anchor is dropped: with "out" sampling and one hop
            # the expanded endpoint of such an edge is a last-hop token.
            param("no_edges_into_anchors", sampling_direction="out", hop_distance=1),
        ]
    )
    def test_training_gradients_match_padded_path_when_edges_are_filtered(
        self, _: str, num_edges: int = 80, **overrides: Any
    ) -> None:
        data = _three_relation_batch(0, num_edges)
        if overrides:
            data = AddHeteroHopDistanceEncoding(h_max=1, sampling_direction="out")(data)
        encoder = _relation_encoder(data, _relation_encoder_kwargs(**overrides)).train()
        _assert_training_gradients_match(
            self,
            _gradients(lambda: encoder(data, anchor_node_type=_USER), encoder),
            _gradients(lambda: _run_packed(encoder, data), encoder),
            atol=1e-5,
        )

    @parameterized.expand(
        [
            param("tiny_max_seq_len", max_seq_len=2),
            param("max_seq_len_one", max_seq_len=1),
        ]
    )
    def test_matches_padded_path_with_most_edges_truncated(
        self, _: str, **overrides: Any
    ) -> None:
        data = _three_relation_batch(0)
        encoder = _relation_encoder(data, _relation_encoder_kwargs(**overrides))
        with torch.no_grad():
            torch.testing.assert_close(
                _run_packed(encoder, data),
                encoder(data, anchor_node_type=_USER),
                rtol=0,
                atol=1e-6,
            )

    def test_matches_padded_path_on_a_hetero_graph(self) -> None:
        # Edge types between different node types, and a relation subset that
        # leaves a relation absent from the batch.
        data = _random_batch(0, hetero=True)
        kwargs = _encoder_kwargs(
            hetero=True, relation_message_mode="edge_type_attention"
        )
        encoder = _relation_encoder(data, kwargs)
        with torch.no_grad():
            torch.testing.assert_close(
                _run_packed(encoder, data),
                encoder(data, anchor_node_type=_USER),
                rtol=0,
                atol=1e-6,
            )

    def test_per_layer_checkpointing_matches_plain_gradients(self) -> None:
        class CheckpointedEncoder(GraphTransformerEncoder):
            layer_calls = 0

            def _run_packed_layer(
                self,
                layer: GraphTransformerEncoderLayer,
                x: torch.Tensor,
                packed: PackedLayout,
                first_token_output: bool,
                relation_edges: PackedRelationEdges | None,
                bias: PackedAttentionBias | None = None,
            ) -> torch.Tensor:
                def run(x: torch.Tensor) -> torch.Tensor:
                    self.layer_calls += 1
                    return super(CheckpointedEncoder, self)._run_packed_layer(
                        layer, x, packed, first_token_output, relation_edges, bias
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
        packed = pack_sequences(torch.tensor([2]), 2)
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
        packed = pack_sequences(torch.tensor([2]), 2)
        with self.assertRaises(ValueError):
            layer.eval().forward_packed(torch.randn(2, 16), packed)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class PackedPathCudaTest(TestCase):
    @parameterized.expand(
        [
            param("default"),
            param("three_hops", hop_distance=3),
            param("hop_order", prioritize_hop_order=True, max_seq_len=10),
            # The backward kernel rejects a length bound above the token count.
            param("one_anchor", anchors=[3], max_seq_len=256),
            param("two_anchors", anchors=[3, 40], max_seq_len=256),
            param("neighbor_readout", readout_mode=_NEIGHBOR_READOUT),
            param(
                "csr_input",
                anchor_based_input_attr_names=["hop_times_two"],
                anchor_based_input_embedding_dict=None,
            ),
            param("head_dim_10", hid_dim=20),
            param(
                "anchor_bias",
                anchor_based_attention_bias_attr_names=[_HOP],
            ),
        ]
    )
    def test_cuda_training_gradients_match_cpu(
        self,
        _: str,
        hop_distance: int = 2,
        anchors: Optional[list[int]] = None,
        **overrides: Any,
    ) -> None:
        data = _random_batch(0, h_max=hop_distance)
        cuda_data = _random_batch(0, h_max=hop_distance).to("cuda")
        anchor_node_ids = None
        if anchors is not None:
            anchor_node_ids = torch.tensor(anchors)
        kwargs = _encoder_kwargs(
            **{
                "hid_dim": 128,
                "hop_distance": hop_distance,
                "dropout_rate": 0.0,
                **overrides,
            }
        )
        cpu_encoder = _materialized_encoder(data, kwargs).train()
        cuda_encoder = copy.deepcopy(cpu_encoder).to("cuda")
        self.assertTrue(cuda_encoder._use_packed_backend(torch.device("cuda")))
        cuda_anchor_node_ids = None
        if anchor_node_ids is not None:
            cuda_anchor_node_ids = anchor_node_ids.to("cuda")
        _assert_training_gradients_match(
            self,
            _gradients(
                lambda: cpu_encoder(
                    data, anchor_node_type=_USER, anchor_node_ids=anchor_node_ids
                ),
                cpu_encoder,
            ),
            _gradients(
                lambda: cuda_encoder(
                    cuda_data,
                    anchor_node_type=_USER,
                    anchor_node_ids=cuda_anchor_node_ids,
                ),
                cuda_encoder,
            ),
            atol=1e-4,
        )

    @parameterized.expand(_READOUT_MODES)
    def test_whole_encoder_inductor_compile_matches_eager_without_recompiling(
        self, readout_mode: str
    ) -> None:
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        kwargs = _encoder_kwargs(
            hid_dim=128,
            dropout_rate=0.0,
            readout_mode=readout_mode,
        )
        encoder = _materialized_encoder(_random_batch(0), kwargs).to("cuda")
        compiled = torch.compile(encoder)
        graph_counts = []
        for seed in range(5):
            data = _random_batch(seed).to("cuda")
            with torch.no_grad():
                actual = compiled(data, anchor_node_type=_USER)
                expected = encoder(data, anchor_node_type=_USER)
            graph_counts.append(torch._dynamo.utils.counters["stats"]["unique_graphs"])
            torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
        self.assertEqual(graph_counts[2:], [graph_counts[2]] * 3, graph_counts)

    @parameterized.expand(_READOUT_MODES)
    def test_cuda_forward_matches_cpu_with_two_host_syncs(
        self, readout_mode: str
    ) -> None:
        device = torch.device("cuda")
        batches = [_random_batch(seed) for seed in range(3)]
        warmup_data = _random_batch(0).to(device)
        cuda_batches = [_random_batch(seed).to(device) for seed in range(3)]
        kwargs = _encoder_kwargs(hid_dim=128, readout_mode=readout_mode)
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

    def test_cuda_relation_messages_match_cpu_with_a_fixed_number_of_host_syncs(
        self,
    ) -> None:
        device = torch.device("cuda")
        sync_counts: dict[tuple[int, int], list[int]] = {}
        for num_layers, num_relations in ((1, 3), (3, 3), (3, 1)):
            kwargs = _relation_encoder_kwargs(
                num_relations=num_relations,
                num_layers=num_layers,
                hid_dim=128,
                anchor_based_input_embedding_dict=nn.ModuleDict(
                    {_HOP: nn.Embedding(3, 128)}
                ),
            )
            data = _three_relation_batch(0, num_relations=num_relations)
            cpu_encoder = _relation_encoder(data, kwargs)
            cuda_encoder = copy.deepcopy(cpu_encoder).to(device)
            with torch.no_grad():
                cuda_encoder(
                    _three_relation_batch(0, num_relations=num_relations).to(device),
                    anchor_node_type=_USER,
                )
                for seed in range(3):
                    data = _three_relation_batch(seed, num_relations=num_relations)
                    cuda_data = data.clone().to(device)
                    torch.cuda.synchronize()
                    torch.cuda.set_sync_debug_mode("warn")
                    try:
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            actual = cuda_encoder(cuda_data, anchor_node_type=_USER)
                    finally:
                        torch.cuda.set_sync_debug_mode("default")
                    sync_counts.setdefault((num_layers, num_relations), []).append(
                        len([w for w in caught if "synchroniz" in str(w.message)])
                    )
                    torch.testing.assert_close(
                        actual.cpu(),
                        cpu_encoder(data, anchor_node_type=_USER),
                        rtol=0,
                        atol=1e-4,
                    )
        # Syncs come from building the batch's layout and edges, so they depend on
        # neither the number of layers nor the number of relations.
        self.assertEqual(
            len({count for counts in sync_counts.values() for count in counts}),
            1,
            sync_counts,
        )

    def test_cuda_relation_training_gradients_match_cpu(self) -> None:
        kwargs = _relation_encoder_kwargs(
            hid_dim=128,
            anchor_based_input_embedding_dict=nn.ModuleDict(
                {_HOP: nn.Embedding(3, 128)}
            ),
        )
        data = _three_relation_batch(0)
        cpu_encoder = _relation_encoder(data, kwargs).train()
        cuda_encoder = copy.deepcopy(cpu_encoder).to("cuda")
        cuda_data = _three_relation_batch(0).to("cuda")
        _assert_training_gradients_match(
            self,
            _gradients(lambda: cpu_encoder(data, anchor_node_type=_USER), cpu_encoder),
            _gradients(
                lambda: cuda_encoder(cuda_data, anchor_node_type=_USER), cuda_encoder
            ),
            atol=1e-4,
        )


if __name__ == "__main__":
    absltest.main()
