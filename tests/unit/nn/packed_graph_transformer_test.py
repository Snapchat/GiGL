"""The packed ``GraphTransformerEncoder`` path matches the padded path."""

import copy
import functools
import itertools
import math
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
)
from gigl.nn.packed_attention import PackedAttentionBias
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.transforms.add_positional_encodings import AddHeteroHopDistanceEncoding
from gigl.transforms.graph_transformer import build_ppr_sequence_grids
from gigl.transforms.sequence_layout import (
    PackedLayout,
    PackedRelationEdges,
    RelationSegment,
    pack_sequences,
)
from gigl.transforms.static_khop import KhopLayoutTooLargeError, khop_layout
from tests.test_assets.packed_attention_reference import reference_packed_attention
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
    ``hop_times_two``, a second sparse attribute with every hop value doubled,
    and ``pair_score``, a square sparse attribute over all nodes (by homogeneous
    id) with a random value on a random fifth of the pairs.
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
    num_nodes = hop.shape[0]
    scores = torch.randn(num_nodes, num_nodes, generator=generator)
    scores[torch.rand(num_nodes, num_nodes, generator=generator) > 0.2] = 0.0
    data["pair_score"] = scores.to_sparse_csr()
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
    seed: int,
    num_edges: int = 80,
    num_relations: int = 3,
    has_self_loops: bool = True,
) -> HeteroData:
    """A user graph with ``num_relations`` of three user-to-user edge types and a
    ``pair_score`` attribute like ``_random_batch``'s.

    With ``has_self_loops=False``, edges from a node to itself are dropped.
    """
    generator = torch.Generator().manual_seed(seed)
    data = HeteroData()
    data[_USER].x = torch.randn(60, 8, generator=generator)
    data[_USER].batch_size = 12
    for edge_type in _USER_EDGE_TYPES[:num_relations]:
        edge_index = torch.randint(0, 60, (2, num_edges), generator=generator)
        if not has_self_loops:
            edge_index = edge_index[:, edge_index[0] != edge_index[1]]
        data[edge_type.tuple_repr()].edge_index = edge_index
    data = AddHeteroHopDistanceEncoding(h_max=2, sampling_direction="in")(data)
    num_nodes = data[_HOP].shape[0]
    scores = torch.randn(num_nodes, num_nodes, generator=generator)
    scores[torch.rand(num_nodes, num_nodes, generator=generator) > 0.2] = 0.0
    data["pair_score"] = scores.to_sparse_csr()
    return data


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
            if any(
                part in name
                for part in (
                    "_relation_message",
                    "_relation_attention_matrices",
                    "_relation_hgt_attention",
                )
            ):
                parameter.add_(0.3 * torch.randn(parameter.shape, generator=generator))
    return encoder


_BIAS_CONFIGS = ["pairwise", "pairwise_and_key", "relation_bilinear", "relation_hgt"]


def _bias_encoder(
    config: str, readout_mode: str
) -> tuple[GraphTransformerEncoder, Callable[[int], HeteroData]]:
    """An eval-mode encoder with a per-pair attention bias, and its batch builder.

    ``config`` is one of ``_BIAS_CONFIGS``. Both pairwise bias and relation
    attention add a per-pair score to the attention.
    """
    if config.startswith("relation"):
        kwargs = _relation_encoder_kwargs(
            "none",
            relation_attention_mode=f"edge_type_{config.split('_')[1]}",
            hid_dim=128,
            readout_mode=readout_mode,
            anchor_based_input_embedding_dict=nn.ModuleDict(
                {_HOP: nn.Embedding(3, 128)}
            ),
        )
        return _relation_encoder(
            _three_relation_batch(0), kwargs
        ), _three_relation_batch
    overrides: dict = {"pairwise_attention_bias_attr_names": ["pair_score"]}
    if config == "pairwise_and_key":
        overrides["anchor_based_attention_bias_attr_names"] = [_HOP]
    kwargs = _encoder_kwargs(
        hid_dim=128, readout_mode=readout_mode, dropout_rate=0.0, **overrides
    )
    encoder = _materialized_encoder(_random_batch(0), kwargs)
    _randomize_pairwise_biases(encoder)
    return encoder, _random_batch


def _without_relation_attention(
    encoder: GraphTransformerEncoder,
) -> GraphTransformerEncoder:
    """A copy of ``encoder`` whose relation attention parameters are back at their
    no-op initial values."""
    reset = copy.deepcopy(encoder)
    with torch.no_grad():
        for layer in reset.modules():
            if not isinstance(layer, GraphTransformerEncoderLayer):
                continue
            if layer._relation_attention_matrices is not None:
                layer._relation_attention_matrices.zero_()
            layer._reset_hgt_relation_attention_parameters()
    return reset


def _run_packed(
    encoder: GraphTransformerEncoder, data: HeteroData, **forward_kwargs: object
) -> torch.Tensor:
    """Runs ``encoder`` on the packed path regardless of device."""
    with mock.patch.object(encoder, "_use_packed_backend", return_value=True):
        return encoder(data, anchor_node_type=_USER, **forward_kwargs)


def _randomize_pairwise_biases(encoder: GraphTransformerEncoder) -> None:
    """Gives the pairwise projection and nonmissing bias random values.

    Both start at (near) zero, which would hide a wrong pairwise path.
    """
    generator = torch.Generator().manual_seed(1)
    projection = encoder._pairwise_pe_attention_bias_projection
    nonmissing = encoder._pairwise_nonmissing_attention_bias
    assert projection is not None and nonmissing is not None
    with torch.no_grad():
        projection.weight.copy_(
            torch.randn(projection.weight.shape, generator=generator)
        )
        nonmissing.copy_(torch.randn(nonmissing.shape, generator=generator))


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


_PPR = EdgeType(_USER, Relation("ppr"), _USER)
_PPR_RELATION_FEATURE_DIM = 3


def _ppr_batch(seed: int, typed: bool = False) -> HeteroData:
    """A batch with outgoing PPR edges from each of 12 anchors, as a PPR sampler gives.

    Anchor 0 has no PPR edges, so its sequence is the anchor alone. The others
    have up to 40 PPR targets each, more than ``_MAX_SEQ_LEN``, with weights
    rounded to one decimal so some are tied. ``edge_attr`` holds the weight and
    ``_PPR_RELATION_FEATURE_DIM`` relation-feature columns.

    The three ``_USER_EDGE_TYPES`` carry random relation edges, and every PPR
    edge has a ``ppr_is_expanded`` flag. With ``typed`` the flag is a two-channel
    bitmask and ``ppr_relation_channel_masks`` maps each relation to the channels
    that may fetch it. The batch also has ``hop_distance`` (out direction, two
    hops) and the ``pair_score`` attribute of ``_random_batch``.
    """
    generator = torch.Generator().manual_seed(seed)
    num_nodes, num_anchors = 60, 12
    data = HeteroData()
    data[_USER].x = torch.randn(num_nodes, 8, generator=generator)
    data[_USER].batch_size = num_anchors
    sources: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for anchor in range(1, num_anchors):
        count = int(torch.randint(1, 41, (1,), generator=generator))
        anchor_targets = torch.randperm(num_nodes, generator=generator)[:count]
        anchor_targets = anchor_targets[anchor_targets != anchor]
        sources.append(torch.full_like(anchor_targets, anchor))
        targets.append(anchor_targets)
    edge_index = torch.stack([torch.cat(sources), torch.cat(targets)])
    num_edges = edge_index.size(1)
    weights = (torch.rand(num_edges, generator=generator) * 0.9 + 0.1).round(decimals=1)
    relation_features = torch.randn(
        num_edges, _PPR_RELATION_FEATURE_DIM, generator=generator
    )
    data[_PPR.tuple_repr()].edge_index = edge_index
    data[_PPR.tuple_repr()].edge_attr = torch.cat(
        [weights.unsqueeze(1), relation_features], dim=1
    )
    if typed:
        expanded = torch.randint(0, 4, (num_edges,), generator=generator)
    else:
        expanded = torch.rand(num_edges, generator=generator) > 0.3
    data[_PPR.tuple_repr()].ppr_is_expanded = expanded
    for edge_type in _USER_EDGE_TYPES:
        data[edge_type.tuple_repr()].edge_index = torch.randint(
            0, num_nodes, (2, 120), generator=generator
        )
    if typed:
        data.ppr_relation_channel_masks = {
            edge_type.tuple_repr(): mask
            for edge_type, mask in zip(_USER_EDGE_TYPES, (0b01, 0b10, 0b11))
        }
    data = AddHeteroHopDistanceEncoding(h_max=2, sampling_direction="out")(data)
    scores = torch.randn(num_nodes, num_nodes, generator=generator)
    scores[torch.rand(num_nodes, num_nodes, generator=generator) > 0.2] = 0.0
    data["pair_score"] = scores.to_sparse_csr()
    return data


def _ppr_encoder(data: HeteroData, **overrides: Any) -> GraphTransformerEncoder:
    """An eval-mode PPR encoder whose relation and pairwise parameters are random."""
    kwargs = _encoder_kwargs(
        sequence_construction_method="ppr",
        sampling_direction="out",
        edge_type_to_feat_dim_map={
            _PPR: 0,
            **{edge_type: 0 for edge_type in _USER_EDGE_TYPES},
        },
        anchor_based_input_attr_names=[],
        anchor_based_input_embedding_dict=None,
        dropout_rate=0.0,
    )
    kwargs.update(overrides)
    encoder = _relation_encoder(data, kwargs)
    if kwargs.get("pairwise_attention_bias_attr_names"):
        _randomize_pairwise_biases(encoder)
    return encoder


_PPR_CONFIGS = [
    param("plain"),
    param("neighbor_readout", readout_mode=_NEIGHBOR_READOUT),
    param("weight_bias", anchor_based_attention_bias_attr_names=["ppr_weight"]),
    param("weight_and_hop_input", anchor_based_input_attr_names=["ppr_weight", _HOP]),
    param(
        "hop_embedding_input",
        anchor_based_input_attr_names=[_HOP],
        anchor_based_input_embedding_dict=nn.ModuleDict({_HOP: nn.Embedding(3, 16)}),
    ),
    param(
        "relation_feature_input",
        anchor_based_input_attr_names=["ppr_relation_features"],
    ),
    param("sinusoidal", sequence_positional_encoding_type="sinusoidal"),
    param("pairwise_bias", pairwise_attention_bias_attr_names=["pair_score"]),
    param("relation_bilinear", relation_attention_mode="edge_type_bilinear"),
    param("relation_hgt", relation_attention_mode="edge_type_hgt"),
    param("relation_message", relation_message_mode="edge_type_attention"),
    param("relation_linear_message", relation_message_mode="edge_type_linear"),
]


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
                # The spy checks the contract between the layer loop and the
                # readout: with anchor_only, _run_packed_layers returns one
                # [hid_dim] row per anchor, with or without layers.
                layer_outputs: list[torch.Tensor] = []
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
            param("pairwise_bias", pairwise_attention_bias_attr_names=["pair_score"]),
            param(
                "pairwise_and_anchor_bias",
                pairwise_attention_bias_attr_names=["pair_score"],
                anchor_based_attention_bias_attr_names=[_HOP],
            ),
            param(
                "pairwise_bias_hetero",
                hetero=True,
                pairwise_attention_bias_attr_names=["pair_score"],
            ),
            param(
                "pairwise_bias_neighbor_readout",
                pairwise_attention_bias_attr_names=["pair_score"],
                readout_mode=_NEIGHBOR_READOUT,
            ),
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
            if kwargs.get("pairwise_attention_bias_attr_names"):
                _randomize_pairwise_biases(encoder)
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

    @parameterized.expand(_READOUT_MODES)
    def test_matches_padded_path_with_pairwise_bias_and_explicit_anchors(
        self, readout_mode: str
    ) -> None:
        data = _random_batch(0)
        encoder = _materialized_encoder(
            data,
            _encoder_kwargs(
                readout_mode=readout_mode,
                pairwise_attention_bias_attr_names=["pair_score"],
            ),
        )
        _randomize_pairwise_biases(encoder)
        anchor_node_ids = torch.tensor([7, 3, 3, 40, 0])
        with torch.no_grad():
            torch.testing.assert_close(
                _run_packed(encoder, data, anchor_node_ids=anchor_node_ids),
                encoder(data, anchor_node_type=_USER, anchor_node_ids=anchor_node_ids),
                rtol=0,
                atol=1e-6,
            )

    def test_pairwise_projection_and_nonmissing_bias_each_change_the_output(
        self,
    ) -> None:
        data = _random_batch(0)
        encoder = _materialized_encoder(
            data,
            _encoder_kwargs(pairwise_attention_bias_attr_names=["pair_score"]),
        )
        _randomize_pairwise_biases(encoder)
        projection = encoder._pairwise_pe_attention_bias_projection
        nonmissing = encoder._pairwise_nonmissing_attention_bias
        assert projection is not None and nonmissing is not None
        with torch.no_grad():
            both = _run_packed(encoder, data)
            saved_projection = projection.weight.clone()
            saved_nonmissing = nonmissing.clone()
            projection.weight.zero_()
            self.assertFalse(torch.allclose(_run_packed(encoder, data), both))
            projection.weight.copy_(saved_projection)
            nonmissing.zero_()
            self.assertFalse(torch.allclose(_run_packed(encoder, data), both))
            nonmissing.copy_(saved_nonmissing)
            torch.testing.assert_close(_run_packed(encoder, data), both)

    def test_packed_path_requires_the_pairwise_attribute(self) -> None:
        data = _random_batch(0)
        encoder = _materialized_encoder(
            data,
            _encoder_kwargs(pairwise_attention_bias_attr_names=["pair_score"]),
        )
        del data["pair_score"]
        with self.assertRaisesRegex(ValueError, "pair_score"), torch.no_grad():
            _run_packed(encoder, data)

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
    def test_anchors_without_neighbors_match_padded_path(
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
            param("pairwise_bias", pairwise_attention_bias_attr_names=["pair_score"]),
            param(
                "pairwise_bias_explicit_anchors",
                anchors=[7, 3, 3, 40, 0],
                pairwise_attention_bias_attr_names=["pair_score"],
                anchor_based_attention_bias_attr_names=[_HOP],
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
        if overrides.get("pairwise_attention_bias_attr_names"):
            _randomize_pairwise_biases(encoder)
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
                supported=True,
            ),
            param(
                "relation_attention_bilinear",
                relation_attention_mode="edge_type_bilinear",
                supported=True,
            ),
            param(
                "relation_attention_hgt",
                relation_attention_mode="edge_type_hgt",
                supported=True,
            ),
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
            param("attention_dropout", attention_dropout_rate=0.1, supported=True),
            param(
                "pairwise_bias",
                pairwise_attention_bias_attr_names=["pair_score"],
                supported=True,
            ),
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
                supported=True,
            ),
            # PPR sequences do not grow with hop_distance.
            param(
                "ppr_five_hops",
                sequence_construction_method="ppr",
                sampling_direction="out",
                hop_distance=5,
                supported=True,
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
            self.assertTrue(
                encoder._use_packed_backend(torch.device("cuda"), torch.float32)
            )
            self.assertFalse(
                encoder._use_packed_backend(torch.device("cpu"), torch.float32)
            )

    def test_packed_path_needs_a_dtype_the_varlen_kernel_supports(self) -> None:
        encoder = GraphTransformerEncoder(**_encoder_kwargs())
        cuda = torch.device("cuda")
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                self.assertTrue(encoder._use_packed_backend(cuda, dtype))
        self.assertFalse(encoder._use_packed_backend(cuda, torch.float64))

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

    def test_attention_dropout_runs_packed_with_eval_parity_and_training_noise(
        self,
    ) -> None:
        data = _random_batch(0)
        encoder = _materialized_encoder(
            data, _encoder_kwargs(attention_dropout_rate=0.3, dropout_rate=0.0)
        )
        self.assertTrue(encoder._supports_packed_backend)
        with torch.no_grad():
            torch.testing.assert_close(
                _run_packed(encoder, data),
                encoder(data, anchor_node_type=_USER),
                rtol=0,
                atol=1e-6,
            )
        encoder.train()
        outputs = []
        for seed in (0, 1):
            torch.manual_seed(seed)
            encoder.zero_grad()
            output = _run_packed(encoder, data)
            output.sum().backward()
            outputs.append(output.detach())
            for name, parameter in encoder.named_parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertFalse(torch.allclose(outputs[0], outputs[1]))

    def test_compiled_layer_module_is_rejected_on_the_packed_path(self) -> None:
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        data = _random_batch(0)
        encoder = _materialized_encoder(data, _encoder_kwargs())
        layers = cast(Any, encoder._encoder_layers)
        layers[0] = torch.compile(layers[0], backend="eager")
        with torch.no_grad(), self.assertRaises(ValueError):
            _run_packed(encoder, data)

    def test_compiled_forward_packed_matches_eager(self) -> None:
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        data = _random_batch(0)
        encoder = _materialized_encoder(data, _encoder_kwargs())
        with torch.no_grad():
            expected = _run_packed(encoder, data)
            for layer in cast(Any, encoder._encoder_layers):
                layer.forward_packed = torch.compile(
                    layer.forward_packed, backend="eager"
                )
            torch.testing.assert_close(_run_packed(encoder, data), expected)

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

    @parameterized.expand(
        [
            param("bilinear", attention="edge_type_bilinear"),
            param("hgt", attention="edge_type_hgt"),
            param(
                "bilinear_with_attention_messages",
                attention="edge_type_bilinear",
                mode="edge_type_attention",
            ),
            param(
                "hgt_with_linear_messages",
                attention="edge_type_hgt",
                mode="edge_type_linear",
            ),
            param("hgt_one_layer", attention="edge_type_hgt", num_layers=1),
            param(
                "bilinear_out_direction",
                attention="edge_type_bilinear",
                sampling_direction="out",
            ),
            param(
                "bilinear_neighbor_readout",
                attention="edge_type_bilinear",
                readout_mode="anchor_neighbor_attention",
            ),
            param(
                "bilinear_with_pairwise_bias",
                attention="edge_type_bilinear",
                pairwise_attention_bias_attr_names=["pair_score"],
            ),
            param(
                "hgt_with_pairwise_bias_neighbor_readout",
                attention="edge_type_hgt",
                readout_mode="anchor_neighbor_attention",
                pairwise_attention_bias_attr_names=["pair_score"],
            ),
        ]
    )
    def test_relation_attention_matches_padded_path(
        self, _: str, attention: str, mode: str = "none", **overrides: Any
    ) -> None:
        for seed in range(3):
            data = _three_relation_batch(seed)
            if overrides.get("sampling_direction") == "out":
                data = AddHeteroHopDistanceEncoding(h_max=2, sampling_direction="out")(
                    data
                )
            encoder = _relation_encoder(
                data,
                _relation_encoder_kwargs(
                    mode, relation_attention_mode=attention, **overrides
                ),
            )
            plain = _without_relation_attention(encoder)
            with self.subTest(seed=seed), torch.no_grad():
                expected = encoder(data, anchor_node_type=_USER)
                torch.testing.assert_close(
                    _run_packed(encoder, data), expected, rtol=0, atol=1e-6
                )
                # The relation attention is what the comparison above exercises.
                self.assertGreater(
                    (expected - plain(data, anchor_node_type=_USER)).abs().max(),
                    1e-3,
                )

    @parameterized.expand(
        [
            param("bilinear", attention="edge_type_bilinear"),
            param("hgt", attention="edge_type_hgt"),
        ]
    )
    def test_relation_attention_compile_does_not_recompile_across_batches(
        self, _: str, attention: str
    ) -> None:
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        encoder = _relation_encoder(
            _three_relation_batch(0),
            _relation_encoder_kwargs("none", relation_attention_mode=attention),
        )
        # Batches differ in their per-relation edge counts; hitting the recompile
        # limit raises.
        with (
            torch._dynamo.config.patch(fail_on_recompile_limit_hit=True),
            mock.patch.object(encoder, "_use_packed_backend", return_value=True),
        ):
            compiled = torch.compile(encoder, backend="eager")
            graph_counts = []
            for seed in range(8):
                with torch.no_grad():
                    compiled(_three_relation_batch(seed), anchor_node_type=_USER)
                graph_counts.append(
                    torch._dynamo.utils.counters["stats"]["unique_graphs"]
                )
        settled = graph_counts[3:]
        self.assertEqual(settled, [settled[0]] * len(settled), graph_counts)

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
            param("bilinear", attention="edge_type_bilinear"),
            param("hgt", attention="edge_type_hgt"),
            param(
                "bilinear_with_messages",
                attention="edge_type_bilinear",
                mode="edge_type_attention",
            ),
            param(
                "hgt_with_messages",
                attention="edge_type_hgt",
                mode="edge_type_linear",
            ),
        ]
    )
    def test_relation_attention_training_gradients_match_padded_path(
        self, _: str, attention: str, mode: str = "none"
    ) -> None:
        data = _three_relation_batch(0)
        encoder = _relation_encoder(
            data, _relation_encoder_kwargs(mode, relation_attention_mode=attention)
        ).train()
        expected = _gradients(lambda: encoder(data, anchor_node_type=_USER), encoder)
        actual = _gradients(lambda: _run_packed(encoder, data), encoder)
        name = {
            "edge_type_bilinear": "_relation_attention_matrices",
            "edge_type_hgt": "_relation_hgt_attention_matrices",
        }[attention]
        # The relation attention parameters of every layer are trained.
        self.assertTrue(
            all(
                actual[1][f"_encoder_layers.{layer}.{name}"].abs().max() > 0
                for layer in range(3)
            )
        )
        _assert_training_gradients_match(self, expected, actual, atol=1e-5)

    @parameterized.expand(
        [
            param("no_edges", num_edges=0, expects_all_filtered=False),
            # Every edge into an anchor is dropped: with "out" sampling and one hop
            # the expanded endpoint of such an edge is a last-hop token. Anchor
            # self-loops would survive the filter, so the batch has none.
            param(
                "no_edges_into_anchors",
                has_self_loops=False,
                expects_all_filtered=True,
                sampling_direction="out",
                hop_distance=1,
            ),
        ]
    )
    def test_training_gradients_match_padded_path_when_edges_are_filtered(
        self,
        _: str,
        expects_all_filtered: bool,
        num_edges: int = 80,
        has_self_loops: bool = True,
        **overrides: Any,
    ) -> None:
        self._assert_filtered_edge_gradients_match(
            num_edges,
            has_self_loops=has_self_loops,
            expects_all_filtered=expects_all_filtered,
            **overrides,
        )

    @parameterized.expand(
        [
            param("bilinear", attention="edge_type_bilinear"),
            param("hgt", attention="edge_type_hgt"),
        ]
    )
    def test_relation_attention_training_gradients_match_when_edges_are_filtered(
        self, _: str, attention: str
    ) -> None:
        self._assert_filtered_edge_gradients_match(
            0, mode="none", relation_attention_mode=attention
        )
        self._assert_filtered_edge_gradients_match(
            80,
            mode="none",
            relation_attention_mode=attention,
            sampling_direction="out",
            hop_distance=1,
        )

    def _assert_filtered_edge_gradients_match(
        self,
        num_edges: int,
        mode: str = "edge_type_attention",
        has_self_loops: bool = True,
        expects_all_filtered: Optional[bool] = None,
        **overrides: Any,
    ) -> None:
        """Checks packed gradients against padded ones on a batch whose edges are
        filtered.

        With ``expects_all_filtered``, also checks whether the packed path takes the
        all-filtered branch of the relation message layer.
        """
        data = _three_relation_batch(0, num_edges, has_self_loops=has_self_loops)
        if overrides:
            data = AddHeteroHopDistanceEncoding(h_max=1, sampling_direction="out")(data)
        encoder = _relation_encoder(
            data, _relation_encoder_kwargs(mode, **overrides)
        ).train()
        expected = _gradients(lambda: encoder(data, anchor_node_type=_USER), encoder)
        with mock.patch.object(
            GraphTransformerEncoderLayer,
            "_zero_relation_parameter_dependency",
            autospec=True,
            side_effect=GraphTransformerEncoderLayer._zero_relation_parameter_dependency,
        ) as zero_dependency:
            actual = _gradients(lambda: _run_packed(encoder, data), encoder)
        if expects_all_filtered is not None:
            self.assertEqual(zero_dependency.called, expects_all_filtered)
        _assert_training_gradients_match(self, expected, actual, atol=1e-5)

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
                layout: PackedLayout,
                first_token_output: bool,
                relation_edges: PackedRelationEdges | None,
                bias: PackedAttentionBias | None = None,
            ) -> torch.Tensor:
                def run(x: torch.Tensor) -> torch.Tensor:
                    self.layer_calls += 1
                    return super(CheckpointedEncoder, self)._run_packed_layer(
                        layer, x, layout, first_token_output, relation_edges, bias
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

    @parameterized.expand(
        [
            param("bilinear", relation_attention_mode="edge_type_bilinear"),
            param("hgt", relation_attention_mode="edge_type_hgt"),
        ]
    )
    def test_layer_requires_relation_edges_for_relation_attention(
        self, _: str, **mode: Any
    ) -> None:
        layer = GraphTransformerEncoderLayer(
            model_dim=16,
            num_heads=2,
            feedforward_dim=32,
            num_relations=2,
            **mode,
        )
        packed = pack_sequences(torch.tensor([2]), 2)
        with self.assertRaises(ValueError):
            layer.eval().forward_packed(torch.randn(2, 16), packed)


class PackedPprTest(TestCase):
    @parameterized.expand(_PPR_CONFIGS)
    def test_ppr_config_runs_packed(self, _: str, **overrides: Any) -> None:
        encoder = _ppr_encoder(_ppr_batch(0), **overrides)
        self.assertEqual(encoder._packed_unsupported_reasons(), [])

    @parameterized.expand(
        [
            param(f"{config.args[0]}_{kind}", typed=typed, **config.kwargs)
            for config in _PPR_CONFIGS
            for kind, typed in (("untyped", False), ("typed", True))
        ]
    )
    def test_packed_ppr_matches_padded(
        self, _: str, typed: bool, **overrides: Any
    ) -> None:
        encoder = _ppr_encoder(_ppr_batch(0, typed=typed), **overrides)
        for seed in range(3):
            data = _ppr_batch(seed, typed=typed)
            with self.subTest(seed=seed), torch.no_grad():
                torch.testing.assert_close(
                    _run_packed(encoder, data),
                    encoder(data, anchor_node_type=_USER),
                    rtol=0,
                    atol=1e-6,
                )

    @parameterized.expand(_PPR_CONFIGS)
    def test_packed_ppr_training_gradients_match_padded(
        self, _: str, **overrides: Any
    ) -> None:
        encoder = _ppr_encoder(_ppr_batch(0, typed=True), **overrides).train()
        data = _ppr_batch(1, typed=True)
        _assert_training_gradients_match(
            self,
            _gradients(lambda: encoder(data, anchor_node_type=_USER), encoder),
            _gradients(lambda: _run_packed(encoder, data), encoder),
            atol=1e-5,
        )

    def test_hop_embedding_is_not_size_checked_for_ppr(self) -> None:
        # PPR hops come from the batch, so an embedding of ``hop_distance`` or
        # fewer entries is fine when the batch's hops fit it.
        data = AddHeteroHopDistanceEncoding(h_max=1, sampling_direction="out")(
            _ppr_batch(0)
        )
        encoder = _ppr_encoder(
            data,
            anchor_based_input_attr_names=[_HOP],
            anchor_based_input_embedding_dict=nn.ModuleDict(
                {_HOP: nn.Embedding(2, 16)}
            ),
        )
        self.assertEqual(encoder._hop_distance, 2)
        with torch.no_grad():
            torch.testing.assert_close(
                _run_packed(encoder, data),
                encoder(data, anchor_node_type=_USER),
                rtol=0,
                atol=1e-6,
            )

    def test_packed_ppr_matches_padded_on_a_heterogeneous_batch(self) -> None:
        data = _ppr_batch(0)
        generator = torch.Generator().manual_seed(1)
        num_items = 30
        data[_ITEM].x = torch.randn(num_items, 8, generator=generator)
        item_ppr = EdgeType(_USER, Relation("ppr"), _ITEM)
        sources = torch.randint(1, 12, (200,), generator=generator)
        targets = torch.randint(0, num_items, (200,), generator=generator)
        weights = (torch.rand(200, generator=generator) * 0.9 + 0.1).round(decimals=1)
        data[item_ppr.tuple_repr()].edge_index = torch.stack([sources, targets])
        data[item_ppr.tuple_repr()].edge_attr = torch.cat(
            [
                weights.unsqueeze(1),
                torch.randn(200, _PPR_RELATION_FEATURE_DIM, generator=generator),
            ],
            dim=1,
        )
        data[item_ppr.tuple_repr()].ppr_is_expanded = (
            torch.rand(200, generator=generator) > 0.3
        )
        encoder = _ppr_encoder(
            data,
            node_type_to_feat_dim_map={_USER: 8, _ITEM: 8},
            edge_type_to_feat_dim_map={
                _PPR: 0,
                item_ppr: 0,
                **{edge_type: 0 for edge_type in _USER_EDGE_TYPES},
            },
            anchor_based_input_attr_names=["ppr_weight"],
            relation_attention_mode="edge_type_bilinear",
        )
        with torch.no_grad():
            packed = _run_packed(encoder, data)
            padded = encoder(data, anchor_node_type=_USER)
            torch.testing.assert_close(packed, padded, rtol=0, atol=1e-6)
        # The item tokens take part: dropping the user-to-item PPR edges changes
        # the output.
        del data[item_ppr.tuple_repr()]
        with torch.no_grad():
            self.assertGreater((_run_packed(encoder, data) - packed).abs().max(), 1e-4)

    def test_batch_without_anchors_gives_empty_output_on_both_paths(self) -> None:
        data = _ppr_batch(0)
        encoder = _ppr_encoder(data)
        no_anchors = torch.empty(0, dtype=torch.long)
        with torch.no_grad():
            padded = encoder(data, anchor_node_type=_USER, anchor_node_ids=no_anchors)
            packed = _run_packed(encoder, data, anchor_node_ids=no_anchors)
        self.assertEqual(padded.shape, (0, 8))
        self.assertEqual(packed.shape, (0, 8))

    def test_relation_features_with_a_single_token_sequence_raise_a_clear_error(
        self,
    ) -> None:
        data = _ppr_batch(0)
        encoder = GraphTransformerEncoder(
            **_encoder_kwargs(
                sequence_construction_method="ppr",
                sampling_direction="out",
                edge_type_to_feat_dim_map={_PPR: 0},
                max_seq_len=1,
                anchor_based_input_attr_names=["ppr_relation_features"],
                anchor_based_input_embedding_dict=None,
            )
        )
        with self.assertRaisesRegex(ValueError, "max_seq_len=1"):
            _run_packed(encoder, data)

    def test_relation_leak_filter_changes_the_packed_output(self) -> None:
        # Without ppr_is_expanded the relation filter keeps the edges a
        # co-seeded node leaks, so the output differs; the parity tests above
        # are only meaningful if the filter is exercised.
        encoder = _ppr_encoder(
            _ppr_batch(0), relation_attention_mode="edge_type_bilinear"
        )
        data = _ppr_batch(0)
        unfiltered = _ppr_batch(0)
        del unfiltered[_PPR.tuple_repr()].ppr_is_expanded
        with torch.no_grad():
            self.assertGreater(
                (_run_packed(encoder, data) - _run_packed(encoder, unfiltered))
                .abs()
                .max(),
                1e-4,
            )


_REFERENCE_LENGTHS = [4, 3, 1, 5]
# (sequence, target position, source position, relation), sorted by relation.
# Some targets have several sources in one relation or edges of several
# relations, and sequence 2 (one token) has no edges.
_REFERENCE_EDGES = [
    (0, 0, 1, 0),
    (0, 0, 2, 0),
    (0, 2, 3, 0),
    (1, 1, 0, 0),
    (3, 0, 4, 0),
    (3, 3, 0, 0),
    (3, 3, 1, 0),
    (3, 3, 2, 0),
    (0, 0, 3, 1),
    (0, 3, 0, 1),
    (1, 0, 2, 1),
    (3, 2, 4, 1),
    (0, 0, 1, 2),
    (1, 0, 1, 2),
    (1, 0, 2, 2),
    (3, 0, 1, 2),
    (3, 4, 0, 2),
]
_REFERENCE_NUM_RELATIONS = 3


def _reference_layer(
    message_mode: str = "none", attention_mode: str = "none"
) -> GraphTransformerEncoderLayer:
    """A float64 eval-mode layer with random relation parameters."""
    torch.manual_seed(0)
    layer = GraphTransformerEncoderLayer(
        model_dim=8,
        num_heads=2,
        feedforward_dim=16,
        dropout_rate=0.0,
        relation_attention_mode=attention_mode,  # ty: ignore[invalid-argument-type]
        relation_message_mode=message_mode,  # ty: ignore[invalid-argument-type]
        num_relations=_REFERENCE_NUM_RELATIONS,
    )
    layer = layer.double().eval()
    with torch.no_grad():
        for name, parameter in layer.named_parameters():
            if "relation" in name:
                parameter.add_(0.5 * torch.randn_like(parameter))
    return layer


def _reference_inputs() -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    return [
        torch.randn(length, 8, generator=generator, dtype=torch.float64)
        for length in _REFERENCE_LENGTHS
    ]


def _reference_layer_outputs(
    layer: GraphTransformerEncoderLayer,
    sequences: list[torch.Tensor],
    edges: list[tuple[int, int, int, int]],
    first_token_only: bool = False,
    pair_biases: Optional[list[torch.Tensor]] = None,
) -> torch.Tensor:
    """The layer output by plain loops over sequences, heads and edges.

    Reads the layer's weights directly and calls none of its attention or
    relation helpers. Returns the rows of all tokens, or of every sequence's
    first token with ``first_token_only``. ``pair_biases[i]`` is an optional
    ``[num_heads, length, length]`` score bias of sequence ``i``.
    """
    num_heads, head_dim = layer._num_heads, layer._head_dim
    scale = 1.0 / math.sqrt(head_dim)
    outputs: list[torch.Tensor] = []
    for sequence_index, x in enumerate(sequences):
        length = x.size(0)
        sequence_edges = [edge[1:] for edge in edges if edge[0] == sequence_index]
        x_norm = layer._attention_norm(x)
        query, key, value = (
            (x_norm @ projection.weight.T + projection.bias).view(
                length, num_heads, head_dim
            )
            for projection in (
                layer._query_projection,
                layer._key_projection,
                layer._value_projection,
            )
        )
        bias = torch.zeros(1, num_heads, length, length, dtype=x.dtype)
        if pair_biases is not None:
            bias[0] += pair_biases[sequence_index]
        for target, source, relation in sequence_edges:
            for head in range(num_heads):
                head_query, head_key = query[target, head], key[source, head]
                score = torch.zeros((), dtype=x.dtype)
                if layer._relation_attention_mode == "edge_type_bilinear":
                    matrix = layer._relation_attention_matrices[relation, head]
                    score = head_query @ matrix @ head_key
                elif layer._relation_attention_mode == "edge_type_hgt":
                    matrix = layer._relation_hgt_attention_matrices[relation]
                    prior = layer._relation_hgt_attention_priors[relation]
                    score = prior * (head_query @ matrix @ head_key) - (
                        head_query @ head_key
                    )
                bias[0, head, target, source] += score * scale
        attention = reference_packed_attention(
            query,
            key,
            value,
            torch.tensor([0, length], dtype=torch.int32),
            bias=bias,
        ).reshape(length, num_heads * head_dim)
        hidden = x + attention @ layer._output_projection.weight.T
        hidden = hidden + layer._output_projection.bias
        if layer._relation_message_mode != "none":
            messages = torch.zeros_like(hidden)
            for target in range(length):
                for relation in range(_REFERENCE_NUM_RELATIONS):
                    sources = [
                        source
                        for edge_target, source, edge_relation in sequence_edges
                        if edge_target == target and edge_relation == relation
                    ]
                    if not sources:
                        continue
                    transform = layer._relation_message_matrices[relation]
                    source_messages = [transform @ x_norm[source] for source in sources]
                    for head in range(num_heads):
                        span = slice(head * head_dim, (head + 1) * head_dim)
                        if layer._relation_message_mode == "edge_type_linear":
                            weights = torch.full(
                                (len(sources),), 1.0 / len(sources), dtype=x.dtype
                            )
                        else:
                            matrix = layer._relation_message_attention_matrices[
                                relation, head
                            ]
                            prior = layer._relation_message_attention_priors[
                                relation, head
                            ]
                            scores = torch.stack(
                                [
                                    prior
                                    * (
                                        (key[source, head] @ matrix)
                                        @ query[target, head]
                                    )
                                    * scale
                                    for source in sources
                                ]
                            )
                            weights = torch.softmax(scores, dim=0)
                        for weight, message in zip(weights, source_messages):
                            messages[target, span] += weight * message[span]
            hidden = hidden + messages
        output = hidden + layer._ffn(layer._ffn_norm(hidden))
        if first_token_only:
            output = output[:1]
        outputs.append(output)
    return torch.cat(outputs)


def _reference_padded_outputs(
    layer: GraphTransformerEncoderLayer,
    sequences: list[torch.Tensor],
    edges: list[tuple[int, int, int, int]],
) -> torch.Tensor:
    """The padded layer's output rows of all real tokens."""
    max_length = max(sequence.size(0) for sequence in sequences)
    valid = torch.zeros(len(sequences), max_length, dtype=torch.bool)
    padded = torch.zeros(len(sequences), max_length, 8, dtype=torch.float64)
    for index, sequence in enumerate(sequences):
        valid[index, : sequence.size(0)] = True
        padded[index, : sequence.size(0)] = sequence
    attention_bias = torch.zeros(len(sequences), 1, 1, max_length, dtype=torch.float64)
    attention_bias.masked_fill_(~valid.view(-1, 1, 1, max_length), -1e30)
    with torch.no_grad():
        return layer(
            padded,
            attn_bias=attention_bias,
            valid_mask=valid,
            pairwise_relation_indices=torch.tensor(edges, dtype=torch.long),
        )[valid]


def _reference_packed_outputs(
    layer: GraphTransformerEncoderLayer,
    sequences: list[torch.Tensor],
    edges: list[tuple[int, int, int, int]],
    first_token_output: bool = False,
) -> torch.Tensor:
    """``forward_packed`` on the hand-built edges, in packed token rows."""
    lengths = torch.tensor([sequence.size(0) for sequence in sequences])
    layout = pack_sequences(lengths, int(lengths.max()))
    starts = layout.cu_seqlens.tolist()
    if first_token_output:
        edges = [edge for edge in edges if edge[1] == 0]
    target_rows = []
    for sequence, target, _, _ in edges:
        if first_token_output:
            target_rows.append(sequence)
        else:
            target_rows.append(starts[sequence] + target)
    segments = tuple(
        RelationSegment(
            relation,
            sum(edge[3] < relation for edge in edges),
            sum(edge[3] <= relation for edge in edges),
        )
        for relation in range(_REFERENCE_NUM_RELATIONS)
    )
    layout = layout._replace(
        relation_edges=PackedRelationEdges(
            target=torch.tensor(target_rows),
            source=torch.tensor([starts[edge[0]] + edge[2] for edge in edges]),
            relation=torch.tensor([edge[3] for edge in edges]),
            segments=segments,
        )
    )
    with torch.no_grad():
        return layer.forward_packed(
            torch.cat(sequences), layout, first_token_output=first_token_output
        )


class ReferenceRelationLayerTest(TestCase):
    """Layer outputs match plain loops over a hand-written edge list.

    The references call none of the layer's relation helpers.
    """

    @parameterized.expand(
        [
            param("mean_messages", message_mode="edge_type_linear"),
            param("attention_messages", message_mode="edge_type_attention"),
            param("bilinear_attention", attention_mode="edge_type_bilinear"),
            param("hgt_attention", attention_mode="edge_type_hgt"),
            param(
                "bilinear_attention_and_attention_messages",
                message_mode="edge_type_attention",
                attention_mode="edge_type_bilinear",
            ),
            param(
                "hgt_attention_and_mean_messages",
                message_mode="edge_type_linear",
                attention_mode="edge_type_hgt",
            ),
        ]
    )
    def test_layer_matches_edge_loop_reference(
        self, _: str, message_mode: str = "none", attention_mode: str = "none"
    ) -> None:
        layer = _reference_layer(message_mode, attention_mode)
        sequences = _reference_inputs()
        expected = _reference_layer_outputs(layer, sequences, _REFERENCE_EDGES)
        torch.testing.assert_close(
            _reference_padded_outputs(layer, sequences, _REFERENCE_EDGES),
            expected,
            rtol=0,
            atol=1e-10,
        )
        torch.testing.assert_close(
            _reference_packed_outputs(layer, sequences, _REFERENCE_EDGES),
            expected,
            rtol=0,
            atol=1e-10,
        )
        # The relation channel changes the output; a no-op reference would pass.
        plain = _reference_layer_outputs(
            _reference_layer(), sequences, _REFERENCE_EDGES
        )
        self.assertGreater((expected - plain).abs().max(), 1e-2)

    @parameterized.expand(
        [
            param("attention_messages", message_mode="edge_type_attention"),
            param("hgt_attention", attention_mode="edge_type_hgt"),
        ]
    )
    def test_first_token_output_matches_edge_loop_reference(
        self, _: str, message_mode: str = "none", attention_mode: str = "none"
    ) -> None:
        layer = _reference_layer(message_mode, attention_mode)
        sequences = _reference_inputs()
        expected = _reference_layer_outputs(
            layer, sequences, _REFERENCE_EDGES, first_token_only=True
        )
        torch.testing.assert_close(
            _reference_packed_outputs(
                layer, sequences, _REFERENCE_EDGES, first_token_output=True
            ),
            expected,
            rtol=0,
            atol=1e-10,
        )


class ReferencePairwiseBiasTest(TestCase):
    def test_pairwise_bias_matches_the_hand_built_bias(self) -> None:
        data = _random_batch(0)
        encoder = _materialized_encoder(
            data,
            _encoder_kwargs(pairwise_attention_bias_attr_names=["pair_score"]),
        )
        _randomize_pairwise_biases(encoder)
        projection = encoder._pairwise_pe_attention_bias_projection
        nonmissing = encoder._pairwise_nonmissing_attention_bias
        assert projection is not None and nonmissing is not None
        captured: list[tuple] = []
        real_run_layers = encoder._run_packed_layers

        def record(x, layout, bias):
            captured.append((x, layout, bias))
            return real_run_layers(x, layout, bias)

        with (
            mock.patch.object(encoder, "_run_packed_layers", side_effect=record),
            torch.no_grad(),
        ):
            _run_packed(encoder, data)
        x, layout, bias = captured[0]
        assert bias is not None and bias.per_pair is not None

        csr = data["pair_score"]
        stored: dict[tuple[int, int], float] = {}
        crow, col, values = csr.crow_indices(), csr.col_indices(), csr.values()
        for row in range(csr.size(0)):
            for entry in range(int(crow[row]), int(crow[row + 1])):
                stored[(row, int(col[entry]))] = float(values[entry])
        num_heads = projection.weight.size(0)
        cu_seqlens = layout.cu_seqlens.tolist()
        hand_biases: list[torch.Tensor] = []
        for sequence in range(len(cu_seqlens) - 1):
            start, end = cu_seqlens[sequence], cu_seqlens[sequence + 1]
            nodes = layout.node_index[start:end].tolist()
            hand = torch.zeros(num_heads, end - start, end - start)
            for row, row_node in enumerate(nodes):
                for column, column_node in enumerate(nodes):
                    value = stored.get((row_node, column_node), 0.0)
                    for head in range(num_heads):
                        hand[head, row, column] = projection.weight[head, 0] * value
                        if (row_node, column_node) in stored or row == column:
                            hand[head, row, column] += nonmissing[head]
            hand_biases.append(hand.detach())
            torch.testing.assert_close(
                bias.per_pair[sequence, :, : end - start, : end - start],
                hand_biases[-1],
                rtol=0,
                atol=1e-6,
            )
        self.assertGreater(max(hand.abs().max() for hand in hand_biases), 0.1)

        layer = encoder._encoder_layers[0]
        sequences = [
            x[cu_seqlens[i] : cu_seqlens[i + 1]] for i in range(len(cu_seqlens) - 1)
        ]
        with torch.no_grad():
            expected = _reference_layer_outputs(
                layer, sequences, [], pair_biases=hand_biases
            )
            actual = layer.forward_packed(x, layout, bias=bias)
        torch.testing.assert_close(actual, expected, rtol=0, atol=2e-5)


class ReferencePprSequenceTest(TestCase):
    """PPR sequences of a hand-written graph, derived from the edge weights.

    Each anchor lists its outgoing ``ppr`` targets by descending weight, ties
    in edge order, keeps ``max_seq_len - 1`` of them behind the anchor, and the
    sequence ends there.
    """

    # (anchor, target, weight, ppr_is_expanded), in edge order. Anchor 0 has
    # a tie at 0.9 (3 before 7) and one at 0.5 (4 before 6), and loses its
    # lowest-weight target to the length limit. Anchor 2 has no edges.
    _EDGES = [
        (0, 4, 0.5, 1),
        (1, 5, 0.3, 0),
        (0, 3, 0.9, 1),
        (0, 6, 0.5, 0),
        (1, 0, 0.8, 1),
        (0, 5, 0.2, 1),
        (0, 7, 0.9, 0),
    ]
    _MAX_SEQ_LEN = 4
    _NODE_GRID = [[0, 3, 7, 4], [1, 0, 5, -1], [2, -1, -1, -1]]
    _WEIGHT_GRID = [[0.0, 0.9, 0.9, 0.5], [0.0, 0.8, 0.3, 0.0], [0.0] * 4]
    # The anchor has every bit set; a padded cell is 0.
    _EXPANDED_GRID = [[-1, 1, 0, 1], [-1, 1, 0, 0], [-1, 0, 0, 0]]

    def _data(self) -> HeteroData:
        data = HeteroData()
        data[_USER].x = torch.randn(8, 8)
        data[_USER].batch_size = 3
        data[_PPR.tuple_repr()].edge_index = torch.tensor(
            [[edge[0] for edge in self._EDGES], [edge[1] for edge in self._EDGES]]
        )
        data[_PPR.tuple_repr()].edge_attr = torch.tensor(
            [[edge[2]] for edge in self._EDGES]
        )
        data[_PPR.tuple_repr()].ppr_is_expanded = torch.tensor(
            [edge[3] for edge in self._EDGES]
        )
        return data

    def test_grids_match_the_hand_built_sequences(self) -> None:
        data = self._data()
        grids = build_ppr_sequence_grids(
            edge_index_dict={_PPR.tuple_repr(): data[_PPR.tuple_repr()].edge_index},
            edge_attr_dict={_PPR.tuple_repr(): data[_PPR.tuple_repr()].edge_attr},
            ppr_is_expanded_dict={
                _PPR.tuple_repr(): data[_PPR.tuple_repr()].ppr_is_expanded
            },
            node_type_offsets={_USER: 0},
            anchor_indices=torch.arange(3),
            num_nodes=8,
            max_seq_len=self._MAX_SEQ_LEN,
            include_anchor_first=True,
            device=torch.device("cpu"),
            return_edge_weights=True,
            return_relation_features=False,
        )
        expected_nodes = torch.tensor(self._NODE_GRID)
        torch.testing.assert_close(grids.node_index, expected_nodes)
        torch.testing.assert_close(grids.valid_mask, expected_nodes >= 0)
        assert grids.ppr_weight is not None and grids.expanded_token_mask is not None
        torch.testing.assert_close(
            grids.ppr_weight.squeeze(-1),
            torch.tensor(self._WEIGHT_GRID),
            rtol=0,
            atol=1e-6,
        )
        torch.testing.assert_close(
            grids.expanded_token_mask, torch.tensor(self._EXPANDED_GRID)
        )

    def test_packed_layout_matches_the_hand_built_sequences(self) -> None:
        data = self._data()
        encoder = _ppr_encoder(
            data,
            max_seq_len=self._MAX_SEQ_LEN,
            edge_type_to_feat_dim_map={_PPR: 0},
            anchor_based_attention_bias_attr_names=["ppr_weight"],
        )
        captured: list[PackedLayout] = []
        real_run_layers = encoder._run_packed_layers

        def record(x, layout, bias):
            captured.append(layout)
            return real_run_layers(x, layout, bias)

        with (
            mock.patch.object(encoder, "_run_packed_layers", side_effect=record),
            torch.no_grad(),
        ):
            _run_packed(encoder, data)
        layout = captured[0]
        expected_nodes = torch.tensor(self._NODE_GRID)
        self.assertEqual(layout.cu_seqlens.tolist(), [0, 4, 7, 8])
        self.assertEqual(
            layout.node_index.tolist(), expected_nodes[expected_nodes >= 0].tolist()
        )
        self.assertEqual(layout.position.tolist(), [0, 1, 2, 3, 0, 1, 2, 0])
        torch.testing.assert_close(
            layout.token_attrs["ppr_weight"].flatten(),
            torch.tensor(self._WEIGHT_GRID)[expected_nodes >= 0],
            rtol=0,
            atol=1e-6,
        )


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
            param(
                "pairwise_bias",
                pairwise_attention_bias_attr_names=["pair_score"],
            ),
            param(
                "pairwise_bias_neighbor_readout",
                pairwise_attention_bias_attr_names=["pair_score"],
                readout_mode=_NEIGHBOR_READOUT,
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
        encoder_overrides: dict[str, Any] = {
            "hid_dim": 128,
            "hop_distance": hop_distance,
            "dropout_rate": 0.0,
            **overrides,
        }
        kwargs = _encoder_kwargs(**encoder_overrides)
        cpu_encoder = _materialized_encoder(data, kwargs).train()
        if overrides.get("pairwise_attention_bias_attr_names"):
            _randomize_pairwise_biases(cpu_encoder)
        cuda_encoder = copy.deepcopy(cpu_encoder).to("cuda")
        self.assertTrue(
            cuda_encoder._use_packed_backend(torch.device("cuda"), torch.float32)
        )
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

    @parameterized.expand(_PPR_CONFIGS)
    def test_cuda_ppr_training_gradients_match_cpu(
        self, _: str, **overrides: Any
    ) -> None:
        cpu_encoder = _ppr_encoder(_ppr_batch(0, typed=True), **overrides).train()
        cuda_encoder = copy.deepcopy(cpu_encoder).to("cuda")
        self.assertTrue(
            cuda_encoder._use_packed_backend(torch.device("cuda"), torch.float32)
        )
        data = _ppr_batch(1, typed=True)
        cuda_data = _ppr_batch(1, typed=True).to("cuda")
        _assert_training_gradients_match(
            self,
            _gradients(lambda: cpu_encoder(data, anchor_node_type=_USER), cpu_encoder),
            _gradients(
                lambda: cuda_encoder(cuda_data, anchor_node_type=_USER), cuda_encoder
            ),
            atol=1e-4,
        )

    @parameterized.expand(_PPR_CONFIGS)
    def test_cuda_ppr_inference_matches_cpu(self, _: str, **overrides: Any) -> None:
        cpu_encoder = _ppr_encoder(_ppr_batch(0), **overrides)
        cuda_encoder = copy.deepcopy(cpu_encoder).to("cuda")
        for seed in range(2):
            with self.subTest(seed=seed), torch.no_grad():
                torch.testing.assert_close(
                    cuda_encoder(
                        _ppr_batch(seed).to("cuda"), anchor_node_type=_USER
                    ).cpu(),
                    cpu_encoder(_ppr_batch(seed), anchor_node_type=_USER),
                    rtol=0,
                    atol=1e-4,
                )

    @parameterized.expand(
        [
            param(f"{config}_{readout_mode}", config, readout_mode)
            for config in _BIAS_CONFIGS
            for readout_mode in _READOUT_MODES
        ]
    )
    def test_cuda_inference_with_a_per_pair_bias_matches_cpu(
        self, _: str, config: str, readout_mode: str
    ) -> None:
        cpu_encoder, make_batch = _bias_encoder(config, readout_mode)
        cuda_encoder = copy.deepcopy(cpu_encoder).to("cuda")
        for dtype, atol in ((None, 1e-4), (torch.bfloat16, 1e-1)):
            for seed in range(2):
                with self.subTest(dtype=dtype, seed=seed), torch.no_grad():
                    expected = cpu_encoder(make_batch(seed), anchor_node_type=_USER)
                    with torch.autocast(
                        "cuda", dtype=torch.bfloat16, enabled=dtype is not None
                    ):
                        actual = cuda_encoder(
                            make_batch(seed).to("cuda"), anchor_node_type=_USER
                        )
                    torch.testing.assert_close(
                        actual.float().cpu(), expected, rtol=0, atol=atol
                    )

    @parameterized.expand(_BIAS_CONFIGS)
    def test_whole_encoder_compile_with_a_per_pair_bias_does_not_recompile(
        self, config: str
    ) -> None:
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        encoder, make_batch = _bias_encoder(config, "anchor_only")
        encoder = encoder.to("cuda")
        compiled = torch.compile(encoder)
        graph_counts = []
        for seed in range(5):
            data = make_batch(seed).to("cuda")
            with torch.no_grad():
                actual = compiled(data, anchor_node_type=_USER)
                expected = encoder(data, anchor_node_type=_USER)
            graph_counts.append(torch._dynamo.utils.counters["stats"]["unique_graphs"])
            torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
        self.assertEqual(graph_counts[2:], [graph_counts[2]] * 3, graph_counts)

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

    @parameterized.expand(
        [
            param("anchor_only"),
            param("neighbor_readout", readout_mode=_NEIGHBOR_READOUT),
            param("anchor_bias", anchor_based_attention_bias_attr_names=[_HOP]),
            param("pairwise_bias", pairwise_attention_bias_attr_names=["pair_score"]),
            param(
                "pairwise_bias_neighbor_readout",
                pairwise_attention_bias_attr_names=["pair_score"],
                readout_mode=_NEIGHBOR_READOUT,
            ),
            param("relation_bilinear", relation_attention_mode="edge_type_bilinear"),
            param("relation_hgt", relation_attention_mode="edge_type_hgt"),
            param("relation_message", relation_message_mode="edge_type_attention"),
        ]
    )
    def test_whole_encoder_inductor_compile_training_gradients_match_eager(
        self, _: str, **overrides: Any
    ) -> None:
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        if any(name.startswith("relation") for name in overrides):
            relation_message_mode = overrides.pop("relation_message_mode", "none")
            kwargs = _relation_encoder_kwargs(
                relation_message_mode,
                hid_dim=128,
                anchor_based_input_embedding_dict=nn.ModuleDict(
                    {_HOP: nn.Embedding(3, 128)}
                ),
                **overrides,
            )
            encoder = _relation_encoder(_three_relation_batch(0), kwargs)
            make_batch: Callable[[int], HeteroData] = _three_relation_batch
        else:
            kwargs = _encoder_kwargs(hid_dim=128, dropout_rate=0.0, **overrides)
            encoder = _materialized_encoder(_random_batch(0), kwargs)
            if "pairwise_attention_bias_attr_names" in overrides:
                _randomize_pairwise_biases(encoder)
            make_batch = _random_batch
        encoder = encoder.to("cuda").train()
        compiled = torch.compile(encoder)
        graph_counts: list[int] = []
        for seed in range(5):
            data = make_batch(seed).to("cuda")
            with self.subTest(seed=seed):
                _assert_training_gradients_match(
                    self,
                    _gradients(
                        functools.partial(encoder, data, anchor_node_type=_USER),
                        encoder,
                    ),
                    _gradients(
                        functools.partial(compiled, data, anchor_node_type=_USER),
                        encoder,
                    ),
                    atol=1e-3,
                )
            graph_counts.append(torch._dynamo.utils.counters["stats"]["unique_graphs"])
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

    @parameterized.expand(
        [
            param("messages"),
            param("bilinear", mode="none", attention="edge_type_bilinear"),
            param("hgt", mode="none", attention="edge_type_hgt"),
            param(
                "bilinear_with_messages",
                attention="edge_type_bilinear",
            ),
            param(
                "hgt_with_messages",
                mode="edge_type_linear",
                attention="edge_type_hgt",
            ),
        ]
    )
    def test_cuda_relation_training_gradients_match_cpu(
        self,
        _: str,
        mode: str = "edge_type_attention",
        attention: str = "none",
    ) -> None:
        kwargs = _relation_encoder_kwargs(
            mode,
            relation_attention_mode=attention,
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
