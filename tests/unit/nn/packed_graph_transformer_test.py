"""The packed ``GraphTransformerEncoder`` path matches the padded path."""

import copy
import importlib
import itertools
import multiprocessing
import unittest
import warnings
from typing import Any, Callable, Literal, cast
from unittest import mock

import torch
import torch.nn as nn
from absl.testing import absltest
from parameterized import param, parameterized
from torch._subclasses.fake_tensor import FakeTensorMode
from torch_geometric.data import HeteroData

import gigl.nn.graph_transformer
import gigl.nn.packed_attention
from gigl.nn.graph_transformer import (
    GraphTransformerEncoder,
    GraphTransformerEncoderLayer,
)
from gigl.nn.packed_attention import (
    _padded_packed_attention,
    _varlen_attention,
    _varlen_attention_backward,
    _varlen_attention_forward,
)
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.transforms.add_positional_encodings import AddHeteroHopDistanceEncoding
from gigl.transforms.static_khop import (
    KhopLayoutTooLargeError,
    PackedSequences,
    pack_sequences,
)
from tests.test_assets.test_case import TestCase

_USER = NodeType("user")
_ITEM = NodeType("item")
_FOLLOWS = EdgeType(_USER, Relation("follows"), _USER)
_BUYS = EdgeType(_USER, Relation("buys"), _ITEM)
_BOUGHT_BY = EdgeType(_ITEM, Relation("bought_by"), _USER)
_HOP = "hop_distance"
_MAX_SEQ_LEN = 24


def _check_custom_ops_in_child(result_queue: Any) -> None:
    """Imports the packed attention module in a fresh process and runs a fake op."""
    from torch._subclasses.fake_tensor import FakeTensorMode as ChildFakeTensorMode

    from gigl.nn.packed_attention import _varlen_attention_forward as forward_op

    with ChildFakeTensorMode():
        query = torch.empty(6, 2, 64)
        cu_seqlens = torch.empty(3, dtype=torch.int32)
        output = forward_op(query, query, query, cu_seqlens, cu_seqlens, 4, 4, True)[0]
    result_queue.put(tuple(output.shape))


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
            param("relation_message", relation_message_mode="edge_type_linear"),
            param("relation_attention", relation_attention_mode="edge_type_bilinear"),
            param("anchor_bias", anchor_based_attention_bias_attr_names=[_HOP]),
            param("head_dim_not_multiple_of_8", num_heads=4),
            param("three_hops", hop_distance=3),
            param("attention_dropout", attention_dropout_rate=0.1),
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

    @parameterized.expand([param("one_anchor", [3]), param("two_anchors", [3, 40])])
    def test_training_gradients_match_padded_path_on_tiny_batches(
        self, _: str, anchors: list[int]
    ) -> None:
        data = _random_batch(0)
        anchor_node_ids = torch.tensor(anchors)
        encoder = _materialized_encoder(
            data, _encoder_kwargs(dropout_rate=0.0, max_seq_len=256)
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

    def test_batch_with_an_oversized_layout_runs_on_the_padded_path(self) -> None:
        # One hub with more neighbours than the layout budget allows at two hops.
        num_nodes = 6000
        data = HeteroData()
        data[_USER].x = torch.randn(num_nodes, 8)
        data[_USER].batch_size = 1
        data[_FOLLOWS.tuple_repr()].edge_index = torch.stack(
            [torch.arange(1, num_nodes), torch.zeros(num_nodes - 1, dtype=torch.long)]
        )
        data = AddHeteroHopDistanceEncoding(h_max=2, sampling_direction="in")(data)
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

    def test_varlen_attention_fake_forward_shapes(self) -> None:
        with FakeTensorMode():
            cu_seqlens = torch.empty(4, dtype=torch.int32)
            query, key, value = (torch.empty(30, 2, 64) for _ in range(3))
            for compute_logsumexp, padded_length in ((True, 32), (False, 0)):
                output, logsumexp = _varlen_attention_forward(
                    query,
                    key,
                    value,
                    cu_seqlens,
                    cu_seqlens,
                    24,
                    24,
                    compute_logsumexp,
                )
                with self.subTest(compute_logsumexp=compute_logsumexp):
                    self.assertEqual(output.shape, query.shape)
                    self.assertEqual(logsumexp.shape, (3, 2, padded_length))

    def test_custom_ops_register_in_a_spawned_process(self) -> None:
        context = multiprocessing.get_context("spawn")
        result_queue = context.Queue()
        process = context.Process(
            target=_check_custom_ops_in_child, args=(result_queue,)
        )
        process.start()
        try:
            self.assertEqual(result_queue.get(timeout=300), (6, 2, 64))
        finally:
            process.join(timeout=60)
        self.assertEqual(process.exitcode, 0)

    def test_custom_ops_survive_a_module_reload(self) -> None:
        importlib.reload(gigl.nn.packed_attention)
        with FakeTensorMode():
            query = torch.empty(6, 2, 64)
            cu_seqlens = torch.empty(3, dtype=torch.int32)
            output = gigl.nn.packed_attention._varlen_attention_forward(
                query, query, query, cu_seqlens, cu_seqlens, 4, 4, True
            )[0]
        self.assertEqual(output.shape, query.shape)

    def test_varlen_attention_fake_backward_shapes(self) -> None:
        with FakeTensorMode():
            cu_seqlens = torch.empty(4, dtype=torch.int32)
            query, key, value = (torch.empty(30, 2, 64) for _ in range(3))
            gradients = _varlen_attention_backward(
                torch.empty(30, 2, 64),
                query,
                key,
                value,
                torch.empty(30, 2, 64),
                torch.empty(3, 2, 32),
                cu_seqlens,
                cu_seqlens,
                24,
                24,
            )
        for gradient in gradients:
            self.assertEqual(gradient.shape, query.shape)

    def test_whole_encoder_compile_does_not_recompile_across_batches(self) -> None:
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        encoder = _materialized_encoder(_random_batch(0), _encoder_kwargs())
        graph_counts = []
        with mock.patch.object(encoder, "_use_packed_backend", return_value=True):
            compiled = torch.compile(encoder, backend="eager")
            for seed in range(5):
                with torch.no_grad():
                    compiled(_random_batch(seed), anchor_node_type=_USER)
                graph_counts.append(
                    torch._dynamo.utils.counters["stats"]["unique_graphs"]
                )
        # The first batches may compile (static, then dynamic shapes); later ones reuse the graphs.
        self.assertEqual(graph_counts[2:], [graph_counts[2]] * 3, graph_counts)

    def test_fallback_under_compile_does_not_keep_recompiling(self) -> None:
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        encoder = _materialized_encoder(_random_batch(0), _encoder_kwargs())
        real_khop_layout = gigl.nn.graph_transformer.khop_layout
        fall_back = {"now": False}

        def khop_layout_or_too_large(*args: Any, **kwargs: Any) -> Any:
            if fall_back["now"]:
                raise KhopLayoutTooLargeError("too many candidates")
            return real_khop_layout(*args, **kwargs)

        graph_counts = []
        with (
            mock.patch.object(encoder, "_use_packed_backend", return_value=True),
            mock.patch.object(
                gigl.nn.graph_transformer,
                "khop_layout",
                side_effect=khop_layout_or_too_large,
            ),
        ):
            compiled = torch.compile(encoder, backend="eager")
            for seed in range(8):
                fall_back["now"] = seed == 3
                with torch.no_grad():
                    compiled(_random_batch(seed), anchor_node_type=_USER)
                graph_counts.append(
                    torch._dynamo.utils.counters["stats"]["unique_graphs"]
                )
        self.assertEqual(encoder._packed_fallback_count, 1)
        # The fallback batch may add graphs; batches after it reuse them.
        self.assertEqual(graph_counts[5:], [graph_counts[5]] * 3, graph_counts)


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

    def test_cuda_training_step_with_fewer_tokens_than_max_seq_len(self) -> None:
        # The backward kernel rejects a length bound above the token count.
        data = _random_batch(0)
        kwargs = _encoder_kwargs(
            hid_dim=128,
            max_seq_len=256,
            dropout_rate=0.0,
            anchor_based_input_embedding_dict=nn.ModuleDict(
                {_HOP: nn.Embedding(3, 128)}
            ),
        )
        cpu_encoder = _materialized_encoder(data, kwargs).train()
        cuda_encoder = copy.deepcopy(cpu_encoder).to("cuda")
        cuda_data = _random_batch(0).to("cuda")
        for anchors in ([3], [3, 40]):
            anchor_node_ids = torch.tensor(anchors)
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
                        anchor_node_ids=anchor_node_ids.to("cuda"),
                    ),
                    cuda_encoder,
                ),
                atol=1e-4,
            )

    def test_varlen_attention_without_grad_matches_padded_packed_attention(
        self,
    ) -> None:
        device = torch.device("cuda")
        lengths = torch.tensor([5, 1, 24, 3], device=device)
        num_tokens = int(lengths.sum())
        _, packed = pack_sequences(lengths, _MAX_SEQ_LEN)
        query, key, value = (
            torch.randn(num_tokens, 2, 64, device=device, requires_grad=True)
            for _ in range(3)
        )
        with torch.no_grad():
            actual = _varlen_attention(
                query,
                key,
                value,
                packed.cu_seqlens,
                packed.cu_seqlens,
                _MAX_SEQ_LEN,
                _MAX_SEQ_LEN,
            )
            expected = _padded_packed_attention(query, key, value, packed, False)
        self.assertFalse(actual.requires_grad)
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-4)

    def test_varlen_attention_ops_pass_opcheck(self) -> None:
        device = torch.device("cuda")
        lengths = torch.tensor([5, 1, 24, 3], device=device)
        num_tokens = int(lengths.sum())
        _, packed = pack_sequences(lengths, _MAX_SEQ_LEN)
        query, key, value = (
            torch.randn(num_tokens, 2, 64, device=device) for _ in range(3)
        )
        forward_args = (
            query,
            key,
            value,
            packed.cu_seqlens,
            packed.cu_seqlens,
            _MAX_SEQ_LEN,
            _MAX_SEQ_LEN,
            True,
        )
        test_utils = ("test_schema", "test_faketensor")
        torch.library.opcheck(
            _varlen_attention_forward, forward_args, test_utils=test_utils
        )
        output, logsumexp = _varlen_attention_forward(*forward_args)
        torch.library.opcheck(
            _varlen_attention_backward,
            (
                torch.randn_like(output),
                query,
                key,
                value,
                output,
                logsumexp,
                packed.cu_seqlens,
                packed.cu_seqlens,
                _MAX_SEQ_LEN,
                _MAX_SEQ_LEN,
            ),
            test_utils=test_utils,
        )

    def test_whole_encoder_inductor_compile_matches_eager_without_recompiling(
        self,
    ) -> None:
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        kwargs = _encoder_kwargs(
            hid_dim=128,
            dropout_rate=0.0,
            anchor_based_input_embedding_dict=nn.ModuleDict(
                {_HOP: nn.Embedding(3, 128)}
            ),
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
