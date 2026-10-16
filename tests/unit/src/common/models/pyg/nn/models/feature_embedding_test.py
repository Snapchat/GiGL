import copy

import tensorflow as tf
import torch
import torch._dynamo
import torch.nn.functional as F
from absl.testing import parameterized
from tensorflow_metadata.proto.v0.schema_pb2 import Feature

from gigl.common.utils.tensorflow_schema import feature_spec_to_feature_index_map
from gigl.src.common.models.pyg.nn.models.feature_embedding import (
    FeatureEmbeddingLayer,
)
from gigl.src.common.utils.data.training import filter_features
from gigl.src.data_preprocessor.lib.types import FeatureSchema
from tests.test_assets.test_case import TestCase

_VOCAB_SIZE = 7
_EMB_DIM = 5
# (name, number of columns, embedded?) in schema order. Embedded and passthrough features are interleaved
# so the passthrough columns are not contiguous.
_FEATURES = [
    ("dense_a", 3, False),
    ("single_cat", 1, True),
    ("dense_b", 2, False),
    ("multi_cat", 4, True),
    ("dense_c", 1, False),
]
_EMBEDDED_NAMES = [name for name, _, embedded in _FEATURES if embedded]
_INPUT_DIM = sum(width for _, width, _ in _FEATURES)


def _build_schema() -> FeatureSchema:
    feature_spec = {
        name: tf.io.FixedLenFeature(
            shape=[width] if width > 1 else [],
            dtype=tf.int64 if embedded else tf.float32,
        )
        for name, width, embedded in _FEATURES
    }
    schema = {}
    for name, _, embedded in _FEATURES:
        feature = Feature()
        if embedded:
            feature.int_domain.min = 0
            feature.int_domain.max = _VOCAB_SIZE - 1
        schema[name] = feature
    return FeatureSchema(
        schema=schema,
        feature_spec=feature_spec,
        feature_index=feature_spec_to_feature_index_map(feature_spec),
        feature_vocab={},
    )


def _build_layer() -> FeatureEmbeddingLayer:
    return FeatureEmbeddingLayer(
        features_to_embed={name: _EMB_DIM for name in _EMBEDDED_NAMES},
        feature_schema=_build_schema(),
        feature_dim=_INPUT_DIM,
        padding_idx=0,
    )


def _random_input(num_rows: int, seed: int = 0) -> torch.Tensor:
    """Dense columns are floats; embedded columns are vocab ids with 0 as the padding id."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(num_rows, _INPUT_DIM, generator=generator)
    schema = _build_schema()
    for name in _EMBEDDED_NAMES:
        start, end = schema.feature_index[name]
        ids = torch.randint(
            0, _VOCAB_SIZE, (num_rows, end - start), generator=generator
        )
        x[:, start:end] = ids.float()
    # Make sure the multi-valued feature has some fully padded rows and some partially padded ones.
    start, end = schema.feature_index["multi_cat"]
    x[0, start:end] = 0
    x[1, start + 1 : end] = 0
    return x


def _reference_forward(layer: FeatureEmbeddingLayer, x: torch.Tensor) -> torch.Tensor:
    """Independent re-statement of the layer's semantics using the layer's embedding weights."""
    schema = _build_schema()
    state = layer.state_dict()
    non_embed_names = [name for name, _, embedded in _FEATURES if not embedded]
    outputs = [filter_features(schema, non_embed_names, x)]
    for name in _EMBEDDED_NAMES:
        weight_key = next(key for key in state if f".{name}." in key)
        start, end = schema.feature_index[name]
        emb = F.embedding(x[:, start:end].long(), state[weight_key], padding_idx=0)
        non_padding = (emb != 0).all(dim=2).sum(dim=1, keepdim=True).float()
        divisor = torch.where(
            non_padding == 0, torch.full_like(non_padding, 1e-8), non_padding
        )
        outputs.append(emb.sum(dim=1) / divisor)
    return torch.cat(outputs, dim=1)


class FeatureEmbeddingLayerTest(TestCase, parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(0)

    def test_forward_matches_reference_for_all_feature_kinds(self) -> None:
        layer = _build_layer()
        x = _random_input(64)
        out = layer(x)
        self.assertEqual(out.shape, (64, layer.out_dim))
        torch.testing.assert_close(out, _reference_forward(layer, x))
        # Fully padded multi-valued rows aggregate to zeros rather than NaN.
        self.assertTrue(torch.isfinite(out).all())

    def test_state_dict_has_no_index_buffers_and_old_checkpoints_load(self) -> None:
        layer = _build_layer()
        keys = list(layer.state_dict().keys())
        self.assertTrue(all("embedding_layers" in key for key in keys), keys)
        self.assertEqual(len(keys), len(_EMBEDDED_NAMES))

        # A checkpoint saved before the index buffer existed has exactly these keys.
        other = _build_layer()
        other.load_state_dict(copy.deepcopy(layer.state_dict()), strict=True)
        x = _random_input(8)
        torch.testing.assert_close(other(x), layer(x))

    def test_index_buffer_follows_dtype_and_device_moves(self) -> None:
        layer = _build_layer()
        x = _random_input(8)
        expected = layer(x)
        layer.to(torch.float64)
        self.assertEqual(layer._non_embed_index.dtype, torch.long)
        torch.testing.assert_close(
            layer(x.double()), expected.double(), rtol=1e-6, atol=1e-6
        )

        meta_layer = _build_layer().to("meta")
        self.assertEqual(meta_layer._non_embed_index.device.type, "meta")

    @parameterized.parameters(
        {"device": "cpu"},
        {"device": "cuda"},
    )
    def test_to_device_keeps_forward_working(self, device: str) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        layer = _build_layer()
        x = _random_input(8)
        expected = layer(x)
        layer.to(device)
        self.assertEqual(layer._non_embed_index.device.type, device)
        torch.testing.assert_close(layer(x.to(device)).cpu(), expected)

    def test_torch_compile_fullgraph_dynamic_matches_eager_without_recompiles(
        self,
    ) -> None:
        torch._dynamo.reset()
        layer = _build_layer()
        compiled = torch.compile(layer, fullgraph=True, dynamic=True, backend="eager")
        torch._dynamo.utils.counters.clear()
        for num_rows in (16, 37, 101):
            x = _random_input(num_rows, seed=num_rows)
            torch.testing.assert_close(compiled(x), layer(x))
        # fullgraph=True already raises on a graph break; a single unique graph means no recompiles.
        self.assertEqual(torch._dynamo.utils.counters["stats"]["unique_graphs"], 1)
        torch._dynamo.reset()
