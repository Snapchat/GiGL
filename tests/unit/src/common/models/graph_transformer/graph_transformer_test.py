"""Tests for GraphTransformerEncoder."""

from typing import cast

import torch
import torch.nn as nn
from absl.testing import absltest
from torch import Tensor
from torch_geometric.data import HeteroData

from gigl.src.common.models.graph_transformer.graph_transformer import (
    FeedForwardNetwork,
    GraphTransformerEncoder,
    GraphTransformerEncoderLayer,
)
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from tests.test_assets.test_case import TestCase


def _create_simple_hetero_data() -> HeteroData:
    """Create a simple heterogeneous graph for testing.

    Graph structure:
        user0 -- item0
        user1 -- item0, item1
        user2 -- item1

    With reverse edges.
    """
    data = HeteroData()

    data["user"].x = torch.randn(3, 16)
    data["item"].x = torch.randn(2, 8)

    data["user", "buys", "item"].edge_index = torch.tensor(
        [
            [0, 1, 1, 2],
            [0, 0, 1, 1],
        ]
    )
    data["item", "bought_by", "user"].edge_index = torch.tensor(
        [
            [0, 0, 1, 1],
            [0, 1, 1, 2],
        ]
    )

    return data


class TestGraphTransformerEncoder(TestCase):
    def setUp(self) -> None:
        self._user_node_type = NodeType("user")
        self._item_node_type = NodeType("item")
        self._user_to_item_edge_type = EdgeType(
            self._user_node_type, Relation("buys"), self._item_node_type
        )
        self._item_to_user_edge_type = EdgeType(
            self._item_node_type, Relation("bought_by"), self._user_node_type
        )

        self._node_type_to_feat_dim_map: dict[NodeType, int] = {
            self._user_node_type: 16,
            self._item_node_type: 8,
        }
        self._edge_type_to_feat_dim_map: dict[EdgeType, int] = {
            self._user_to_item_edge_type: 0,
            self._item_to_user_edge_type: 0,
        }

        self._hid_dim = 32
        self._out_dim = 16
        self._device = torch.device("cpu")

    def _create_encoder(self, **kwargs: object) -> GraphTransformerEncoder:
        """Create a GraphTransformerEncoder with default test parameters."""
        defaults: dict = dict(
            node_type_to_feat_dim_map=self._node_type_to_feat_dim_map,
            edge_type_to_feat_dim_map=self._edge_type_to_feat_dim_map,
            hid_dim=self._hid_dim,
            out_dim=self._out_dim,
            num_layers=2,
            num_heads=4,
            max_seq_len=10,
            hop_distance=2,
            dropout_rate=0.0,
            attention_dropout_rate=0.0,
        )
        defaults.update(kwargs)
        return GraphTransformerEncoder(**defaults)

    def test_forward_single_node_type(self) -> None:
        """Test forward pass for a single anchor node type."""
        data = _create_simple_hetero_data()
        encoder = self._create_encoder()

        embeddings = encoder(
            data=data,
            anchor_node_type=self._user_node_type,
            device=self._device,
        )

        self.assertEqual(embeddings.shape, (3, self._out_dim))
        self.assertFalse(torch.isnan(embeddings).any())

    def test_forward_different_anchor_node_types(self) -> None:
        """Test forward pass for different anchor node types."""
        data = _create_simple_hetero_data()
        encoder = self._create_encoder()

        user_embeddings = encoder(
            data=data,
            anchor_node_type=self._user_node_type,
            device=self._device,
        )
        item_embeddings = encoder(
            data=data,
            anchor_node_type=self._item_node_type,
            device=self._device,
        )

        self.assertEqual(user_embeddings.shape, (3, self._out_dim))
        self.assertEqual(item_embeddings.shape, (2, self._out_dim))
        self.assertFalse(torch.isnan(user_embeddings).any())
        self.assertFalse(torch.isnan(item_embeddings).any())

    def test_forward_with_l2_normalization(self) -> None:
        """Test that L2 normalization produces unit-length embeddings."""
        data = _create_simple_hetero_data()
        encoder = self._create_encoder(
            should_l2_normalize_embedding_layer_output=True,
        )

        embeddings = encoder(
            data=data,
            anchor_node_type=self._user_node_type,
            device=self._device,
        )

        norms = torch.norm(embeddings, p=2, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_forward_defaults_to_first_node_type(self) -> None:
        """Test that omitted anchor node type defaults to the first node type."""
        data = _create_simple_hetero_data()
        encoder = self._create_encoder()

        embeddings = encoder(
            data=data,
            device=self._device,
        )

        self.assertEqual(embeddings.shape, (3, self._out_dim))
        self.assertFalse(torch.isnan(embeddings).any())

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through the model."""
        data = _create_simple_hetero_data()
        encoder = self._create_encoder()
        encoder.train()

        embeddings = encoder(
            data=data,
            anchor_node_type=self._user_node_type,
            device=self._device,
        )

        loss = embeddings.sum()
        loss.backward()

        # Check gradients exist for encoder parameters
        for name, param in encoder.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(
                    param.grad,
                    f"Parameter {name} has no gradient",
                )

    def test_different_feature_dims(self) -> None:
        """Test that node types with different input dims are projected correctly."""
        data = HeteroData()
        data["a"].x = torch.randn(5, 4)
        data["b"].x = torch.randn(3, 64)
        data["a", "to", "b"].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])
        data["b", "to", "a"].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])

        node_type_a = NodeType("a")
        node_type_b = NodeType("b")

        encoder = GraphTransformerEncoder(
            node_type_to_feat_dim_map={node_type_a: 4, node_type_b: 64},
            edge_type_to_feat_dim_map={
                EdgeType(node_type_a, Relation("to"), node_type_b): 0,
                EdgeType(node_type_b, Relation("to"), node_type_a): 0,
            },
            hid_dim=32,
            out_dim=16,
            num_layers=1,
            num_heads=4,
            max_seq_len=10,
            hop_distance=1,
            dropout_rate=0.0,
        )

        embeddings_a = encoder(
            data=data,
            anchor_node_type=node_type_a,
            device=self._device,
        )
        embeddings_b = encoder(
            data=data,
            anchor_node_type=node_type_b,
            device=self._device,
        )

        self.assertEqual(embeddings_a.shape, (5, 16))
        self.assertEqual(embeddings_b.shape, (3, 16))
        self.assertFalse(torch.isnan(embeddings_a).any())
        self.assertFalse(torch.isnan(embeddings_b).any())

    def test_deterministic_eval_mode(self) -> None:
        """Test that eval mode produces deterministic outputs."""
        data = _create_simple_hetero_data()
        encoder = self._create_encoder()
        encoder.eval()

        with torch.no_grad():
            result_1 = encoder(
                data=data,
                anchor_node_type=self._user_node_type,
                device=self._device,
            )
            result_2 = encoder(
                data=data,
                anchor_node_type=self._user_node_type,
                device=self._device,
            )

        self.assertTrue(torch.allclose(result_1, result_2))


def _create_user_graph_with_pe() -> HeteroData:
    data = HeteroData()

    data["user"].x = torch.tensor(
        [[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.5, 0.0], [0.0, 0.0, 1.0, 1.0]]
    )
    data["user"].random_walk_pe = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    data["user", "connects", "user"].edge_index = torch.tensor(
        [[0, 1, 1, 2], [1, 0, 2, 1]]
    )

    hop_distance = torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    data.hop_distance = hop_distance.to_sparse_csr()
    pairwise_distance = torch.tensor(
        [[0.0, 0.2, 0.4], [0.2, 0.0, 0.3], [0.4, 0.3, 0.0]]
    )
    data.pairwise_distance = pairwise_distance.to_sparse_csr()

    return data


def _create_user_graph_with_ppr_edges() -> HeteroData:
    data = HeteroData()

    data["user"].x = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    data["user", "ppr", "user"].edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]])
    data["user", "ppr", "user"].edge_attr = torch.tensor([0.9, 0.4, 0.7])
    hop_distance = torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    data.hop_distance = hop_distance.to_sparse_csr()

    return data


class TestGraphTransformerEncoderPEModes(TestCase):
    def setUp(self) -> None:
        self._node_type = NodeType("user")
        self._edge_type = EdgeType(
            self._node_type,
            Relation("connects"),
            self._node_type,
        )
        self._device = torch.device("cpu")

    def _create_encoder(self, **kwargs: object) -> GraphTransformerEncoder:
        defaults: dict = dict(
            node_type_to_feat_dim_map={self._node_type: 4},
            edge_type_to_feat_dim_map={self._edge_type: 0},
            hid_dim=8,
            out_dim=6,
            num_layers=1,
            num_heads=2,
            max_seq_len=4,
            hop_distance=1,
            dropout_rate=0.0,
            attention_dropout_rate=0.0,
        )
        defaults.update(kwargs)
        return GraphTransformerEncoder(**defaults)

    def test_additive_mode_matches_base_encoder_when_node_pe_projection_is_zero(
        self,
    ) -> None:
        data = _create_user_graph_with_pe()

        base_encoder = self._create_encoder()
        additive_encoder = self._create_encoder(
            pe_attr_names=["random_walk_pe"],
            pe_integration_mode="add",
        )
        additive_encoder.load_state_dict(base_encoder.state_dict(), strict=False)

        base_encoder.eval()
        additive_encoder.eval()

        with torch.no_grad():
            _ = additive_encoder(
                data=data,
                anchor_node_type=self._node_type,
                device=self._device,
            )
            self.assertIsNotNone(additive_encoder._pe_projection)
            assert additive_encoder._pe_projection is not None
            assert isinstance(additive_encoder._pe_projection, nn.Linear)
            additive_encoder._pe_projection.weight.data.zero_()

            base_embeddings = base_encoder(
                data=data,
                anchor_node_type=self._node_type,
                device=self._device,
            )
            additive_embeddings = additive_encoder(
                data=data,
                anchor_node_type=self._node_type,
                device=self._device,
            )

        self.assertEqual(base_embeddings.shape, (3, 6))
        self.assertTrue(torch.allclose(base_embeddings, additive_embeddings, atol=1e-6))

    def test_forward_accepts_pairwise_attention_bias(self) -> None:
        data = _create_user_graph_with_pe()

        encoder = self._create_encoder(
            pe_attr_names=["random_walk_pe"],
            anchor_based_attention_bias_attr_names=["hop_distance"],
            pairwise_attention_bias_attr_names=["pairwise_distance"],
            pe_integration_mode="add",
        )
        encoder.eval()

        with torch.no_grad():
            embeddings = encoder(
                data=data,
                anchor_node_type=self._node_type,
                device=self._device,
            )

        self.assertEqual(embeddings.shape, (3, 6))
        self.assertFalse(torch.isnan(embeddings).any())

    def test_concat_mode_infers_sequence_width_without_explicit_pe_dim(self) -> None:
        data = _create_user_graph_with_pe()

        encoder = self._create_encoder(
            pe_attr_names=["random_walk_pe"],
            pe_integration_mode="concat",
        )
        encoder.eval()

        with torch.no_grad():
            embeddings = encoder(
                data=data,
                anchor_node_type=self._node_type,
                device=self._device,
            )

        self.assertEqual(embeddings.shape, (3, 6))
        self.assertFalse(torch.isnan(embeddings).any())

    def test_attention_bias_features_are_projected_per_head(self) -> None:
        encoder = self._create_encoder(
            anchor_based_attention_bias_attr_names=["hop_distance"],
            pairwise_attention_bias_attr_names=["pairwise_distance"],
        )

        assert encoder._anchor_pe_attention_bias_projection is not None
        assert encoder._pairwise_pe_attention_bias_projection is not None

        with torch.no_grad():
            encoder._anchor_pe_attention_bias_projection.weight.copy_(
                torch.tensor([[1.0], [2.0]])
            )
            encoder._pairwise_pe_attention_bias_projection.weight.copy_(
                torch.tensor([[3.0], [4.0]])
            )

            attn_bias = encoder._build_attention_bias(
                valid_mask=torch.ones((1, 3), dtype=torch.bool),
                sequences=torch.zeros((1, 3, 8), dtype=torch.float),
                attention_bias_data={
                    "anchor_bias": torch.tensor([[[1.0], [2.0], [3.0]]]),
                    "pairwise_bias": torch.tensor(
                        [
                            [
                                [[0.0], [1.0], [2.0]],
                                [[3.0], [4.0], [5.0]],
                                [[6.0], [7.0], [8.0]],
                            ]
                        ]
                    ),
                    "token_input": None,
                },
            )

        self.assertEqual(attn_bias.shape, (1, 2, 3, 3))
        self.assertEqual(attn_bias[0, 0, 0, 1].item(), 5.0)
        self.assertEqual(attn_bias[0, 1, 0, 1].item(), 8.0)
        self.assertEqual(attn_bias[0, 0, 2, 2].item(), 27.0)
        self.assertEqual(attn_bias[0, 1, 2, 2].item(), 38.0)

    def test_attention_bias_supports_anchor_relative_attrs_and_ppr_weights(
        self,
    ) -> None:
        encoder = self._create_encoder(
            edge_type_to_feat_dim_map={
                EdgeType(self._node_type, Relation("ppr"), self._node_type): 0
            },
            sequence_construction_method="ppr",
            anchor_based_attention_bias_attr_names=["hop_distance", "ppr_weight"],
        )

        assert encoder._anchor_pe_attention_bias_projection is not None

        with torch.no_grad():
            encoder._anchor_pe_attention_bias_projection.weight.copy_(
                torch.tensor([[1.0, 10.0], [2.0, 20.0]])
            )

            attn_bias = encoder._build_attention_bias(
                valid_mask=torch.ones((1, 3), dtype=torch.bool),
                sequences=torch.zeros((1, 3, 8), dtype=torch.float),
                attention_bias_data={
                    "anchor_bias": torch.tensor(
                        [[[1.0, 0.5], [2.0, 0.25], [3.0, 0.125]]]
                    ),
                    "pairwise_bias": None,
                    "token_input": None,
                },
            )

        self.assertEqual(attn_bias.shape, (1, 2, 1, 3))
        self.assertEqual(attn_bias[0, 0, 0, 1].item(), 4.5)
        self.assertEqual(attn_bias[0, 1, 0, 1].item(), 9.0)
        self.assertEqual(attn_bias[0, 0, 0, 2].item(), 4.25)
        self.assertEqual(attn_bias[0, 1, 0, 2].item(), 8.5)

    def test_sinusoidal_sequence_positional_encoding_masks_padding(self) -> None:
        encoder = self._create_encoder(
            sequence_construction_method="ppr",
            sequence_positional_encoding_type="sinusoidal",
        )

        sequence_positional_encoding = encoder._get_sequence_positional_encoding(
            valid_mask=torch.tensor([[True, True, False, False]]),
            sequences=torch.zeros((1, 4, 8), dtype=torch.float),
        )

        assert sequence_positional_encoding is not None
        expected_position_zero = torch.tensor(
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            dtype=torch.float,
        )
        self.assertEqual(sequence_positional_encoding.shape, (1, 4, 8))
        self.assertTrue(
            torch.allclose(sequence_positional_encoding[0, 0], expected_position_zero)
        )
        self.assertFalse(
            torch.allclose(
                sequence_positional_encoding[0, 1],
                torch.zeros(8, dtype=torch.float),
            )
        )
        self.assertTrue(
            torch.allclose(
                sequence_positional_encoding[0, 2:],
                torch.zeros((2, 8), dtype=torch.float),
            )
        )

    def test_khop_sequence_construction_rejects_sequence_position_encoding(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "sequence_positional_encoding_type requires "
            "sequence_construction_method='ppr'",
        ):
            self._create_encoder(
                sequence_construction_method="khop",
                sequence_positional_encoding_type="sinusoidal",
            )

    def test_forward_supports_ppr_sequence_construction(self) -> None:
        data = _create_user_graph_with_ppr_edges()

        encoder = self._create_encoder(
            edge_type_to_feat_dim_map={
                EdgeType(self._node_type, Relation("ppr"), self._node_type): 0
            },
            sequence_construction_method="ppr",
        )
        encoder.eval()

        with torch.no_grad():
            embeddings = encoder(
                data=data,
                anchor_node_type=self._node_type,
                device=self._device,
            )

        self.assertEqual(embeddings.shape, (3, 6))
        self.assertFalse(torch.isnan(embeddings).any())

    def test_forward_supports_sinusoidal_sequence_position_encoding_in_ppr_mode(
        self,
    ) -> None:
        data = _create_user_graph_with_ppr_edges()
        encoder = self._create_encoder(
            edge_type_to_feat_dim_map={
                EdgeType(self._node_type, Relation("ppr"), self._node_type): 0
            },
            sequence_construction_method="ppr",
            sequence_positional_encoding_type="sinusoidal",
            anchor_based_attention_bias_attr_names=["hop_distance", "ppr_weight"],
            anchor_based_input_attr_names=["hop_distance", "ppr_weight"],
        )
        encoder.eval()

        with torch.no_grad():
            _ = encoder(
                data=data,
                anchor_node_type=self._node_type,
                device=self._device,
            )
            assert encoder._sequence_positional_encoding_table is not None
            position_table = cast(Tensor, encoder._sequence_positional_encoding_table)
            original_position_table = torch.clone(position_table)

            embeddings_with_position_encoding = encoder(
                data=data,
                anchor_node_type=self._node_type,
                device=self._device,
            )
            position_table[...] = 0
            embeddings_without_position_encoding = encoder(
                data=data,
                anchor_node_type=self._node_type,
                device=self._device,
            )
            position_table[...] = original_position_table

        self.assertEqual(embeddings_with_position_encoding.shape, (3, 6))
        self.assertFalse(torch.isnan(embeddings_with_position_encoding).any())
        self.assertFalse(
            torch.allclose(
                embeddings_with_position_encoding,
                embeddings_without_position_encoding,
            )
        )

    def test_forward_supports_anchor_relative_and_ppr_token_input_features(
        self,
    ) -> None:
        data = _create_user_graph_with_ppr_edges()
        ppr_edge_type = EdgeType(self._node_type, Relation("ppr"), self._node_type)

        base_encoder = self._create_encoder(
            edge_type_to_feat_dim_map={ppr_edge_type: 0},
            sequence_construction_method="ppr",
        )
        augmented_encoder = self._create_encoder(
            edge_type_to_feat_dim_map={ppr_edge_type: 0},
            sequence_construction_method="ppr",
            anchor_based_input_attr_names=["hop_distance", "ppr_weight"],
        )
        augmented_encoder.load_state_dict(base_encoder.state_dict(), strict=False)

        base_encoder.eval()
        augmented_encoder.eval()

        with torch.no_grad():
            _ = augmented_encoder(
                data=data,
                anchor_node_type=self._node_type,
                device=self._device,
            )
            assert augmented_encoder._token_input_projection is not None
            assert isinstance(augmented_encoder._token_input_projection, nn.Linear)
            augmented_encoder._token_input_projection.weight.data.zero_()

            base_embeddings = base_encoder(
                data=data,
                anchor_node_type=self._node_type,
                device=self._device,
            )
            augmented_embeddings = augmented_encoder(
                data=data,
                anchor_node_type=self._node_type,
                device=self._device,
            )

        self.assertEqual(augmented_embeddings.shape, (3, 6))
        self.assertTrue(
            torch.allclose(base_embeddings, augmented_embeddings, atol=1e-6)
        )

    def test_forward_supports_mixed_embedded_and_continuous_token_input_features(
        self,
    ) -> None:
        data = _create_user_graph_with_ppr_edges()
        ppr_edge_type = EdgeType(self._node_type, Relation("ppr"), self._node_type)

        base_encoder = self._create_encoder(
            edge_type_to_feat_dim_map={ppr_edge_type: 0},
            sequence_construction_method="ppr",
        )
        augmented_encoder = self._create_encoder(
            edge_type_to_feat_dim_map={ppr_edge_type: 0},
            sequence_construction_method="ppr",
            anchor_based_input_attr_names=["hop_distance", "ppr_weight"],
            anchor_based_input_embedding_dict=nn.ModuleDict(
                {"hop_distance": nn.Embedding(8, 8)}
            ),
        )
        augmented_encoder.load_state_dict(base_encoder.state_dict(), strict=False)

        base_encoder.eval()
        augmented_encoder.eval()

        with torch.no_grad():
            _ = augmented_encoder(
                data=data,
                anchor_node_type=self._node_type,
                device=self._device,
            )
            assert augmented_encoder._anchor_based_input_embedding_dict is not None
            hop_distance_embedding = (
                augmented_encoder._anchor_based_input_embedding_dict["hop_distance"]
            )
            assert isinstance(hop_distance_embedding, nn.Embedding)
            hop_distance_embedding.weight[...] = 0
            assert augmented_encoder._token_input_projection is not None
            assert isinstance(augmented_encoder._token_input_projection, nn.Linear)
            augmented_encoder._token_input_projection.weight[...] = 0

            base_embeddings = base_encoder(
                data=data,
                anchor_node_type=self._node_type,
                device=self._device,
            )
            augmented_embeddings = augmented_encoder(
                data=data,
                anchor_node_type=self._node_type,
                device=self._device,
            )

        self.assertEqual(augmented_embeddings.shape, (3, 6))
        self.assertTrue(
            torch.allclose(base_embeddings, augmented_embeddings, atol=1e-6)
        )


class TestFeedForwardNetwork(TestCase):
    """Tests for FeedForwardNetwork with various activations."""

    def test_standard_activation_gelu(self) -> None:
        """Test FFN with default GELU activation."""
        ffn = FeedForwardNetwork(model_dim=32, feedforward_dim=128, activation="gelu")
        x = torch.randn(2, 10, 32)
        out = ffn(x)
        self.assertEqual(out.shape, (2, 10, 32))
        self.assertFalse(torch.isnan(out).any())

    def test_standard_activation_relu(self) -> None:
        """Test FFN with ReLU activation."""
        ffn = FeedForwardNetwork(model_dim=32, feedforward_dim=128, activation="relu")
        x = torch.randn(2, 10, 32)
        out = ffn(x)
        self.assertEqual(out.shape, (2, 10, 32))
        self.assertFalse(torch.isnan(out).any())

    def test_standard_activation_silu(self) -> None:
        """Test FFN with SiLU/Swish activation."""
        ffn = FeedForwardNetwork(model_dim=32, feedforward_dim=128, activation="silu")
        x = torch.randn(2, 10, 32)
        out = ffn(x)
        self.assertEqual(out.shape, (2, 10, 32))
        self.assertFalse(torch.isnan(out).any())

    def test_xglu_activation_swiglu(self) -> None:
        """Test FFN with SwiGLU activation."""
        ffn = FeedForwardNetwork(model_dim=32, feedforward_dim=128, activation="swiglu")
        x = torch.randn(2, 10, 32)
        out = ffn(x)
        self.assertEqual(out.shape, (2, 10, 32))
        self.assertFalse(torch.isnan(out).any())

    def test_xglu_activation_geglu(self) -> None:
        """Test FFN with GeGLU activation."""
        ffn = FeedForwardNetwork(model_dim=32, feedforward_dim=128, activation="geglu")
        x = torch.randn(2, 10, 32)
        out = ffn(x)
        self.assertEqual(out.shape, (2, 10, 32))
        self.assertFalse(torch.isnan(out).any())

    def test_xglu_activation_reglu(self) -> None:
        """Test FFN with ReGLU activation."""
        ffn = FeedForwardNetwork(model_dim=32, feedforward_dim=128, activation="reglu")
        x = torch.randn(2, 10, 32)
        out = ffn(x)
        self.assertEqual(out.shape, (2, 10, 32))
        self.assertFalse(torch.isnan(out).any())

    def test_invalid_activation_raises_error(self) -> None:
        """Test that invalid activation name raises ValueError."""
        with self.assertRaises(ValueError) as context:
            FeedForwardNetwork(model_dim=32, feedforward_dim=128, activation="invalid")
        self.assertIn("Unsupported activation", str(context.exception))

    def test_xglu_has_double_input_projection(self) -> None:
        """Test that XGLU activations project to 2x feedforward_dim."""
        ffn = FeedForwardNetwork(model_dim=32, feedforward_dim=128, activation="swiglu")
        # XGLU projects to 2 * feedforward_dim for gating
        assert ffn._linear_in is not None
        assert ffn._linear_out is not None
        self.assertEqual(ffn._linear_in.out_features, 256)  # 2 * 128
        self.assertEqual(ffn._linear_out.in_features, 128)

    def test_gradient_flow_xglu(self) -> None:
        """Test that gradients flow through XGLU activation."""
        ffn = FeedForwardNetwork(model_dim=32, feedforward_dim=128, activation="swiglu")
        x = torch.randn(2, 10, 32, requires_grad=True)
        out = ffn(x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        assert x.grad is not None
        self.assertFalse(torch.isnan(x.grad).any())


class TestGraphTransformerEncoderLayerActivations(TestCase):
    """Tests for GraphTransformerEncoderLayer with various activations."""

    def test_layer_with_gelu(self) -> None:
        """Test encoder layer with GELU activation."""
        layer = GraphTransformerEncoderLayer(
            model_dim=32, num_heads=4, feedforward_dim=128, activation="gelu"
        )
        x = torch.randn(2, 10, 32)
        out = layer(x)
        self.assertEqual(out.shape, (2, 10, 32))

    def test_layer_with_swiglu(self) -> None:
        """Test encoder layer with SwiGLU activation."""
        layer = GraphTransformerEncoderLayer(
            model_dim=32, num_heads=4, feedforward_dim=128, activation="swiglu"
        )
        x = torch.randn(2, 10, 32)
        out = layer(x)
        self.assertEqual(out.shape, (2, 10, 32))


class TestGraphTransformerEncoderFeedforwardRatio(TestCase):
    """Tests for GraphTransformerEncoder feedforward_ratio parameter."""

    def setUp(self) -> None:
        self._node_type = NodeType("user")
        self._edge_type = EdgeType(
            self._node_type, Relation("connects"), self._node_type
        )
        self._device = torch.device("cpu")

    def _create_data(self) -> HeteroData:
        data = HeteroData()
        data["user"].x = torch.randn(3, 16)
        data["user", "connects", "user"].edge_index = torch.tensor(
            [[0, 1, 2], [1, 2, 0]]
        )
        return data

    def test_default_ratio_for_gelu_is_4(self) -> None:
        """Test that default feedforward_ratio is 4.0 for GELU."""
        encoder = GraphTransformerEncoder(
            node_type_to_feat_dim_map={self._node_type: 16},
            edge_type_to_feat_dim_map={self._edge_type: 0},
            hid_dim=32,
            out_dim=16,
            num_layers=1,
            num_heads=4,
            activation="gelu",
        )
        # Check internal layer's FFN dimension
        layer = encoder._encoder_layers[0]
        assert isinstance(layer, GraphTransformerEncoderLayer)
        # For standard activation, _ffn is a Sequential with Linear as first layer
        assert layer._ffn._ffn is not None
        self.assertEqual(layer._ffn._ffn[0].out_features, 128)  # 32 * 4

    def test_default_ratio_for_swiglu_is_8_over_3(self) -> None:
        """Test that default feedforward_ratio is 8/3 for SwiGLU."""
        encoder = GraphTransformerEncoder(
            node_type_to_feat_dim_map={self._node_type: 16},
            edge_type_to_feat_dim_map={self._edge_type: 0},
            hid_dim=32,
            out_dim=16,
            num_layers=1,
            num_heads=4,
            activation="swiglu",
        )
        # Check internal layer's FFN dimension
        layer = encoder._encoder_layers[0]
        assert isinstance(layer, GraphTransformerEncoderLayer)
        # For XGLU, _linear_in projects to 2 * feedforward_dim
        # feedforward_dim = int(32 * 8/3) = 85
        expected_feedforward_dim = int(32 * 8.0 / 3.0)
        assert layer._ffn._linear_in is not None
        self.assertEqual(
            layer._ffn._linear_in.out_features, expected_feedforward_dim * 2
        )

    def test_custom_feedforward_ratio(self) -> None:
        """Test custom feedforward_ratio overrides default."""
        encoder = GraphTransformerEncoder(
            node_type_to_feat_dim_map={self._node_type: 16},
            edge_type_to_feat_dim_map={self._edge_type: 0},
            hid_dim=32,
            out_dim=16,
            num_layers=1,
            num_heads=4,
            activation="gelu",
            feedforward_ratio=2.0,
        )
        layer = encoder._encoder_layers[0]
        assert isinstance(layer, GraphTransformerEncoderLayer)
        assert layer._ffn._ffn is not None
        self.assertEqual(layer._ffn._ffn[0].out_features, 64)  # 32 * 2

    def test_encoder_forward_with_swiglu(self) -> None:
        """Test encoder forward pass with SwiGLU activation."""
        data = self._create_data()
        encoder = GraphTransformerEncoder(
            node_type_to_feat_dim_map={self._node_type: 16},
            edge_type_to_feat_dim_map={self._edge_type: 0},
            hid_dim=32,
            out_dim=16,
            num_layers=2,
            num_heads=4,
            activation="swiglu",
        )
        encoder.eval()

        with torch.no_grad():
            result = encoder(
                data=data,
                anchor_node_type=self._node_type,
                device=self._device,
            )

        self.assertEqual(result.shape, (3, 16))
        self.assertFalse(torch.isnan(result).any())

    def test_encoder_forward_with_geglu_and_custom_ratio(self) -> None:
        """Test encoder forward pass with GeGLU and custom feedforward_ratio."""
        data = self._create_data()
        encoder = GraphTransformerEncoder(
            node_type_to_feat_dim_map={self._node_type: 16},
            edge_type_to_feat_dim_map={self._edge_type: 0},
            hid_dim=32,
            out_dim=16,
            num_layers=2,
            num_heads=4,
            activation="geglu",
            feedforward_ratio=3.0,
        )
        encoder.eval()

        with torch.no_grad():
            result = encoder(
                data=data,
                anchor_node_type=self._node_type,
                device=self._device,
            )

        self.assertEqual(result.shape, (3, 16))
        self.assertFalse(torch.isnan(result).any())


if __name__ == "__main__":
    absltest.main()
