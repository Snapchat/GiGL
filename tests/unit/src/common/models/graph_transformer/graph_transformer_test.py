"""Tests for GraphTransformerEncoder."""

import torch
from absl.testing import absltest
from torch_geometric.data import HeteroData

from gigl.src.common.models.graph_transformer.graph_transformer import (
    GraphTransformerEncoder,
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
        """Test forward pass requesting a single node type."""
        data = _create_simple_hetero_data()
        encoder = self._create_encoder()

        result = encoder(
            data=data,
            output_node_types=[self._user_node_type],
            device=self._device,
        )

        self.assertIn(self._user_node_type, result)
        # 3 user nodes, out_dim=16
        self.assertEqual(result[self._user_node_type].shape, (3, self._out_dim))
        self.assertFalse(torch.isnan(result[self._user_node_type]).any())

    def test_forward_multiple_node_types(self) -> None:
        """Test forward pass requesting multiple node types."""
        data = _create_simple_hetero_data()
        encoder = self._create_encoder()

        result = encoder(
            data=data,
            output_node_types=[self._user_node_type, self._item_node_type],
            device=self._device,
        )

        self.assertIn(self._user_node_type, result)
        self.assertIn(self._item_node_type, result)
        self.assertEqual(result[self._user_node_type].shape, (3, self._out_dim))
        self.assertEqual(result[self._item_node_type].shape, (2, self._out_dim))
        self.assertFalse(torch.isnan(result[self._user_node_type]).any())
        self.assertFalse(torch.isnan(result[self._item_node_type]).any())

    def test_forward_with_l2_normalization(self) -> None:
        """Test that L2 normalization produces unit-length embeddings."""
        data = _create_simple_hetero_data()
        encoder = self._create_encoder(
            should_l2_normalize_embedding_layer_output=True,
        )

        result = encoder(
            data=data,
            output_node_types=[self._user_node_type],
            device=self._device,
        )

        norms = torch.norm(result[self._user_node_type], p=2, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_missing_node_type_returns_empty_tensor(self) -> None:
        """Test that requesting a missing node type returns an empty tensor."""
        data = _create_simple_hetero_data()
        encoder = self._create_encoder()

        missing_type = NodeType("nonexistent")
        result = encoder(
            data=data,
            output_node_types=[missing_type],
            device=self._device,
        )

        self.assertIn(missing_type, result)
        self.assertEqual(result[missing_type].numel(), 0)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through the model."""
        data = _create_simple_hetero_data()
        encoder = self._create_encoder()
        encoder.train()

        result = encoder(
            data=data,
            output_node_types=[self._user_node_type],
            device=self._device,
        )

        loss = result[self._user_node_type].sum()
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

        result = encoder(
            data=data,
            output_node_types=[node_type_a, node_type_b],
            device=self._device,
        )

        self.assertEqual(result[node_type_a].shape, (5, 16))
        self.assertEqual(result[node_type_b].shape, (3, 16))
        self.assertFalse(torch.isnan(result[node_type_a]).any())
        self.assertFalse(torch.isnan(result[node_type_b]).any())

    def test_deterministic_eval_mode(self) -> None:
        """Test that eval mode produces deterministic outputs."""
        data = _create_simple_hetero_data()
        encoder = self._create_encoder()
        encoder.eval()

        with torch.no_grad():
            result_1 = encoder(
                data=data,
                output_node_types=[self._user_node_type],
                device=self._device,
            )
            result_2 = encoder(
                data=data,
                output_node_types=[self._user_node_type],
                device=self._device,
            )

        self.assertTrue(
            torch.allclose(
                result_1[self._user_node_type],
                result_2[self._user_node_type],
            )
        )


def _create_user_graph_with_pe() -> HeteroData:
    data = HeteroData()

    data["user"].x = torch.tensor(
        [[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.5, 0.0], [0.0, 0.0, 1.0, 1.0]]
    )
    data["user"].random_walk_pe = torch.tensor(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    )
    data["user", "connects", "user"].edge_index = torch.tensor(
        [[0, 1, 1, 2], [1, 0, 2, 1]]
    )

    hop_distance = torch.tensor(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]
    )
    data.hop_distance = hop_distance.to_sparse_csr()
    pairwise_distance = torch.tensor(
        [[0.0, 0.2, 0.4], [0.2, 0.0, 0.3], [0.4, 0.3, 0.0]]
    )
    data.pairwise_distance = pairwise_distance.to_sparse_csr()

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

    def test_additive_mode_matches_base_encoder_when_node_pe_projection_is_zero(self) -> None:
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
            self.assertIsNotNone(additive_encoder._node_pe_projection)
            assert additive_encoder._node_pe_projection is not None
            additive_encoder._node_pe_projection.weight.data.zero_()

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
            anchor_based_pe_attr_names=["hop_distance"],
            pairwise_pe_attr_names=["pairwise_distance"],
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
            anchor_based_pe_attr_names=["hop_distance"],
            pairwise_pe_attr_names=["pairwise_distance"],
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
                },
            )

        self.assertEqual(attn_bias.shape, (1, 2, 3, 3))
        self.assertEqual(attn_bias[0, 0, 0, 1].item(), 4.0)
        self.assertEqual(attn_bias[0, 1, 0, 1].item(), 8.0)
        self.assertEqual(attn_bias[0, 0, 2, 2].item(), 27.0)
        self.assertEqual(attn_bias[0, 1, 2, 2].item(), 38.0)


if __name__ == "__main__":
    absltest.main()
