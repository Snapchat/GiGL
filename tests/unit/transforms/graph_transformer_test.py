"""
Tests for heterodata_to_graph_transformer_input transform.
"""

import torch
import torch.nn as nn
from absl.testing import absltest
from torch_geometric.data import HeteroData

from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.transforms.graph_transformer import (
    _get_k_hop_neighbors_sparse,
    heterodata_to_graph_transformer_input,
)
from tests.test_assets.test_case import TestCase


def create_simple_hetero_data() -> HeteroData:
    """Create a simple heterogeneous graph for testing.

    Graph structure:
        user0 -- item0
        user1 -- item0, item1
        user2 -- item1

    With reverse edges.
    """
    data = HeteroData()

    # Node features
    data["user"].x = torch.randn(3, 16)  # 3 users with 16-dim features
    data["item"].x = torch.randn(2, 16)  # 2 items with 16-dim features

    # Edges: user -> item
    data["user", "buys", "item"].edge_index = torch.tensor(
        [
            [0, 1, 1, 2],  # source (user)
            [0, 0, 1, 1],  # target (item)
        ]
    )

    # Reverse edges: item -> user
    data["item", "bought_by", "user"].edge_index = torch.tensor(
        [
            [0, 0, 1, 1],  # source (item)
            [0, 1, 1, 2],  # target (user)
        ]
    )

    return data


def create_larger_hetero_data(num_users: int = 10, num_items: int = 5) -> HeteroData:
    """Create a larger heterogeneous graph for testing."""
    data = HeteroData()

    # Node features
    data["user"].x = torch.randn(num_users, 32)
    data["item"].x = torch.randn(num_items, 32)

    # Create random edges (each user connects to ~2 items on average)
    num_edges = num_users * 2
    src = torch.randint(0, num_users, (num_edges,))
    dst = torch.randint(0, num_items, (num_edges,))

    data["user", "buys", "item"].edge_index = torch.stack([src, dst])
    data["item", "bought_by", "user"].edge_index = torch.stack([dst, src])

    return data


def create_ppr_sequence_hetero_data() -> HeteroData:
    """Create a graph with explicit PPR edges for sequence-construction tests."""
    data = HeteroData()

    data["user"].x = torch.tensor([[10.0, 0.0], [11.0, 0.0]])
    data["item"].x = torch.tensor([[0.0, 20.0], [0.0, 21.0]])

    data["user", "ppr", "item"].edge_index = torch.tensor(
        [
            [0, 0, 1],
            [1, 0, 0],
        ]
    )
    data["user", "ppr", "item"].edge_attr = torch.tensor([0.9, 0.6, 0.8])

    data["user", "ppr", "user"].edge_index = torch.tensor(
        [
            [0, 1],
            [1, 0],
        ]
    )
    data["user", "ppr", "user"].edge_attr = torch.tensor([0.4, 0.3])

    homo_data = data.to_homogeneous()
    node_type_order = list(getattr(homo_data, "_node_type_names", data.node_types))
    offsets = {}
    offset = 0
    for node_type in node_type_order:
        offsets[node_type] = offset
        offset += data[node_type].num_nodes

    total_nodes = homo_data.num_nodes
    hop_distance = torch.zeros((total_nodes, total_nodes), dtype=torch.float)

    user0_idx = offsets["user"] + 0
    user1_idx = offsets["user"] + 1
    item0_idx = offsets["item"] + 0
    item1_idx = offsets["item"] + 1

    hop_distance[user0_idx, user0_idx] = 0.0
    hop_distance[user0_idx, user1_idx] = 1.0
    hop_distance[user0_idx, item0_idx] = 2.0
    hop_distance[user0_idx, item1_idx] = 3.0

    hop_distance[user1_idx, user0_idx] = 1.0
    hop_distance[user1_idx, user1_idx] = 0.0
    hop_distance[user1_idx, item0_idx] = 4.0
    hop_distance[user1_idx, item1_idx] = 5.0

    data.hop_distance = hop_distance.to_sparse_csr()

    return data


def create_relation_mask_hetero_data() -> HeteroData:
    """Create a single-node-type graph with overlapping directed relations."""
    data = HeteroData()

    data["user"].x = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    data["user", "follows", "user"].edge_index = torch.tensor([[0, 1], [1, 2]])
    data["user", "likes", "user"].edge_index = torch.tensor([[0], [1]])

    return data


class TestGetKHopNeighborsSparse(TestCase):
    """Tests for _get_k_hop_neighbors_sparse helper function."""

    def test_one_hop(self):
        """Test 1-hop neighbor extraction with sparse implementation."""
        # Simple chain: 0 -> 1 -> 2 -> 3
        edge_index = torch.tensor(
            [
                [0, 1, 2],
                [1, 2, 3],
            ]
        )
        anchor_indices = torch.tensor([0])

        reachable = _get_k_hop_neighbors_sparse(
            anchor_indices=anchor_indices,
            edge_index=edge_index,
            num_nodes=4,
            k=1,
            device=torch.device("cpu"),
        )

        # Should include anchor (0) and 1-hop neighbor (1)
        mask_indices = reachable.indices()
        nodes_reached = mask_indices[1].tolist()
        self.assertIn(0, nodes_reached)  # anchor
        self.assertIn(1, nodes_reached)  # 1-hop neighbor

    def test_two_hop(self):
        """Test 2-hop neighbor extraction with sparse implementation."""
        # Simple chain: 0 -> 1 -> 2 -> 3
        edge_index = torch.tensor(
            [
                [0, 1, 2],
                [1, 2, 3],
            ]
        )
        anchor_indices = torch.tensor([0])

        reachable = _get_k_hop_neighbors_sparse(
            anchor_indices=anchor_indices,
            edge_index=edge_index,
            num_nodes=4,
            k=2,
            device=torch.device("cpu"),
        )

        # Should include anchor (0), 1-hop (1), and 2-hop (2)
        mask_indices = reachable.indices()
        nodes_reached = mask_indices[1].tolist()
        self.assertIn(0, nodes_reached)  # anchor
        self.assertIn(1, nodes_reached)  # 1-hop
        self.assertIn(2, nodes_reached)  # 2-hop
        self.assertNotIn(3, nodes_reached)  # 3-hop, not reachable

    def test_batched_anchors(self):
        """Test sparse k-hop with multiple anchors."""
        # Star graph: center (0) connects to 1, 2, 3, 4
        edge_index = torch.tensor(
            [
                [0, 0, 0, 0, 1, 2, 3, 4],
                [1, 2, 3, 4, 0, 0, 0, 0],
            ]
        )
        anchor_indices = torch.tensor([1, 2])  # Two anchors

        reachable = _get_k_hop_neighbors_sparse(
            anchor_indices=anchor_indices,
            edge_index=edge_index,
            num_nodes=5,
            k=2,
            device=torch.device("cpu"),
        )

        # Both anchors should reach center (0) in 1 hop
        # And reach other leaf nodes in 2 hops
        self.assertTrue(reachable._nnz() > 0)

    def test_disconnected_node(self):
        """Test neighbor extraction for disconnected node."""
        edge_index = torch.tensor(
            [
                [1],
                [2],
            ]
        )
        anchor_indices = torch.tensor([0])  # Node 0 is disconnected

        reachable = _get_k_hop_neighbors_sparse(
            anchor_indices=anchor_indices,
            edge_index=edge_index,
            num_nodes=3,
            k=2,
            device=torch.device("cpu"),
        )

        # Only anchor itself should be reachable
        mask_indices = reachable.indices()
        nodes_reached = mask_indices[1].tolist()
        self.assertEqual(nodes_reached, [0])


class TestHeteroToGraphTransformerInput(TestCase):
    """Tests for heterodata_to_graph_transformer_input function."""

    def test_basic_transform(self):
        """Test basic transformation with simple data."""
        data = create_simple_hetero_data()

        (
            sequences,
            valid_mask,
            attention_bias_data,
        ) = heterodata_to_graph_transformer_input(
            data=data,
            batch_size=2,
            max_seq_len=10,
            anchor_node_type="user",
            hop_distance=2,
        )

        # Check output shapes
        self.assertEqual(sequences.shape[0], 2)  # batch_size
        self.assertEqual(sequences.shape[1], 10)  # max_seq_len
        self.assertEqual(sequences.shape[2], 16)  # feature_dim (inferred)

        self.assertEqual(valid_mask.shape, (2, 10))
        self.assertIsInstance(attention_bias_data, dict)
        self.assertIn("anchor_bias", attention_bias_data)
        self.assertIn("pairwise_bias", attention_bias_data)
        self.assertIn("pairwise_relation_mask", attention_bias_data)

    def test_attention_mask_validity(self):
        """Test that attention mask correctly identifies valid positions."""
        data = create_simple_hetero_data()

        (
            sequences,
            valid_mask,
            attention_bias_data,
        ) = heterodata_to_graph_transformer_input(
            data=data,
            batch_size=1,
            max_seq_len=20,
            anchor_node_type="user",
            hop_distance=2,
        )

        # Valid mask should be boolean with True for valid positions, False for padding
        num_valid = int(valid_mask[0].sum().item())
        self.assertGreater(num_valid, 0)  # At least anchor should be valid

        # Padding positions should have padding value (0.0)
        if num_valid < 20:
            padding_features = sequences[0, num_valid:]
            self.assertTrue(
                torch.allclose(padding_features, torch.zeros_like(padding_features))
            )

    def test_anchor_first(self):
        """Test that anchor node is first when include_anchor_first=True."""
        data = create_simple_hetero_data()

        # Get anchor node feature
        homo_data = data.to_homogeneous()
        node_type_order = list(getattr(homo_data, "_node_type_names", data.node_types))
        user_offset = 0
        for node_type in node_type_order:
            if node_type == "user":
                break
            user_offset += data[node_type].num_nodes
        anchor_feature = homo_data.x[user_offset]  # First user node

        sequences, _, _ = heterodata_to_graph_transformer_input(
            data=data,
            batch_size=1,
            max_seq_len=10,
            anchor_node_type="user",
            include_anchor_first=True,
        )

        # First position should be anchor node
        self.assertTrue(torch.allclose(sequences[0, 0], anchor_feature))

    def test_different_anchor_types(self):
        """Test with different anchor node types."""
        data = create_simple_hetero_data()

        # Test with 'item' as anchor
        (
            sequences,
            valid_mask,
            attention_bias_data,
        ) = heterodata_to_graph_transformer_input(
            data=data,
            batch_size=1,
            max_seq_len=10,
            anchor_node_type="item",
            hop_distance=2,
        )

        self.assertEqual(sequences.shape[0], 1)
        self.assertEqual(sequences.shape[2], 16)

    def test_larger_batch(self):
        """Test with larger batch size."""
        data = create_larger_hetero_data(num_users=20, num_items=10)

        (
            sequences,
            valid_mask,
            attention_bias_data,
        ) = heterodata_to_graph_transformer_input(
            data=data,
            batch_size=8,
            max_seq_len=50,
            anchor_node_type="user",
            hop_distance=2,
        )

        self.assertEqual(sequences.shape[0], 8)
        self.assertEqual(sequences.shape[1], 50)
        self.assertEqual(valid_mask.shape, (8, 50))

    def test_truncation(self):
        """Test that sequences are truncated to max_seq_len."""
        data = create_larger_hetero_data(num_users=50, num_items=50)

        sequences, valid_mask, _ = heterodata_to_graph_transformer_input(
            data=data,
            batch_size=5,
            max_seq_len=10,  # Small max_seq_len
            anchor_node_type="user",
            hop_distance=2,
        )

        # Output should respect max_seq_len
        self.assertEqual(sequences.shape[1], 10)

        # All valid counts should be <= 10
        self.assertTrue((valid_mask.sum(dim=1) <= 10).all())

    def test_custom_padding_value(self):
        """Test custom padding value."""
        data = create_simple_hetero_data()

        (
            sequences,
            valid_mask,
            attention_bias_data,
        ) = heterodata_to_graph_transformer_input(
            data=data,
            batch_size=1,
            max_seq_len=50,
            anchor_node_type="user",
            padding_value=-1.0,
        )

        num_valid = int(valid_mask[0].sum().item())
        if num_valid < 50:
            # Padding should be -1.0
            padding_features = sequences[0, num_valid:]
            expected_padding = torch.full_like(padding_features, -1.0)
            self.assertTrue(torch.allclose(padding_features, expected_padding))

    def test_ppr_sequence_construction_sorts_tokens_by_weight(self):
        """Test that PPR mode uses outgoing PPR edges ordered by descending weight."""
        data = create_ppr_sequence_hetero_data()

        sequences, valid_mask, _ = heterodata_to_graph_transformer_input(
            data=data,
            batch_size=2,
            max_seq_len=4,
            anchor_node_type="user",
            sequence_construction_method="ppr",
        )

        expected_anchor_0 = torch.tensor(
            [
                [10.0, 0.0],  # anchor user0
                [0.0, 21.0],  # item1, weight 0.9
                [0.0, 20.0],  # item0, weight 0.6
                [11.0, 0.0],  # user1, weight 0.4
            ]
        )
        expected_anchor_1 = torch.tensor(
            [
                [11.0, 0.0],  # anchor user1
                [0.0, 20.0],  # item0, weight 0.8
                [10.0, 0.0],  # user0, weight 0.3
                [0.0, 0.0],  # padding
            ]
        )

        self.assertTrue(torch.allclose(sequences[0], expected_anchor_0))
        self.assertTrue(torch.allclose(sequences[1], expected_anchor_1))
        self.assertTrue(
            torch.equal(valid_mask[0], torch.tensor([True, True, True, True]))
        )
        self.assertTrue(
            torch.equal(valid_mask[1], torch.tensor([True, True, True, False]))
        )

    def test_ppr_sequence_construction_requires_only_ppr_relations(self):
        data = create_simple_hetero_data()

        with self.assertRaisesRegex(ValueError, "contain only PPR edges"):
            heterodata_to_graph_transformer_input(
                data=data,
                batch_size=1,
                max_seq_len=4,
                anchor_node_type="user",
                sequence_construction_method="ppr",
            )

    def test_ppr_sequence_can_return_token_input_and_attention_bias_features(self):
        data = create_ppr_sequence_hetero_data()

        _, valid_mask, sequence_auxiliary_data = heterodata_to_graph_transformer_input(
            data=data,
            batch_size=2,
            max_seq_len=4,
            anchor_node_type="user",
            sequence_construction_method="ppr",
            anchor_based_attention_bias_attr_names=["hop_distance", "ppr_weight"],
            anchor_based_input_attr_names=["hop_distance", "ppr_weight"],
        )

        anchor_bias = sequence_auxiliary_data["anchor_bias"]
        token_input = sequence_auxiliary_data["token_input"]
        assert anchor_bias is not None
        assert token_input is not None

        expected_anchor_0 = torch.tensor(
            [
                [0.0, 0.0],
                [3.0, 0.9],
                [2.0, 0.6],
                [1.0, 0.4],
            ]
        )
        expected_anchor_1 = torch.tensor(
            [
                [0.0, 0.0],
                [4.0, 0.8],
                [1.0, 0.3],
                [0.0, 0.0],
            ]
        )

        self.assertEqual(anchor_bias.shape, (2, 4, 2))
        self.assertTrue(torch.allclose(anchor_bias[0], expected_anchor_0))
        self.assertTrue(torch.allclose(anchor_bias[1], expected_anchor_1))
        self.assertEqual(set(token_input.keys()), {"hop_distance", "ppr_weight"})
        self.assertTrue(
            torch.allclose(
                token_input["hop_distance"],
                anchor_bias[..., 0:1],
            )
        )
        self.assertTrue(
            torch.allclose(
                token_input["ppr_weight"],
                anchor_bias[..., 1:2],
            )
        )
        self.assertTrue(
            torch.equal(valid_mask[1], torch.tensor([True, True, True, False]))
        )

    def test_relation_mask_outputs_follow_requested_order_and_direction(self):
        data = create_relation_mask_hetero_data()
        user_node_type = NodeType("user")
        follows_edge_type = EdgeType(
            user_node_type, Relation("follows"), user_node_type
        )
        likes_edge_type = EdgeType(user_node_type, Relation("likes"), user_node_type)

        sequences, valid_mask, attention_bias_data = (
            heterodata_to_graph_transformer_input(
                data=data,
                batch_size=1,
                max_seq_len=4,
                anchor_node_type="user",
                hop_distance=2,
                relation_edge_types=[likes_edge_type, follows_edge_type],
            )
        )

        relation_mask = attention_bias_data["pairwise_relation_mask"]
        assert relation_mask is not None

        expected_sequences = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ]
        )
        self.assertTrue(torch.allclose(sequences, expected_sequences))
        self.assertTrue(
            torch.equal(valid_mask[0], torch.tensor([True, True, True, False]))
        )
        self.assertEqual(relation_mask.shape, (1, 4, 4, 2))
        self.assertEqual(relation_mask.dtype, torch.bool)
        self.assertTrue(
            torch.equal(relation_mask[0, 1, 0], torch.tensor([True, True]))
        )
        self.assertTrue(
            torch.equal(relation_mask[0, 2, 1], torch.tensor([False, True]))
        )
        self.assertTrue(
            torch.equal(relation_mask[0, 0, 1], torch.zeros(2, dtype=torch.bool))
        )
        self.assertTrue(torch.all(relation_mask[0, 3] == 0))


class TestPyTorchTransformerIntegration(TestCase):
    """Tests for integration with PyTorch TransformerEncoderLayer."""

    def test_transformer_encoder_layer_forward(self):
        """Test that transform output works with TransformerEncoderLayer."""
        data = create_larger_hetero_data(num_users=10, num_items=5)
        batch_size = 4
        max_seq_len = 20
        feature_dim = 32  # Must match the feature dim in create_larger_hetero_data

        # Transform HeteroData to sequences
        (
            sequences,
            valid_mask,
            attention_bias_data,
        ) = heterodata_to_graph_transformer_input(
            data=data,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            anchor_node_type="user",
            hop_distance=2,
        )

        # Verify shapes
        self.assertEqual(sequences.shape, (batch_size, max_seq_len, feature_dim))
        self.assertEqual(valid_mask.shape, (batch_size, max_seq_len))

        # Create a TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=4,  # feature_dim must be divisible by nhead
            dim_feedforward=128,
            batch_first=True,  # Important: we use (batch, seq, feature) format
        )

        # Convert valid_mask to the format expected by PyTorch Transformer
        # PyTorch expects: True = ignore (padding), False = attend
        # Our mask: True = valid, False = padding
        src_key_padding_mask = ~valid_mask  # Invert: True where padding

        # Forward pass through encoder layer
        output = encoder_layer(
            sequences,
            src_key_padding_mask=src_key_padding_mask,
        )

        # Output should have same shape as input
        self.assertEqual(output.shape, sequences.shape)

        # Output should not have NaN values
        self.assertFalse(torch.isnan(output).any())

    def test_transformer_encoder_full_stack(self):
        """Test with full TransformerEncoder (multiple layers)."""
        data = create_larger_hetero_data(num_users=10, num_items=5)
        batch_size = 4
        max_seq_len = 20
        feature_dim = 32

        sequences, valid_mask, _ = heterodata_to_graph_transformer_input(
            data=data,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            anchor_node_type="user",
            hop_distance=2,
        )

        # Create full TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=4,
            dim_feedforward=128,
            batch_first=True,
        )
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Convert mask: True = padding (ignore)
        src_key_padding_mask = ~valid_mask

        # Forward pass
        output = transformer_encoder(
            sequences,
            src_key_padding_mask=src_key_padding_mask,
        )

        self.assertEqual(output.shape, sequences.shape)
        self.assertFalse(torch.isnan(output).any())

    def test_transformer_with_custom_attention_mask(self):
        """Test with causal attention mask combined with padding mask."""
        data = create_larger_hetero_data(num_users=10, num_items=5)
        batch_size = 4
        max_seq_len = 20
        feature_dim = 32

        sequences, valid_mask, _ = heterodata_to_graph_transformer_input(
            data=data,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            anchor_node_type="user",
            hop_distance=2,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=4,
            dim_feedforward=128,
            batch_first=True,
        )

        # Create causal mask (optional for graph transformers, but test compatibility)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(max_seq_len)
        src_key_padding_mask = ~valid_mask

        # Forward pass with both masks
        output = encoder_layer(
            sequences,
            src_mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

        self.assertEqual(output.shape, sequences.shape)
        self.assertFalse(torch.isnan(output).any())

    def test_gradient_flow(self):
        """Test that gradients flow correctly through the transformer."""
        data = create_larger_hetero_data(num_users=10, num_items=5)
        batch_size = 4
        max_seq_len = 20
        feature_dim = 32

        sequences, valid_mask, _ = heterodata_to_graph_transformer_input(
            data=data,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            anchor_node_type="user",
            hop_distance=2,
        )

        # Make sequences require gradients
        sequences = sequences.clone().requires_grad_(True)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=4,
            dim_feedforward=128,
            batch_first=True,
        )

        src_key_padding_mask = ~valid_mask

        output = encoder_layer(
            sequences,
            src_key_padding_mask=src_key_padding_mask,
        )

        # Compute loss and backward
        loss = output.mean()
        loss.backward()

        # Check that gradients exist and are not NaN
        self.assertIsNotNone(sequences.grad)
        assert sequences.grad is not None  # Type narrowing for mypy
        self.assertFalse(torch.isnan(sequences.grad).any())

    def test_transformer_with_classification_head(self):
        """Test end-to-end: transform -> transformer -> classification."""
        data = create_larger_hetero_data(num_users=10, num_items=5)
        batch_size = 4
        max_seq_len = 20
        feature_dim = 32
        num_classes = 5

        sequences, valid_mask, _ = heterodata_to_graph_transformer_input(
            data=data,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            anchor_node_type="user",
            hop_distance=2,
        )

        # Build a simple graph transformer classifier
        class SimpleGraphTransformerClassifier(nn.Module):
            def __init__(self, d_model, nhead, num_classes):
                super().__init__()
                self.encoder = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=128,
                    batch_first=True,
                )
                self.classifier = nn.Linear(d_model, num_classes)

            def forward(self, x, padding_mask):
                # x: (batch, seq, d_model)
                # padding_mask: (batch, seq) - True where padding
                encoded = self.encoder(x, src_key_padding_mask=padding_mask)
                # Use first token (anchor) for classification
                anchor_repr = encoded[:, 0, :]  # (batch, d_model)
                logits = self.classifier(anchor_repr)  # (batch, num_classes)
                return logits

        model = SimpleGraphTransformerClassifier(
            d_model=feature_dim,
            nhead=4,
            num_classes=num_classes,
        )

        src_key_padding_mask = ~valid_mask
        logits = model(sequences, src_key_padding_mask)

        # Check output shape
        self.assertEqual(logits.shape, (batch_size, num_classes))

        # Check no NaN
        self.assertFalse(torch.isnan(logits).any())

        # Check that we can compute loss and backward
        labels = torch.randint(0, num_classes, (batch_size,))
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()

        # Check gradients exist in model parameters
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)

    def test_batch_first_format(self):
        """Verify that output is in batch-first format (batch, seq, feature)."""
        data = create_larger_hetero_data(num_users=10, num_items=5)
        batch_size = 4
        max_seq_len = 20
        feature_dim = 32

        sequences, valid_mask, _ = heterodata_to_graph_transformer_input(
            data=data,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            anchor_node_type="user",
            hop_distance=2,
        )

        # Verify batch-first format
        self.assertEqual(sequences.dim(), 3)
        self.assertEqual(sequences.shape[0], batch_size)  # batch dimension
        self.assertEqual(sequences.shape[1], max_seq_len)  # sequence dimension
        self.assertEqual(sequences.shape[2], feature_dim)  # feature dimension

        # Verify mask is also batch-first
        self.assertEqual(valid_mask.dim(), 2)
        self.assertEqual(valid_mask.shape[0], batch_size)
        self.assertEqual(valid_mask.shape[1], max_seq_len)


def _create_hetero_data_with_relative_pe() -> HeteroData:
    data = HeteroData()

    data["user"].x = torch.tensor(
        [[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.5, 0.0], [0.0, 0.0, 1.0, 1.0]]
    )
    data["item"].x = torch.tensor([[2.0, 0.0, 0.0, 1.0], [0.0, 2.0, 1.0, 0.0]])

    data["user"].random_walk_pe = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    data["item"].random_walk_pe = torch.tensor([[1.0, 1.1], [1.2, 1.3]])

    data["user", "buys", "item"].edge_index = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]])
    data["item", "bought_by", "user"].edge_index = torch.tensor(
        [[0, 0, 1, 1], [0, 1, 1, 2]]
    )

    hop_distance = torch.arange(25, dtype=torch.float).reshape(5, 5)
    data.hop_distance = hop_distance.to_sparse_csr()
    pairwise_distance = (torch.arange(25, dtype=torch.float).reshape(5, 5) + 1.0) / 10.0
    data.pairwise_distance = pairwise_distance.to_sparse_csr()

    return data


class TestGraphTransformerRelativeBiasAssembly(TestCase):
    def test_transform_returns_base_sequences_and_anchor_relative_bias(self) -> None:
        data = _create_hetero_data_with_relative_pe()

        (
            sequences,
            valid_mask,
            attention_bias_data,
        ) = heterodata_to_graph_transformer_input(
            data=data,
            batch_size=1,
            max_seq_len=4,
            anchor_node_type="user",
            hop_distance=2,
            anchor_based_attention_bias_attr_names=["hop_distance"],
        )

        self.assertEqual(sequences.shape, (1, 4, 4))
        anchor_bias = attention_bias_data["anchor_bias"]
        assert anchor_bias is not None
        self.assertEqual(anchor_bias.shape, (1, 4, 1))
        self.assertIsNone(attention_bias_data["pairwise_bias"])
        self.assertTrue(valid_mask[0, 0].item())

    def test_attention_bias_outputs_include_valid_mask_and_relative_features(
        self,
    ) -> None:
        data = _create_hetero_data_with_relative_pe()

        (
            sequences,
            valid_mask,
            attention_bias_data,
        ) = heterodata_to_graph_transformer_input(
            data=data,
            batch_size=1,
            max_seq_len=4,
            anchor_node_type="user",
            hop_distance=2,
            anchor_based_attention_bias_attr_names=["hop_distance"],
            pairwise_attention_bias_attr_names=["pairwise_distance"],
        )

        self.assertEqual(sequences.shape, (1, 4, 4))
        self.assertEqual(valid_mask.shape, (1, 4))
        anchor_bias = attention_bias_data["anchor_bias"]
        pairwise_bias = attention_bias_data["pairwise_bias"]
        assert anchor_bias is not None
        assert pairwise_bias is not None
        self.assertEqual(anchor_bias.shape, (1, 4, 1))
        self.assertEqual(pairwise_bias.shape, (1, 4, 4, 1))
        self.assertAlmostEqual(anchor_bias[0, 0, 0].item(), 0.0, places=5)
        self.assertAlmostEqual(anchor_bias[0, 1, 0].item(), 1.0, places=5)
        self.assertAlmostEqual(anchor_bias[0, 2, 0].item(), 3.0, places=5)
        self.assertAlmostEqual(pairwise_bias[0, 0, 0, 0].item(), 0.1, places=5)

        invalid_pair_mask = ~(valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1))
        self.assertTrue(torch.all(pairwise_bias[..., 0][invalid_pair_mask] == 0))


if __name__ == "__main__":
    absltest.main()
