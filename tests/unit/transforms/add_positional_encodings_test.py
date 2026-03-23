import torch
from absl.testing import absltest
from torch_geometric.data import HeteroData

from gigl.transforms.add_positional_encodings import (
    AddHeteroHopDistanceEncoding,
    AddHeteroRandomWalkEncodings,
)
from tests.test_assets.test_case import TestCase


def create_simple_hetero_data() -> HeteroData:
    """Create a simple heterogeneous graph for testing.

    Graph structure:
    - 3 'user' nodes
    - 2 'item' nodes
    - Edges: user -> item (bipartite)
    """
    data = HeteroData()

    # Node features
    data["user"].x = torch.randn(3, 4)
    data["item"].x = torch.randn(2, 4)

    # Edges: user -> item
    data["user", "buys", "item"].edge_index = torch.tensor(
        [
            [0, 1, 2],  # source (user)
            [0, 0, 1],  # target (item)
        ]
    )

    # Edges: item -> user (reverse)
    data["item", "bought_by", "user"].edge_index = torch.tensor(
        [
            [0, 0, 1],  # source (item)
            [0, 1, 2],  # target (user)
        ]
    )

    return data


def create_empty_hetero_data() -> HeteroData:
    """Create an empty heterogeneous graph for testing edge cases."""
    data = HeteroData()
    data["user"].x = torch.zeros(0, 4)
    data["item"].x = torch.zeros(0, 4)
    return data


class TestAddHeteroRandomWalkEncodings(TestCase):
    """Tests for AddHeteroRandomWalkEncodings (consolidated PE and SE in single pass)."""

    def test_forward_basic(self):
        """Test basic forward pass computes both PE and SE."""
        data = create_simple_hetero_data()
        transform = AddHeteroRandomWalkEncodings(walk_length=4)

        result = transform(data)

        # Check that both PE and SE were added to both node types
        self.assertTrue(hasattr(result["user"], "random_walk_pe"))
        self.assertTrue(hasattr(result["item"], "random_walk_pe"))
        self.assertTrue(hasattr(result["user"], "random_walk_se"))
        self.assertTrue(hasattr(result["item"], "random_walk_se"))

        # Check shapes
        self.assertEqual(result["user"].random_walk_pe.shape, (3, 4))
        self.assertEqual(result["item"].random_walk_pe.shape, (2, 4))
        self.assertEqual(result["user"].random_walk_se.shape, (3, 4))
        self.assertEqual(result["item"].random_walk_se.shape, (2, 4))

        # SE values should be probabilities (between 0 and 1)
        self.assertTrue((result["user"].random_walk_se >= 0).all())
        self.assertTrue((result["user"].random_walk_se <= 1).all())

    def test_forward_with_custom_attr_names(self):
        """Test forward pass with custom attribute names."""
        data = create_simple_hetero_data()
        transform = AddHeteroRandomWalkEncodings(
            walk_length=3,
            pe_attr_name="rw_pe",
            se_attr_name="rw_se",
        )

        result = transform(data)

        self.assertTrue(hasattr(result["user"], "rw_pe"))
        self.assertTrue(hasattr(result["item"], "rw_pe"))
        self.assertTrue(hasattr(result["user"], "rw_se"))
        self.assertTrue(hasattr(result["item"], "rw_se"))
        self.assertFalse(hasattr(result["user"], "random_walk_pe"))
        self.assertFalse(hasattr(result["user"], "random_walk_se"))

    def test_forward_undirected(self):
        """Test forward pass with undirected graph setting."""
        data = create_simple_hetero_data()
        transform = AddHeteroRandomWalkEncodings(walk_length=3, is_undirected=True)

        result = transform(data)

        self.assertEqual(result["user"].random_walk_pe.shape, (3, 3))
        self.assertEqual(result["item"].random_walk_pe.shape, (2, 3))
        self.assertEqual(result["user"].random_walk_se.shape, (3, 3))
        self.assertEqual(result["item"].random_walk_se.shape, (2, 3))

    def test_forward_empty_graph(self):
        """Test forward pass with empty graph."""
        data = create_empty_hetero_data()
        transform = AddHeteroRandomWalkEncodings(walk_length=3)

        result = transform(data)

        self.assertEqual(result["user"].random_walk_pe.shape, (0, 3))
        self.assertEqual(result["item"].random_walk_pe.shape, (0, 3))
        self.assertEqual(result["user"].random_walk_se.shape, (0, 3))
        self.assertEqual(result["item"].random_walk_se.shape, (0, 3))

    def test_repr(self):
        """Test string representation."""
        transform = AddHeteroRandomWalkEncodings(walk_length=10)
        self.assertEqual(
            repr(transform),
            "AddHeteroRandomWalkEncodings(walk_length=10, attach_to_x=False)",
        )

    def test_forward_attach_to_x(self):
        """Test forward pass with attach_to_x=True concatenates both PE and SE to node features."""
        data = create_simple_hetero_data()
        original_user_dim = data["user"].x.shape[1]  # 4
        original_item_dim = data["item"].x.shape[1]  # 4
        walk_length = 3
        transform = AddHeteroRandomWalkEncodings(
            walk_length=walk_length, attach_to_x=True
        )

        result = transform(data)

        # Check that PE and SE were NOT added as separate attributes
        self.assertFalse(hasattr(result["user"], "random_walk_pe"))
        self.assertFalse(hasattr(result["item"], "random_walk_pe"))
        self.assertFalse(hasattr(result["user"], "random_walk_se"))
        self.assertFalse(hasattr(result["item"], "random_walk_se"))

        # Check that x was expanded with both PE and SE dimensions
        # PE is added first, then SE
        self.assertEqual(
            result["user"].x.shape, (3, original_user_dim + walk_length * 2)
        )
        self.assertEqual(
            result["item"].x.shape, (2, original_item_dim + walk_length * 2)
        )

    def test_forward_attach_to_x_no_existing_features(self):
        """Test forward pass with attach_to_x=True when nodes have no existing features."""
        data = HeteroData()
        data["user"].num_nodes = 3
        data["item"].num_nodes = 2
        data["user", "buys", "item"].edge_index = torch.tensor(
            [
                [0, 1, 2],
                [0, 0, 1],
            ]
        )
        data["item", "bought_by", "user"].edge_index = torch.tensor(
            [
                [0, 0, 1],
                [0, 1, 2],
            ]
        )

        walk_length = 4
        transform = AddHeteroRandomWalkEncodings(
            walk_length=walk_length, attach_to_x=True
        )

        result = transform(data)

        # Check that x was created with both PE and SE as features
        self.assertTrue(hasattr(result["user"], "x"))
        self.assertTrue(hasattr(result["item"], "x"))
        self.assertEqual(result["user"].x.shape, (3, walk_length * 2))
        self.assertEqual(result["item"].x.shape, (2, walk_length * 2))

    def test_forward_attach_to_x_empty_graph(self):
        """Test forward pass with attach_to_x=True on empty graph."""
        data = create_empty_hetero_data()
        original_user_dim = data["user"].x.shape[1]  # 4
        original_item_dim = data["item"].x.shape[1]  # 4
        walk_length = 3
        transform = AddHeteroRandomWalkEncodings(
            walk_length=walk_length, attach_to_x=True
        )

        result = transform(data)

        # Check shapes on empty graph
        self.assertEqual(
            result["user"].x.shape, (0, original_user_dim + walk_length * 2)
        )
        self.assertEqual(
            result["item"].x.shape, (0, original_item_dim + walk_length * 2)
        )

    def test_repr_attach_to_x(self):
        """Test string representation with attach_to_x=True."""
        transform = AddHeteroRandomWalkEncodings(walk_length=10, attach_to_x=True)
        self.assertEqual(
            repr(transform),
            "AddHeteroRandomWalkEncodings(walk_length=10, attach_to_x=True)",
        )


class TestAddHeteroHopDistanceEncoding(TestCase):
    def test_forward_basic(self):
        """Test basic forward pass returns sparse matrix."""
        data = create_simple_hetero_data()
        transform = AddHeteroHopDistanceEncoding(h_max=3)

        result = transform(data)

        # Check that sparse pairwise distance matrix is stored as graph-level attribute
        self.assertTrue(hasattr(result, "hop_distance"))
        # Total nodes: 3 users + 2 items = 5 nodes
        self.assertEqual(result.hop_distance.shape, (5, 5))
        # Should be sparse
        self.assertTrue(result.hop_distance.is_sparse)

    def test_forward_sparse_values(self):
        """Test that sparse matrix has correct values (0 for unreachable, 1-h_max for reachable)."""
        data = create_simple_hetero_data()
        transform = AddHeteroHopDistanceEncoding(h_max=3)

        result = transform(data)

        # Convert to dense for easier testing
        dense = result.hop_distance.to_dense()

        # Diagonal should be 0 (distance to self, not stored in sparse = 0)
        self.assertTrue((dense.diag() == 0).all())

        # Non-zero values (reachable pairs) should be in [1, h_max]
        nonzero_vals = result.hop_distance.values()
        if nonzero_vals.numel() > 0:
            self.assertTrue((nonzero_vals >= 1).all())
            self.assertTrue((nonzero_vals <= 3).all())

    def test_forward_with_custom_attr_name(self):
        """Test forward pass with custom attribute name."""
        data = create_simple_hetero_data()
        transform = AddHeteroHopDistanceEncoding(h_max=2, attr_name="custom_hop")

        result = transform(data)

        self.assertTrue(hasattr(result, "custom_hop"))
        self.assertFalse(hasattr(result, "hop_distance"))
        self.assertTrue(result.custom_hop.is_sparse)

    def test_forward_undirected(self):
        """Test forward pass with undirected graph setting."""
        data = create_simple_hetero_data()
        transform = AddHeteroHopDistanceEncoding(h_max=2, is_undirected=True)

        result = transform(data)

        self.assertTrue(result.hop_distance.is_sparse)
        self.assertEqual(result.hop_distance.shape, (5, 5))

    def test_forward_empty_graph(self):
        """Test forward pass with empty graph."""
        data = create_empty_hetero_data()
        # Add empty edge types
        data["user", "buys", "item"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        transform = AddHeteroHopDistanceEncoding(h_max=3)

        result = transform(data)

        self.assertTrue(hasattr(result, "hop_distance"))
        self.assertTrue(result.hop_distance.is_sparse)
        self.assertEqual(result.hop_distance.shape, (0, 0))

    def test_repr(self):
        """Test string representation."""
        transform = AddHeteroHopDistanceEncoding(h_max=5)
        self.assertEqual(repr(transform), "AddHeteroHopDistanceEncoding(h_max=5)")


if __name__ == "__main__":
    absltest.main()
