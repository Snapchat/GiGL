import torch
from absl.testing import absltest
from torch_geometric.data import HeteroData

from gigl.transforms.add_positional_encodings import (
    AddHeteroHopDistanceEncoding,
    AddHeteroRandomWalkPE,
    AddHeteroRandomWalkSE,
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
    data['user'].x = torch.randn(3, 4)
    data['item'].x = torch.randn(2, 4)

    # Edges: user -> item
    data['user', 'buys', 'item'].edge_index = torch.tensor([
        [0, 1, 2],  # source (user)
        [0, 0, 1],  # target (item)
    ])

    # Edges: item -> user (reverse)
    data['item', 'bought_by', 'user'].edge_index = torch.tensor([
        [0, 0, 1],  # source (item)
        [0, 1, 2],  # target (user)
    ])

    return data


def create_empty_hetero_data() -> HeteroData:
    """Create an empty heterogeneous graph for testing edge cases."""
    data = HeteroData()
    data['user'].x = torch.zeros(0, 4)
    data['item'].x = torch.zeros(0, 4)
    return data



class TestAddHeteroRandomWalkPE(TestCase):
    """Tests for AddHeteroRandomWalkPE (Positional Encoding - column sum of non-diagonal)."""

    def test_forward_basic(self):
        """Test basic forward pass."""
        data = create_simple_hetero_data()
        transform = AddHeteroRandomWalkPE(walk_length=4)

        result = transform(data)

        # Check that PE was added to both node types
        self.assertTrue(hasattr(result['user'], 'random_walk_pe'))
        self.assertTrue(hasattr(result['item'], 'random_walk_pe'))

        # Check shapes
        self.assertEqual(result['user'].random_walk_pe.shape, (3, 4))
        self.assertEqual(result['item'].random_walk_pe.shape, (2, 4))

    def test_forward_with_custom_attr_name(self):
        """Test forward pass with custom attribute name."""
        data = create_simple_hetero_data()
        transform = AddHeteroRandomWalkPE(walk_length=3, attr_name='rw_pe')

        result = transform(data)

        self.assertTrue(hasattr(result['user'], 'rw_pe'))
        self.assertTrue(hasattr(result['item'], 'rw_pe'))
        self.assertFalse(hasattr(result['user'], 'random_walk_pe'))

    def test_forward_undirected(self):
        """Test forward pass with undirected graph setting."""
        data = create_simple_hetero_data()
        transform = AddHeteroRandomWalkPE(walk_length=3, is_undirected=True)

        result = transform(data)

        self.assertEqual(result['user'].random_walk_pe.shape, (3, 3))
        self.assertEqual(result['item'].random_walk_pe.shape, (2, 3))

    def test_forward_empty_graph(self):
        """Test forward pass with empty graph."""
        data = create_empty_hetero_data()
        transform = AddHeteroRandomWalkPE(walk_length=3)

        result = transform(data)

        self.assertEqual(result['user'].random_walk_pe.shape, (0, 3))
        self.assertEqual(result['item'].random_walk_pe.shape, (0, 3))

    def test_repr(self):
        """Test string representation."""
        transform = AddHeteroRandomWalkPE(walk_length=10)
        self.assertEqual(repr(transform), 'AddHeteroRandomWalkPE(walk_length=10)')


class TestAddHeteroRandomWalkSE(TestCase):
    """Tests for AddHeteroRandomWalkSE (Structural Encoding - diagonal elements)."""

    def test_forward_basic(self):
        """Test basic forward pass."""
        data = create_simple_hetero_data()
        transform = AddHeteroRandomWalkSE(walk_length=4)

        result = transform(data)

        # Check that SE was added to both node types
        self.assertTrue(hasattr(result['user'], 'random_walk_se'))
        self.assertTrue(hasattr(result['item'], 'random_walk_se'))

        # Check shapes
        self.assertEqual(result['user'].random_walk_se.shape, (3, 4))
        self.assertEqual(result['item'].random_walk_se.shape, (2, 4))

        # Values should be probabilities (between 0 and 1)
        self.assertTrue((result['user'].random_walk_se >= 0).all())
        self.assertTrue((result['user'].random_walk_se <= 1).all())

    def test_forward_with_custom_attr_name(self):
        """Test forward pass with custom attribute name."""
        data = create_simple_hetero_data()
        transform = AddHeteroRandomWalkSE(walk_length=3, attr_name='rw_se')

        result = transform(data)

        self.assertTrue(hasattr(result['user'], 'rw_se'))
        self.assertTrue(hasattr(result['item'], 'rw_se'))
        self.assertFalse(hasattr(result['user'], 'random_walk_se'))

    def test_forward_undirected(self):
        """Test forward pass with undirected graph setting."""
        data = create_simple_hetero_data()
        transform = AddHeteroRandomWalkSE(walk_length=3, is_undirected=True)

        result = transform(data)

        self.assertEqual(result['user'].random_walk_se.shape, (3, 3))
        self.assertEqual(result['item'].random_walk_se.shape, (2, 3))

    def test_forward_empty_graph(self):
        """Test forward pass with empty graph."""
        data = create_empty_hetero_data()
        transform = AddHeteroRandomWalkSE(walk_length=3)

        result = transform(data)

        self.assertEqual(result['user'].random_walk_se.shape, (0, 3))
        self.assertEqual(result['item'].random_walk_se.shape, (0, 3))

    def test_repr(self):
        """Test string representation."""
        transform = AddHeteroRandomWalkSE(walk_length=10)
        self.assertEqual(repr(transform), 'AddHeteroRandomWalkSE(walk_length=10)')


class TestAddHeteroHopDistanceEncoding(TestCase):
    def test_forward_basic(self):
        """Test basic forward pass with full_matrix=True (default)."""
        data = create_simple_hetero_data()
        transform = AddHeteroHopDistanceEncoding(h_max=3)

        result = transform(data)

        # Check that full pairwise distance matrix is stored as graph-level attribute
        self.assertTrue(hasattr(result, 'hop_distance'))
        # Total nodes: 3 users + 2 items = 5 nodes
        self.assertEqual(result.hop_distance.shape, (5, 5))

    def test_forward_full_matrix_false(self):
        """Test forward pass with full_matrix=False (edge-level)."""
        data = create_simple_hetero_data()
        transform = AddHeteroHopDistanceEncoding(h_max=3, full_matrix=False)

        result = transform(data)

        # Check that PE was added to both edge types
        self.assertTrue(hasattr(result['user', 'buys', 'item'], 'hop_distance'))
        self.assertTrue(hasattr(result['item', 'bought_by', 'user'], 'hop_distance'))

        # Check shapes (3 edges each)
        self.assertEqual(result['user', 'buys', 'item'].hop_distance.shape, (3, 1))
        self.assertEqual(result['item', 'bought_by', 'user'].hop_distance.shape, (3, 1))

        # Direct edges should have distance <= h_max
        self.assertTrue((result['user', 'buys', 'item'].hop_distance <= 3).all())

    def test_forward_with_custom_attr_name(self):
        """Test forward pass with custom attribute name."""
        data = create_simple_hetero_data()
        transform = AddHeteroHopDistanceEncoding(h_max=2, attr_name='custom_hop', full_matrix=False)

        result = transform(data)

        self.assertTrue(hasattr(result['user', 'buys', 'item'], 'custom_hop'))
        self.assertTrue(hasattr(result['item', 'bought_by', 'user'], 'custom_hop'))
        self.assertFalse(hasattr(result['user', 'buys', 'item'], 'hop_distance'))

    def test_forward_undirected(self):
        """Test forward pass with undirected graph setting."""
        data = create_simple_hetero_data()
        transform = AddHeteroHopDistanceEncoding(h_max=2, is_undirected=True, full_matrix=False)

        result = transform(data)

        self.assertEqual(result['user', 'buys', 'item'].hop_distance.shape, (3, 1))
        self.assertEqual(result['item', 'bought_by', 'user'].hop_distance.shape, (3, 1))

    def test_forward_empty_graph(self):
        """Test forward pass with empty graph."""
        data = create_empty_hetero_data()
        # Add empty edge types
        data['user', 'buys', 'item'].edge_index = torch.zeros((2, 0), dtype=torch.long)
        transform = AddHeteroHopDistanceEncoding(h_max=3, full_matrix=False)

        result = transform(data)

        self.assertEqual(result['user', 'buys', 'item'].hop_distance.shape, (0,))

    def test_forward_full_matrix(self):
        """Test forward pass with full_matrix=True for Graph Transformer use."""
        data = create_simple_hetero_data()
        transform = AddHeteroHopDistanceEncoding(h_max=3, full_matrix=True)

        result = transform(data)

        # Check that full pairwise distance matrix is stored as graph-level attribute
        self.assertTrue(hasattr(result, 'hop_distance'))
        # Total nodes: 3 users + 2 items = 5 nodes
        self.assertEqual(result.hop_distance.shape, (5, 5))
        # Diagonal should be 0 (distance to self)
        self.assertTrue((result.hop_distance.diag() == 0).all())
        # All distances should be <= h_max
        self.assertTrue((result.hop_distance <= 3).all())

    def test_forward_full_matrix_empty_graph(self):
        """Test forward pass with full_matrix=True on empty graph."""
        data = create_empty_hetero_data()
        data['user', 'buys', 'item'].edge_index = torch.zeros((2, 0), dtype=torch.long)
        transform = AddHeteroHopDistanceEncoding(h_max=3, full_matrix=True)

        result = transform(data)

        self.assertTrue(hasattr(result, 'hop_distance'))
        self.assertEqual(result.hop_distance.shape, (0, 0))

    def test_forward_node_type_aware(self):
        """Test forward pass with node_type_aware=True for heterogeneous Graph Transformers."""
        data = create_simple_hetero_data()
        transform = AddHeteroHopDistanceEncoding(h_max=3, full_matrix=True, node_type_aware=True)

        result = transform(data)

        # Check that hop distance matrix is stored
        self.assertTrue(hasattr(result, 'hop_distance'))
        self.assertEqual(result.hop_distance.shape, (5, 5))

        # Check that node type information is stored
        self.assertTrue(hasattr(result, 'node_type_ids'))
        self.assertEqual(result.node_type_ids.shape, (5,))

        # Check that node type pair matrix is stored
        self.assertTrue(hasattr(result, 'node_type_pair'))
        self.assertEqual(result.node_type_pair.shape, (5, 5))

        # Check that node type names are stored
        self.assertTrue(hasattr(result, 'node_type_names'))
        self.assertEqual(result.node_type_names, ['item', 'user'])  # Sorted alphabetically

        # Verify node_type_ids values are valid (0 or 1 for 2 node types)
        self.assertTrue((result.node_type_ids >= 0).all())
        self.assertTrue((result.node_type_ids < 2).all())

        # Verify node_type_pair encodes (src_type, dst_type) correctly
        # For 2 node types, pair values should be in [0, 3] (0*2+0, 0*2+1, 1*2+0, 1*2+1)
        self.assertTrue((result.node_type_pair >= 0).all())
        self.assertTrue((result.node_type_pair < 4).all())

    def test_forward_node_type_aware_empty_graph(self):
        """Test forward pass with node_type_aware=True on empty graph."""
        data = create_empty_hetero_data()
        data['user', 'buys', 'item'].edge_index = torch.zeros((2, 0), dtype=torch.long)
        transform = AddHeteroHopDistanceEncoding(h_max=3, full_matrix=True, node_type_aware=True)

        result = transform(data)

        self.assertTrue(hasattr(result, 'hop_distance'))
        self.assertEqual(result.hop_distance.shape, (0, 0))
        self.assertTrue(hasattr(result, 'node_type_ids'))
        self.assertEqual(result.node_type_ids.shape, (0,))
        self.assertTrue(hasattr(result, 'node_type_pair'))
        self.assertEqual(result.node_type_pair.shape, (0, 0))

    def test_node_type_aware_only_with_full_matrix(self):
        """Test that node_type_aware only takes effect when full_matrix=True."""
        data = create_simple_hetero_data()
        # node_type_aware=True but full_matrix=False should not add node type info
        transform = AddHeteroHopDistanceEncoding(h_max=3, full_matrix=False, node_type_aware=True)

        result = transform(data)

        # Should have edge-level hop_distance
        self.assertTrue(hasattr(result['user', 'buys', 'item'], 'hop_distance'))
        # Should NOT have node type information (only added when full_matrix=True)
        self.assertFalse(hasattr(result, 'node_type_ids'))
        self.assertFalse(hasattr(result, 'node_type_pair'))

    def test_repr(self):
        """Test string representation."""
        transform = AddHeteroHopDistanceEncoding(h_max=5, full_matrix=False)
        self.assertEqual(
            repr(transform),
            'AddHeteroHopDistanceEncoding(h_max=5, full_matrix=False, node_type_aware=False)'
        )

        transform_full = AddHeteroHopDistanceEncoding(h_max=3, full_matrix=True)
        self.assertEqual(
            repr(transform_full),
            'AddHeteroHopDistanceEncoding(h_max=3, full_matrix=True, node_type_aware=False)'
        )

        transform_type_aware = AddHeteroHopDistanceEncoding(h_max=3, full_matrix=True, node_type_aware=True)
        self.assertEqual(
            repr(transform_type_aware),
            'AddHeteroHopDistanceEncoding(h_max=3, full_matrix=True, node_type_aware=True)'
        )


if __name__ == '__main__':
    absltest.main()
