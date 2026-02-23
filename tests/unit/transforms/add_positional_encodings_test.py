import torch
from absl.testing import absltest
from torch_geometric.data import HeteroData

from gigl.transforms.add_positional_encodings import (
    AddHeteroHopDistancePE,
    AddHeteroRandomWalkPE,
    add_node_attr,
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


class TestAddNodeAttr(TestCase):
    def test_add_node_attr_with_attr_name(self):
        """Test adding a node attribute with a specific attribute name."""
        data = create_simple_hetero_data()

        # Create values in homogeneous order (3 users + 2 items = 5 nodes)
        values = torch.randn(5, 8)

        add_node_attr(data, values, attr_name='test_attr')

        self.assertEqual(data['user'].test_attr.shape, (3, 8))
        self.assertEqual(data['item'].test_attr.shape, (2, 8))
        self.assert_tensor_equality(data['user'].test_attr, values[:3])
        self.assert_tensor_equality(data['item'].test_attr, values[3:])

    def test_add_node_attr_concatenate_to_x(self):
        """Test adding a node attribute by concatenating to existing x."""
        data = create_simple_hetero_data()
        original_user_x = data['user'].x.clone()
        original_item_x = data['item'].x.clone()

        # Create values in homogeneous order
        values = torch.randn(5, 8)

        add_node_attr(data, values, attr_name=None)

        # Check that x was concatenated
        self.assertEqual(data['user'].x.shape, (3, 12))  # 4 + 8
        self.assertEqual(data['item'].x.shape, (2, 12))  # 4 + 8

        # Check original features are preserved
        self.assert_tensor_equality(data['user'].x[:, :4], original_user_x)
        self.assert_tensor_equality(data['item'].x[:, :4], original_item_x)

    def test_add_node_attr_create_x_if_none(self):
        """Test creating x attribute if it doesn't exist."""
        data = HeteroData()
        data['user'].num_nodes = 3
        data['item'].num_nodes = 2

        values = torch.randn(5, 8)

        add_node_attr(data, values, attr_name=None)

        self.assertEqual(data['user'].x.shape, (3, 8))
        self.assertEqual(data['item'].x.shape, (2, 8))


class TestAddHeteroHopDistancePE(TestCase):
    def test_forward_basic(self):
        """Test basic forward pass."""
        data = create_simple_hetero_data()
        transform = AddHeteroHopDistancePE(k=3)

        result = transform(data)

        # Check that PE was added to both node types
        self.assertTrue(hasattr(result['user'], 'hop_pe'))
        self.assertTrue(hasattr(result['item'], 'hop_pe'))

        # Check shapes
        self.assertEqual(result['user'].hop_pe.shape, (3, 3))
        self.assertEqual(result['item'].hop_pe.shape, (2, 3))

    def test_forward_with_custom_attr_name(self):
        """Test forward pass with custom attribute name."""
        data = create_simple_hetero_data()
        transform = AddHeteroHopDistancePE(k=2, attr_name='custom_pe')

        result = transform(data)

        self.assertTrue(hasattr(result['user'], 'custom_pe'))
        self.assertTrue(hasattr(result['item'], 'custom_pe'))
        self.assertFalse(hasattr(result['user'], 'hop_pe'))

    def test_forward_undirected(self):
        """Test forward pass with undirected graph setting."""
        data = create_simple_hetero_data()
        transform = AddHeteroHopDistancePE(k=2, is_undirected=True)

        result = transform(data)

        self.assertEqual(result['user'].hop_pe.shape, (3, 2))
        self.assertEqual(result['item'].hop_pe.shape, (2, 2))

    def test_forward_empty_graph(self):
        """Test forward pass with empty graph."""
        data = create_empty_hetero_data()
        transform = AddHeteroHopDistancePE(k=3)

        result = transform(data)

        self.assertEqual(result['user'].hop_pe.shape, (0, 3))
        self.assertEqual(result['item'].hop_pe.shape, (0, 3))

    def test_repr(self):
        """Test string representation."""
        transform = AddHeteroHopDistancePE(k=5)
        self.assertEqual(repr(transform), 'AddHeteroHopDistancePE(k=5)')


class TestAddHeteroRandomWalkPE(TestCase):
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


if __name__ == '__main__':
    absltest.main()
