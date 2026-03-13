import torch
from absl.testing import absltest
from torch_geometric.data import HeteroData

from gigl.transforms.utils import add_node_attr, add_edge_attr
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

    def test_add_node_attr_with_dict(self):
        """Test adding a node attribute with dictionary input."""
        data = create_simple_hetero_data()

        # Create values as dictionary per node type
        values = {
            'user': torch.randn(3, 8),
            'item': torch.randn(2, 8),
        }

        add_node_attr(data, values, attr_name='test_attr')

        self.assertEqual(data['user'].test_attr.shape, (3, 8))
        self.assertEqual(data['item'].test_attr.shape, (2, 8))
        self.assert_tensor_equality(data['user'].test_attr, values['user'])
        self.assert_tensor_equality(data['item'].test_attr, values['item'])

    def test_add_node_attr_with_dict_partial(self):
        """Test adding a node attribute with dictionary containing only some node types."""
        data = create_simple_hetero_data()

        # Only provide values for 'user' node type
        values = {
            'user': torch.randn(3, 8),
        }

        add_node_attr(data, values, attr_name='test_attr')

        self.assertTrue(hasattr(data['user'], 'test_attr'))
        self.assertEqual(data['user'].test_attr.shape, (3, 8))
        self.assertFalse(hasattr(data['item'], 'test_attr'))

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


class TestAddEdgeAttr(TestCase):
    def test_add_edge_attr_with_attr_name(self):
        """Test adding an edge attribute with a specific attribute name."""
        data = create_simple_hetero_data()

        # Create values in homogeneous order (3 + 3 = 6 edges)
        values = torch.randn(6, 8)

        add_edge_attr(data, values, attr_name='test_attr')

        self.assertEqual(data['user', 'buys', 'item'].test_attr.shape, (3, 8))
        self.assertEqual(data['item', 'bought_by', 'user'].test_attr.shape, (3, 8))
        self.assert_tensor_equality(data['user', 'buys', 'item'].test_attr, values[:3])
        self.assert_tensor_equality(data['item', 'bought_by', 'user'].test_attr, values[3:])

    def test_add_edge_attr_with_dict(self):
        """Test adding an edge attribute with dictionary input."""
        data = create_simple_hetero_data()

        # Create values as dictionary per edge type
        values = {
            ('user', 'buys', 'item'): torch.randn(3, 8),
            ('item', 'bought_by', 'user'): torch.randn(3, 8),
        }

        add_edge_attr(data, values, attr_name='test_attr')

        self.assertEqual(data['user', 'buys', 'item'].test_attr.shape, (3, 8))
        self.assertEqual(data['item', 'bought_by', 'user'].test_attr.shape, (3, 8))
        self.assert_tensor_equality(data['user', 'buys', 'item'].test_attr, values[('user', 'buys', 'item')])
        self.assert_tensor_equality(data['item', 'bought_by', 'user'].test_attr, values[('item', 'bought_by', 'user')])

    def test_add_edge_attr_with_dict_partial(self):
        """Test adding an edge attribute with dictionary containing only some edge types."""
        data = create_simple_hetero_data()

        # Only provide values for one edge type
        values = {
            ('user', 'buys', 'item'): torch.randn(3, 8),
        }

        add_edge_attr(data, values, attr_name='test_attr')

        self.assertTrue(hasattr(data['user', 'buys', 'item'], 'test_attr'))
        self.assertEqual(data['user', 'buys', 'item'].test_attr.shape, (3, 8))
        self.assertFalse(hasattr(data['item', 'bought_by', 'user'], 'test_attr'))

    def test_add_edge_attr_concatenate_to_edge_attr(self):
        """Test adding an edge attribute by concatenating to existing edge_attr."""
        data = create_simple_hetero_data()

        # Add initial edge attributes
        data['user', 'buys', 'item'].edge_attr = torch.randn(3, 4)
        data['item', 'bought_by', 'user'].edge_attr = torch.randn(3, 4)

        original_buys_attr = data['user', 'buys', 'item'].edge_attr.clone()
        original_bought_by_attr = data['item', 'bought_by', 'user'].edge_attr.clone()

        # Create values in homogeneous order
        values = torch.randn(6, 8)

        add_edge_attr(data, values, attr_name=None)

        # Check that edge_attr was concatenated
        self.assertEqual(data['user', 'buys', 'item'].edge_attr.shape, (3, 12))  # 4 + 8
        self.assertEqual(data['item', 'bought_by', 'user'].edge_attr.shape, (3, 12))  # 4 + 8

        # Check original features are preserved
        self.assert_tensor_equality(data['user', 'buys', 'item'].edge_attr[:, :4], original_buys_attr)
        self.assert_tensor_equality(data['item', 'bought_by', 'user'].edge_attr[:, :4], original_bought_by_attr)

    def test_add_edge_attr_create_edge_attr_if_none(self):
        """Test creating edge_attr attribute if it doesn't exist."""
        data = create_simple_hetero_data()

        # Ensure no edge_attr exists
        if hasattr(data['user', 'buys', 'item'], 'edge_attr'):
            del data['user', 'buys', 'item'].edge_attr
        if hasattr(data['item', 'bought_by', 'user'], 'edge_attr'):
            del data['item', 'bought_by', 'user'].edge_attr

        values = torch.randn(6, 8)

        add_edge_attr(data, values, attr_name=None)

        self.assertEqual(data['user', 'buys', 'item'].edge_attr.shape, (3, 8))
        self.assertEqual(data['item', 'bought_by', 'user'].edge_attr.shape, (3, 8))


if __name__ == '__main__':
    absltest.main()
