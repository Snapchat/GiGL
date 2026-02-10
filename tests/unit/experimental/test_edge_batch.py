"""
Tests for EdgeBatch class.

This module contains comprehensive tests for the EdgeBatch class,
focusing on construction to and from edge tensors round-trip functionality.
"""

import torch
import torchrec
from absl.testing import absltest

from gigl.experimental.knowledge_graph_embedding.lib.data.edge_batch import EdgeBatch
from gigl.src.common.types.graph_data import (
    CondensedEdgeType,
    CondensedNodeType,
    NodeType,
)
from tests.test_assets.test_case import TestCase


class TestEdgeBatch(TestCase):
    """Test suite for the EdgeBatch class."""

    def setUp(self):
        """Set up test fixtures with sample data."""
        # Sample condensed node type to node type mapping
        self.condensed_node_type_to_node_type_map = {
            CondensedNodeType(0): NodeType("user"),
            CondensedNodeType(1): NodeType("item"),
            CondensedNodeType(2): NodeType("category"),
        }

        # Sample condensed edge type to condensed node type mapping
        self.condensed_edge_type_to_condensed_node_type_map = {
            CondensedEdgeType(0): (
                CondensedNodeType(0),
                CondensedNodeType(1),
            ),  # user -> item
            CondensedEdgeType(1): (
                CondensedNodeType(1),
                CondensedNodeType(2),
            ),  # item -> category
            CondensedEdgeType(2): (
                CondensedNodeType(0),
                CondensedNodeType(2),
            ),  # user -> category
        }

    def test_basic_round_trip_conversion(self):
        """Test basic round-trip conversion from tensors to EdgeBatch and back."""
        # Create sample input tensors
        edges = torch.tensor([[100, 200], [101, 201], [102, 202]], dtype=torch.int32)
        condensed_edge_types = torch.tensor(
            [0, 0, 0], dtype=torch.int32
        )  # all user -> item
        edge_labels = torch.tensor([1, 1, 1], dtype=torch.int32)

        # Create EdgeBatch from tensors
        edge_batch = EdgeBatch.from_edge_tensors(
            edges=edges,
            condensed_edge_types=condensed_edge_types,
            edge_labels=edge_labels,
            condensed_node_type_to_node_type_map=self.condensed_node_type_to_node_type_map,
            condensed_edge_type_to_condensed_node_type_map=self.condensed_edge_type_to_condensed_node_type_map,
        )

        # Verify EdgeBatch structure
        self.assertIsInstance(edge_batch.src_dst_pairs, torchrec.KeyedJaggedTensor)
        self.assertTrue(
            torch.equal(edge_batch.condensed_edge_types, condensed_edge_types)
        )
        self.assertTrue(torch.equal(edge_batch.labels, edge_labels))

        # Convert back to tensors
        (
            reconstructed_edges,
            reconstructed_edge_types,
            reconstructed_labels,
        ) = edge_batch.to_edge_tensors(
            condensed_edge_type_to_condensed_node_type_map=self.condensed_edge_type_to_condensed_node_type_map
        )

        # Verify round-trip preservation
        self.assertTrue(torch.equal(reconstructed_edges, edges))
        self.assertTrue(torch.equal(reconstructed_edge_types, condensed_edge_types))
        self.assertTrue(torch.equal(reconstructed_labels, edge_labels))

    def test_single_edge_conversion(self):
        """Test conversion with a single edge."""
        edges = torch.tensor([[42, 84]], dtype=torch.int32)
        condensed_edge_types = torch.tensor([1], dtype=torch.int32)  # item -> category
        edge_labels = torch.tensor([1], dtype=torch.int32)

        # Create and convert back
        edge_batch = EdgeBatch.from_edge_tensors(
            edges=edges,
            condensed_edge_types=condensed_edge_types,
            edge_labels=edge_labels,
            condensed_node_type_to_node_type_map=self.condensed_node_type_to_node_type_map,
            condensed_edge_type_to_condensed_node_type_map=self.condensed_edge_type_to_condensed_node_type_map,
        )

        (
            reconstructed_edges,
            reconstructed_edge_types,
            reconstructed_labels,
        ) = edge_batch.to_edge_tensors(
            condensed_edge_type_to_condensed_node_type_map=self.condensed_edge_type_to_condensed_node_type_map
        )

        # Verify single edge is preserved
        self.assertTrue(torch.equal(reconstructed_edges, edges))
        self.assertTrue(torch.equal(reconstructed_edge_types, condensed_edge_types))
        self.assertTrue(torch.equal(reconstructed_labels, edge_labels))
        self.assertEqual(reconstructed_edges.shape, torch.Size([1, 2]))

    def test_multiple_edge_types(self):
        """Test with different condensed edge types in the same batch."""
        edges = torch.tensor(
            [
                [10, 20],  # user -> item (type 0)
                [20, 30],  # item -> category (type 1)
                [10, 30],  # user -> category (type 2)
                [11, 21],  # user -> item (type 0)
            ],
            dtype=torch.int32,
        )
        condensed_edge_types = torch.tensor([0, 1, 2, 0], dtype=torch.int32)
        edge_labels = torch.tensor([1, 1, 1, 1], dtype=torch.int32)

        edge_batch = EdgeBatch.from_edge_tensors(
            edges=edges,
            condensed_edge_types=condensed_edge_types,
            edge_labels=edge_labels,
            condensed_node_type_to_node_type_map=self.condensed_node_type_to_node_type_map,
            condensed_edge_type_to_condensed_node_type_map=self.condensed_edge_type_to_condensed_node_type_map,
        )

        (
            reconstructed_edges,
            reconstructed_edge_types,
            reconstructed_labels,
        ) = edge_batch.to_edge_tensors(
            condensed_edge_type_to_condensed_node_type_map=self.condensed_edge_type_to_condensed_node_type_map
        )

        self.assertTrue(torch.equal(reconstructed_edges, edges))
        self.assertTrue(torch.equal(reconstructed_edge_types, condensed_edge_types))
        self.assertTrue(torch.equal(reconstructed_labels, edge_labels))

    def test_keyed_jagged_tensor_structure(self):
        """Test that the KeyedJaggedTensor is structured correctly."""
        edges = torch.tensor([[100, 200], [101, 201]], dtype=torch.int32)
        condensed_edge_types = torch.tensor(
            [0, 0], dtype=torch.int32
        )  # both user -> item
        edge_labels = torch.tensor([1, 1], dtype=torch.int32)

        edge_batch = EdgeBatch.from_edge_tensors(
            edges=edges,
            condensed_edge_types=condensed_edge_types,
            edge_labels=edge_labels,
            condensed_node_type_to_node_type_map=self.condensed_node_type_to_node_type_map,
            condensed_edge_type_to_condensed_node_type_map=self.condensed_edge_type_to_condensed_node_type_map,
        )

        # Verify KeyedJaggedTensor structure
        kjt = edge_batch.src_dst_pairs
        self.assertEqual(
            len(kjt.keys()), len(self.condensed_node_type_to_node_type_map)
        )
        self.assertEqual(
            sorted(kjt.keys()), sorted(self.condensed_node_type_to_node_type_map.keys())
        )

        # For 2 edges, we should have 2 * 2 = 4 length entries per node type (src, dst for each edge)
        expected_total_length = 2 * len(edges) * len(kjt.keys())
        self.assertEqual(len(kjt.lengths()), expected_total_length)

        # Verify that src and dst nodes are placed correctly
        # For edge type 0 (user -> item), we expect:
        # - User nodes (condensed type 0) in positions 0, 2 (src positions)
        # - Item nodes (condensed type 1) in positions 1, 3 (dst positions)
        # - Category nodes (condensed type 2) should be empty
        kjt_dict = kjt.to_dict()

        # Check user nodes (should contain src nodes: 100, 101)
        user_values = kjt_dict[CondensedNodeType(0)].values()
        expected_user_values = torch.tensor([100, 101], dtype=torch.int32)
        self.assertTrue(torch.equal(user_values, expected_user_values))

        # Check item nodes (should contain dst nodes: 200, 201)
        item_values = kjt_dict[CondensedNodeType(1)].values()
        expected_item_values = torch.tensor([200, 201], dtype=torch.int32)
        self.assertTrue(torch.equal(item_values, expected_item_values))

        # Check category nodes (should be empty for this test)
        category_values = kjt_dict[CondensedNodeType(2)].values()
        self.assertEqual(len(category_values), 0)

    def test_mixed_positive_negative_labels(self):
        """Test with mixed positive and negative edge labels."""
        edges = torch.tensor([[10, 20], [11, 21], [12, 22]], dtype=torch.int32)
        condensed_edge_types = torch.tensor([0, 0, 0], dtype=torch.int32)
        edge_labels = torch.tensor(
            [1, 0, 1], dtype=torch.int32
        )  # positive, negative, positive

        edge_batch = EdgeBatch.from_edge_tensors(
            edges=edges,
            condensed_edge_types=condensed_edge_types,
            edge_labels=edge_labels,
            condensed_node_type_to_node_type_map=self.condensed_node_type_to_node_type_map,
            condensed_edge_type_to_condensed_node_type_map=self.condensed_edge_type_to_condensed_node_type_map,
        )

        (
            reconstructed_edges,
            reconstructed_edge_types,
            reconstructed_labels,
        ) = edge_batch.to_edge_tensors(
            condensed_edge_type_to_condensed_node_type_map=self.condensed_edge_type_to_condensed_node_type_map
        )

        # Verify mixed labels are preserved
        self.assertTrue(torch.equal(reconstructed_labels, edge_labels))
        self.assertTrue(torch.equal(reconstructed_edges, edges))
        self.assertTrue(torch.equal(reconstructed_edge_types, condensed_edge_types))

    def test_tensor_dtypes_preservation(self):
        """Test that tensor dtypes are preserved correctly."""
        edges = torch.tensor([[1000, 2000], [3000, 4000]], dtype=torch.int32)
        condensed_edge_types = torch.tensor([1, 1], dtype=torch.int32)
        edge_labels = torch.tensor([1, 0], dtype=torch.int32)

        edge_batch = EdgeBatch.from_edge_tensors(
            edges=edges,
            condensed_edge_types=condensed_edge_types,
            edge_labels=edge_labels,
            condensed_node_type_to_node_type_map=self.condensed_node_type_to_node_type_map,
            condensed_edge_type_to_condensed_node_type_map=self.condensed_edge_type_to_condensed_node_type_map,
        )

        (
            reconstructed_edges,
            reconstructed_edge_types,
            reconstructed_labels,
        ) = edge_batch.to_edge_tensors(
            condensed_edge_type_to_condensed_node_type_map=self.condensed_edge_type_to_condensed_node_type_map
        )

        # Verify dtypes are preserved
        self.assertEqual(reconstructed_edges.dtype, torch.int32)
        self.assertEqual(reconstructed_edge_types.dtype, torch.int32)
        self.assertEqual(reconstructed_labels.dtype, torch.int32)

        # Also verify the dtypes in the original batch
        self.assertEqual(edge_batch.condensed_edge_types.dtype, torch.int32)
        self.assertEqual(edge_batch.labels.dtype, torch.int32)

    def test_large_edge_batch(self):
        """Test with a larger batch of edges."""
        # Create a larger batch
        num_edges = 1000
        edges = torch.stack(
            [
                torch.arange(0, num_edges, dtype=torch.int32),
                torch.arange(num_edges, 2 * num_edges, dtype=torch.int32),
            ],
            dim=1,
        )
        condensed_edge_types = torch.zeros(num_edges, dtype=torch.int32)  # all type 0
        edge_labels = torch.ones(num_edges, dtype=torch.int32)

        edge_batch = EdgeBatch.from_edge_tensors(
            edges=edges,
            condensed_edge_types=condensed_edge_types,
            edge_labels=edge_labels,
            condensed_node_type_to_node_type_map=self.condensed_node_type_to_node_type_map,
            condensed_edge_type_to_condensed_node_type_map=self.condensed_edge_type_to_condensed_node_type_map,
        )

        (
            reconstructed_edges,
            reconstructed_edge_types,
            reconstructed_labels,
        ) = edge_batch.to_edge_tensors(
            condensed_edge_type_to_condensed_node_type_map=self.condensed_edge_type_to_condensed_node_type_map
        )

        # Verify large batch is handled correctly
        self.assertTrue(torch.equal(reconstructed_edges, edges))
        self.assertTrue(torch.equal(reconstructed_edge_types, condensed_edge_types))
        self.assertTrue(torch.equal(reconstructed_labels, edge_labels))
        self.assertEqual(len(reconstructed_edges), num_edges)

    def test_edge_batch_attributes(self):
        """Test that EdgeBatch has the expected attributes and types."""
        edges = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        condensed_edge_types = torch.tensor([0, 1], dtype=torch.int32)
        edge_labels = torch.tensor([1, 0], dtype=torch.int32)

        edge_batch = EdgeBatch.from_edge_tensors(
            edges=edges,
            condensed_edge_types=condensed_edge_types,
            edge_labels=edge_labels,
            condensed_node_type_to_node_type_map=self.condensed_node_type_to_node_type_map,
            condensed_edge_type_to_condensed_node_type_map=self.condensed_edge_type_to_condensed_node_type_map,
        )

        # Test that all expected attributes exist
        self.assertTrue(hasattr(edge_batch, "src_dst_pairs"))
        self.assertTrue(hasattr(edge_batch, "condensed_edge_types"))
        self.assertTrue(hasattr(edge_batch, "labels"))

        # Test attribute types
        self.assertIsInstance(edge_batch.src_dst_pairs, torchrec.KeyedJaggedTensor)
        self.assertIsInstance(edge_batch.condensed_edge_types, torch.Tensor)
        self.assertIsInstance(edge_batch.labels, torch.Tensor)

    def test_sequential_conversions(self):
        """Test multiple sequential conversions to ensure stability."""
        edges = torch.tensor([[50, 51], [52, 53]], dtype=torch.int32)
        condensed_edge_types = torch.tensor([1, 2], dtype=torch.int32)
        edge_labels = torch.tensor([1, 0], dtype=torch.int32)

        # Store original values
        original_edges = edges.clone()
        original_edge_types = condensed_edge_types.clone()
        original_labels = edge_labels.clone()

        # Perform multiple round-trips
        current_edges = edges
        current_edge_types = condensed_edge_types
        current_labels = edge_labels

        for i in range(3):
            with self.subTest(iteration=i):
                edge_batch = EdgeBatch.from_edge_tensors(
                    edges=current_edges,
                    condensed_edge_types=current_edge_types,
                    edge_labels=current_labels,
                    condensed_node_type_to_node_type_map=self.condensed_node_type_to_node_type_map,
                    condensed_edge_type_to_condensed_node_type_map=self.condensed_edge_type_to_condensed_node_type_map,
                )

                (
                    current_edges,
                    current_edge_types,
                    current_labels,
                ) = edge_batch.to_edge_tensors(
                    condensed_edge_type_to_condensed_node_type_map=self.condensed_edge_type_to_condensed_node_type_map
                )

                # Verify data is still preserved after multiple conversions
                self.assertTrue(torch.equal(current_edges, original_edges))
                self.assertTrue(torch.equal(current_edge_types, original_edge_types))
                self.assertTrue(torch.equal(current_labels, original_labels))

    def test_heterogeneous_edge_distribution(self):
        """Test with edges distributed across all edge types."""
        edges = torch.tensor(
            [
                [10, 20],  # user -> item (type 0)
                [20, 30],  # item -> category (type 1)
                [15, 35],  # user -> category (type 2)
                [11, 21],  # user -> item (type 0)
                [25, 36],  # item -> category (type 1)
            ],
            dtype=torch.int32,
        )
        condensed_edge_types = torch.tensor([0, 1, 2, 0, 1], dtype=torch.int32)
        edge_labels = torch.tensor([1, 1, 0, 1, 0], dtype=torch.int32)

        edge_batch = EdgeBatch.from_edge_tensors(
            edges=edges,
            condensed_edge_types=condensed_edge_types,
            edge_labels=edge_labels,
            condensed_node_type_to_node_type_map=self.condensed_node_type_to_node_type_map,
            condensed_edge_type_to_condensed_node_type_map=self.condensed_edge_type_to_condensed_node_type_map,
        )

        # Verify KeyedJaggedTensor has entries for all node types
        kjt_dict = edge_batch.src_dst_pairs.to_dict()

        # All node types should have some values since we use all edge types
        for node_type in self.condensed_node_type_to_node_type_map.keys():
            self.assertGreater(
                len(kjt_dict[node_type].values()),
                0,
                f"Node type {node_type} should have values",
            )

        (
            reconstructed_edges,
            reconstructed_edge_types,
            reconstructed_labels,
        ) = edge_batch.to_edge_tensors(
            condensed_edge_type_to_condensed_node_type_map=self.condensed_edge_type_to_condensed_node_type_map
        )

        # Verify all data is preserved
        self.assertTrue(torch.equal(reconstructed_edges, edges))
        self.assertTrue(torch.equal(reconstructed_edge_types, condensed_edge_types))
        self.assertTrue(torch.equal(reconstructed_labels, edge_labels))

    def test_validation_assertions_in_to_edge_tensors(self):
        """Test the validation logic in to_edge_tensors method."""
        edges = torch.tensor([[100, 200]], dtype=torch.int32)
        condensed_edge_types = torch.tensor([0], dtype=torch.int32)
        edge_labels = torch.tensor([1], dtype=torch.int32)

        edge_batch = EdgeBatch.from_edge_tensors(
            edges=edges,
            condensed_edge_types=condensed_edge_types,
            edge_labels=edge_labels,
            condensed_node_type_to_node_type_map=self.condensed_node_type_to_node_type_map,
            condensed_edge_type_to_condensed_node_type_map=self.condensed_edge_type_to_condensed_node_type_map,
        )

        # This should not raise any assertions since the batch is valid
        (
            reconstructed_edges,
            reconstructed_edge_types,
            reconstructed_labels,
        ) = edge_batch.to_edge_tensors(
            condensed_edge_type_to_condensed_node_type_map=self.condensed_edge_type_to_condensed_node_type_map
        )

        # Verify the validation logic by checking internal consistency
        num_edges = len(edge_batch.labels)
        num_edge_types = len(edge_batch.condensed_edge_types)
        self.assertEqual(
            num_edges,
            num_edge_types,
            "Number of edges should match number of edge types",
        )
        self.assertEqual(
            len(reconstructed_edges),
            num_edges,
            "Reconstructed edges should match original count",
        )


if __name__ == "__main__":
    absltest.main()
