"""
Tests for NodeBatch class.

This module contains comprehensive tests for the NodeBatch class,
focusing on construction to and from node tensors round-trip functionality.
"""

import torch
import torchrec

from gigl.experimental.knowledge_graph_embedding.lib.data.node_batch import NodeBatch
from gigl.src.common.types.graph_data import CondensedNodeType, NodeType
from tests.test_assets.test_case import TestCase


class TestNodeBatch(TestCase):
    """Test suite for the NodeBatch class."""

    def setUp(self):
        """Set up test fixtures with sample data."""
        # Sample condensed node type to node type mapping
        self.condensed_node_type_to_node_type_map = {
            CondensedNodeType(0): NodeType("user"),
            CondensedNodeType(1): NodeType("item"),
            CondensedNodeType(2): NodeType("category"),
        }

    def test_basic_round_trip_conversion(self):
        """Test basic round-trip conversion from tensors to NodeBatch and back."""
        # Create sample input tensors
        nodes = torch.tensor([100, 101, 102], dtype=torch.int32)
        condensed_node_type = torch.tensor(0, dtype=torch.int32)
        condensed_edge_type = torch.tensor(5, dtype=torch.int32)

        # Create NodeBatch from tensors
        node_batch = NodeBatch.from_node_tensors(
            nodes=nodes,
            condensed_node_type=condensed_node_type,
            condensed_edge_type=condensed_edge_type,
            condensed_node_type_to_node_type_map=self.condensed_node_type_to_node_type_map,
        )

        # Verify NodeBatch structure
        self.assertIsInstance(node_batch.nodes, torchrec.KeyedJaggedTensor)
        self.assertEqual(node_batch.condensed_node_type, condensed_node_type)
        self.assertEqual(node_batch.condensed_edge_type, condensed_edge_type)

        # Convert back to tensors
        (
            reconstructed_nodes,
            reconstructed_cnt,
            reconstructed_cet,
        ) = node_batch.to_node_tensors()

        # Verify round-trip preservation
        self.assertTrue(torch.equal(reconstructed_nodes, nodes))
        self.assertTrue(torch.equal(reconstructed_cnt, condensed_node_type))
        self.assertTrue(torch.equal(reconstructed_cet, condensed_edge_type))

    def test_single_node_conversion(self):
        """Test conversion with a single node."""
        nodes = torch.tensor([42], dtype=torch.int32)
        condensed_node_type = torch.tensor(1, dtype=torch.int32)
        condensed_edge_type = torch.tensor(3, dtype=torch.int32)

        # Create and convert back
        node_batch = NodeBatch.from_node_tensors(
            nodes=nodes,
            condensed_node_type=condensed_node_type,
            condensed_edge_type=condensed_edge_type,
            condensed_node_type_to_node_type_map=self.condensed_node_type_to_node_type_map,
        )

        (
            reconstructed_nodes,
            reconstructed_cnt,
            reconstructed_cet,
        ) = node_batch.to_node_tensors()

        # Verify single node is preserved
        self.assertTrue(torch.equal(reconstructed_nodes, nodes))
        self.assertTrue(torch.equal(reconstructed_cnt, condensed_node_type))
        self.assertTrue(torch.equal(reconstructed_cet, condensed_edge_type))
        self.assertEqual(reconstructed_nodes.shape, torch.Size([1]))

    def test_multiple_nodes_different_types(self):
        """Test with different condensed node types."""
        test_cases = [
            {
                "nodes": torch.tensor([10, 11, 12, 13], dtype=torch.int32),
                "condensed_node_type": torch.tensor(0, dtype=torch.int32),
                "condensed_edge_type": torch.tensor(1, dtype=torch.int32),
            },
            {
                "nodes": torch.tensor([20, 21], dtype=torch.int32),
                "condensed_node_type": torch.tensor(1, dtype=torch.int32),
                "condensed_edge_type": torch.tensor(2, dtype=torch.int32),
            },
            {
                "nodes": torch.tensor([30, 31, 32, 33, 34, 35], dtype=torch.int32),
                "condensed_node_type": torch.tensor(2, dtype=torch.int32),
                "condensed_edge_type": torch.tensor(0, dtype=torch.int32),
            },
        ]

        for i, case in enumerate(test_cases):
            with self.subTest(case=i):
                node_batch = NodeBatch.from_node_tensors(
                    nodes=case["nodes"],
                    condensed_node_type=case["condensed_node_type"],
                    condensed_edge_type=case["condensed_edge_type"],
                    condensed_node_type_to_node_type_map=self.condensed_node_type_to_node_type_map,
                )

                (
                    reconstructed_nodes,
                    reconstructed_cnt,
                    reconstructed_cet,
                ) = node_batch.to_node_tensors()

                self.assertTrue(torch.equal(reconstructed_nodes, case["nodes"]))
                self.assertTrue(
                    torch.equal(reconstructed_cnt, case["condensed_node_type"])
                )
                self.assertTrue(
                    torch.equal(reconstructed_cet, case["condensed_edge_type"])
                )

    def test_keyed_jagged_tensor_structure(self):
        """Test that the KeyedJaggedTensor is structured correctly."""
        nodes = torch.tensor([100, 101, 102], dtype=torch.int32)
        condensed_node_type = torch.tensor(1, dtype=torch.int32)  # Maps to "item"
        condensed_edge_type = torch.tensor(0, dtype=torch.int32)

        node_batch = NodeBatch.from_node_tensors(
            nodes=nodes,
            condensed_node_type=condensed_node_type,
            condensed_edge_type=condensed_edge_type,
            condensed_node_type_to_node_type_map=self.condensed_node_type_to_node_type_map,
        )

        # Verify KeyedJaggedTensor structure
        kjt = node_batch.nodes
        self.assertEqual(
            len(kjt.keys()), len(self.condensed_node_type_to_node_type_map)
        )
        self.assertEqual(
            sorted(kjt.keys()), sorted(self.condensed_node_type_to_node_type_map.keys())
        )

        # Verify that only the correct condensed node type has values
        lengths_per_key = kjt.length_per_key()
        for i, key in enumerate(kjt.keys()):
            if key == condensed_node_type.item():
                # This key should have length 3 for 3 nodes
                expected_length = len(nodes)
                actual_length = lengths_per_key[i]
                self.assertEqual(actual_length, expected_length)
            else:
                # Other keys should have empty lengths
                actual_length = lengths_per_key[i]
                self.assertEqual(actual_length, 0)

    def test_tensor_dtypes_preservation(self):
        """Test that tensor dtypes are preserved correctly."""
        nodes = torch.tensor([1000, 2000, 3000], dtype=torch.int32)
        condensed_node_type = torch.tensor(0, dtype=torch.int32)
        condensed_edge_type = torch.tensor(7, dtype=torch.int32)

        node_batch = NodeBatch.from_node_tensors(
            nodes=nodes,
            condensed_node_type=condensed_node_type,
            condensed_edge_type=condensed_edge_type,
            condensed_node_type_to_node_type_map=self.condensed_node_type_to_node_type_map,
        )

        (
            reconstructed_nodes,
            reconstructed_cnt,
            reconstructed_cet,
        ) = node_batch.to_node_tensors()

        # Verify dtypes are preserved
        self.assertEqual(reconstructed_nodes.dtype, torch.int32)
        self.assertEqual(reconstructed_cnt.dtype, torch.int32)
        self.assertEqual(reconstructed_cet.dtype, torch.int32)

        # Also verify the dtypes in the original batch
        self.assertEqual(node_batch.condensed_node_type.dtype, torch.int32)
        self.assertEqual(node_batch.condensed_edge_type.dtype, torch.int32)

    def test_large_node_batch(self):
        """Test with a larger batch of nodes."""
        # Create a larger batch
        num_nodes = 1000
        nodes = torch.arange(0, num_nodes, dtype=torch.int32)
        condensed_node_type = torch.tensor(2, dtype=torch.int32)
        condensed_edge_type = torch.tensor(1, dtype=torch.int32)

        node_batch = NodeBatch.from_node_tensors(
            nodes=nodes,
            condensed_node_type=condensed_node_type,
            condensed_edge_type=condensed_edge_type,
            condensed_node_type_to_node_type_map=self.condensed_node_type_to_node_type_map,
        )

        (
            reconstructed_nodes,
            reconstructed_cnt,
            reconstructed_cet,
        ) = node_batch.to_node_tensors()

        # Verify large batch is handled correctly
        self.assertTrue(torch.equal(reconstructed_nodes, nodes))
        self.assertTrue(torch.equal(reconstructed_cnt, condensed_node_type))
        self.assertTrue(torch.equal(reconstructed_cet, condensed_edge_type))
        self.assertEqual(len(reconstructed_nodes), num_nodes)

    def test_validation_in_to_node_tensors(self):
        """Test the validation logic in to_node_tensors method."""
        nodes = torch.tensor([100, 101], dtype=torch.int32)
        condensed_node_type = torch.tensor(0, dtype=torch.int32)
        condensed_edge_type = torch.tensor(1, dtype=torch.int32)

        node_batch = NodeBatch.from_node_tensors(
            nodes=nodes,
            condensed_node_type=condensed_node_type,
            condensed_edge_type=condensed_edge_type,
            condensed_node_type_to_node_type_map=self.condensed_node_type_to_node_type_map,
        )

        # This should not raise any assertions since the batch is valid
        (
            reconstructed_nodes,
            reconstructed_cnt,
            reconstructed_cet,
        ) = node_batch.to_node_tensors()

        # Verify the validation is working by checking the internal structure
        lengths_per_key = torch.tensor(node_batch.nodes.length_per_key())
        non_zero_keys = lengths_per_key.argwhere().ravel()
        self.assertEqual(len(non_zero_keys), 1, "Should have exactly one non-zero key")

        argmax_key = lengths_per_key.argmax().item()
        self.assertEqual(
            argmax_key,
            condensed_node_type.item(),
            "The key with maximum length should match condensed_node_type",
        )

    def test_empty_mapping_edge_case(self):
        """Test behavior with minimal mapping."""
        # Test with single entry mapping
        minimal_mapping = {CondensedNodeType(0): NodeType("single_type")}

        nodes = torch.tensor([42], dtype=torch.int32)
        condensed_node_type = torch.tensor(0, dtype=torch.int32)
        condensed_edge_type = torch.tensor(0, dtype=torch.int32)

        node_batch = NodeBatch.from_node_tensors(
            nodes=nodes,
            condensed_node_type=condensed_node_type,
            condensed_edge_type=condensed_edge_type,
            condensed_node_type_to_node_type_map=minimal_mapping,
        )

        (
            reconstructed_nodes,
            reconstructed_cnt,
            reconstructed_cet,
        ) = node_batch.to_node_tensors()

        self.assertTrue(torch.equal(reconstructed_nodes, nodes))
        self.assertTrue(torch.equal(reconstructed_cnt, condensed_node_type))
        self.assertTrue(torch.equal(reconstructed_cet, condensed_edge_type))

    def test_node_batch_attributes(self):
        """Test that NodeBatch has the expected attributes and types."""
        nodes = torch.tensor([1, 2, 3], dtype=torch.int32)
        condensed_node_type = torch.tensor(0, dtype=torch.int32)
        condensed_edge_type = torch.tensor(1, dtype=torch.int32)

        node_batch = NodeBatch.from_node_tensors(
            nodes=nodes,
            condensed_node_type=condensed_node_type,
            condensed_edge_type=condensed_edge_type,
            condensed_node_type_to_node_type_map=self.condensed_node_type_to_node_type_map,
        )

        # Test that all expected attributes exist
        self.assertTrue(hasattr(node_batch, "nodes"))
        self.assertTrue(hasattr(node_batch, "condensed_node_type"))
        self.assertTrue(hasattr(node_batch, "condensed_edge_type"))

        # Test attribute types
        self.assertIsInstance(node_batch.nodes, torchrec.KeyedJaggedTensor)
        self.assertIsInstance(node_batch.condensed_node_type, torch.Tensor)
        self.assertIsInstance(node_batch.condensed_edge_type, torch.Tensor)

    def test_sequential_conversions(self):
        """Test multiple sequential conversions to ensure stability."""
        nodes = torch.tensor([50, 51, 52], dtype=torch.int32)
        condensed_node_type = torch.tensor(1, dtype=torch.int32)
        condensed_edge_type = torch.tensor(2, dtype=torch.int32)

        # Perform multiple round-trips
        original_nodes = nodes.clone()
        original_cnt = condensed_node_type.clone()
        original_cet = condensed_edge_type.clone()

        for i in range(3):
            with self.subTest(iteration=i):
                node_batch = NodeBatch.from_node_tensors(
                    nodes=nodes,
                    condensed_node_type=condensed_node_type,
                    condensed_edge_type=condensed_edge_type,
                    condensed_node_type_to_node_type_map=self.condensed_node_type_to_node_type_map,
                )

                (
                    nodes,
                    condensed_node_type,
                    condensed_edge_type,
                ) = node_batch.to_node_tensors()

                # Verify data is still preserved after multiple conversions
                self.assertTrue(torch.equal(nodes, original_nodes))
                self.assertTrue(torch.equal(condensed_node_type, original_cnt))
                self.assertTrue(torch.equal(condensed_edge_type, original_cet))
