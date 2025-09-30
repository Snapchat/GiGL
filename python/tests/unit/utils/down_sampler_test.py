import unittest
from unittest.mock import Mock

import torch
from graphlearn_torch.utils import id2idx
from torch.testing import assert_close

from gigl.src.common.types.graph_data import NodeType
from gigl.utils.down_sampler import (
    down_sample_node_ids_from_dataset_labels,
    down_sample_node_ids_from_labels,
)


class TestNodeLabelDownSampler(unittest.TestCase):
    def test_basic_down_sampling(self):
        """Test basic homogeneous down_sampling with default >= 0 filter."""

        node_ids = torch.tensor([0, 20, 10, 3])

        node_label_feats = torch.tensor([1, -1, 2, 4, 1, 6, -1, -2]).unsqueeze(1)
        node_label_ids = torch.tensor([10, 6, 20, 4, 3, 15, 1, 0])

        id2idx_tensor = id2idx(node_label_ids)

        down_sampled_node_ids = down_sample_node_ids_from_labels(
            node_ids=node_ids,
            id2idx=id2idx_tensor,
            node_label_feats=node_label_feats,
        )

        expected_down_sampled_node_ids = torch.tensor([20, 10, 3])

        assert_close(down_sampled_node_ids, expected_down_sampled_node_ids)

    def test_down_sampling_with_homogeneous_dataset_default_filter(self):
        """Test down_sampling with homogeneous dataset and default >= 0 filter."""

        # Test data
        node_ids = torch.tensor([0, 20, 10, 3])
        node_label_feats = torch.tensor([1, -1, 2, 4, 1, 6, -1, -2]).unsqueeze(1)
        node_label_ids = torch.tensor([10, 6, 20, 4, 3, 15, 1, 0])

        id2idx_tensor = id2idx(node_label_ids)

        # Mock the Feature object
        mock_labels = Mock()
        mock_labels.lazy_init_with_ipc_handle = Mock()
        mock_labels.feature_tensor = node_label_feats
        mock_labels.id2idx = id2idx_tensor

        # Mock the DistDataset object for homogeneous case
        mock_dataset = Mock()
        mock_dataset.node_labels = mock_labels  # Not a Mapping, so homogeneous

        # Call the function
        down_sampled_node_ids = down_sample_node_ids_from_dataset_labels(
            dataset=mock_dataset,
            node_ids=node_ids,
        )

        # 0 has a negative label values and should be excluded. The remaining node ids have positive label values (20 -> 2, 10 -> 1, 3 -> 1) and should be included
        expected_down_sampled_node_ids = torch.tensor([20, 10, 3])

        # Verify the result
        assert_close(down_sampled_node_ids, expected_down_sampled_node_ids)

    def test_down_sampling_with_heterogeneous_dataset_default_filter(self):
        """Test down_sampling with heterogeneous dataset requiring node_type."""

        # Test data
        node_ids = torch.tensor([0, 20, 10, 3])
        node_label_feats = torch.tensor([1, -1, 2, 4, 1, 6, -1, -2]).unsqueeze(1)
        node_label_ids = torch.tensor([10, 6, 20, 4, 3, 15, 1, 0])

        id2idx_tensor = id2idx(node_label_ids)

        # Mock the Feature object for the specific node type
        mock_labels = Mock()
        mock_labels.lazy_init_with_ipc_handle = Mock()
        mock_labels.feature_tensor = node_label_feats
        mock_labels.id2idx = id2idx_tensor

        # Mock the DistDataset object for heterogeneous case
        mock_dataset = Mock()
        # Make node_labels a dict to simulate heterogeneous dataset
        mock_dataset.node_labels = {NodeType("user"): mock_labels}

        # Call the function with node_type specified
        down_sampled_node_ids = down_sample_node_ids_from_dataset_labels(
            dataset=mock_dataset,
            node_ids=node_ids,
            node_type=NodeType("user"),
        )

        # 0 has a negative label values and should be excluded. The remaining node ids have positive label values (20 -> 2, 10 -> 1, 3 -> 1) and should be included
        expected_down_sampled_node_ids = torch.tensor([20, 10, 3])

        # Verify the result
        assert_close(down_sampled_node_ids, expected_down_sampled_node_ids)

    def test_down_sampling_with_custom_label_filter(self):
        """Test down_sampling with custom label filter function."""

        # Test data - using labels where we want to filter for values > 2
        node_ids = torch.tensor([1, 2, 3])
        node_label_feats = torch.tensor([1, 3, 5]).unsqueeze(1)  # 1, 3, 5
        node_label_ids = torch.tensor([1, 2, 3])

        id2idx_tensor = id2idx(node_label_ids)

        # Mock objects
        mock_labels = Mock()
        mock_labels.lazy_init_with_ipc_handle = Mock()
        mock_labels.feature_tensor = node_label_feats
        mock_labels.id2idx = id2idx_tensor

        mock_dataset = Mock()
        mock_dataset.node_labels = mock_labels

        # Custom filter: only include labels > 2
        def custom_filter(labels):
            return labels > 2

        # Call function with custom filter
        down_sampled_node_ids = down_sample_node_ids_from_dataset_labels(
            dataset=mock_dataset,
            node_ids=node_ids,
            label_filter_fn=custom_filter,
        )

        # Expected: only nodes 2 and 3 have labels > 2 (labels 3 and 5)
        expected_down_sampled_node_ids = torch.tensor([2, 3])

        # Verify the result
        assert_close(down_sampled_node_ids, expected_down_sampled_node_ids)


if __name__ == "__main__":
    unittest.main()
