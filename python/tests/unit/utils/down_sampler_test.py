import unittest
from typing import Callable, Optional
from unittest.mock import Mock

import torch
from graphlearn_torch.utils import id2idx
from parameterized import param, parameterized
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

        id2index = id2idx(node_label_ids)

        down_sampled_node_ids = down_sample_node_ids_from_labels(
            node_ids=node_ids,
            id2idx=id2index,
            node_label_feats=node_label_feats,
        )

        # 0 has a negative label values and should be excluded. The remaining node ids have positive label values (20 -> 2, 10 -> 1, 3 -> 1) and should be included
        expected_down_sampled_node_ids = torch.tensor([20, 10, 3])

        assert_close(down_sampled_node_ids, expected_down_sampled_node_ids)

    @parameterized.expand(
        [
            param(
                "homogeneous_dataset_default_filter",
                node_ids=torch.tensor([0, 20, 10, 3]),
                node_label_feats=torch.tensor([1, -1, 2, 4, 1, 6, -1, -2]),
                node_label_ids=torch.tensor([10, 6, 20, 4, 3, 15, 1, 0]),
                node_type=None,
                label_filter_fn=None,
                # 0 has a negative label values and should be excluded.
                # The remaining node ids have positive label values (20 -> 2, 10 -> 1, 3 -> 1) and should be included
                expected_result=torch.tensor([20, 10, 3]),
            ),
            param(
                "heterogeneous_dataset_default_filter",
                node_ids=torch.tensor([0, 20, 10, 3]),
                node_label_feats=torch.tensor([1, -1, 2, 4, 1, 6, -1, -2]),
                node_label_ids=torch.tensor([10, 6, 20, 4, 3, 15, 1, 0]),
                node_type=NodeType("user"),
                label_filter_fn=None,
                # 0 has a negative label values and should be excluded.
                # The remaining node ids have positive label values (20 -> 2, 10 -> 1, 3 -> 1) and should be included
                expected_result=torch.tensor([20, 10, 3]),
            ),
            param(
                "custom_label_filter",
                node_ids=torch.tensor([1, 2, 3]),
                node_label_feats=torch.tensor([1, 3, 5]),
                node_label_ids=torch.tensor([1, 2, 3]),
                node_type=None,
                label_filter_fn=lambda labels: labels > 2,
                # Only nodes 2 and 3 have labels greater than 2 (2 -> 3 and 3 -> 5)
                expected_result=torch.tensor([2, 3]),
            ),
        ]
    )
    def test_down_sampling_from_dataset_labels(
        self,
        _,
        node_ids: torch.Tensor,
        node_label_feats: torch.Tensor,
        node_label_ids: torch.Tensor,
        node_type: Optional[NodeType],
        label_filter_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
        expected_result: torch.Tensor,
    ):
        """
        Test down_sampling with various dataset configurations and filters.
        """

        # Prepare label features with correct dimensions
        node_label_feats = node_label_feats.unsqueeze(1)
        id2index = id2idx(node_label_ids)

        # Mock the Feature object
        mock_labels = Mock()
        mock_labels.lazy_init_with_ipc_handle = Mock()
        mock_labels.feature_tensor = node_label_feats
        mock_labels.id2index = id2index

        # Mock the DistDataset object
        mock_dataset = Mock()
        if node_type is not None:
            mock_dataset.node_labels = {node_type: mock_labels}
        else:
            mock_dataset.node_labels = mock_labels

        if label_filter_fn is None:
            # Use the default label filter function, which filters out node ids with negative label values
            down_sampled_node_ids = down_sample_node_ids_from_dataset_labels(
                dataset=mock_dataset,
                node_ids=node_ids,
                node_type=node_type,
            )
        else:
            down_sampled_node_ids = down_sample_node_ids_from_dataset_labels(
                dataset=mock_dataset,
                node_ids=node_ids,
                node_type=node_type,
                label_filter_fn=label_filter_fn,
            )

        assert_close(down_sampled_node_ids, expected_result)


if __name__ == "__main__":
    unittest.main()
