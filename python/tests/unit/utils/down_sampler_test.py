import unittest

import torch
from torch.testing import assert_close

from gigl.utils.down_sampler import NodeLabelDownsampler


class TestNodeLabelDownsampler(unittest.TestCase):
    def test_homogeneous_default_filter(self):
        """Test basic homogeneous downsampling with default >= 0 filter."""
        downsampler = NodeLabelDownsampler()

        train_node_ids = torch.tensor([0, 20, 10, 3])
        val_node_ids = torch.tensor([1, 4])
        test_node_ids = torch.tensor([15, 6])

        node_label_feats = torch.tensor([1, -1, 2, 4, 1, 6, -1, -2]).unsqueeze(1)
        node_label_ids = torch.tensor([10, 6, 20, 4, 3, 15, 1, 0])

        homogeneous_splits = (train_node_ids, val_node_ids, test_node_ids)

        downsampled_splits = downsampler(
            splits=homogeneous_splits,
            node_label_ids=node_label_ids,
            node_label_feats=node_label_feats,
        )

        # Verify return type
        self.assertIsInstance(downsampled_splits, tuple)
        self.assertEqual(len(downsampled_splits), 3)

        downsampled_train, downsampled_val, downsampled_test = downsampled_splits

        expected_train = torch.tensor([20, 10, 3])
        expected_val = torch.tensor([4])
        expected_test = torch.tensor([15])

        assert_close(downsampled_train, expected_train)
        assert_close(downsampled_val, expected_val)
        assert_close(downsampled_test, expected_test)


if __name__ == "__main__":
    unittest.main()
