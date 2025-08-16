import unittest

import torch
from parameterized import param, parameterized

from gigl.utils.node_labels import get_labels_from_features
from tests.test_assets.distributed.utils import assert_tensor_equality


class NodeLabelsTest(unittest.TestCase):
    @parameterized.expand(
        [
            param(
                "Basic test with default label_dim=1",
                feature_and_label_tensor=torch.tensor(
                    [
                        [1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [9.0, 10.0, 11.0, 12.0],
                    ]
                ),
                label_dim=1,
                expected_features=torch.tensor(
                    [[1.0, 2.0, 3.0], [5.0, 6.0, 7.0], [9.0, 10.0, 11.0]]
                ),
                expected_labels=torch.tensor([[4.0], [8.0], [12.0]]),
            ),
            param(
                "Test with label_dim=2",
                feature_and_label_tensor=torch.tensor(
                    [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]
                ),
                label_dim=2,
                expected_features=torch.tensor([[1.0, 2.0, 3.0], [6.0, 7.0, 8.0]]),
                expected_labels=torch.tensor([[4.0, 5.0], [9.0, 10.0]]),
            ),
            param(
                "Test with single feature column",
                feature_and_label_tensor=torch.tensor(
                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
                ),
                label_dim=1,
                expected_features=torch.tensor([[1.0], [3.0], [5.0]]),
                expected_labels=torch.tensor([[2.0], [4.0], [6.0]]),
            ),
            param(
                "Test with no features and labels",
                feature_and_label_tensor=torch.tensor([[3.0], [6.0]]),
                label_dim=1,
                expected_features=torch.empty((2, 0)),
                expected_labels=torch.tensor([[3.0], [6.0]]),
            ),
            param(
                "Test with minimal tensor (1x2)",
                feature_and_label_tensor=torch.tensor([[1.0, 2.0]]),
                label_dim=1,
                expected_features=torch.tensor([[1.0]]),
                expected_labels=torch.tensor([[2.0]]),
            ),
        ]
    )
    def test_get_labels_from_features(
        self,
        _,
        feature_and_label_tensor: torch.Tensor,
        label_dim: int,
        expected_features: torch.Tensor,
        expected_labels: torch.Tensor,
    ):
        features, labels = get_labels_from_features(feature_and_label_tensor, label_dim)
        assert_tensor_equality(features, expected_features)
        assert_tensor_equality(labels, expected_labels)


if __name__ == "__main__":
    unittest.main()
