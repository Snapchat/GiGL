import unittest
from collections import abc

import torch
from parameterized import param, parameterized

from gigl.distributed.dataset_factory import (
    _get_labels_from_features,
    build_dataset_from_task_config_uri,
)
from gigl.distributed.dist_context import DistributedContext
from gigl.src.mocking.lib.versioning import get_mocked_dataset_artifact_metadata
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO,
)
from gigl.types.graph import DEFAULT_HOMOGENEOUS_NODE_TYPE
from tests.test_assets.distributed.utils import assert_tensor_equality


# TODO(kmonte, mkolodner): Add more tests for heterogeneous datasets.
class TestDatasetFactory(unittest.TestCase):
    def setUp(self):
        # Set up any necessary context or mock data
        self._dist_context = DistributedContext(
            main_worker_ip_address="localhost", global_rank=0, global_world_size=1
        )

    @parameterized.expand(
        [
            param("training", is_inference=False),
            param("inference", is_inference=True),
        ]
    )
    def test_build_dataset_from_task_config_uri_homogeneous(
        self, _, is_inference: bool
    ):
        # Test with a valid task config URI
        task_config_uri = get_mocked_dataset_artifact_metadata()[
            CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO.name
        ].frozen_gbml_config_uri

        dataset = build_dataset_from_task_config_uri(
            task_config_uri,
            self._dist_context,
            is_inference=is_inference,
            _tfrecord_uri_pattern=".*data.tfrecord$",
        )

        if is_inference:
            self.assertIsNone(dataset.train_node_ids)
            self.assertIsNone(dataset.val_node_ids)
            self.assertIsNone(dataset.test_node_ids)
        else:
            # Mapping despite being "homogeneous" as ABLP uses labels as edge types.
            # Use assert isinstance instead of self.assertIsInstance to type narrow.
            assert isinstance(dataset.train_node_ids, abc.Mapping)
            self.assertTrue(
                dataset.train_node_ids.keys() == set([DEFAULT_HOMOGENEOUS_NODE_TYPE])
            )
            assert isinstance(dataset.val_node_ids, abc.Mapping)
            self.assertTrue(
                dataset.val_node_ids.keys() == set([DEFAULT_HOMOGENEOUS_NODE_TYPE])
            )
            assert isinstance(dataset.test_node_ids, abc.Mapping)
            self.assertTrue(
                dataset.val_node_ids.keys() == set([DEFAULT_HOMOGENEOUS_NODE_TYPE])
            )

    @parameterized.expand(
        [
            param(
                "Basic test with label_dim=1",
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
        features, labels = _get_labels_from_features(
            feature_and_label_tensor, label_dim
        )
        assert_tensor_equality(features, expected_features)
        assert_tensor_equality(labels, expected_labels)


if __name__ == "__main__":
    unittest.main()
