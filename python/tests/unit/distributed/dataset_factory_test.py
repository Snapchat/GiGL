import unittest
from collections import abc

from parameterized import param, parameterized

from gigl.distributed.dataset_factory import build_dataset_from_task_config_uri
from gigl.distributed.dist_context import DistributedContext
from gigl.src.mocking.lib.versioning import get_mocked_dataset_artifact_metadata
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO,
)


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
            self.assertIsInstance(dataset.train_node_ids, abc.Mapping)
            self.assertIsInstance(dataset.val_node_ids, abc.Mapping)
            self.assertIsInstance(dataset.test_node_ids, abc.Mapping)


if __name__ == "__main__":
    unittest.main()
