import os
from collections import abc
from unittest import mock

import torch
from absl.testing import absltest
from parameterized import param, parameterized

from gigl.distributed.dataset_factory import build_dataset_from_task_config_uri
from gigl.distributed.utils.lossless_compression import FEATURE_COMPRESSION_ENV
from gigl.src.mocking.lib.versioning import get_mocked_dataset_artifact_metadata
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO,
)
from gigl.types.graph import DEFAULT_HOMOGENEOUS_NODE_TYPE, FeatureQuantizationMetadata
from tests.test_assets.distributed.utils import create_test_process_group
from tests.test_assets.test_case import TestCase


# TODO(kmonte, mkolodner): Add more tests for heterogeneous datasets.
class TestDatasetFactory(TestCase):
    def setUp(self) -> None:
        # A process group on a free port lets the factory pick free dataset building ports.
        create_test_process_group()

    def tearDown(self) -> None:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

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

    def test_node_features_are_compressed_by_default_and_off_keeps_float32(
        self,
    ) -> None:
        task_config_uri = get_mocked_dataset_artifact_metadata()[
            CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO.name
        ].frozen_gbml_config_uri
        with mock.patch.dict(os.environ):
            os.environ.pop(FEATURE_COMPRESSION_ENV, None)
            compressed = build_dataset_from_task_config_uri(
                task_config_uri,
                is_inference=False,
                _tfrecord_uri_pattern=".*data.tfrecord$",
            )
        with mock.patch.dict(os.environ, {FEATURE_COMPRESSION_ENV: "OFF"}):
            uncompressed = build_dataset_from_task_config_uri(
                task_config_uri,
                is_inference=False,
                _tfrecord_uri_pattern=".*data.tfrecord$",
            )

        # Cora's features are all 0/1, so every column is inferred as a lossless bit column.
        metadata = compressed.node_quantization_metadata
        assert isinstance(metadata, FeatureQuantizationMetadata)
        self.assertEqual(
            metadata.lossless_feature_indices, tuple(range(metadata.feature_dim))
        )
        self.assertEqual(
            metadata.lossless_column_kinds, ("bit",) * metadata.feature_dim
        )
        self.assertIsNone(compressed.node_features)
        self.assertIsNone(uncompressed.node_quantization_metadata)
        self.assertIsNotNone(uncompressed.node_features)


if __name__ == "__main__":
    absltest.main()
