import os
from collections import abc
from unittest import mock

import torch
from absl.testing import absltest

from gigl.common.utils.feature_quantization.lossless_compression import (
    FEATURE_COMPRESSION_ENV,
)
from gigl.distributed.dataset_factory import build_dataset_from_task_config_uri
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

    def test_cora_dataset_has_splits_for_training_and_none_for_inference_and_compresses_features(
        self,
    ) -> None:
        """The default (training, compressed) build and the compression-off inference build share one Cora load each."""
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
                is_inference=True,
                _tfrecord_uri_pattern=".*data.tfrecord$",
            )

        # Mapping despite being "homogeneous" as ABLP uses labels as edge types.
        # Use assert isinstance instead of self.assertIsInstance to type narrow.
        assert isinstance(compressed.train_node_ids, abc.Mapping)
        self.assertEqual(
            compressed.train_node_ids.keys(), {DEFAULT_HOMOGENEOUS_NODE_TYPE}
        )
        assert isinstance(compressed.val_node_ids, abc.Mapping)
        self.assertEqual(
            compressed.val_node_ids.keys(), {DEFAULT_HOMOGENEOUS_NODE_TYPE}
        )
        assert isinstance(compressed.test_node_ids, abc.Mapping)
        self.assertEqual(
            compressed.test_node_ids.keys(), {DEFAULT_HOMOGENEOUS_NODE_TYPE}
        )
        self.assertIsNone(uncompressed.train_node_ids)
        self.assertIsNone(uncompressed.val_node_ids)
        self.assertIsNone(uncompressed.test_node_ids)

        # Cora's features are all 0/1, so every column is inferred as a lossless bit column.
        metadata = compressed.node_quantization_metadata
        assert isinstance(metadata, FeatureQuantizationMetadata)
        self.assertEqual(
            metadata.lossless_feature_indices, tuple(range(metadata.feature_dim))
        )
        self.assertEqual(
            metadata.lossless_column_encodings, ("bit",) * metadata.feature_dim
        )
        self.assertIsNone(compressed.node_features)
        self.assertIsNone(uncompressed.node_quantization_metadata)
        self.assertIsNotNone(uncompressed.node_features)

        edge_metadata = compressed.edge_quantization_metadata
        assert isinstance(edge_metadata, FeatureQuantizationMetadata)
        self.assertEqual(edge_metadata.feature_dim, 4)
        self.assertEqual(
            edge_metadata.lossless_feature_indices,
            tuple(range(edge_metadata.feature_dim)),
        )
        self.assertIsNone(compressed.edge_features)
        self.assertIsNone(uncompressed.edge_quantization_metadata)
        self.assertIsNotNone(uncompressed.edge_features)


if __name__ == "__main__":
    absltest.main()
