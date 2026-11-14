import os
from unittest import mock

import torch
from absl.testing import absltest

from gigl.common.utils.feature_quantization.lossless_compression import (
    FEATURE_COMPRESSION_ENV,
)
from gigl.distributed.graph_store.storage_utils import build_storage_dataset
from gigl.src.mocking.lib.versioning import get_mocked_dataset_artifact_metadata
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO,
)
from gigl.types.graph import FeatureQuantizationMetadata
from tests.test_assets.distributed.utils import create_test_process_group
from tests.test_assets.test_case import TestCase


class BuildStorageDatasetTest(TestCase):
    def setUp(self) -> None:
        # A process group on a free port lets the factory pick free dataset building ports.
        create_test_process_group()

    def tearDown(self) -> None:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def test_node_features_are_compressed_unless_feature_compression_is_off(
        self,
    ) -> None:
        task_config_uri = get_mocked_dataset_artifact_metadata()[
            CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO.name
        ].frozen_gbml_config_uri
        with mock.patch.dict(os.environ):
            os.environ.pop(FEATURE_COMPRESSION_ENV, None)
            compressed = build_storage_dataset(
                task_config_uri,
                sample_edge_direction="out",
                tf_record_uri_pattern=".*data.tfrecord$",
            )
        with mock.patch.dict(os.environ, {FEATURE_COMPRESSION_ENV: "off"}):
            uncompressed = build_storage_dataset(
                task_config_uri,
                sample_edge_direction="out",
                tf_record_uri_pattern=".*data.tfrecord$",
            )

        # Cora's features are all 0/1, so every column is packed as a bit.
        metadata = compressed.node_quantization_metadata
        assert isinstance(metadata, FeatureQuantizationMetadata)
        self.assertEqual(
            metadata.lossless_feature_indices, tuple(range(metadata.feature_dim))
        )
        self.assertIsNone(compressed.node_features)
        self.assertIsNone(uncompressed.node_quantization_metadata)
        self.assertIsNotNone(uncompressed.node_features)


if __name__ == "__main__":
    absltest.main()
