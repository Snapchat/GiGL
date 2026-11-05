import tempfile
from typing import Optional

import torch
from absl.testing import absltest
from parameterized import param, parameterized

from gigl.common import LocalUri
from gigl.common.utils.proto_utils import ProtoUtils
from gigl.distributed.graph_store.storage_utils import build_storage_dataset
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.mocking.lib.versioning import get_mocked_dataset_artifact_metadata
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO,
)
from gigl.types.graph import FeatureQuantizationMetadata
from snapchat.research.gbml import gbml_config_pb2
from tests.test_assets.distributed.utils import create_test_process_group
from tests.test_assets.test_case import TestCase


class BuildStorageDatasetTest(TestCase):
    def setUp(self) -> None:
        # A process group on a free port lets the factory pick free dataset building ports.
        create_test_process_group()

    def tearDown(self) -> None:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    @parameterized.expand(
        [
            param(
                "default_on", trainer_value=None, inferencer_value=None, expected=True
            ),
            param(
                "explicitly_on",
                trainer_value="True",
                inferencer_value="True",
                expected=True,
            ),
            param(
                "trainer_opt_out",
                trainer_value="False",
                inferencer_value=None,
                expected=False,
            ),
            param(
                "inferencer_opt_out",
                trainer_value=None,
                inferencer_value="False",
                expected=False,
            ),
        ]
    )
    def test_compress_node_features_task_arg_decides_whether_features_are_packed(
        self,
        _,
        trainer_value: Optional[str],
        inferencer_value: Optional[str],
        expected: bool,
    ) -> None:
        original = GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
            get_mocked_dataset_artifact_metadata()[
                CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO.name
            ].frozen_gbml_config_uri
        )
        config = gbml_config_pb2.GbmlConfig()
        config.CopyFrom(original.gbml_config_pb)
        if trainer_value is not None:
            config.trainer_config.trainer_args["compress_node_features"] = trainer_value
        if inferencer_value is not None:
            config.inferencer_config.inferencer_args["compress_node_features"] = (
                inferencer_value
            )
        config_uri = LocalUri(
            f"{self.enterContext(tempfile.TemporaryDirectory())}/config.yaml"
        )
        ProtoUtils().write_proto_to_yaml(config, config_uri)

        dataset = build_storage_dataset(
            config_uri,
            sample_edge_direction="out",
            tf_record_uri_pattern=".*data.tfrecord$",
        )

        if expected:
            # Cora's features are all 0/1, so every column is packed as a bit.
            metadata = dataset.node_quantization_metadata
            assert isinstance(metadata, FeatureQuantizationMetadata)
            self.assertEqual(
                metadata.lossless_feature_indices, tuple(range(metadata.feature_dim))
            )
            self.assertIsNone(dataset.node_features)
        else:
            self.assertIsNone(dataset.node_quantization_metadata)
            self.assertIsNotNone(dataset.node_features)


if __name__ == "__main__":
    absltest.main()
