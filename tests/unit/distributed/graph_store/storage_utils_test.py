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

# (trainer arg, inferencer arg, whether the entity is packed). The flag defaults to on and one opt-out wins.
_OPT_OUT_CASES = (
    ("default_on", None, None, True),
    ("explicitly_on", "True", "True", True),
    ("trainer_opt_out", "False", None, False),
    ("inferencer_opt_out", None, "False", False),
)
_COMPRESS_ARG_CASES: list[param] = []
for _entity in ("node", "edge"):
    for _case, _trainer_value, _inferencer_value, _expected in _OPT_OUT_CASES:
        _COMPRESS_ARG_CASES.append(
            param(
                f"{_entity}_{_case}",
                arg_name=f"compress_{_entity}_features",
                trainer_value=_trainer_value,
                inferencer_value=_inferencer_value,
                expected=_expected,
            )
        )


class BuildStorageDatasetTest(TestCase):
    def setUp(self) -> None:
        # A process group on a free port lets the factory pick free dataset building ports.
        create_test_process_group()

    def tearDown(self) -> None:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    @parameterized.expand(_COMPRESS_ARG_CASES)
    def test_compress_features_task_arg_decides_whether_features_are_packed(
        self,
        _,
        arg_name: str,
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
            config.trainer_config.trainer_args[arg_name] = trainer_value
        if inferencer_value is not None:
            config.inferencer_config.inferencer_args[arg_name] = inferencer_value
        config_uri = LocalUri(
            f"{self.enterContext(tempfile.TemporaryDirectory())}/config.yaml"
        )
        ProtoUtils().write_proto_to_yaml(config, config_uri)

        dataset = build_storage_dataset(
            config_uri,
            sample_edge_direction="out",
            tf_record_uri_pattern=".*data.tfrecord$",
        )

        # Each flag only decides its own entity, so the other one stays packed.
        if arg_name == "compress_node_features":
            decided = (dataset.node_quantization_metadata, dataset.node_features)
            other = dataset.edge_quantization_metadata
        else:
            decided = (dataset.edge_quantization_metadata, dataset.edge_features)
            other = dataset.node_quantization_metadata
        self.assertIsNotNone(other)
        metadata, raw_features = decided
        if expected:
            # Cora's features are small whole numbers, so every column is packed.
            assert isinstance(metadata, FeatureQuantizationMetadata)
            self.assertEqual(
                metadata.lossless_feature_indices, tuple(range(metadata.feature_dim))
            )
            self.assertIsNone(raw_features)
        else:
            self.assertIsNone(metadata)
            self.assertIsNotNone(raw_features)


if __name__ == "__main__":
    absltest.main()
