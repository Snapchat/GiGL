import tempfile

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
from snapchat.research.gbml import gbml_config_pb2
from tests.test_assets.test_case import TestCase


class BuildStorageDatasetTest(TestCase):
    @parameterized.expand(
        [
            param("node_trainer_args", "compress_node_features", is_inference=False),
            param("node_inferencer_args", "compress_node_features", is_inference=True),
            param("edge_trainer_args", "compress_edge_features", is_inference=False),
            param("edge_inferencer_args", "compress_edge_features", is_inference=True),
        ]
    )
    def test_compress_features_task_arg_is_rejected(
        self, _, arg_name: str, is_inference: bool
    ) -> None:
        original = GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
            get_mocked_dataset_artifact_metadata()[
                CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO.name
            ].frozen_gbml_config_uri
        )
        config = gbml_config_pb2.GbmlConfig()
        config.CopyFrom(original.gbml_config_pb)
        if is_inference:
            config.inferencer_config.inferencer_args[arg_name] = "True"
        else:
            config.trainer_config.trainer_args[arg_name] = "True"
        config_uri = LocalUri(
            f"{self.enterContext(tempfile.TemporaryDirectory())}/config.yaml"
        )
        ProtoUtils().write_proto_to_yaml(config, config_uri)

        with self.assertRaises(ValueError):
            build_storage_dataset(config_uri, sample_edge_direction="out")


if __name__ == "__main__":
    absltest.main()
