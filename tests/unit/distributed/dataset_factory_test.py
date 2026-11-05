import tempfile
from collections import abc

import torch
from absl.testing import absltest
from google.protobuf import text_format
from parameterized import param, parameterized

from gigl.common import LocalUri, Uri, UriFactory
from gigl.common.utils.proto_utils import ProtoUtils
from gigl.common.utils.tensorflow_schema import load_tf_schema_uri_str_to_feature_spec
from gigl.distributed.dataset_factory import build_dataset_from_task_config_uri
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.mocking.lib.versioning import get_mocked_dataset_artifact_metadata
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO,
)
from gigl.types.graph import (
    DEFAULT_HOMOGENEOUS_NODE_TYPE,
    FeatureQuantizationMetadata,
)
from snapchat.research.gbml import gbml_config_pb2, preprocessed_metadata_pb2
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

    @parameterized.expand(
        [
            param("training", is_inference=False),
            param("inference", is_inference=True),
        ]
    )
    def test_compress_node_features_arg_compresses_columns_declared_as_ids(
        self, _, is_inference: bool
    ) -> None:
        task_config_uri = self._task_config_uri_with_id_features(
            is_inference, "compress_node_features"
        )

        dataset = build_dataset_from_task_config_uri(
            task_config_uri,
            is_inference=is_inference,
            _tfrecord_uri_pattern=".*data.tfrecord$",
        )

        metadata = dataset.node_quantization_metadata
        assert isinstance(metadata, FeatureQuantizationMetadata)
        self.assertEqual(
            metadata.lossless_feature_indices, tuple(range(metadata.feature_dim))
        )
        self.assertIsNone(dataset.node_features)
        self.assertIsNone(dataset.edge_quantization_metadata)

    @parameterized.expand(
        [
            param("training", is_inference=False),
            param("inference", is_inference=True),
        ]
    )
    def test_compress_edge_features_arg_compresses_columns_declared_as_ids(
        self, _, is_inference: bool
    ) -> None:
        task_config_uri = self._task_config_uri_with_id_features(
            is_inference, "compress_edge_features"
        )

        dataset = build_dataset_from_task_config_uri(
            task_config_uri,
            is_inference=is_inference,
            _tfrecord_uri_pattern=".*data.tfrecord$",
        )

        metadata = dataset.edge_quantization_metadata
        assert isinstance(metadata, FeatureQuantizationMetadata)
        self.assertEqual(
            metadata.lossless_feature_indices, tuple(range(metadata.feature_dim))
        )
        self.assertIsNone(dataset.edge_features)
        self.assertEqual(metadata.feature_dim, 4)
        self.assertIsNone(dataset.node_quantization_metadata)

    def _task_config_uri_with_id_features(
        self, is_inference: bool, compress_arg: str
    ) -> Uri:
        """Copies the Cora task config, declaring the features as ids and setting ``compress_arg``.

        Node features are 0/1 and the four edge features are small counts, so both fit a narrow integer.
        """
        proto_utils = ProtoUtils()
        tmp_dir = self.enterContext(tempfile.TemporaryDirectory())
        original = GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
            get_mocked_dataset_artifact_metadata()[
                CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO.name
            ].frozen_gbml_config_uri
        )

        preprocessed = preprocessed_metadata_pb2.PreprocessedMetadata()
        preprocessed.CopyFrom(
            original.preprocessed_metadata_pb_wrapper.preprocessed_metadata
        )
        for (
            node_metadata
        ) in preprocessed.condensed_node_type_to_preprocessed_metadata.values():
            schema, _ = load_tf_schema_uri_str_to_feature_spec(
                UriFactory.create_uri(node_metadata.schema_uri)
            )
            for feature in schema.feature:
                feature.int_domain.min = 0
                feature.int_domain.max = 1
            node_metadata.schema_uri = f"{tmp_dir}/node_schema.pbtxt"
            with open(node_metadata.schema_uri, "w") as schema_file:
                schema_file.write(text_format.MessageToString(schema))
        for (
            edge_metadata
        ) in preprocessed.condensed_edge_type_to_preprocessed_metadata.values():
            schema, _ = load_tf_schema_uri_str_to_feature_spec(
                UriFactory.create_uri(edge_metadata.main_edge_info.schema_uri)
            )
            for feature in schema.feature:
                feature.int_domain.min = 0
                feature.int_domain.max = 4
            edge_metadata.main_edge_info.schema_uri = f"{tmp_dir}/edge_schema.pbtxt"
            with open(edge_metadata.main_edge_info.schema_uri, "w") as schema_file:
                schema_file.write(text_format.MessageToString(schema))
        preprocessed_uri = LocalUri(f"{tmp_dir}/preprocessed_metadata.yaml")
        proto_utils.write_proto_to_yaml(preprocessed, preprocessed_uri)

        config = gbml_config_pb2.GbmlConfig()
        config.CopyFrom(original.gbml_config_pb)
        config.shared_config.preprocessed_metadata_uri = preprocessed_uri.uri
        if is_inference:
            config.inferencer_config.inferencer_args[compress_arg] = "True"
        else:
            config.trainer_config.trainer_args[compress_arg] = "True"
        config_uri = LocalUri(f"{tmp_dir}/frozen_gbml_config.yaml")
        proto_utils.write_proto_to_yaml(config, config_uri)
        return config_uri


if __name__ == "__main__":
    absltest.main()
