import platform
import tempfile
import unittest
from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_data_validation as tfdv
from tensorflow_transform.tf_metadata import schema_utils

import gigl.common.utils.local_fs as local_fs_utils
import gigl.src.common.constants.gcs as gcs_consts
import gigl.src.common.constants.local_fs as local_fs_constants
from gigl.common import GcsUri, LocalUri, Uri, UriFactory
from gigl.common.logger import Logger
from gigl.common.utils.gcs import GcsUtils
from gigl.common.utils.proto_utils import ProtoUtils
from gigl.src.common.constants.graph_metadata import DEFAULT_CONDENSED_NODE_TYPE
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.utils.time import current_formatted_datetime
from gigl.src.common.utils.timeout import timeout
from gigl.src.data_preprocessor.data_preprocessor import DataPreprocessor
from gigl.src.data_preprocessor.lib.ingest.reference import NodeDataReference
from gigl.src.data_preprocessor.lib.types import NodeDataPreprocessingSpec, TFTensorDict
from gigl.src.mocking.lib.versioning import get_mocked_dataset_artifact_metadata
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    CORA_NODE_CLASSIFICATION_MOCKED_DATASET_INFO,
)
from gigl.src.mocking.mocking_assets.passthrough_preprocessor_config_for_mocked_assets import (
    PassthroughPreprocessorConfigForMockedAssets,
)
from snapchat.research.gbml import gbml_config_pb2, preprocessed_metadata_pb2
from tests.test_assets.test_case import TestCase
from tests.test_assets.uri_constants import DEFAULT_TEST_RESOURCE_CONFIG_URI

logger = Logger()

DATA_PREPROCESSOR_PIPELINE_TIMEOUT_SECONDS = 1200

_PRETRAINED_NODE_TFT_MODEL_URI_ARG = "pretrained_node_tft_model_uri"
# `tft_beam.WriteTransformFn` writes the transformed schema at this path under the
# transform directory that `tft_beam.ReadTransformFn` reads back.
_TRANSFORMED_SCHEMA_SUFFIX = "/transformed_metadata/schema.pbtxt"

_PASSTHROUGH_CONFIG_CLS_PATH = (
    "gigl.src.mocking.mocking_assets.passthrough_preprocessor_config_for_mocked_assets."
    "PassthroughPreprocessorConfigForMockedAssets"
)
_PRETRAINED_CONFIG_CLS_PATH = (
    "tests.integration.pipeline.data_preprocessor.pretrained_transform_fn_test."
    "PretrainedNodeTransformFnPassthroughConfig"
)


def _preprocessing_fn_that_must_not_be_traced(inputs: TFTensorDict) -> TFTensorDict:
    raise AssertionError(
        "preprocessing_fn was traced, so the pretrained transform_fn was not used."
    )


class PretrainedNodeTransformFnPassthroughConfig(
    PassthroughPreprocessorConfigForMockedAssets
):
    """Passthrough config whose node specs load a transform_fn instead of analyzing one.

    Each node spec's `preprocessing_fn` raises if traced, so a run that analyzes the data
    instead of reading `pretrained_node_tft_model_uri` fails loudly.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._pretrained_node_tft_model_uri: Uri = UriFactory.create_uri(
            kwargs.pop(_PRETRAINED_NODE_TFT_MODEL_URI_ARG)
        )
        super().__init__(**kwargs)

    def get_nodes_preprocessing_spec(
        self,
    ) -> dict[NodeDataReference, NodeDataPreprocessingSpec]:
        return {
            node_data_ref: spec._replace(
                pretrained_tft_model_uri=self._pretrained_node_tft_model_uri,
                preprocessing_fn=_preprocessing_fn_that_must_not_be_traced,
            )
            for node_data_ref, spec in super().get_nodes_preprocessing_spec().items()
        }


@unittest.skipIf(
    platform.machine() == "arm64",
    "Skipping this test on M1 Mac. TFT is known to stall - need to investigate",
)
class PretrainedTransformFnTest(TestCase):
    """
    Runs the data preprocessor on Cora twice: once analyzing the node data to build a
    transform_fn, then again loading that transform_fn through `pretrained_tft_model_uri`.
    Both runs must produce the same transformed node schema and rows.
    """

    def setUp(self) -> None:
        self._gcs_utils = GcsUtils()
        self._proto_utils = ProtoUtils()
        self._applied_tasks_to_cleanup: list[AppliedTaskIdentifier] = []

    def _write_task_config(
        self,
        applied_task_identifier: AppliedTaskIdentifier,
        data_preprocessor_config_cls_path: str,
        extra_data_preprocessor_args: dict[str, str],
    ) -> LocalUri:
        mocked_dataset_name = CORA_NODE_CLASSIFICATION_MOCKED_DATASET_INFO.name
        artifact_metadata = get_mocked_dataset_artifact_metadata()[mocked_dataset_name]
        gbml_config_pb = self._proto_utils.read_proto_from_yaml(
            uri=artifact_metadata.frozen_gbml_config_uri,
            proto_cls=gbml_config_pb2.GbmlConfig,
        )
        data_preprocessor_config = (
            gbml_config_pb.dataset_config.data_preprocessor_config
        )
        data_preprocessor_config.data_preprocessor_config_cls_path = (
            data_preprocessor_config_cls_path
        )
        data_preprocessor_config.data_preprocessor_args["mocked_dataset_name"] = (
            mocked_dataset_name
        )
        data_preprocessor_config.data_preprocessor_args.update(
            extra_data_preprocessor_args
        )
        gbml_config_pb.shared_config.preprocessed_metadata_uri = GcsUri.join(
            gcs_consts.get_applied_task_temp_gcs_path(
                applied_task_identifier=applied_task_identifier
            ),
            mocked_dataset_name,
            "preprocessed_metadata.yaml",
        ).uri

        task_config_uri = LocalUri(tempfile.NamedTemporaryFile(delete=False).name)
        self._proto_utils.write_proto_to_yaml(proto=gbml_config_pb, uri=task_config_uri)
        return task_config_uri

    def _run_data_preprocessor(
        self,
        run_name: str,
        data_preprocessor_config_cls_path: str,
        extra_data_preprocessor_args: dict[str, str],
    ) -> preprocessed_metadata_pb2.PreprocessedMetadata.NodeMetadataOutput:
        applied_task_identifier = AppliedTaskIdentifier(
            f"pretrained_transform_fn_test_{run_name}_{current_formatted_datetime()}"
        )
        self._applied_tasks_to_cleanup.append(applied_task_identifier)
        task_config_uri = self._write_task_config(
            applied_task_identifier=applied_task_identifier,
            data_preprocessor_config_cls_path=data_preprocessor_config_cls_path,
            extra_data_preprocessor_args=extra_data_preprocessor_args,
        )

        @timeout(
            DATA_PREPROCESSOR_PIPELINE_TIMEOUT_SECONDS,
            error_message="Data Preprocessor pipeline timed out",
        )
        def run_with_timeout() -> Uri:
            return DataPreprocessor().run(
                applied_task_identifier=applied_task_identifier,
                task_config_uri=task_config_uri,
                resource_config_uri=DEFAULT_TEST_RESOURCE_CONFIG_URI,
            )

        preprocessed_metadata_pb = self._proto_utils.read_proto_from_yaml(
            uri=run_with_timeout(),
            proto_cls=preprocessed_metadata_pb2.PreprocessedMetadata,
        )
        node_metadata_outputs = (
            preprocessed_metadata_pb.condensed_node_type_to_preprocessed_metadata
        )
        self.assertIn(int(DEFAULT_CONDENSED_NODE_TYPE), node_metadata_outputs)
        return node_metadata_outputs[int(DEFAULT_CONDENSED_NODE_TYPE)]

    @staticmethod
    def _read_transformed_nodes(
        node_metadata_output: preprocessed_metadata_pb2.PreprocessedMetadata.NodeMetadataOutput,
    ) -> dict[str, np.ndarray]:
        schema = tfdv.load_schema_text(node_metadata_output.schema_uri)
        feature_spec = schema_utils.schema_as_feature_spec(schema).feature_spec
        tfrecord_files = tf.io.gfile.glob(
            f"{node_metadata_output.tfrecord_uri_prefix}*.tfrecord"
        )
        batches = list(
            tf.data.TFRecordDataset(tfrecord_files)
            .batch(4096)
            .map(lambda records: tf.io.parse_example(records, feature_spec))
            .as_numpy_iterator()
        )
        return {
            key: np.concatenate([batch[key] for batch in batches])
            for key in feature_spec
        }

    @staticmethod
    def _sorted_rows(columns: dict[str, np.ndarray], keys: list[str]) -> np.ndarray:
        # Node order in the output shards is not deterministic, and enumeration may map
        # the same node to a different id on each run, so rows are compared as a sorted
        # multiset of feature and label values.
        rows = np.stack([columns[key].astype(np.float64) for key in keys], axis=1)
        return rows[np.lexsort(rows.T[::-1])]

    def test_pretrained_transform_fn_reproduces_analyzed_output(self) -> None:
        analyzed = self._run_data_preprocessor(
            run_name="analyzed",
            data_preprocessor_config_cls_path=_PASSTHROUGH_CONFIG_CLS_PATH,
            extra_data_preprocessor_args={},
        )
        self.assertTrue(
            analyzed.schema_uri.endswith(_TRANSFORMED_SCHEMA_SUFFIX),
            analyzed.schema_uri,
        )
        transform_directory_uri = analyzed.schema_uri.removesuffix(
            _TRANSFORMED_SCHEMA_SUFFIX
        )

        pretrained = self._run_data_preprocessor(
            run_name="pretrained",
            data_preprocessor_config_cls_path=_PRETRAINED_CONFIG_CLS_PATH,
            extra_data_preprocessor_args={
                _PRETRAINED_NODE_TFT_MODEL_URI_ARG: transform_directory_uri
            },
        )

        self.assertNotEqual(pretrained.schema_uri, analyzed.schema_uri)
        self.assertEqual(
            tfdv.load_schema_text(pretrained.schema_uri),
            tfdv.load_schema_text(analyzed.schema_uri),
        )
        self.assertEqual(pretrained.node_id_key, analyzed.node_id_key)
        self.assertEqual(list(pretrained.feature_keys), list(analyzed.feature_keys))
        self.assertEqual(list(pretrained.label_keys), list(analyzed.label_keys))
        self.assertEqual(pretrained.feature_dim, analyzed.feature_dim)

        analyzed_nodes = self._read_transformed_nodes(analyzed)
        pretrained_nodes = self._read_transformed_nodes(pretrained)
        self.assertEqual(
            len(analyzed_nodes[analyzed.node_id_key]),
            CORA_NODE_CLASSIFICATION_MOCKED_DATASET_INFO.num_nodes[
                CORA_NODE_CLASSIFICATION_MOCKED_DATASET_INFO.default_node_type
            ],
        )
        np.testing.assert_array_equal(
            np.sort(pretrained_nodes[pretrained.node_id_key]),
            np.sort(analyzed_nodes[analyzed.node_id_key]),
        )
        value_keys = list(analyzed.feature_keys) + list(analyzed.label_keys)
        np.testing.assert_array_equal(
            self._sorted_rows(pretrained_nodes, value_keys),
            self._sorted_rows(analyzed_nodes, value_keys),
        )

    def tearDown(self) -> None:
        for applied_task_identifier in self._applied_tasks_to_cleanup:
            self._gcs_utils.delete_files_in_bucket_dir(
                gcs_path=gcs_consts.get_applied_task_temp_gcs_path(
                    applied_task_identifier=applied_task_identifier
                )
            )
            self._gcs_utils.delete_files_in_bucket_dir(
                gcs_path=gcs_consts.get_applied_task_perm_gcs_path(
                    applied_task_identifier=applied_task_identifier
                )
            )
            local_fs_utils.delete_local_directory(
                local_fs_constants.get_gbml_task_local_tmp_path(
                    applied_task_identifier=applied_task_identifier
                )
            )
        return super().tearDown()
