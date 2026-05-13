import tempfile
from pathlib import Path

from absl.testing import absltest

from gigl.src.common.constants.graph_metadata import DEFAULT_CONDENSED_NODE_TYPE
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.types.pb_wrappers.preprocessed_metadata import (
    PreprocessedMetadataPbWrapper,
)
from gigl.src.mocking.lib.versioning import get_mocked_dataset_artifact_metadata
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO,
)
from snapchat.research.gbml import preprocessed_metadata_pb2
from tests.test_assets.test_case import TestCase


class PreprocessedMetadataTest(TestCase):
    def test_feature_schema_keys_match_original_keys(self):
        """
        We currently observe a bug in the FeatureEmbeddingLayer which occurs if we sort the feature keys, leading
        to training failures that use DDP. This test ensures that we don't sort the feature keys as part of the
        preprocessed metadata pb wrapper, and that they are in the same order as the original feature keys.

        TODO (mkolodner-sc): Once the reason for why sorting the feature keys breaks training is understood and fixed, this test should be removed.
        """

        cora_dataset_info = get_mocked_dataset_artifact_metadata()[
            CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO.name
        ]

        gbml_config_pb_wrapper = (
            GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
                gbml_config_uri=cora_dataset_info.frozen_gbml_config_uri
            )
        )

        preprocessed_metadata_wrapper = (
            gbml_config_pb_wrapper.preprocessed_metadata_pb_wrapper
        )
        feature_schema = (
            preprocessed_metadata_wrapper.condensed_node_type_to_feature_schema_map[
                DEFAULT_CONDENSED_NODE_TYPE
            ]
        )

        # Get the original feature keys from the base proto
        feature_spec_keys = list(feature_schema.feature_spec.keys())
        feature_index_keys = list(feature_schema.feature_index.keys())
        feature_schema_keys = list(feature_schema.schema.keys())

        # The feature vocab keys are not required to match the feature keys, so we shouldn't check them.

        original_feature_keys = preprocessed_metadata_wrapper.preprocessed_metadata_pb.condensed_node_type_to_preprocessed_metadata[
            DEFAULT_CONDENSED_NODE_TYPE
        ].feature_keys

        # Assert that all the keys in the feature schema match the original feature keys from the base proto

        self.assertEqual(feature_spec_keys, original_feature_keys)
        self.assertEqual(feature_index_keys, original_feature_keys)
        self.assertEqual(feature_schema_keys, original_feature_keys)


class PreprocessedMetadataLocalUriBranchTest(TestCase):
    """Regression test for the ``LocalUri`` branch of
    ``PreprocessedMetadataPbWrapper.__get_feature_to_vocab_list_map``.

    The branch builds a ``functools.partial`` over
    ``LocalFsUtils.list_at_path`` and was passing the kwarg as ``entity``
    while the actual signature accepts ``file_system_entity``. ``partial``
    doesn't validate kwarg names at construction time, so the mismatch was
    only surfaced when ``list_files_fn(uri)`` was actually called -- which
    only happens when ``transform_fn_assets_uri`` is a ``LocalUri``. All
    production callers point at GCS, so the bug was latent until something
    exercised the local-path branch.

    This test exercises that branch end-to-end so the kwarg name stays
    correct.
    """

    def test_wrapper_instantiates_with_local_transform_fn_assets_uri(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            # An empty file is a valid (zero-feature) Schema textproto.
            schema_path = tmp / "schema.pbtxt"
            schema_path.write_text("")
            transform_fn_assets_dir = tmp / "transform_fn_assets"
            transform_fn_assets_dir.mkdir()

            pb = preprocessed_metadata_pb2.PreprocessedMetadata()
            node_entry = pb.condensed_node_type_to_preprocessed_metadata[0]
            node_entry.node_id_key = "node_id"
            node_entry.schema_uri = str(schema_path)
            node_entry.tfrecord_uri_prefix = "placeholder/not-used"
            node_entry.transform_fn_assets_uri = str(transform_fn_assets_dir)
            node_entry.enumerated_node_ids_bq_table = "placeholder.not_used"
            node_entry.enumerated_node_data_bq_table = "placeholder.not_used"

            edge_entry = pb.condensed_edge_type_to_preprocessed_metadata[0]
            edge_entry.src_node_id_key = "src"
            edge_entry.dst_node_id_key = "dst"
            edge_entry.main_edge_info.schema_uri = str(schema_path)
            edge_entry.main_edge_info.tfrecord_uri_prefix = "placeholder/not-used"
            edge_entry.main_edge_info.transform_fn_assets_uri = str(
                transform_fn_assets_dir
            )
            edge_entry.main_edge_info.enumerated_edge_data_bq_table = (
                "placeholder.not_used"
            )

            # Should not raise. Previously raised
            # ``TypeError: list_at_path() got an unexpected keyword argument 'entity'``
            # from ``__get_feature_to_vocab_list_map``.
            PreprocessedMetadataPbWrapper(preprocessed_metadata_pb=pb)


if __name__ == "__main__":
    absltest.main()
