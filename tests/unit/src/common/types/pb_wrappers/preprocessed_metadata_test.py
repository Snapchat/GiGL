from absl.testing import absltest

from gigl.src.common.constants.graph_metadata import DEFAULT_CONDENSED_NODE_TYPE
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.mocking.lib.versioning import get_mocked_dataset_artifact_metadata
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO,
)
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


if __name__ == "__main__":
    absltest.main()
