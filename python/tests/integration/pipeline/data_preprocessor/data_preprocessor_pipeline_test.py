import platform
import tempfile
import unittest
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf
import tensorflow_data_validation as tfdv
import tensorflow_transform as tft
import torch

import gigl.common.utils.local_fs as local_fs_utils
import gigl.src.common.constants.gcs as gcs_consts
import gigl.src.common.constants.local_fs as local_fs_constants
from gigl.common import GcsUri, LocalUri, Uri
from gigl.common.logger import Logger
from gigl.common.utils.gcs import GcsUtils
from gigl.common.utils.proto_utils import ProtoUtils
from gigl.src.common.constants.graph_metadata import (
    DEFAULT_CONDENSED_EDGE_TYPE,
    DEFAULT_CONDENSED_NODE_TYPE,
)
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.graph_data import (
    CondensedEdgeType,
    CondensedNodeType,
    EdgeType,
    EdgeUsageType,
    NodeType,
)
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.types.pb_wrappers.graph_metadata import GraphMetadataPbWrapper
from gigl.src.common.types.pb_wrappers.preprocessed_metadata import (
    PreprocessedMetadataPbWrapper,
)
from gigl.src.common.utils.bq import BqUtils
from gigl.src.common.utils.time import current_formatted_datetime
from gigl.src.common.utils.timeout import timeout
from gigl.src.data_preprocessor.data_preprocessor import DataPreprocessor
from gigl.src.mocking.lib.feature_handling import get_feature_field_name
from gigl.src.mocking.lib.mocked_dataset_resources import MockedDatasetInfo
from gigl.src.mocking.lib.versioning import get_mocked_dataset_artifact_metadata
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    CORA_NODE_ANCHOR_MOCKED_DATASET_INFO,
    CORA_NODE_CLASSIFICATION_MOCKED_DATASET_INFO,
    CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO,
)
from snapchat.research.gbml import (
    gbml_config_pb2,
    graph_schema_pb2,
    preprocessed_metadata_pb2,
)
from tests.test_assets.uri_constants import DEFAULT_TEST_RESOURCE_CONFIG_URI

logger = Logger()


DATA_PREPROCESSOR_PIPELINE_TIMEOUT_SECONDS = 1200


# TODO (svij-sc) Figure out how we can re-enable this.
@unittest.skipIf(
    platform.machine() == "arm64",
    "Skipping this test on M1 Mac. TFT is known to stall - need to investigate",
)
class DataPreprocessorPipelineTest(unittest.TestCase):
    """
    This test checks the completion of preprocess pipeline with the Planetoid Cora dataset.
    Test will error out if it takes too long, or output files missing.
    """

    def setUp(self) -> None:
        self.gcs_utils = GcsUtils()
        self.bq_utils = BqUtils()
        self.proto_utils = ProtoUtils()
        self.__gcs_dirs_to_cleanup: List[GcsUri] = []
        self.__applied_tasks_to_cleanup: List[AppliedTaskIdentifier] = []

    @staticmethod
    def __get_np_arrays_from_tfrecords(
        schema_path: GcsUri,
        tfrecord_files: List[str],
        # This is set larger than cora num nodes & edges so as to read the whole dataset in 1 go.
        max_batch_size=16384,
    ) -> Dict[str, np.ndarray]:
        schema = tfdv.load_schema_text(schema_path.uri)
        feature_spec = tft.tf_metadata.schema_utils.schema_as_feature_spec(
            schema
        ).feature_spec
        dataset = (
            tf.data.TFRecordDataset(tfrecord_files)
            .map(lambda record: tf.io.parse_example(record, feature_spec))
            .batch(max_batch_size)
            .as_numpy_iterator()
        )
        return next(dataset)

    def __generate_gbml_config_pb_for_mocked_dataset_using_passthrough_preprocessor(
        self,
        applied_task_identifier: AppliedTaskIdentifier,
        mocked_dataset_info: MockedDatasetInfo,
    ) -> LocalUri:
        """
        Overwrites the output paths for a mocked gbml config to ensure each test gets a unique write destination.
        """

        task_name = mocked_dataset_info.name
        artifact_metadata = get_mocked_dataset_artifact_metadata()[task_name]
        logger.info(f"Trying to generate gbml_config_pb for task: {task_name}")
        # Overwrite output preprocessed_metadata_path
        output_directory = GcsUri.join(
            gcs_consts.get_applied_task_temp_gcs_path(
                applied_task_identifier=applied_task_identifier
            ),
            task_name,
        )
        self.__applied_tasks_to_cleanup.append(applied_task_identifier)
        self.__gcs_dirs_to_cleanup.append(output_directory)

        gbml_config_pb = self.proto_utils.read_proto_from_yaml(
            uri=artifact_metadata.frozen_gbml_config_uri,
            proto_cls=gbml_config_pb2.GbmlConfig,
        )

        passthrough_preprocessor_config_cls_path = (
            "gigl."
            "src."
            "mocking."
            "mocking_assets."
            "passthrough_preprocessor_config_for_mocked_assets."
            "PassthroughPreprocessorConfigForMockedAssets"
        )

        # Inject Data Preprocessor Config cls path.
        gbml_config_pb.dataset_config.data_preprocessor_config.data_preprocessor_config_cls_path = (
            passthrough_preprocessor_config_cls_path
        )
        gbml_config_pb.dataset_config.data_preprocessor_config.data_preprocessor_args[
            "mocked_dataset_name"
        ] = task_name

        gbml_config_pb.shared_config.preprocessed_metadata_uri = GcsUri.join(
            output_directory, "preprocessed_metadata.yaml"
        ).uri

        logger.info(f"Will be using gbml_config_pb: {gbml_config_pb} to run the test.")

        # Write it out locally.
        f = tempfile.NamedTemporaryFile(delete=False)
        frozen_gbml_config_uri = LocalUri(f.name)
        self.proto_utils.write_proto_to_yaml(
            proto=gbml_config_pb, uri=frozen_gbml_config_uri
        )

        return frozen_gbml_config_uri

    def __assert_graph_metadata_reflects_mocked_dataset_info(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        mocked_dataset_info: MockedDatasetInfo,
    ):
        logger.info(
            f"Analyzing {graph_schema_pb2.GraphMetadata.__name__}: {gbml_config_pb_wrapper.graph_metadata} for correctness;"
            + f"Source mock dataset info: {mocked_dataset_info}"
        )

        condensed_node_type_to_node_type_map: Dict[
            CondensedNodeType, NodeType
        ] = (
            gbml_config_pb_wrapper.graph_metadata_pb_wrapper.condensed_node_type_to_node_type_map
        )
        condensed_edge_type_to_edge_type_map: Dict[
            CondensedEdgeType, EdgeType
        ] = (
            gbml_config_pb_wrapper.graph_metadata_pb_wrapper.condensed_edge_type_to_edge_type_map
        )

        self.assertEquals(
            len(condensed_node_type_to_node_type_map),
            len(mocked_dataset_info.node_types),
        )
        self.assertEquals(
            len(condensed_edge_type_to_edge_type_map),
            len(mocked_dataset_info.edge_types),
        )

        self.assertEquals(
            condensed_node_type_to_node_type_map[DEFAULT_CONDENSED_NODE_TYPE],
            mocked_dataset_info.default_node_type,
        )

        self.assertEquals(
            condensed_edge_type_to_edge_type_map[DEFAULT_CONDENSED_EDGE_TYPE].relation,
            mocked_dataset_info.default_edge_type.relation,
        )

    def __assert_node_metadata_output_reflects_mocked_dataset_info(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        mocked_dataset_info: MockedDatasetInfo,
    ):
        preprocessed_metadata_pb_wrapper: PreprocessedMetadataPbWrapper = (
            gbml_config_pb_wrapper.preprocessed_metadata_pb_wrapper
        )
        preprocessed_metadata_pb: preprocessed_metadata_pb2.PreprocessedMetadata = (
            preprocessed_metadata_pb_wrapper.preprocessed_metadata_pb
        )
        graph_metadata_pb_wrapper: GraphMetadataPbWrapper = (
            gbml_config_pb_wrapper.graph_metadata_pb_wrapper
        )
        _condensed_node_type: int
        node_metadata_output_pb: (
            preprocessed_metadata_pb2.PreprocessedMetadata.NodeMetadataOutput
        )
        for (
            _condensed_node_type,
            node_metadata_output_pb,
        ) in (
            preprocessed_metadata_pb.condensed_node_type_to_preprocessed_metadata.items()
        ):
            condensed_node_type = CondensedNodeType(_condensed_node_type)
            node_type: NodeType = (
                graph_metadata_pb_wrapper.condensed_node_type_to_node_type_map[
                    condensed_node_type
                ]
            )

            expected_node_feature_names: List[str] = [
                get_feature_field_name(n=n)
                for n in range(mocked_dataset_info.num_node_features[node_type])
            ]
            expected_node_feats_indexed_by_feat_name: Dict[str, tf.Tensor] = {}
            for column_tensor, node_feat_name in zip(
                mocked_dataset_info.node_feats[node_type].T, expected_node_feature_names
            ):
                expected_node_feats_indexed_by_feat_name[
                    node_feat_name
                ] = column_tensor.numpy()

            self.assertEqual(
                node_metadata_output_pb.node_id_key,
                mocked_dataset_info.node_id_column_name,
            )
            self.assertEqual(
                list(node_metadata_output_pb.feature_keys), expected_node_feature_names
            )
            self.assertEqual(
                node_metadata_output_pb.label_keys,
                (
                    [mocked_dataset_info.node_label_column_name]
                    if mocked_dataset_info.node_labels
                    else []
                ),
            )
            self.assertIsNotNone(
                node_metadata_output_pb.tfrecord_uri_prefix
            )  # Will load and check the actual data below.
            self.assertIsNotNone(
                node_metadata_output_pb.schema_uri
            )  # Will load and check the actual data below.
            self.assertIsNotNone(node_metadata_output_pb.transform_fn_assets_uri)
            self.assertIsNotNone(
                node_metadata_output_pb.enumerated_node_ids_bq_table
            )  # Should be tested in enumeration test.
            self.assertIsNotNone(
                node_metadata_output_pb.enumerated_node_data_bq_table
            )  # Should be tested in enumeration test.
            self.assertEqual(
                node_metadata_output_pb.feature_dim,
                mocked_dataset_info.num_node_features[node_type],
            )  # We use passthrough preprocessor, so the feature dim should be the same as mocked dataset.

            node_data_features_prefix = GcsUri(
                node_metadata_output_pb.tfrecord_uri_prefix
            )
            node_data_schema = GcsUri(node_metadata_output_pb.schema_uri)

            # Check node data is same.
            node_tfrecords = tf.io.gfile.glob(
                f"{node_data_features_prefix.uri}*.tfrecord"
            )
            self.assertIsNotNone(node_tfrecords)
            node_info: Dict[
                str, np.ndarray
            ] = DataPreprocessorPipelineTest.__get_np_arrays_from_tfrecords(
                schema_path=node_data_schema, tfrecord_files=node_tfrecords
            )

            self.assertEqual(
                node_info[node_metadata_output_pb.node_id_key].shape[0],
                mocked_dataset_info.num_nodes[node_type],
                f"The number of nodes does not match what is expected for node type: {node_type}.",
            )
            fields_to_check = (
                [  # Ensure all expected fields exist in preprocessed TfExample outputs
                    mocked_dataset_info.node_id_column_name,
                ]
            )
            if mocked_dataset_info.node_labels is not None:
                fields_to_check.append(mocked_dataset_info.node_label_column_name)
            for field in fields_to_check:
                self.assertIn(field, node_info)

            for (
                expected_feat_name,
                expected_feat_tensor,
            ) in expected_node_feats_indexed_by_feat_name.items():
                # We sum the feature tensors since the order of nodes in the read tfrecords is not guaranteed
                # We may in the future want to do more strict equality checks for features, but those
                # can be tested through unit tests, and as such an approximate check suffices.
                self.assertEqual(
                    np.sum(node_info[expected_feat_name]),
                    np.sum(expected_feat_tensor),
                    f"Node features do not match what is expected for feature: {expected_feat_name} "
                    f"for node type: {node_type}. Expected: {expected_feat_tensor}, "
                    + f"actual {node_info[expected_feat_name]}.",
                )

    def __assert_edge_metadata_info_reflects_mocked_dataset_info(
        self,
        edge_metadata_info_pb: preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataInfo,
        src_node_id_key: str,
        dst_node_id_key: str,
        expected_num_edge_features: int,
        expected_num_edges: int,
        expected_edge_feature_names: List[str],
        expected_edge_feats_indexed_by_feat_name: Dict[str, tf.Tensor],
    ):
        logger.info(
            f"Analyzing {preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataInfo.__name__}: "
            + f"{edge_metadata_info_pb} for correctness; Source mock info: "
            + f"src_node_id_key={src_node_id_key}, dst_node_id_key={dst_node_id_key}, "
            + f"expected_num_edge_features={expected_num_edge_features}, "
            + f"expected_num_edges={expected_num_edges}, "
            + f"expected_edge_feature_names={expected_edge_feature_names}, "
            + f"expected_edge_feats_indexed_by_feat_name={expected_edge_feats_indexed_by_feat_name}"
        )

        self.assertEqual(
            list(edge_metadata_info_pb.feature_keys), expected_edge_feature_names
        )
        # TODO: (svij-sc) Add check for label_keys, once it is supported by `MockedDatasetInfo`.
        self.assertIsNotNone(
            edge_metadata_info_pb.tfrecord_uri_prefix
        )  # Will load and check the actual data below.
        self.assertIsNotNone(
            edge_metadata_info_pb.schema_uri
        )  # Will load and check the actual data below.
        self.assertIsNotNone(edge_metadata_info_pb.transform_fn_assets_uri)
        self.assertIsNotNone(
            edge_metadata_info_pb.enumerated_edge_data_bq_table
        )  # Should be tested in enumeration test.
        self.assertEqual(
            edge_metadata_info_pb.feature_dim,
            expected_num_edge_features,
        )  # We use passthrough preprocessor, so the feature dim should be the same as mocked dataset.
        edge_data_prefix = GcsUri(edge_metadata_info_pb.tfrecord_uri_prefix)
        edge_data_schema = GcsUri(edge_metadata_info_pb.schema_uri)

        # Check node data is same.
        edge_tfrecords = tf.io.gfile.glob(f"{edge_data_prefix.uri}*.tfrecord")
        self.assertIsNotNone(edge_tfrecords)
        edge_info: Dict[
            str, np.ndarray
        ] = DataPreprocessorPipelineTest.__get_np_arrays_from_tfrecords(
            schema_path=edge_data_schema, tfrecord_files=edge_tfrecords
        )

        self.assertEqual(
            edge_info[src_node_id_key].shape[0],
            expected_num_edges,
            f"The number of source edges does not match what is expected",
        )
        self.assertEqual(
            edge_info[dst_node_id_key].shape[0],
            expected_num_edges,
            f"The number of dest edges does not match what is expected",
        )

        for (
            field
        ) in [  # Ensure all expected fields exist in preprocessed TfExample outputs
            src_node_id_key,
            dst_node_id_key,
        ]:
            self.assertIn(field, edge_info)

        for (
            expected_feat_name,
            expected_feat_tensor,
        ) in expected_edge_feats_indexed_by_feat_name.items():
            # We sum the feature tensors since the order of nodes in the read tfrecords is not guaranteed
            # We may in the future want to do more strict equality checks for features, but those
            # can be tested through unit tests, and as such an approximate check suffices.
            self.assertEqual(
                np.sum(edge_info[expected_feat_name]),
                np.sum(expected_feat_tensor),
                f"Edge features do not match what is expected for feature: {expected_feat_name} "
                f"Expected: {expected_feat_tensor}, actual {edge_info[expected_feat_name]}.",
            )

    def __assert_user_defined_edges_in_preprocessed_metadata_reflects_mocked_dataset_info(
        self,
        edge_type: EdgeType,
        edge_usage_type: EdgeUsageType,
        mocked_dataset_info: MockedDatasetInfo,
        edge_metadata_output_pb: preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataOutput,
    ):
        assert edge_usage_type in [
            EdgeUsageType.POSITIVE,
            EdgeUsageType.NEGATIVE,
        ], f"Invalid edge usage type: {edge_usage_type}; expected one of {EdgeUsageType.POSITIVE}, {EdgeUsageType.NEGATIVE}"
        logger.info(
            f"assert_user_defined_edges_in_preprocessed_metadata_reflects_mocked_dataset_info for {edge_usage_type} edges"
        )
        user_defined_edge_index: Optional[
            Dict[EdgeType, Dict[EdgeUsageType, torch.Tensor]]
        ] = mocked_dataset_info.user_defined_edge_index
        src_node_id_key = edge_metadata_output_pb.src_node_id_key
        dst_node_id_key = edge_metadata_output_pb.dst_node_id_key

        expected_num_edges = 0
        if (
            user_defined_edge_index is not None
            and edge_type in user_defined_edge_index
            and edge_usage_type in user_defined_edge_index[edge_type]
        ):
            expected_num_edges = user_defined_edge_index[edge_type][
                edge_usage_type
            ].shape[1]

        if expected_num_edges > 0:
            expected_num_user_def_edge_features_for_label_type = (
                mocked_dataset_info.num_user_def_edge_features[edge_type][
                    edge_usage_type
                ]
            )

            expected_edge_feature_names: List[str] = [
                get_feature_field_name(n=n)
                for n in range(expected_num_user_def_edge_features_for_label_type)
            ]
            expected_positive_edge_feats_indexed_by_feat_name: Dict[str, tf.Tensor] = {}
            if expected_num_user_def_edge_features_for_label_type > 0:
                assert mocked_dataset_info.user_defined_edge_feats is not None
                for column_tensor, edge_feat_name in zip(
                    mocked_dataset_info.user_defined_edge_feats[edge_type][
                        edge_usage_type
                    ].T,
                    expected_edge_feature_names,
                ):
                    expected_positive_edge_feats_indexed_by_feat_name[
                        edge_feat_name
                    ] = column_tensor.numpy()

            edge_metadata_info: (
                preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataInfo
            ) = (
                edge_metadata_output_pb.positive_edge_info
                if edge_usage_type == EdgeUsageType.POSITIVE
                else edge_metadata_output_pb.negative_edge_info
            )
            self.__assert_edge_metadata_info_reflects_mocked_dataset_info(
                edge_metadata_info_pb=edge_metadata_info,
                src_node_id_key=src_node_id_key,
                dst_node_id_key=dst_node_id_key,
                expected_num_edge_features=expected_num_user_def_edge_features_for_label_type,
                expected_num_edges=expected_num_edges,
                expected_edge_feature_names=expected_edge_feature_names,
                expected_edge_feats_indexed_by_feat_name=expected_positive_edge_feats_indexed_by_feat_name,
            )

    def __assert_edge_metadata_output_reflects_mocked_dataset_info(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        mocked_dataset_info: MockedDatasetInfo,
    ):
        preprocessed_metadata_pb_wrapper = (
            gbml_config_pb_wrapper.preprocessed_metadata_pb_wrapper
        )
        preprocessed_metadata_pb: preprocessed_metadata_pb2.PreprocessedMetadata = (
            preprocessed_metadata_pb_wrapper.preprocessed_metadata_pb
        )
        graph_metadata_pb_wrapper: GraphMetadataPbWrapper = (
            gbml_config_pb_wrapper.graph_metadata_pb_wrapper
        )
        _condensed_edge_type: int
        edge_metadata_output_pb: (
            preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataOutput
        )
        for (
            _condensed_edge_type,
            edge_metadata_output_pb,
        ) in (
            preprocessed_metadata_pb.condensed_edge_type_to_preprocessed_metadata.items()
        ):
            condensed_edge_type = CondensedEdgeType(_condensed_edge_type)
            edge_type: EdgeType = (
                graph_metadata_pb_wrapper.condensed_edge_type_to_edge_type_map[
                    condensed_edge_type
                ]
            )

            self.assertEqual(
                edge_metadata_output_pb.src_node_id_key,
                mocked_dataset_info.edge_src_column_name,
            )
            self.assertEqual(
                edge_metadata_output_pb.dst_node_id_key,
                mocked_dataset_info.edge_dst_column_name,
            )

            src_node_id_key = edge_metadata_output_pb.src_node_id_key
            dst_node_id_key = edge_metadata_output_pb.dst_node_id_key

            expected_main_edge_feature_names: List[str] = [
                get_feature_field_name(n=n)
                for n in range(mocked_dataset_info.num_edge_features[edge_type])
            ]
            expected_main_edge_feats_indexed_by_feat_name: Dict[str, tf.Tensor] = {}
            if (
                mocked_dataset_info.edge_feats is not None
                and len(mocked_dataset_info.edge_feats) > 0
            ):
                for column_tensor, edge_feat_name in zip(
                    mocked_dataset_info.edge_feats[edge_type].T,
                    expected_main_edge_feature_names,
                ):
                    expected_main_edge_feats_indexed_by_feat_name[
                        edge_feat_name
                    ] = column_tensor.numpy()

            expected_num_main_edge_features = mocked_dataset_info.num_edge_features[
                edge_type
            ]
            expected_num_main_edges = mocked_dataset_info.get_num_edges(
                edge_type=edge_type, edge_usage_type=EdgeUsageType.MAIN
            )
            self.__assert_edge_metadata_info_reflects_mocked_dataset_info(
                edge_metadata_info_pb=edge_metadata_output_pb.main_edge_info,
                src_node_id_key=src_node_id_key,
                dst_node_id_key=dst_node_id_key,
                expected_num_edge_features=expected_num_main_edge_features,
                expected_num_edges=expected_num_main_edges,
                expected_edge_feature_names=expected_main_edge_feature_names,
                expected_edge_feats_indexed_by_feat_name=expected_main_edge_feats_indexed_by_feat_name,
            )

            self.__assert_user_defined_edges_in_preprocessed_metadata_reflects_mocked_dataset_info(
                edge_type=edge_type,
                edge_usage_type=EdgeUsageType.NEGATIVE,
                mocked_dataset_info=mocked_dataset_info,
                edge_metadata_output_pb=edge_metadata_output_pb,
            )

            # Assert if positive edges are present, they are correctly reflected in the metadata.
            self.__assert_user_defined_edges_in_preprocessed_metadata_reflects_mocked_dataset_info(
                edge_type=edge_type,
                edge_usage_type=EdgeUsageType.POSITIVE,
                mocked_dataset_info=mocked_dataset_info,
                edge_metadata_output_pb=edge_metadata_output_pb,
            )

    def __assert_preprocessed_metadata_reflects_mocked_dataset_info(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        preprocessed_metadata_pb: preprocessed_metadata_pb2.PreprocessedMetadata,
        mocked_dataset_info: MockedDatasetInfo,
    ):
        self.__assert_graph_metadata_reflects_mocked_dataset_info(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            mocked_dataset_info=mocked_dataset_info,
        )
        logger.info(
            f"{preprocessed_metadata_pb2.PreprocessedMetadata.__name__} result: {preprocessed_metadata_pb}"
        )

        self.__assert_node_metadata_output_reflects_mocked_dataset_info(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            mocked_dataset_info=mocked_dataset_info,
        )
        self.__assert_edge_metadata_output_reflects_mocked_dataset_info(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            mocked_dataset_info=mocked_dataset_info,
        )

    def __run_data_preprocessor_pipeline(
        self, applied_task_identifier: AppliedTaskIdentifier, task_config_uri: Uri
    ) -> preprocessed_metadata_pb2.PreprocessedMetadata:
        # create inner function, so args can be passed into timeout decorator
        @timeout(
            DATA_PREPROCESSOR_PIPELINE_TIMEOUT_SECONDS,
            error_message="Data Preprocessor pipeline timed out",
        )
        def __run_data_preprocessor_pipeline_w_timeout() -> Uri:
            # Run data preprocessor
            data_preprocessor = DataPreprocessor()
            preprocessed_metadata_output_uri = data_preprocessor.run(
                applied_task_identifier=applied_task_identifier,
                task_config_uri=task_config_uri,
                resource_config_uri=DEFAULT_TEST_RESOURCE_CONFIG_URI,
            )
            return preprocessed_metadata_output_uri

        # call run with timeout
        preprocessed_metadata_output_uri = __run_data_preprocessor_pipeline_w_timeout()
        # Read the generated preprocessed metadata file and assert that all expected assets were
        # generated and accurate.
        preprocessed_metadata_pb: preprocessed_metadata_pb2.PreprocessedMetadata = (
            self.proto_utils.read_proto_from_yaml(
                uri=preprocessed_metadata_output_uri,
                proto_cls=preprocessed_metadata_pb2.PreprocessedMetadata,
            )
        )
        return preprocessed_metadata_pb

    def __run_test_for_mocked_dataset(self, mocked_dataset_info: MockedDatasetInfo):
        applied_task_id = AppliedTaskIdentifier(
            f"data_preprocessor_pipeline_test_{mocked_dataset_info.name}_{current_formatted_datetime()}"
        )
        frozen_gbml_config_uri: LocalUri = self.__generate_gbml_config_pb_for_mocked_dataset_using_passthrough_preprocessor(
            applied_task_identifier=applied_task_id,
            mocked_dataset_info=mocked_dataset_info,
        )
        preprocessed_metadata_pb: preprocessed_metadata_pb2.PreprocessedMetadata = (
            self.__run_data_preprocessor_pipeline(
                applied_task_identifier=applied_task_id,
                task_config_uri=frozen_gbml_config_uri,
            )
        )
        gbml_config_pb_wrapper = (
            GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
                gbml_config_uri=frozen_gbml_config_uri
            )
        )
        self.__assert_preprocessed_metadata_reflects_mocked_dataset_info(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            preprocessed_metadata_pb=preprocessed_metadata_pb,
            mocked_dataset_info=mocked_dataset_info,
        )

    def test_homogeneous_supervised_node_classification_preprocessor(self):
        mocked_dataset_info: MockedDatasetInfo = (
            CORA_NODE_CLASSIFICATION_MOCKED_DATASET_INFO
        )
        self.__run_test_for_mocked_dataset(mocked_dataset_info=mocked_dataset_info)

    def test_homogeneous_node_anchor_edge_features(self):
        mocked_dataset_info: MockedDatasetInfo = CORA_NODE_ANCHOR_MOCKED_DATASET_INFO
        self.__run_test_for_mocked_dataset(mocked_dataset_info=mocked_dataset_info)

    def test_homogeneous_node_link_pred_with_user_defined_labels_preprocessor(
        self,
    ):
        mocked_dataset_info: MockedDatasetInfo = (
            CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO
        )
        self.__run_test_for_mocked_dataset(mocked_dataset_info=mocked_dataset_info)

    def tearDown(self) -> None:
        logger.info(f"Cleaning up test assets. in : {self.__gcs_dirs_to_cleanup}")
        for gcs_dir in self.__gcs_dirs_to_cleanup:
            self.gcs_utils.delete_files_in_bucket_dir(gcs_path=gcs_dir)

        for applied_task_identifier in self.__applied_tasks_to_cleanup:
            logger.info(
                f"Cleaning up files for applied task: {local_fs_constants.get_gbml_task_local_tmp_path(applied_task_identifier=applied_task_identifier)}"
            )
            self.gcs_utils.delete_files_in_bucket_dir(
                gcs_path=gcs_consts.get_applied_task_temp_gcs_path(
                    applied_task_identifier=applied_task_identifier
                )
            )

            self.gcs_utils.delete_files_in_bucket_dir(
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
