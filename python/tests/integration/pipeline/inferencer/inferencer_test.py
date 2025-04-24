import tempfile
import unittest
from typing import Dict

import gigl.src.common.constants.gcs as gcs_constants
from gigl.common import LocalUri
from gigl.common.logger import Logger
from gigl.common.utils.gcs import GcsUtils
from gigl.common.utils.proto_utils import ProtoUtils
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.constants.bq import get_embeddings_table, get_predictions_table
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.graph_data import NodeType
from gigl.src.common.types.pb_wrappers.task_metadata import TaskMetadataPbWrapper
from gigl.src.common.types.task_metadata import TaskMetadataType
from gigl.src.common.utils.bq import BqUtils
from gigl.src.common.utils.time import current_formatted_datetime
from gigl.src.inference.v1.gnn_inferencer import InferencerV1
from gigl.src.mocking.lib.mocked_dataset_resources import MockedDatasetInfo
from gigl.src.mocking.lib.versioning import get_mocked_dataset_artifact_metadata
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    CORA_NODE_ANCHOR_MOCKED_DATASET_INFO,
    CORA_NODE_CLASSIFICATION_MOCKED_DATASET_INFO,
    DBLP_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
)
from snapchat.research.gbml import gbml_config_pb2
from snapchat.research.gbml.inference_metadata_pb2 import InferenceOutput

logger = Logger()


class InferencerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.__bq_utils = BqUtils()
        self.__gcs_utils = GcsUtils()
        self.__proto_utils = ProtoUtils()
        self.__inferencer = InferencerV1(bq_gcp_project=get_resource_config().project)

    def __clean_up_inferencer_test_assets(
        self,
        applied_task_identifier: AppliedTaskIdentifier,
        node_type_to_inferencer_output_map: Dict[str, InferenceOutput],
    ):
        self.__gcs_utils.delete_files_in_bucket_dir(
            gcs_path=gcs_constants.get_applied_task_temp_gcs_path(
                applied_task_identifier=applied_task_identifier
            )
        )

        self.__gcs_utils.delete_files_in_bucket_dir(
            gcs_path=gcs_constants.get_applied_task_perm_gcs_path(
                applied_task_identifier=applied_task_identifier
            )
        )
        for node_type in node_type_to_inferencer_output_map:
            if node_type_to_inferencer_output_map[node_type].embeddings_path:
                self.__bq_utils.delete_bq_table_if_exist(
                    bq_table_path=node_type_to_inferencer_output_map[
                        node_type
                    ].embeddings_path
                )
            if node_type_to_inferencer_output_map[node_type].predictions_path:
                self.__bq_utils.delete_bq_table_if_exist(
                    bq_table_path=node_type_to_inferencer_output_map[
                        node_type
                    ].predictions_path
                )

    def tearDown(self) -> None:
        pass

    def __populate_gbml_config_with_inference_paths(
        self,
        applied_task_identifier: AppliedTaskIdentifier,
        gbml_config_pb: gbml_config_pb2.GbmlConfig,
    ) -> LocalUri:
        task_metadata_pb_wrapper = TaskMetadataPbWrapper(
            task_metadata_pb=gbml_config_pb.task_metadata
        )
        node_type_to_inferencer_output_info_map = (
            gbml_config_pb.shared_config.inference_metadata.node_type_to_inferencer_output_info_map
        )
        if (
            task_metadata_pb_wrapper.task_metadata_type
            == TaskMetadataType.NODE_BASED_TASK
        ):
            node_types = list(
                task_metadata_pb_wrapper.task_metadata_pb.node_based_task_metadata.supervision_node_types
            )
            assert (
                len(node_types) == 1
            ), "Node classification only supports single node types for inference output"
            for node_type in node_types:
                node_type_to_inferencer_output_info_map[
                    node_type
                ].embeddings_path = get_embeddings_table(
                    applied_task_identifier=applied_task_identifier,
                    node_type=NodeType(node_type),
                )
                node_type_to_inferencer_output_info_map[
                    node_type
                ].predictions_path = get_predictions_table(
                    applied_task_identifier=applied_task_identifier,
                    node_type=NodeType(node_type),
                )

        elif (
            task_metadata_pb_wrapper.task_metadata_type
            == TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK
        ):
            for node_type in task_metadata_pb_wrapper.get_supervision_edge_node_types(
                should_include_src_nodes=True,
                should_include_dst_nodes=True,
            ):
                node_type_to_inferencer_output_info_map[
                    node_type
                ].embeddings_path = get_embeddings_table(
                    applied_task_identifier=applied_task_identifier,
                    node_type=NodeType(node_type),
                )
        else:
            raise ValueError(
                f"Expected one of {TaskMetadataType.NODE_BASED_TASK} or {TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK}, got {task_metadata_pb_wrapper.task_metadata_type}"
            )

        logger.info(f"Using proto for test: {gbml_config_pb}")

        # Write out the config.
        f = tempfile.NamedTemporaryFile(delete=False)
        gbml_config_local_uri = LocalUri(f.name)
        self.__proto_utils.write_proto_to_yaml(
            proto=gbml_config_pb, uri=gbml_config_local_uri
        )
        return gbml_config_local_uri

    def __generate_mocked_dataset_info_gbml_config_pb(
        self, mocked_dataset_info: MockedDatasetInfo
    ) -> gbml_config_pb2.GbmlConfig:
        task_name = mocked_dataset_info.name
        artifact_metadata = get_mocked_dataset_artifact_metadata()[task_name]
        gbml_config_pb = self.__proto_utils.read_proto_from_yaml(
            uri=artifact_metadata.frozen_gbml_config_uri,
            proto_cls=gbml_config_pb2.GbmlConfig,
        )
        return gbml_config_pb

    def __validate_inferencer_for_mocked_dataset(
        self, applied_task_id: AppliedTaskIdentifier, mocked_dataset: MockedDatasetInfo
    ):
        gbml_config_pb = self.__generate_mocked_dataset_info_gbml_config_pb(
            mocked_dataset_info=mocked_dataset
        )
        frozen_gbml_config_uri = self.__populate_gbml_config_with_inference_paths(
            applied_task_identifier=applied_task_id,
            gbml_config_pb=gbml_config_pb,
        )

        self.__inferencer.run(
            applied_task_identifier=applied_task_id,
            task_config_uri=frozen_gbml_config_uri,
        )
        node_type_to_inferencer_output_info_map = dict(
            gbml_config_pb.shared_config.inference_metadata.node_type_to_inferencer_output_info_map
        )

        task_metadata_pb_wrapper = TaskMetadataPbWrapper(
            task_metadata_pb=gbml_config_pb.task_metadata
        )
        if (
            task_metadata_pb_wrapper.task_metadata_type
            == TaskMetadataType.NODE_BASED_TASK
        ):
            mocked_dataset_inference_node_types = [
                NodeType(node_type)
                for node_type in list(
                    task_metadata_pb_wrapper.task_metadata_pb.node_based_task_metadata.supervision_node_types
                )
            ]
        elif (
            task_metadata_pb_wrapper.task_metadata_type
            == TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK
        ):
            mocked_dataset_inference_node_types = list(
                task_metadata_pb_wrapper.get_supervision_edge_node_types(
                    should_include_src_nodes=True,
                    should_include_dst_nodes=True,
                )
            )
        else:
            raise ValueError(
                f"Expected one of {TaskMetadataType.NODE_BASED_TASK} or {TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK}, got {task_metadata_pb_wrapper.task_metadata_type}"
            )

        for node_type in mocked_dataset_inference_node_types:
            assert (
                node_type in mocked_dataset.node_types
            ), f"Task metadata node type {node_type} not found in mocked dataset node types: {mocked_dataset.node_types}"
            should_assert_predictions = bool(
                node_type_to_inferencer_output_info_map[node_type].predictions_path
            )
            should_assert_embeddings = bool(
                node_type_to_inferencer_output_info_map[node_type].embeddings_path
            )
            if should_assert_embeddings:
                self.assertEquals(
                    self.__bq_utils.count_number_of_rows_in_bq_table(
                        bq_table=node_type_to_inferencer_output_info_map[
                            node_type
                        ].embeddings_path,
                        labels=get_resource_config().get_resource_labels(),
                    ),
                    mocked_dataset.num_nodes[node_type],
                    f"Found unexpected number of rows for node type {node_type} in embedding table.",
                )
            if should_assert_predictions:
                self.assertEquals(
                    self.__bq_utils.count_number_of_rows_in_bq_table(
                        bq_table=node_type_to_inferencer_output_info_map[
                            node_type
                        ].predictions_path,
                        labels=get_resource_config().get_resource_labels(),
                    ),
                    mocked_dataset.num_nodes[node_type],
                    f"Found unexpected number of rows for node type {node_type} in prediction table.",
                )
        self.__clean_up_inferencer_test_assets(
            applied_task_identifier=applied_task_id,
            node_type_to_inferencer_output_map=node_type_to_inferencer_output_info_map,
        )

    def test_supervised_node_classification_inference(self):
        applied_task_id = AppliedTaskIdentifier(
            f"inferencer_v2_test_supervised_node_classification_{(current_formatted_datetime())}"
        )
        mocked_dataset = CORA_NODE_CLASSIFICATION_MOCKED_DATASET_INFO
        self.__validate_inferencer_for_mocked_dataset(
            applied_task_id=applied_task_id, mocked_dataset=mocked_dataset
        )

    def test_homogeneous_node_anchor_based_link_prediction_inference(self):
        applied_task_id = AppliedTaskIdentifier(
            f"inferencer_test_homogeneous_node_anchor_based_link_prediction_{(current_formatted_datetime())}"
        )
        mocked_dataset = CORA_NODE_ANCHOR_MOCKED_DATASET_INFO
        self.__validate_inferencer_for_mocked_dataset(
            applied_task_id=applied_task_id, mocked_dataset=mocked_dataset
        )

    def test_heterogeneous_node_anchor_based_link_prediction_inference(self):
        applied_task_id = AppliedTaskIdentifier(
            f"inferencer_test_heterogeneous_node_anchor_based_link_prediction_{(current_formatted_datetime())}"
        )
        mocked_dataset = DBLP_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO
        self.__validate_inferencer_for_mocked_dataset(
            applied_task_id=applied_task_id, mocked_dataset=mocked_dataset
        )
