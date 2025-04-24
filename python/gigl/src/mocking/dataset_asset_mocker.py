from typing import Optional

import gigl.src.mocking.lib.constants as mocking_constants
from gigl.common import GcsUri
from gigl.common.logger import Logger
from gigl.common.utils.gcs import GcsUtils
from gigl.common.utils.proto_utils import ProtoUtils
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.src.common.types.pb_wrappers.graph_metadata import GraphMetadataPbWrapper
from gigl.src.common.types.task_metadata import TaskMetadataType
from gigl.src.common.utils.bq import BqUtils
from gigl.src.config_populator.config_populator import ConfigPopulator
from gigl.src.mocking.lib import (
    mock_input_for_data_preprocessor,
    mock_input_for_inference,
    mock_input_for_split_generator,
    mock_input_for_subgraph_sampler,
    mock_input_for_trainer,
    mock_output_for_inference,
)
from gigl.src.mocking.lib.mocked_dataset_resources import MockedDatasetInfo
from snapchat.research.gbml import gbml_config_pb2, graph_schema_pb2

logger = Logger()


class DatasetAssetMocker:
    """
    Enables functionality to mock the input / output assets of all components based on input graph data.
    Useful to (re-)generate assets which can be used for testing.
    """

    def __init__(self) -> None:
        self.__proto_utils = ProtoUtils()

    def _update_supervised_node_classification_config_paths(
        self,
        pb: gbml_config_pb2.GbmlConfig,
        root_node_type: Optional[NodeType],
    ):
        modeling_task_spec_path = (
            "gigl."
            "src."
            "common."
            "modeling_task_specs."
            "node_classification_modeling_task_spec."
            "NodeClassificationModelingTaskSpec"
        )

        assert (
            root_node_type in self._mocked_dataset_info.num_node_distinct_labels
        ), f"Need labels for node type {root_node_type} to mock for supervised tasks."
        kwargs = {
            "batch_size": "16",
            "out_dim": str(
                self._mocked_dataset_info.num_node_distinct_labels[root_node_type]
            ),
            "num_epochs": "1",
        }

        pb.trainer_config.trainer_cls_path = modeling_task_spec_path
        pb.trainer_config.trainer_args.update(kwargs)
        pb.inferencer_config.inferencer_cls_path = modeling_task_spec_path
        pb.inferencer_config.inferencer_args.update(kwargs)

        task_output = (
            pb.shared_config.flattened_graph_metadata.supervised_node_classification_output
        )
        task_output.labeled_tfrecord_uri_prefix = (
            mocking_constants.update_gcs_uri_with_test_assets_and_version(
                uri_str=task_output.labeled_tfrecord_uri_prefix, version=self._version
            )
        )
        task_output.unlabeled_tfrecord_uri_prefix = (
            mocking_constants.update_gcs_uri_with_test_assets_and_version(
                uri_str=task_output.unlabeled_tfrecord_uri_prefix, version=self._version
            )
        )

        task_dataset = (
            pb.shared_config.dataset_metadata.supervised_node_classification_dataset
        )
        task_dataset.train_data_uri = (
            mocking_constants.update_gcs_uri_with_test_assets_and_version(
                uri_str=task_dataset.train_data_uri, version=self._version
            )
        )
        task_dataset.val_data_uri = (
            mocking_constants.update_gcs_uri_with_test_assets_and_version(
                uri_str=task_dataset.val_data_uri, version=self._version
            )
        )
        task_dataset.test_data_uri = (
            mocking_constants.update_gcs_uri_with_test_assets_and_version(
                uri_str=task_dataset.test_data_uri, version=self._version
            )
        )

        node_type_to_inferencer_output_info_map = (
            pb.shared_config.inference_metadata.node_type_to_inferencer_output_info_map
        )
        for node_type in node_type_to_inferencer_output_info_map:
            node_type_to_inferencer_output_info_map[
                node_type
            ].predictions_path = mocking_constants.update_bq_table_with_test_assets_and_version(
                bq_table=node_type_to_inferencer_output_info_map[
                    node_type
                ].predictions_path,
                version=self._version,
            )
            node_type_to_inferencer_output_info_map[
                node_type
            ].embeddings_path = mocking_constants.update_bq_table_with_test_assets_and_version(
                bq_table=node_type_to_inferencer_output_info_map[
                    node_type
                ].embeddings_path,
                version=self._version,
            )

    def _update_node_anchor_based_link_prediction_config_paths(
        self,
        pb: gbml_config_pb2.GbmlConfig,
    ):
        modeling_task_spec_path = (
            "gigl."
            "src."
            "common."
            "modeling_task_specs."
            "node_anchor_based_link_prediction_modeling_task_spec."
            "NodeAnchorBasedLinkPredictionModelingTaskSpec"
        )
        kwargs = {
            "main_sample_batch_size": "4",
            "random_negative_sample_batch_size": "4",
            "random_negative_sample_batch_size_for_evaluation": "4",
            "num_val_batches": "4",
            "num_test_batches": "4",
            "val_every_num_batches": "4",
            "early_stop_patience": "1",
        }
        graph_metadata_pb_wrapper = GraphMetadataPbWrapper(
            graph_metadata_pb=pb.graph_metadata
        )
        if graph_metadata_pb_wrapper.is_heterogeneous:
            kwargs.update(
                {"gnn_model_class_path": "gigl.src.common.models.pyg.heterogeneous.HGT"}
            )

        pb.trainer_config.trainer_cls_path = modeling_task_spec_path
        pb.trainer_config.trainer_args.update(kwargs)
        pb.inferencer_config.inferencer_cls_path = modeling_task_spec_path
        pb.inferencer_config.inferencer_args.update(kwargs)

        task_output = (
            pb.shared_config.flattened_graph_metadata.node_anchor_based_link_prediction_output
        )
        task_output.tfrecord_uri_prefix = (
            mocking_constants.update_gcs_uri_with_test_assets_and_version(
                uri_str=task_output.tfrecord_uri_prefix, version=self._version
            )
        )
        for (
            node_type,
            random_negative_tfrecord_uri_prefix,
        ) in task_output.node_type_to_random_negative_tfrecord_uri_prefix.items():
            task_output.node_type_to_random_negative_tfrecord_uri_prefix[
                node_type
            ] = mocking_constants.update_gcs_uri_with_test_assets_and_version(
                uri_str=random_negative_tfrecord_uri_prefix, version=self._version
            )
        task_dataset = (
            pb.shared_config.dataset_metadata.node_anchor_based_link_prediction_dataset
        )
        task_dataset.train_main_data_uri = (
            mocking_constants.update_gcs_uri_with_test_assets_and_version(
                uri_str=task_dataset.train_main_data_uri, version=self._version
            )
        )
        task_dataset.test_main_data_uri = (
            mocking_constants.update_gcs_uri_with_test_assets_and_version(
                uri_str=task_dataset.test_main_data_uri, version=self._version
            )
        )
        task_dataset.val_main_data_uri = (
            mocking_constants.update_gcs_uri_with_test_assets_and_version(
                uri_str=task_dataset.val_main_data_uri, version=self._version
            )
        )
        for (
            node_type,
            random_negative_tfrecord_uri_prefix,
        ) in task_dataset.train_node_type_to_random_negative_data_uri.items():
            task_dataset.train_node_type_to_random_negative_data_uri[
                node_type
            ] = mocking_constants.update_gcs_uri_with_test_assets_and_version(
                uri_str=random_negative_tfrecord_uri_prefix, version=self._version
            )

        for (
            node_type,
            random_negative_tfrecord_uri_prefix,
        ) in task_dataset.val_node_type_to_random_negative_data_uri.items():
            task_dataset.val_node_type_to_random_negative_data_uri[
                node_type
            ] = mocking_constants.update_gcs_uri_with_test_assets_and_version(
                uri_str=random_negative_tfrecord_uri_prefix, version=self._version
            )

        for (
            node_type,
            random_negative_tfrecord_uri_prefix,
        ) in task_dataset.test_node_type_to_random_negative_data_uri.items():
            task_dataset.test_node_type_to_random_negative_data_uri[
                node_type
            ] = mocking_constants.update_gcs_uri_with_test_assets_and_version(
                uri_str=random_negative_tfrecord_uri_prefix, version=self._version
            )

        inference_metadata = pb.shared_config.inference_metadata
        for node_type in inference_metadata.node_type_to_inferencer_output_info_map:
            inference_metadata.node_type_to_inferencer_output_info_map[
                node_type
            ].embeddings_path = mocking_constants.update_bq_table_with_test_assets_and_version(
                bq_table=inference_metadata.node_type_to_inferencer_output_info_map[
                    node_type
                ].embeddings_path,
                version=self._version,
            )

    def _prepare_frozen_gbml_config_shared(
        self, task_metadata_pb: gbml_config_pb2.GbmlConfig.TaskMetadata
    ) -> gbml_config_pb2.GbmlConfig:
        applied_task_identifier = AppliedTaskIdentifier(self._mocked_dataset_info.name)
        graph_metadata_pb = (
            self._mocked_dataset_info.graph_metadata_pb_wrapper.graph_metadata_pb
        )
        template_gbml_config_pb = gbml_config_pb2.GbmlConfig(
            task_metadata=task_metadata_pb,
            graph_metadata=graph_metadata_pb,
        )

        config_populator = ConfigPopulator()
        frozen_gbml_config_pb = config_populator._populate_frozen_gbml_config_pb(
            applied_task_identifier=applied_task_identifier,
            template_gbml_config_pb=template_gbml_config_pb,
        )

        frozen_gbml_config_pb.shared_config.preprocessed_metadata_uri = (
            mocking_constants.update_gcs_uri_with_test_assets_and_version(
                uri_str=frozen_gbml_config_pb.shared_config.preprocessed_metadata_uri,
                version=self._version,
            )
        )
        trained_model_metadata = (
            frozen_gbml_config_pb.shared_config.trained_model_metadata
        )
        trained_model_metadata.trained_model_uri = (
            mocking_constants.update_gcs_uri_with_test_assets_and_version(
                uri_str=trained_model_metadata.trained_model_uri, version=self._version
            )
        )
        trained_model_metadata.scripted_model_uri = (
            mocking_constants.update_gcs_uri_with_test_assets_and_version(
                uri_str=trained_model_metadata.scripted_model_uri, version=self._version
            )
        )
        trained_model_metadata.eval_metrics_uri = (
            mocking_constants.update_gcs_uri_with_test_assets_and_version(
                uri_str=trained_model_metadata.eval_metrics_uri, version=self._version
            )
        )
        trained_model_metadata.tensorboard_logs_uri = (
            mocking_constants.update_gcs_uri_with_test_assets_and_version(
                uri_str=trained_model_metadata.tensorboard_logs_uri,
                version=self._version,
            )
        )

        return frozen_gbml_config_pb

    def _populate_and_write_frozen_gbml_config(
        self, frozen_gbml_config_pb: gbml_config_pb2.GbmlConfig
    ) -> None:
        self._frozen_gbml_config_pb = frozen_gbml_config_pb
        logger.info(self._frozen_gbml_config_pb)

        frozen_gbml_config_gcs_uri = (
            mocking_constants.get_example_task_frozen_gbml_config_gcs_path(
                task_name=self._mocked_dataset_info.name, version=self._version
            )
        )
        self.__proto_utils.write_proto_to_yaml(
            proto=self._frozen_gbml_config_pb, uri=frozen_gbml_config_gcs_uri
        )

    def _prepare_supervised_node_classification_frozen_gbml_config(
        self, sample_node_type: NodeType
    ):
        task_metadata_pb = gbml_config_pb2.GbmlConfig.TaskMetadata(
            node_based_task_metadata=gbml_config_pb2.GbmlConfig.TaskMetadata.NodeBasedTaskMetadata(
                supervision_node_types=[str(sample_node_type)]
            )
        )
        frozen_gbml_config_pb = self._prepare_frozen_gbml_config_shared(
            task_metadata_pb=task_metadata_pb
        )

        self._update_supervised_node_classification_config_paths(
            pb=frozen_gbml_config_pb, root_node_type=sample_node_type
        )
        self._populate_and_write_frozen_gbml_config(frozen_gbml_config_pb)

    def _prepare_node_anchor_based_link_prediction_frozen_gbml_config(
        self, sample_edge_type: EdgeType
    ):
        task_metadata_pb = gbml_config_pb2.GbmlConfig.TaskMetadata(
            node_anchor_based_link_prediction_task_metadata=gbml_config_pb2.GbmlConfig.TaskMetadata.NodeAnchorBasedLinkPredictionTaskMetadata(
                supervision_edge_types=[
                    graph_schema_pb2.EdgeType(
                        src_node_type=sample_edge_type.src_node_type,
                        relation=sample_edge_type.relation,
                        dst_node_type=sample_edge_type.dst_node_type,
                    )
                ]
            )
        )
        frozen_gbml_config_pb = self._prepare_frozen_gbml_config_shared(
            task_metadata_pb=task_metadata_pb
        )
        self._update_node_anchor_based_link_prediction_config_paths(
            pb=frozen_gbml_config_pb
        )
        self._populate_and_write_frozen_gbml_config(frozen_gbml_config_pb)

    def _mock_supervised_node_classification_assets(self):
        # Prepare GCS and BQ assets / environment.
        self._prepare_env()

        # Prepare frozen GbmlConfig.
        assert (
            self._mocked_dataset_info.sample_node_type is not None
        ), f"Need defined sample_node_type to mock for {TaskMetadataType.NODE_BASED_TASK} task."
        self._prepare_supervised_node_classification_frozen_gbml_config(
            sample_node_type=self._mocked_dataset_info.sample_node_type
        )

        # Upload assets to BQ
        mock_input_for_data_preprocessor.generate_bigquery_assets(
            mocked_dataset_info=self._mocked_dataset_info, version=self._version
        )

        # Mock SubgraphSampler inputs ("run Data Preprocessor")
        mock_input_for_subgraph_sampler.generate_preprocessed_tfrecord_data(
            mocked_dataset_info=self._mocked_dataset_info,
            version=self._version,
            gbml_config_pb=self._frozen_gbml_config_pb,
        )

        # Mock SplitGenerator inputs ("run Subgraph Sampler")
        hetero_data = mock_input_for_split_generator.build_and_write_supervised_node_classification_subgraph_samples_from_mocked_dataset_info(
            mocked_dataset_info=self._mocked_dataset_info,
            root_node_type=self._mocked_dataset_info.sample_node_type,
            gbml_config_pb=self._frozen_gbml_config_pb,
        )

        # Mock Trainer inputs ("run Split Generator")
        mock_input_for_trainer.split_and_write_supervised_node_classification_subgraph_samples_from_mocked_dataset_info(
            mocked_dataset_info=self._mocked_dataset_info,
            root_node_type=self._mocked_dataset_info.sample_node_type,
            gbml_config_pb=self._frozen_gbml_config_pb,
            hetero_data=hetero_data,
        )

        # Mock Inferencer inputs ("run Trainer")
        mock_input_for_inference.train_model(
            gbml_config_pb=self._frozen_gbml_config_pb,
        )

        # Mock Inferencer outputs ("run Inferencer")
        mock_output_for_inference.infer_model(
            gbml_config_pb=self._frozen_gbml_config_pb,
        )

    def _mock_node_anchor_based_link_prediction_assets(self):
        # Prepare GCS and BQ assets / environment.
        self._prepare_env()

        # Prepare frozen GbmlConfig.
        assert (
            self._mocked_dataset_info.sample_edge_type is not None
        ), f"Need defined sample_edge_type to mock for {TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK} task."

        self._prepare_node_anchor_based_link_prediction_frozen_gbml_config(
            sample_edge_type=self._mocked_dataset_info.sample_edge_type
        )

        # Upload assets to BQ
        mock_input_for_data_preprocessor.generate_bigquery_assets(
            mocked_dataset_info=self._mocked_dataset_info, version=self._version
        )

        # Mock SubgraphSampler inputs ("run Data Preprocessor")
        mock_input_for_subgraph_sampler.generate_preprocessed_tfrecord_data(
            mocked_dataset_info=self._mocked_dataset_info,
            version=self._version,
            gbml_config_pb=self._frozen_gbml_config_pb,
        )

        # Mock SplitGenerator inputs ("run Subgraph Sampler")
        hetero_data = mock_input_for_split_generator.build_and_write_node_anchor_link_prediction_subgraph_samples_from_mocked_dataset_info(
            mocked_dataset_info=self._mocked_dataset_info,
            sample_edge_type=self._mocked_dataset_info.sample_edge_type,
            gbml_config_pb=self._frozen_gbml_config_pb,
        )

        # Mock Trainer inputs ("run Split Generator")
        mock_input_for_trainer.split_and_write_node_anchor_link_prediction_subgraph_samples_from_mocked_dataset_info(
            mocked_dataset_info=self._mocked_dataset_info,
            sample_edge_type=self._mocked_dataset_info.sample_edge_type,
            gbml_config_pb=self._frozen_gbml_config_pb,
            hetero_data=hetero_data,
        )

        # Mock Inferencer inputs ("run Trainer")
        mock_input_for_inference.train_model(
            gbml_config_pb=self._frozen_gbml_config_pb,
        )

        # Mock Inferencer outputs ("run Inferencer")
        mock_output_for_inference.infer_model(
            gbml_config_pb=self._frozen_gbml_config_pb,
        )

    def _prepare_env(self):
        bq_utils = BqUtils()
        bq_utils.create_bq_dataset(
            dataset_id=mocking_constants.MOCK_DATA_BQ_DATASET_NAME, exists_ok=True
        )
        gcs_utils = GcsUtils()
        gcs_utils.delete_files_in_bucket_dir(
            gcs_path=mocking_constants.get_example_task_static_assets_gcs_dir(
                task_name=self._mocked_dataset_info.name, version=self._version
            )
        )

    def mock_assets(self, mocked_dataset_info: MockedDatasetInfo) -> GcsUri:
        self._mocked_dataset_info = mocked_dataset_info
        assert (
            mocked_dataset_info.version is not None
        ), "Need defined version to mock assets."
        self._version = mocked_dataset_info.version

        if mocked_dataset_info.task_metadata_type == TaskMetadataType.NODE_BASED_TASK:
            assert (
                mocked_dataset_info.sample_node_type is not None
            ), f"Need defined sample_node_type to mock for {TaskMetadataType.NODE_BASED_TASK} task."
            self._mock_supervised_node_classification_assets()
        elif (
            mocked_dataset_info.task_metadata_type
            == TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK
        ):
            assert (
                mocked_dataset_info.sample_edge_type is not None
            ), f"Need defined sample_edge_type to mock for {TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK} task."
            self._mock_node_anchor_based_link_prediction_assets()
        else:
            raise NotImplementedError

        frozen_gbml_config_uri = (
            mocking_constants.get_example_task_frozen_gbml_config_gcs_path(
                task_name=self._mocked_dataset_info.name, version=self._version
            )
        )
        return frozen_gbml_config_uri
