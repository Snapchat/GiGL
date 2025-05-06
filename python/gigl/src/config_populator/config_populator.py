import argparse
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

import gigl.src.common.constants.bq as bq_constants
import gigl.src.common.constants.gcs as gcs_constants
from gigl.common import GcsUri, LocalUri, Uri, UriFactory
from gigl.common.logger import Logger
from gigl.common.metrics.decorators import flushes_metrics, profileit
from gigl.common.utils.proto_utils import ProtoUtils
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.constants.metrics import TIMER_CONFIG_POPULATOR_S
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.dataset_split import DatasetSplit
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.types.pb_wrappers.task_metadata import TaskMetadataPbWrapper
from gigl.src.common.types.task_metadata import TaskMetadataType
from gigl.src.common.utils.metrics_service_provider import (
    get_metrics_service_instance,
    initialize_metrics,
)
from snapchat.research.gbml import (
    dataset_metadata_pb2,
    flattened_graph_metadata_pb2,
    gbml_config_pb2,
    graph_schema_pb2,
    inference_metadata_pb2,
    postprocessed_metadata_pb2,
    trained_model_metadata_pb2,
)

logger = Logger()


class ConfigPopulator:
    """
    GiGL Component that takes in a template GbmlConfig and outputs a frozen GbmlConfig by populating all job related metadata paths in sharedConfig.
    """

    __template_gbml_config: gbml_config_pb2.GbmlConfig
    __applied_task_identifier: AppliedTaskIdentifier

    def __init__(self) -> None:
        self.__proto_utils = ProtoUtils()

    @property
    def template_gbml_config(self) -> gbml_config_pb2.GbmlConfig:
        if not self.__template_gbml_config:
            raise ValueError(f"template_gbml_config is not initialized before use.")
        return self.__template_gbml_config

    @property
    def task_metadata_pb_wrapper(self) -> TaskMetadataPbWrapper:
        return TaskMetadataPbWrapper(
            task_metadata_pb=self.template_gbml_config.task_metadata
        )

    @property
    def applied_task_identifier(self) -> AppliedTaskIdentifier:
        if not self.__applied_task_identifier:
            raise ValueError(f"applied_task_identifier is not initialized before use.")
        return self.__applied_task_identifier

    def __populate_flattened_graph_metadata_pb(
        self,
    ) -> flattened_graph_metadata_pb2.FlattenedGraphMetadata:
        flattened_graph_metadata_pb: flattened_graph_metadata_pb2.FlattenedGraphMetadata
        supervised_node_class_pb: Optional[
            flattened_graph_metadata_pb2.SupervisedNodeClassificationOutput
        ] = None
        node_anchor_pb: Optional[
            flattened_graph_metadata_pb2.NodeAnchorBasedLinkPredictionOutput
        ] = None
        supervised_link_based_pb: Optional[
            flattened_graph_metadata_pb2.SupervisedLinkBasedTaskOutput
        ] = None

        if (
            self.task_metadata_pb_wrapper.task_metadata_type
            == TaskMetadataType.NODE_BASED_TASK
        ):
            labeled_tfrecord_uri_prefix = gcs_constants.get_subgraph_sampler_supervised_node_classification_labeled_samples_prefix(
                applied_task_identifier=self.applied_task_identifier
            )
            unlabeled_tfrecord_uri_prefix = gcs_constants.get_subgraph_sampler_supervised_node_classification_unlabeled_samples_prefix(
                applied_task_identifier=self.applied_task_identifier
            )

            supervised_node_class_pb = (
                flattened_graph_metadata_pb2.SupervisedNodeClassificationOutput(
                    labeled_tfrecord_uri_prefix=labeled_tfrecord_uri_prefix.uri,
                    unlabeled_tfrecord_uri_prefix=unlabeled_tfrecord_uri_prefix.uri,
                )
            )
        elif (
            self.task_metadata_pb_wrapper.task_metadata_type
            == TaskMetadataType.LINK_BASED_TASK
        ):
            labeled_tfrecord_uri_prefix = gcs_constants.get_subgraph_sampler_supervised_link_based_task_labeled_samples_prefix(
                applied_task_identifier=self.applied_task_identifier
            )
            unlabeled_tfrecord_uri_prefix = gcs_constants.get_subgraph_sampler_supervised_link_based_task_unlabeled_samples_prefix(
                applied_task_identifier=self.applied_task_identifier
            )

            supervised_link_based_pb = (
                flattened_graph_metadata_pb2.SupervisedLinkBasedTaskOutput(
                    labeled_tfrecord_uri_prefix=labeled_tfrecord_uri_prefix.uri,
                    unlabeled_tfrecord_uri_prefix=unlabeled_tfrecord_uri_prefix.uri,
                )
            )
        elif (
            self.task_metadata_pb_wrapper.task_metadata_type
            == TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK
        ):
            tfrecord_uri_prefix = gcs_constants.get_subgraph_sampler_node_anchor_based_link_prediction_samples_prefix(
                applied_task_identifier=self.applied_task_identifier
            )

            # node types for which random negative samples are generated
            # All anchor node types and target node types are considered for random negative samples in Subgraph Sampler
            random_negative_node_types = (
                self.task_metadata_pb_wrapper.get_supervision_edge_node_types(
                    should_include_src_nodes=True,
                    should_include_dst_nodes=True,
                )
            )

            node_type_to_random_negative_tfrecord_uri_prefix: Dict[str, str] = {}
            for node_type in random_negative_node_types:
                node_type_to_random_negative_tfrecord_uri_prefix[
                    str(node_type)
                ] = gcs_constants.get_subgraph_sampler_node_anchor_based_link_prediction_random_negatives_samples_prefix(
                    applied_task_identifier=self.applied_task_identifier,
                    node_type=NodeType(node_type),
                ).uri

            node_anchor_pb = flattened_graph_metadata_pb2.NodeAnchorBasedLinkPredictionOutput(
                tfrecord_uri_prefix=tfrecord_uri_prefix.uri,
                node_type_to_random_negative_tfrecord_uri_prefix=node_type_to_random_negative_tfrecord_uri_prefix,
            )
        else:
            raise TypeError(
                f"Found un-supported training task type: {self.task_metadata_type}; it has to be one of {[option.value for option in TaskMetadataType]}"
            )

        flattened_graph_metadata_pb = (
            flattened_graph_metadata_pb2.FlattenedGraphMetadata(
                supervised_node_classification_output=supervised_node_class_pb,
                node_anchor_based_link_prediction_output=node_anchor_pb,
                supervised_link_based_task_output=supervised_link_based_pb,
            )
        )
        return flattened_graph_metadata_pb

    def __populate_dataset_metadata_pb(
        self,
    ) -> dataset_metadata_pb2.DatasetMetadata:
        dataset_metadata_pb: dataset_metadata_pb2.DatasetMetadata
        supervised_node_class_pb: Optional[
            dataset_metadata_pb2.SupervisedNodeClassificationDataset
        ] = None
        node_anchor_pb: Optional[
            dataset_metadata_pb2.NodeAnchorBasedLinkPredictionDataset
        ] = None
        supervised_link_based_pb: Optional[
            dataset_metadata_pb2.SupervisedLinkBasedTaskSplitDataset
        ] = None

        train_data_uri = gcs_constants.get_split_dataset_output_gcs_file_prefix(
            applied_task_identifier=self.applied_task_identifier,
            dataset_split=DatasetSplit.TRAIN,
        )
        val_data_uri = gcs_constants.get_split_dataset_output_gcs_file_prefix(
            applied_task_identifier=self.applied_task_identifier,
            dataset_split=DatasetSplit.VAL,
        )
        test_data_uri = gcs_constants.get_split_dataset_output_gcs_file_prefix(
            applied_task_identifier=self.applied_task_identifier,
            dataset_split=DatasetSplit.TEST,
        )

        if (
            self.task_metadata_pb_wrapper.task_metadata_type
            == TaskMetadataType.NODE_BASED_TASK
        ):
            supervised_node_class_pb = (
                dataset_metadata_pb2.SupervisedNodeClassificationDataset(
                    train_data_uri=train_data_uri.uri,
                    val_data_uri=val_data_uri.uri,
                    test_data_uri=test_data_uri.uri,
                )
            )
        elif (
            self.task_metadata_pb_wrapper.task_metadata_type
            == TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK
        ):
            main_train_data_tfrecord_uri_prefix = (
                gcs_constants.get_split_dataset_main_samples_gcs_file_prefix(
                    applied_task_identifier=self.applied_task_identifier,
                    dataset_split=DatasetSplit.TRAIN,
                ).uri
            )

            main_val_data_tfrecord_uri_prefix = (
                gcs_constants.get_split_dataset_main_samples_gcs_file_prefix(
                    applied_task_identifier=self.applied_task_identifier,
                    dataset_split=DatasetSplit.VAL,
                ).uri
            )

            main_test_data_tfrecord_uri_prefix = (
                gcs_constants.get_split_dataset_main_samples_gcs_file_prefix(
                    applied_task_identifier=self.applied_task_identifier,
                    dataset_split=DatasetSplit.TEST,
                ).uri
            )

            # node types for which random negative samples are generated
            # Only target node types are considered for random negative samples in Split Genenator
            random_negative_node_types = (
                self.task_metadata_pb_wrapper.get_supervision_edge_node_types(
                    should_include_src_nodes=False,
                    should_include_dst_nodes=True,
                )
            )

            train_node_type_to_random_negative_data_uri: Dict[str, str] = {}
            val_node_type_to_random_negative_data_uri: Dict[str, str] = {}
            test_node_type_to_random_negative_data_uri: Dict[str, str] = {}

            for node_type in random_negative_node_types:
                train_node_type_to_random_negative_data_uri[
                    str(node_type)
                ] = gcs_constants.get_split_dataset_random_negatives_gcs_file_prefix(
                    applied_task_identifier=self.applied_task_identifier,
                    node_type=NodeType(node_type),
                    dataset_split=DatasetSplit.TRAIN,
                ).uri

                val_node_type_to_random_negative_data_uri[
                    str(node_type)
                ] = gcs_constants.get_split_dataset_random_negatives_gcs_file_prefix(
                    applied_task_identifier=self.applied_task_identifier,
                    node_type=NodeType(node_type),
                    dataset_split=DatasetSplit.VAL,
                ).uri

                test_node_type_to_random_negative_data_uri[
                    str(node_type)
                ] = gcs_constants.get_split_dataset_random_negatives_gcs_file_prefix(
                    applied_task_identifier=self.applied_task_identifier,
                    node_type=NodeType(node_type),
                    dataset_split=DatasetSplit.TEST,
                ).uri

            node_anchor_pb = dataset_metadata_pb2.NodeAnchorBasedLinkPredictionDataset(
                train_main_data_uri=main_train_data_tfrecord_uri_prefix,
                val_main_data_uri=main_val_data_tfrecord_uri_prefix,
                test_main_data_uri=main_test_data_tfrecord_uri_prefix,
                train_node_type_to_random_negative_data_uri=train_node_type_to_random_negative_data_uri,
                val_node_type_to_random_negative_data_uri=val_node_type_to_random_negative_data_uri,
                test_node_type_to_random_negative_data_uri=test_node_type_to_random_negative_data_uri,
            )

        elif (
            self.task_metadata_pb_wrapper.task_metadata_type
            == TaskMetadataType.LINK_BASED_TASK
        ):
            supervised_link_based_pb = (
                dataset_metadata_pb2.SupervisedLinkBasedTaskSplitDataset(
                    train_data_uri=train_data_uri.uri,
                    val_data_uri=val_data_uri.uri,
                    test_data_uri=test_data_uri.uri,
                )
            )

        dataset_metadata_pb = dataset_metadata_pb2.DatasetMetadata(
            supervised_node_classification_dataset=supervised_node_class_pb,
            node_anchor_based_link_prediction_dataset=node_anchor_pb,
            supervised_link_based_task_dataset=supervised_link_based_pb,
        )

        return dataset_metadata_pb

    def __populate_trained_model_metadata_pb(
        self,
        template_trained_model_metadata_pb: trained_model_metadata_pb2.TrainedModelMetadata,
    ) -> trained_model_metadata_pb2.TrainedModelMetadata:
        logger.info(
            f"Provided template_trained_model_metadata_pb: {template_trained_model_metadata_pb}"
        )
        trained_model_uri = (
            gcs_constants.get_trained_model_gcs_path(
                applied_task_identifier=self.applied_task_identifier
            )
            if not template_trained_model_metadata_pb.trained_model_uri
            else Uri(template_trained_model_metadata_pb.trained_model_uri)
        )

        scripted_model_uri = gcs_constants.get_trained_scripted_model_gcs_path(
            applied_task_identifier=self.applied_task_identifier
        )

        eval_metrics_uri = gcs_constants.get_trained_model_eval_metrics_gcs_path(
            applied_task_identifier=self.applied_task_identifier
        )

        tensorboard_logs_uri = gcs_constants.get_tensorboard_logs_gcs_path(
            applied_task_identifier=self.applied_task_identifier
        )

        trained_model_metadata_pb = trained_model_metadata_pb2.TrainedModelMetadata(
            trained_model_uri=trained_model_uri.uri,
            scripted_model_uri=scripted_model_uri.uri,
            eval_metrics_uri=eval_metrics_uri.uri,
            tensorboard_logs_uri=tensorboard_logs_uri.uri,
        )
        return trained_model_metadata_pb

    def __populate_inference_metadata_pb(
        self,
    ) -> inference_metadata_pb2.InferenceMetadata:
        """
        Populates the embeddings path and predictions path per inferencer node type in InferenceMetadata.
        """
        node_type_to_inferencer_output_info_map: Dict[
            str, inference_metadata_pb2.InferenceOutput
        ] = {}
        inferencer_node_types = self.task_metadata_pb_wrapper.get_task_root_node_types()
        template_gbml_config_pb_wrapper = GbmlConfigPbWrapper(
            gbml_config_pb=self.template_gbml_config
        )
        for node_type in inferencer_node_types:
            embeddings_path = bq_constants.get_embeddings_table(
                applied_task_identifier=self.applied_task_identifier,
                node_type=node_type,
            )
            predictions_path: Optional[str] = None

            if (
                self.task_metadata_pb_wrapper.task_metadata_type
                == TaskMetadataType.NODE_BASED_TASK
                or template_gbml_config_pb_wrapper.should_populate_predictions_path
            ):
                # TODO: currently, we are overloading the predictions path to store extra embeddings.
                # consider extending InferenceOutput's definition for this purpose.
                predictions_path = bq_constants.get_predictions_table(
                    applied_task_identifier=self.applied_task_identifier,
                    node_type=node_type,
                )
            inference_output_pb = inference_metadata_pb2.InferenceOutput(
                embeddings_path=embeddings_path,
                predictions_path=predictions_path,
            )
            node_type_to_inferencer_output_info_map[
                str(node_type)
            ] = inference_output_pb
        inference_metadata_pb = inference_metadata_pb2.InferenceMetadata(
            node_type_to_inferencer_output_info_map=node_type_to_inferencer_output_info_map
        )
        return inference_metadata_pb

    def __populate_postprocessed_metadata_pb(
        self,
    ) -> postprocessed_metadata_pb2.PostProcessedMetadata:
        """
        Populates the post_processor_log_metrics_uri in PostProcessedMetadata.
        """
        post_processor_log_metrics_uri = (
            gcs_constants.get_post_processor_metrics_gcs_path(
                applied_task_identifier=self.applied_task_identifier
            )
        )
        post_processing_metadata_pb = postprocessed_metadata_pb2.PostProcessedMetadata(
            post_processor_log_metrics_uri=post_processor_log_metrics_uri.uri
        )
        return post_processing_metadata_pb

    def __populate_graph_metadata_pb_condensed_maps(
        self, graph_metadata_pb: graph_schema_pb2.GraphMetadata
    ) -> graph_schema_pb2.GraphMetadata:
        """
        Builds the condensed_edge_type_map and condensed_node_type_map if not present.
        :return:
        """
        if (
            graph_metadata_pb.condensed_edge_type_map
            and graph_metadata_pb.condensed_node_type_map
        ):
            return graph_metadata_pb

        logger.info(
            f"{graph_schema_pb2.GraphMetadata.__name__} instance missing one or both of condensed edge and node type maps.\n"
            f"Will augment the provided GraphMetadata and build maps for {graph_metadata_pb}"
        )

        # User defined condensed_edge_type_map but not condensed_node_type_map. Build the condensed_node_type_map.
        if not graph_metadata_pb.condensed_node_type_map:
            logger.info("Missing condensed_node_type_map; will build it.")
            for condensed_node_type_id, node_type in enumerate(
                graph_metadata_pb.node_types
            ):
                graph_metadata_pb.condensed_node_type_map[condensed_node_type_id] = str(
                    node_type
                )

        # User defined condensed_node_type_map but not condensed_edge_type_map. Build the condensed_edge_type_map.
        if not graph_metadata_pb.condensed_edge_type_map:
            logger.info("Missing condensed_edge_type_map; will build it.")
            for condensed_edge_type_id, edge_type in enumerate(
                graph_metadata_pb.edge_types
            ):
                edge_type_pb = graph_schema_pb2.EdgeType(
                    src_node_type=str(edge_type.src_node_type),
                    relation=str(edge_type.relation),
                    dst_node_type=str(edge_type.dst_node_type),
                )
                graph_metadata_pb.condensed_edge_type_map[
                    condensed_edge_type_id
                ].CopyFrom(edge_type_pb)

        return graph_metadata_pb

    def __validate_graph_metadata_pb_is_coherent(
        self, graph_metadata_pb: graph_schema_pb2.GraphMetadata
    ) -> None:
        """
        Validates that the GraphMetadata pb is coherent.  This checks a few things:
        1. That all node types are unique.
        2. That all edge types are unique.
        3. That each edge type has a src_node_type and dst_node_type that are in the node types.
        4. That each condensed node type corresponds to a valid node type.
        5. That each condensed edge type corresponds to a valid edge type.
        :return:
        """

        # Make sure that all node types are unique.
        node_type_counter = Counter(graph_metadata_pb.node_types)
        duplicate_node_types = [
            node_type for node_type, freq in node_type_counter.items() if freq > 1
        ]
        if duplicate_node_types:
            raise ValueError(
                f"Duplicate node types not allowed in graphMetadata; please revise: {duplicate_node_types}"
            )

        # Make sure that edge types are unique.
        edge_type_counter = Counter(
            [
                EdgeType(
                    src_node_type=NodeType(edge_type_pb.src_node_type),
                    relation=Relation(edge_type_pb.relation),
                    dst_node_type=NodeType(edge_type_pb.dst_node_type),
                )
                for edge_type_pb in graph_metadata_pb.edge_types
            ]
        )
        duplicate_edge_types = [
            edge_type for edge_type, freq in edge_type_counter.items() if freq > 1
        ]
        if duplicate_edge_types:
            raise ValueError(f"Duplicate edge types found: {duplicate_edge_types}")

        # Make sure that each edge type has a src_node_type and dst_node_type that are in the node types.
        valid_node_types = set(graph_metadata_pb.node_types)
        for edge_type_pb in graph_metadata_pb.edge_types:
            if edge_type_pb.src_node_type not in valid_node_types:
                raise ValueError(
                    f"Edge type {edge_type_pb} has a src_node_type that is not in the node types."
                )
            if edge_type_pb.dst_node_type not in valid_node_types:
                raise ValueError(
                    f"Edge type {edge_type_pb} has a dst_node_type that is not in the node types."
                )

        # Make sure that each condensed node type corresponds to a node type.
        for (
            condensed_node_type,
            node_type,
        ) in graph_metadata_pb.condensed_node_type_map.items():
            if node_type not in valid_node_types:
                raise ValueError(
                    f"Condensed node type {condensed_node_type} does not correspond to a valid node type."
                )

        # Make sure that each condensed edge type corresponds to an edge type.
        valid_edge_types = set(edge_type_counter.keys())
        for (
            condensed_edge_type,
            edge_type_pb,
        ) in graph_metadata_pb.condensed_edge_type_map.items():
            edge_type = EdgeType(
                src_node_type=NodeType(edge_type_pb.src_node_type),
                relation=Relation(edge_type_pb.relation),
                dst_node_type=NodeType(edge_type_pb.dst_node_type),
            )
            if edge_type not in valid_edge_types:
                raise ValueError(
                    f"Condensed edge type {condensed_edge_type} does not correspond to a valid edge type."
                )

    def _populate_frozen_gbml_config_pb(
        self,
        applied_task_identifier: AppliedTaskIdentifier,
        template_gbml_config_pb: gbml_config_pb2.GbmlConfig,
    ) -> gbml_config_pb2.GbmlConfig:
        """
        Populates relevant constant URIs in the GbmlConfig.
        :return:
        """

        # Populate module-level variables.
        self.__applied_task_identifier = applied_task_identifier
        self.__template_gbml_config = template_gbml_config_pb

        output_gbml_config_pb = gbml_config_pb2.GbmlConfig()
        output_gbml_config_pb.CopyFrom(self.template_gbml_config)

        # Populate GraphMetadata (after validating and adding condensed node and edge type maps)
        graph_metadata_pb = self.__populate_graph_metadata_pb_condensed_maps(
            graph_metadata_pb=self.__template_gbml_config.graph_metadata
        )
        self.__validate_graph_metadata_pb_is_coherent(
            graph_metadata_pb=graph_metadata_pb
        )
        output_gbml_config_pb.graph_metadata.CopyFrom(graph_metadata_pb)

        preprocessed_metadata_uri = (
            gcs_constants.get_preprocessed_metadata_proto_gcs_path(
                applied_task_identifier=self.applied_task_identifier
            )
        )

        flattened_graph_metadata_pb = self.__populate_flattened_graph_metadata_pb()
        dataset_metadata_pb = self.__populate_dataset_metadata_pb()
        trained_model_metadata_pb = self.__populate_trained_model_metadata_pb(
            template_trained_model_metadata_pb=template_gbml_config_pb.shared_config.trained_model_metadata
        )
        inference_metadata_pb = self.__populate_inference_metadata_pb()
        postprocessed_metadata_pb = self.__populate_postprocessed_metadata_pb()

        # Build SharedConfig from constants, and merge into the content of the template / input GbmlConfig.
        shared_config_pb = gbml_config_pb2.GbmlConfig.SharedConfig(
            preprocessed_metadata_uri=preprocessed_metadata_uri.uri,  # type: ignore
            flattened_graph_metadata=flattened_graph_metadata_pb,
            dataset_metadata=dataset_metadata_pb,
            trained_model_metadata=trained_model_metadata_pb,
            inference_metadata=inference_metadata_pb,
            postprocessed_metadata=postprocessed_metadata_pb,
        )

        output_gbml_config_pb.shared_config.MergeFrom(shared_config_pb)

        return output_gbml_config_pb

    def __write_frozen_gbml_config(
        self, frozen_gbml_config_pb: gbml_config_pb2.GbmlConfig
    ) -> GcsUri:
        frozen_gbml_config_output_uri: GcsUri = (
            gcs_constants.get_frozen_gbml_config_proto_gcs_path(
                applied_task_identifier=self.applied_task_identifier
            )
        )
        self.__proto_utils.write_proto_to_yaml(
            proto=frozen_gbml_config_pb, uri=frozen_gbml_config_output_uri
        )
        logger.info(
            f"{frozen_gbml_config_pb.__class__.__name__} written to {frozen_gbml_config_output_uri.uri}"
        )
        return frozen_gbml_config_output_uri

    def __run(
        self,
        applied_task_identifier: AppliedTaskIdentifier,
        task_config_uri: Uri,
    ) -> GcsUri:
        # Populate the input GbmlConfig with constants, and "freeze" it
        frozen_gbml_config_pb = self._populate_frozen_gbml_config_pb(
            applied_task_identifier=applied_task_identifier,
            template_gbml_config_pb=self.__proto_utils.read_proto_from_yaml(
                uri=task_config_uri, proto_cls=gbml_config_pb2.GbmlConfig
            ),
        )

        # Write the frozen config out
        frozen_gbml_config_uri = self.__write_frozen_gbml_config(
            frozen_gbml_config_pb=frozen_gbml_config_pb
        )
        return frozen_gbml_config_uri

    @flushes_metrics(get_metrics_service_instance_fn=get_metrics_service_instance)
    @profileit(
        metric_name=TIMER_CONFIG_POPULATOR_S,
        get_metrics_service_instance_fn=get_metrics_service_instance,
    )
    def run(
        self,
        applied_task_identifier: AppliedTaskIdentifier,
        task_config_uri: Uri,
        resource_config_uri: Uri,
    ) -> GcsUri:
        """
        Runs the ConfigPopulator; given an input GbmlConfig file, produces a frozen one.

        Args:
            applied_task_identifier (AppliedTaskIdentifier): The job name.
            task_config_uri (Uri): Template GbmlConfig URI.
            resource_config_uri: GiGL resource config Uri

        Returns:
            GcsUri: The URI of the frozen GbmlConfig.
        """
        initialize_metrics(
            task_config_uri=task_config_uri, service_name=applied_task_identifier
        )

        resource_config = get_resource_config(resource_config_uri=resource_config_uri)
        try:
            gbml_config_output_uri = self.__run(
                applied_task_identifier=applied_task_identifier,
                task_config_uri=task_config_uri,
            )
            return gbml_config_output_uri
        except Exception as e:
            logger.error(
                "ConfigPopulator failed due to a raised exception, which will follow"
            )
            logger.error(e)
            logger.error(traceback.format_exc())
            logger.info("Cleaning up ConfigPopulator environment...")
            sys.exit(f"System will now exit: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Program to generate embeddings from a GBML model"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        help="Unique identifier for the job name",
    )
    parser.add_argument(
        "--template_uri",
        type=str,
        help="Gbml config uri",
    )
    parser.add_argument(
        "--output_file_path_frozen_gbml_config_uri",
        type=str,
        help="File to store output frozen gbml config uri",
    )
    parser.add_argument(
        "--resource_config_uri",
        type=str,
        help="Runtime argument for resource and env specifications of each component",
    )

    args = parser.parse_args()

    ati = AppliedTaskIdentifier(args.job_name)
    template_uri = UriFactory.create_uri(args.template_uri)
    resource_config_uri = UriFactory.create_uri(args.resource_config_uri)

    if not args.job_name or not args.template_uri:
        raise RuntimeError("Missing command-line arguments")

    config_populator = ConfigPopulator()
    frozen_gbml_config_uri = config_populator.run(
        applied_task_identifier=ati,
        task_config_uri=template_uri,
        resource_config_uri=resource_config_uri,
    )

    # Write fozen_gbml_config_uri to file where it can be read by subsequent components
    output_file_path_frozen_gbml_config_uri: LocalUri = LocalUri(
        args.output_file_path_frozen_gbml_config_uri
    )
    Path(output_file_path_frozen_gbml_config_uri.uri).parent.mkdir(
        parents=True, exist_ok=True
    )
    with open(output_file_path_frozen_gbml_config_uri.uri, "w+") as f:
        f.write(frozen_gbml_config_uri.uri)
