# Note this class will get deprecated in the future without notice
# Use python/gigl/src/inference/inferencer.py instead

import argparse
import concurrent.futures
import sys
import threading
import traceback
from dataclasses import dataclass
from typing import Any, Optional

from apache_beam.runners.dataflow.dataflow_runner import DataflowPipelineResult
from apache_beam.runners.runner import PipelineState
from google.cloud import bigquery

from gigl.common import GcsUri, Uri, UriFactory
from gigl.common.env_config import get_available_cpus
from gigl.common.logger import Logger
from gigl.common.metrics.decorators import flushes_metrics, profileit
from gigl.common.utils import os_utils
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.constants.metrics import TIMER_INFERENCER_S
from gigl.src.common.graph_builder.graph_builder_factory import GraphBuilderFactory
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.graph_data import NodeType
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.utils.bq import BqUtils
from gigl.src.common.utils.metrics_service_provider import (
    get_metrics_service_instance,
    initialize_metrics,
)
from gigl.src.common.utils.model import load_state_dict_from_uri
from gigl.src.inference.lib.assets import InferenceAssets
from gigl.src.inference.v1.lib.base_inference_blueprint import BaseInferenceBlueprint
from gigl.src.inference.v1.lib.base_inferencer import BaseInferencer
from gigl.src.inference.v1.lib.inference_blueprint_factory import (
    InferenceBlueprintFactory,
)
from gigl.src.inference.v1.lib.utils import (
    get_inferencer_pipeline_component_for_single_node_type,
)
from snapchat.research.gbml.inference_metadata_pb2 import InferenceOutput

logger = Logger()
MAX_INFERENCER_NUM_WORKERS = 4


@dataclass
class InferencerOutputPaths:
    """
    Dataclass containing the output path fields from running inference for a single node type. These fields are used to write files from gcs to bigquery.
    """

    bq_inferencer_output_paths: InferenceOutput
    temp_predictions_gcs_path: Optional[GcsUri]
    temp_embedding_gcs_path: Optional[GcsUri]


class InferencerV1:
    """
    ********** WILL BE DEPRECATED **********
    Note this class will get deprecated in the future without notice
    Use python/gigl/src/inference/inferencer.py instead
    ********** WILL BE DEPRECATED **********
    GiGL Component that runs inference of a trained model on samples generated by the Subgraph Sampler component and outputs embedding and/or prediction assets.
    """

    __bq_utils: BqUtils
    __gbml_config_pb_wrapper: GbmlConfigPbWrapper

    @property
    def bq_utils(self) -> BqUtils:
        if not self.__bq_utils:
            raise ValueError(f"bq_utils is not initialized before use.")
        return self.__bq_utils

    @property
    def gbml_config_pb_wrapper(self) -> GbmlConfigPbWrapper:
        if not self.__gbml_config_pb_wrapper:
            raise ValueError(f"gbml_config_pb_wrapper is not initialized before use.")
        return self.__gbml_config_pb_wrapper

    def write_from_gcs_to_bq(
        self,
        schema: dict[str, list[dict[str, str]]],
        gcs_uri: GcsUri,
        bq_table_uri: str,
    ) -> None:
        """
        Writes embeddings or predictions from gcs folder to bq table with specified schema
        Args:
            schema (Optional[dict[str, list[dict[str, str]]]): BQ Table schema for embeddings or predictions from inference output
            gcs_uri (GcsUri): GCS Folder for embeddings or predictions from inference output
            bq_table_uri (str): Path to the table for embeddings or predictions output
        """
        assert schema is not None
        assert "fields" in schema
        field_schema = schema["fields"]
        logger.info(f"schema = {field_schema}")
        logger.info(f"loading from {gcs_uri} to BQ table: {bq_table_uri}")
        self.bq_utils.load_file_to_bq(
            source_path=GcsUri.join(gcs_uri, "*"),
            bq_path=bq_table_uri,
            job_config=bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
                schema=field_schema,
            ),
            retry=True,
        )
        logger.info(f"Finished loading to BQ table {bq_table_uri}")

    def generate_inferencer_instance(self) -> BaseInferencer:
        kwargs: dict[str, Any] = {}

        inferencer_class_path: str = (
            self.gbml_config_pb_wrapper.inferencer_config.inferencer_cls_path
        )
        kwargs = dict(self.gbml_config_pb_wrapper.inferencer_config.inferencer_args)
        inferencer_cls = os_utils.import_obj(inferencer_class_path)
        inferencer_instance: BaseInferencer
        try:
            inferencer_instance = inferencer_cls(**kwargs)
            assert isinstance(inferencer_instance, BaseInferencer)
        except Exception as e:
            logger.error(f"Could not instantiate class {inferencer_cls}: {e}")
            raise e

        model_save_path_uri = UriFactory.create_uri(
            self.gbml_config_pb_wrapper.shared_config.trained_model_metadata.trained_model_uri
        )
        logger.info(
            f"Loading model state dict from: {model_save_path_uri}, for inferencer: {inferencer_instance}"
        )
        model_state_dict = load_state_dict_from_uri(load_from_uri=model_save_path_uri)
        inferencer_instance.init_model(
            gbml_config_pb_wrapper=self.gbml_config_pb_wrapper,
            state_dict=model_state_dict,
        )
        return inferencer_instance

    def __infer_single_node_type(
        self,
        inference_blueprint: BaseInferenceBlueprint,
        applied_task_identifier: AppliedTaskIdentifier,
        custom_worker_image_uri: Optional[str],
        node_type: NodeType,
        uri_prefix_list: list[Uri],
        lock: threading.Lock,
    ) -> InferencerOutputPaths:
        """
        Runs inference on a single node type
        Args:
            inference_blueprint (BaseInferenceBlueprint): Blueprint for running and saving inference for GBML pipelines
            applied_task_identifier (AppliedTaskIdentifier): Identifier for the GiGL job
            custom_worker_image_uri (Optional[str]): Uri to custom worker image
            node_type (NodeType): Node type being inferred
            uri_prefix_list (list[Uri]): List of prefixes for running inference for given node type
            lock (threading.Lock): lock to prevent race conditions when starting dataflow pipelines
        Returns:
            (InferencerOutputPaths): Dataclass with path fields for writing from gcs to bigquery for given node type
        """
        node_type_to_inferencer_output_info_map = (
            self.gbml_config_pb_wrapper.shared_config.inference_metadata.node_type_to_inferencer_output_info_map
        )
        # Sanity check that we have some paths defined for intended inferred assets.
        if not (
            node_type_to_inferencer_output_info_map[node_type].embeddings_path
            or node_type_to_inferencer_output_info_map[node_type].predictions_path
        ):
            raise ValueError(
                f"Inference metadata for node type {node_type} is missing; must have at least one of "
                "embeddings_path or predictions_path defined."
            )
        should_persist_predictions = bool(
            node_type_to_inferencer_output_info_map[node_type].predictions_path
        )
        should_persist_embeddings = bool(
            node_type_to_inferencer_output_info_map[node_type].embeddings_path
        )
        temp_predictions_gcs_path: Optional[GcsUri]
        temp_embeddings_gcs_path: Optional[GcsUri]
        if should_persist_predictions:
            temp_predictions_gcs_path = InferenceAssets.get_gcs_asset_write_path_prefix(
                applied_task_identifier=applied_task_identifier,
                bq_table_path=node_type_to_inferencer_output_info_map[
                    node_type
                ].predictions_path,
            )
        else:
            temp_predictions_gcs_path = None

        if should_persist_embeddings:
            temp_embeddings_gcs_path = InferenceAssets.get_gcs_asset_write_path_prefix(
                applied_task_identifier=applied_task_identifier,
                bq_table_path=node_type_to_inferencer_output_info_map[
                    node_type
                ].embeddings_path,
            )
        else:
            temp_embeddings_gcs_path = None

        with lock:
            logger.debug(f"Node Type {node_type} acquiring lock.")
            p = get_inferencer_pipeline_component_for_single_node_type(
                gbml_config_pb_wrapper=self.gbml_config_pb_wrapper,
                inference_blueprint=inference_blueprint,
                applied_task_identifier=applied_task_identifier,
                custom_worker_image_uri=custom_worker_image_uri,
                node_type=node_type,
                uri_prefix_list=uri_prefix_list,
                temp_predictions_gcs_path=temp_predictions_gcs_path,
                temp_embeddings_gcs_path=temp_embeddings_gcs_path,
            )
            inferencer_pipeline_result = p.run()
            logger.debug(f"Node Type {node_type} releasing lock.")
        logger.info(f"Starting Dataflow job to run inference on {node_type} node type")
        inferencer_pipeline_result.wait_until_finish()
        logger.info(f"Finished Dataflow job to run inference on {node_type} node type")
        if isinstance(inferencer_pipeline_result, DataflowPipelineResult):
            pipeline_state: str = inferencer_pipeline_result.state
            if pipeline_state != PipelineState.DONE:
                raise RuntimeError(
                    f"A dataflow pipeline failed, has state {pipeline_state}: {inferencer_pipeline_result}"
                )
        return InferencerOutputPaths(
            bq_inferencer_output_paths=node_type_to_inferencer_output_info_map[
                node_type
            ],
            temp_predictions_gcs_path=temp_predictions_gcs_path,
            temp_embedding_gcs_path=temp_embeddings_gcs_path,
        )

    def __run(
        self,
        applied_task_identifier: AppliedTaskIdentifier,
        task_config_uri: Uri,
        custom_worker_image_uri: Optional[str] = None,
    ):
        self.__gbml_config_pb_wrapper = (
            GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
                gbml_config_uri=task_config_uri
            )
        )

        if self.gbml_config_pb_wrapper.shared_config.should_skip_inference:
            logger.info("Skipping inference as flag set in GbmlConfig")
            return

        inferencer_instance: BaseInferencer = self.generate_inferencer_instance()

        graph_builder = GraphBuilderFactory.get_graph_builder(
            backend_name=inferencer_instance.model.graph_backend  # type: ignore
        )

        inference_blueprint: BaseInferenceBlueprint = (
            InferenceBlueprintFactory.get_inference_blueprint(
                gbml_config_pb_wrapper=self.gbml_config_pb_wrapper,
                inferencer_instance=inferencer_instance,
                graph_builder=graph_builder,
            )
        )
        node_type_to_inferencer_output_paths_map: dict[
            NodeType, InferencerOutputPaths
        ] = dict()
        dataflow_setup_lock = threading.Lock()
        # We kick off multiple Inferencer pipelines, each of which kicks off a setup.py sdist run.
        # sdist has race-condition issues for simultaneous runs: https://github.com/pypa/setuptools/issues/1222
        # We have each thread take a lock when kicking off the pipelines to avoid this issue.
        num_workers = min(get_available_cpus(), MAX_INFERENCER_NUM_WORKERS)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            logger.info(f"Using up to {num_workers} threads.")
            futures: dict[
                concurrent.futures.Future[InferencerOutputPaths], NodeType
            ] = dict()
            for (
                node_type,
                uri_prefix_list,
            ) in (
                inference_blueprint.get_inference_data_tf_record_uri_prefixes().items()
            ):
                # Launching one beam pipeline per node type
                future = executor.submit(
                    self.__infer_single_node_type,
                    inference_blueprint=inference_blueprint,
                    applied_task_identifier=applied_task_identifier,
                    custom_worker_image_uri=custom_worker_image_uri,
                    node_type=node_type,
                    uri_prefix_list=uri_prefix_list,
                    lock=dataflow_setup_lock,
                )
                futures.update({future: node_type})
            for future in concurrent.futures.as_completed(futures):
                node_type = futures[future]
                try:
                    inferencer_output_paths: InferencerOutputPaths = future.result()
                    node_type_to_inferencer_output_paths_map[
                        node_type
                    ] = inferencer_output_paths
                except Exception as e:
                    logger.exception(
                        f"{node_type} inferencer job failed due to a raised exception: {e}"
                    )
                    raise e

        for (
            node_type,
            inferencer_output_paths,
        ) in node_type_to_inferencer_output_paths_map.items():
            condensed_node_type = self.__gbml_config_pb_wrapper.graph_metadata_pb_wrapper.node_type_to_condensed_node_type_map[
                node_type
            ]
            should_run_unenumeration = bool(
                self.__gbml_config_pb_wrapper.preprocessed_metadata_pb_wrapper.preprocessed_metadata_pb.condensed_node_type_to_preprocessed_metadata[
                    condensed_node_type
                ].enumerated_node_ids_bq_table
            )

            temp_predictions_gcs_path = (
                inferencer_output_paths.temp_predictions_gcs_path
            )
            temp_embeddings_gcs_path = inferencer_output_paths.temp_embedding_gcs_path
            bq_inferencer_output_paths = (
                inferencer_output_paths.bq_inferencer_output_paths
            )
            if temp_predictions_gcs_path is not None:
                self.write_from_gcs_to_bq(
                    schema=inference_blueprint.get_pred_table_schema(should_run_unenumeration=should_run_unenumeration).schema,  # type: ignore
                    gcs_uri=temp_predictions_gcs_path,
                    bq_table_uri=bq_inferencer_output_paths.predictions_path,
                )
            if temp_embeddings_gcs_path is not None:
                self.write_from_gcs_to_bq(
                    schema=inference_blueprint.get_emb_table_schema(should_run_unenumeration=should_run_unenumeration).schema,  # type: ignore
                    gcs_uri=temp_embeddings_gcs_path,
                    bq_table_uri=bq_inferencer_output_paths.embeddings_path,
                )

    @flushes_metrics(get_metrics_service_instance_fn=get_metrics_service_instance)
    @profileit(
        metric_name=TIMER_INFERENCER_S,
        get_metrics_service_instance_fn=get_metrics_service_instance,
    )
    def run(
        self,
        applied_task_identifier: AppliedTaskIdentifier,
        task_config_uri: Uri,
        custom_worker_image_uri: Optional[str] = None,
    ):
        try:
            return self.__run(
                applied_task_identifier=applied_task_identifier,
                task_config_uri=task_config_uri,
                custom_worker_image_uri=custom_worker_image_uri,
            )

        except Exception as e:
            logger.error(
                "Inference failed due to a raised exception; which will follow"
            )
            logger.error(e)
            logger.error(traceback.format_exc())
            sys.exit(f"System will now exit: {e}")

    def __init__(self, bq_gcp_project: str):
        self.__bq_utils = BqUtils(project=bq_gcp_project if bq_gcp_project else None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Program to run distributed inference")
    parser.add_argument(
        "--job_name",
        type=str,
        help="Unique identifier for the job name",
        required=True,
    )
    parser.add_argument(
        "--task_config_uri",
        type=str,
        help="Gbml config uri",
        required=True,
    )
    parser.add_argument(
        "--resource_config_uri",
        type=str,
        help="Runtime argument for resource and env specifications of each component",
        required=True,
    )
    parser.add_argument(
        "--custom_worker_image_uri",
        type=str,
        help="Docker image to use for the worker harness in dataflow",
        required=False,
    )

    parser.add_argument(
        "--cpu_docker_uri",
        type=str,
        help="User Specified or KFP compiled Docker Image for CPU inference",
        required=False,
    )
    parser.add_argument(
        "--cuda_docker_uri",
        type=str,
        help="User Specified or KFP compiled Docker Image for GPU inference",
        required=False,
    )
    args = parser.parse_args()

    task_config_uri = UriFactory.create_uri(args.task_config_uri)
    resource_config_uri = UriFactory.create_uri(args.resource_config_uri)
    custom_worker_image_uri = args.custom_worker_image_uri

    initialize_metrics(task_config_uri=task_config_uri, service_name=args.job_name)

    applied_task_identifier = AppliedTaskIdentifier(args.job_name)
    inferencer = InferencerV1(bq_gcp_project=get_resource_config().project)
    inferencer.run(
        applied_task_identifier=applied_task_identifier,
        task_config_uri=task_config_uri,
        custom_worker_image_uri=custom_worker_image_uri,
    )
