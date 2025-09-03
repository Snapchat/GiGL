from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Union, overload

from google.cloud import aiplatform
from kfp.compiler import Compiler

import gigl.src.common.constants.local_fs as local_fs_constants
from gigl.common import LocalUri, Uri
from gigl.common.logger import Logger
from gigl.common.services.vertex_ai import VertexAIService
from gigl.common.types.resource_config import CommonPipelineComponentConfigs
from gigl.env.pipelines_config import get_resource_config
from gigl.orchestration.kubeflow.kfp_pipeline import generate_pipeline
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.utils.file_loader import FileLoader
from gigl.src.common.utils.time import current_formatted_datetime
from gigl.src.validation_check.libs.name_checks import (
    check_if_kfp_pipeline_job_name_valid,
)

logger = Logger()


DEFAULT_PIPELINE_VERSION_NAME = (
    f"gigl-pipeline-version-at-{current_formatted_datetime()}"
)

DEFAULT_KFP_COMPILED_PIPELINE_DEST_PATH = LocalUri.join(
    local_fs_constants.get_project_root_directory(),
    "build",
    f"gigl_pipeline_gnn.yaml",
)

DEFAULT_START_AT_COMPONENT = "config_populator"


class KfpOrchestrator:
    """
    Orchestration of Kubeflow Pipelines for GiGL.
    Methods:
        compile: Compiles the Kubeflow pipeline.
        run: Runs the Kubeflow pipeline.
        upload: Uploads the pipeline to KFP.
        wait_for_completion: Waits for the pipeline run to complete.
    """

    @classmethod
    def compile(
        cls,
        cuda_container_image: str,
        cpu_container_image: str,
        dataflow_container_image: str,
        dst_compiled_pipeline_path: Uri = DEFAULT_KFP_COMPILED_PIPELINE_DEST_PATH,
        additional_job_args: Optional[dict[GiGLComponents, dict[str, str]]] = None,
        tag: Optional[str] = None,
    ) -> Uri:
        """
        Compiles the GiGL Kubeflow pipeline.

        Args:
            cuda_container_image (str): Container image for CUDA (see: containers/Dockerfile.cuda).
            cpu_container_image (str): Container image for CPU.
            dataflow_container_image (str): Container image for Dataflow.
            dst_compiled_pipeline_path (Uri): Destination path for the compiled pipeline YAML file. Defaults to
            :data:`~gigl.constants.DEFAULT_KFP_COMPILED_PIPELINE_DEST_PATH`.
            additional_job_args (Optional[dict[GiGLComponents, dict[str, str]]]): Additional arguments to be passed into components, organized by component.
            tag (Optional[str]): Optional tag to include in the pipeline description.

        Returns:
            Uri: The URI of the compiled pipeline.
        """
        local_pipeline_bundle_path: LocalUri = (
            dst_compiled_pipeline_path
            if isinstance(dst_compiled_pipeline_path, LocalUri)
            else DEFAULT_KFP_COMPILED_PIPELINE_DEST_PATH
        )
        Path(local_pipeline_bundle_path.uri).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Compiling pipeline to {local_pipeline_bundle_path.uri}")

        common_pipeline_component_configs = CommonPipelineComponentConfigs(
            cuda_container_image=cuda_container_image,
            cpu_container_image=cpu_container_image,
            dataflow_container_image=dataflow_container_image,
            additional_job_args=additional_job_args or {},
        )

        Compiler().compile(
            generate_pipeline(
                common_pipeline_component_configs=common_pipeline_component_configs,
                tag=tag,
            ),
            local_pipeline_bundle_path.uri,
        )

        logger.info(f"Compiled Kubeflow pipeline to {local_pipeline_bundle_path.uri}")

        logger.info(f"Uploading compiled pipeline to {dst_compiled_pipeline_path.uri}")
        if local_pipeline_bundle_path != dst_compiled_pipeline_path:
            logger.info(f"Will upload pipeline to {dst_compiled_pipeline_path.uri}")
            file_loader = FileLoader()
            file_loader.load_file(
                file_uri_src=local_pipeline_bundle_path,
                file_uri_dst=dst_compiled_pipeline_path,
            )

        return dst_compiled_pipeline_path

    def run(
        self,
        applied_task_identifier: AppliedTaskIdentifier,
        task_config_uri: Uri,
        resource_config_uri: Uri,
        start_at: str = DEFAULT_START_AT_COMPONENT,
        stop_after: Optional[str] = None,
        compiled_pipeline_path: Uri = DEFAULT_KFP_COMPILED_PIPELINE_DEST_PATH,
        labels: Optional[dict[str, str]] = None,
    ) -> aiplatform.PipelineJob:
        """
        Runs the GiGL Kubeflow pipeline.

        Args:
            applied_task_identifier (AppliedTaskIdentifier): Identifier for the task.
            task_config_uri (Uri): URI of the task configuration file.
            resource_config_uri (Uri): URI of the resource configuration file.
            start_at (str): Component to start the pipeline at. Defaults to 'config_populator'.
            stop_after (Optional[str]): Component to stop the pipeline after. Defaults to None i.e. run entire pipeline.
            compiled_pipeline_path (Uri): Path to the compiled pipeline YAML file.
            labels (Optional[dict[str, str]]): Labels to associate with the run.
        Returns:
            aiplatform.PipelineJob: The created pipeline job.
        """
        check_if_kfp_pipeline_job_name_valid(str(applied_task_identifier))

        file_loader = FileLoader()
        assert file_loader.does_uri_exist(
            compiled_pipeline_path
        ), f"Compiled pipeline path {compiled_pipeline_path} does not exist."
        logger.info(f"Skipping pipeline compilation; will use {compiled_pipeline_path}")

        run_keyword_args = {
            "job_name": applied_task_identifier,
            "start_at": start_at,
            "template_or_frozen_config_uri": task_config_uri.uri,
            "resource_config_uri": resource_config_uri.uri,
        }
        if stop_after is not None:
            run_keyword_args["stop_after"] = stop_after

        logger.info(f"Running pipeline with args: {run_keyword_args}")
        resource_config = get_resource_config(resource_config_uri=resource_config_uri)
        vertex_ai_service = VertexAIService(
            project=resource_config.project,
            location=resource_config.region,
            service_account=resource_config.service_account_email,
            staging_bucket=resource_config.temp_assets_regional_bucket_path.uri,
        )
        run = vertex_ai_service.run_pipeline(
            display_name=str(applied_task_identifier),
            template_path=compiled_pipeline_path,
            run_keyword_args=run_keyword_args,
            job_id=str(applied_task_identifier).replace("_", "-"),
            labels=labels,
        )
        return run

    @overload
    def wait_for_completion(self, runs: list[aiplatform.PipelineJob]):
        ...

    @overload
    def wait_for_completion(self, run: list[str]):
        ...

    @overload
    def wait_for_completion(self, run: aiplatform.PipelineJob):
        ...

    @overload
    def wait_for_completion(self, run: str):
        ...

    def wait_for_completion(
        self,
        run: Union[
            aiplatform.PipelineJob, str, list[aiplatform.PipelineJob], list[str]
        ],
    ):
        """
        Waits for the completion of a pipeline run.

        Args:
            run (Union[aiplatform.PipelineJob, str]): The pipeline job or its resource name.

        Returns:
            None
        """
        resource_names: list[str]
        if isinstance(run, str):
            resource_names = [run]
        elif isinstance(run, aiplatform.PipelineJob):
            resource_names = [run.resource_name]
        else:
            resource_names = [
                run.resource_name if isinstance(run, aiplatform.PipelineJob) else run
                for run in run
            ]

        logger.info(
            f"Waiting for {len(resource_names)} pipeline runs to complete: {resource_names}"
        )

        def wait_for_run_completion(resource_name: str) -> str:
            VertexAIService.wait_for_run_completion(resource_name=resource_name)
            return resource_name  # Convenience to return the run name for logging

        with ThreadPoolExecutor(max_workers=len(resource_names)) as executor:
            futures = [
                executor.submit(wait_for_run_completion, resource_name=resource_name)
                for resource_name in resource_names
            ]
            for future in as_completed(futures):
                resource_name = future.result()
                logger.info(f"Pipeline run {resource_name} completed successfully.")

        logger.info(f"All {len(resource_names)} pipeline runs completed successfully.")
