from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

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
            dst_compiled_pipeline_path (Uri): Destination path for where to store the compiled pipeline yaml.
            additional_job_args: Optional additional arguments to be passed into components, by component.
            tag: Optional tag, which is provided will be used to tag the pipeline description.
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
    ) -> aiplatform.PipelineJob:
        """
        Runs the GiGL Kubeflow pipeline.
        Args:
            applied_task_identifier (AppliedTaskIdentifier): Identifier for the task.
            task_config_uri (Uri): URI for the task config.
            resource_config_uri (Uri): URI for the resource config.
            start_at (str): Component to start at.
            stop_after (str): Component to stop after.
            compiled_pipeline_path (Uri): Path to the compiled pipeline.
                If compile is False, this should be provided and is directly used to run the pipeline and skip compilation.
                If compile is True, this flag is optional and if provided, is used as the destination path for where to
                store the compiled pipeline yaml.
            additional_job_args: Optional additional arguements to be passed into components, by component.

        Returns:
            aiplatform.PipelineJob: The job that was created.
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
        )
        return run

    def wait_for_completion(self, run: Union[aiplatform.PipelineJob, str]):
        resource_name = run if isinstance(run, str) else run.resource_name
        VertexAIService.wait_for_run_completion(resource_name)
        logger.info(f"Pipeline run {resource_name} completed successfully.")
