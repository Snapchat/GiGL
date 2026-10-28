from __future__ import annotations

from concurrent.futures import (
    ALL_COMPLETED,
    FIRST_EXCEPTION,
    ThreadPoolExecutor,
    wait,
)
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
from gigl.orchestration.kubeflow.kfp_pipeline import ExitTask, generate_pipeline
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
        exit_task: ExitTask = ExitTask.EMAIL,
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
            exit_task (ExitTask): What the pipeline runs when it exits. See :class:`ExitTask`.
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
                exit_task=exit_task,
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
        notification_emails: Optional[list[str]] = None,
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
            notification_emails (Optional[list[str]]): Emails to send notification to.
                See https://cloud.google.com/vertex-ai/docs/pipelines/email-notifications for more details.
        Returns:
            aiplatform.PipelineJob: The created pipeline job.
        """
        check_if_kfp_pipeline_job_name_valid(str(applied_task_identifier))

        file_loader = FileLoader()
        assert file_loader.does_uri_exist(compiled_pipeline_path), (
            f"Compiled pipeline path {compiled_pipeline_path} does not exist."
        )
        logger.info(f"Skipping pipeline compilation; will use {compiled_pipeline_path}")

        run_keyword_args = {
            "job_name": applied_task_identifier,
            "start_at": start_at,
            "template_or_frozen_config_uri": task_config_uri.uri,
            "resource_config_uri": resource_config_uri.uri,
        }
        # We need to provide *some* notification emails, other wise the cleanup component will fail.
        # Ideally, we'd be able to provide None and have it handle it, but for whatever reason
        # that's not supported atm. Passing in None gives the below error:
        # Notification email "recipients" parameter must specify at least one recipient.
        if notification_emails:
            run_keyword_args["notification_emails"] = notification_emails
        else:
            run_keyword_args["notification_emails"] = [
                get_resource_config(
                    resource_config_uri=resource_config_uri
                ).service_account_email
            ]
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

    def wait_for_completion(
        self,
        run: Union[
            str,
            aiplatform.PipelineJob,
            list[str],
            list[aiplatform.PipelineJob],
        ],
        fail_fast: bool = False,
    ) -> None:
        """
        Waits for the completion of one or more pipeline runs.

        Args:
            run (Union[aiplatform.PipelineJob, str]): The pipeline job or its resource name.
            fail_fast (bool): If True, the first failed run cancels every run that is still
                waiting, then this waits for Vertex AI to report those runs as cancelled,
                which takes a few minutes. If False, every run is awaited.
                Either way the error names each failed run.

        Raises:
            RuntimeError: If any run failed, or a cancel could not be requested. The message
                names every failed run and every failed cancel with its error. Runs this call
                cancelled on purpose are not counted as failures.
        """
        resource_names: list[str] = []
        if isinstance(run, str):
            resource_names = [run]
        elif isinstance(run, aiplatform.PipelineJob):
            resource_names = [run.resource_name]
        else:
            resource_names = [
                r.resource_name if isinstance(r, aiplatform.PipelineJob) else r
                for r in run
            ]

        logger.info(
            f"Waiting for {len(resource_names)} pipeline runs to complete: {resource_names}"
        )

        with ThreadPoolExecutor(max_workers=len(resource_names)) as executor:
            futures = {
                executor.submit(
                    VertexAIService.wait_for_run_completion, resource_name=resource_name
                ): resource_name
                for resource_name in resource_names
            }
            done, not_done = wait(
                futures,
                return_when=FIRST_EXCEPTION if fail_fast else ALL_COMPLETED,
            )
            cancelled_runs: set[str] = set()
            cancel_failures: dict[str, BaseException] = {}
            if not_done and any(future.exception() is not None for future in done):
                logger.warning(
                    f"A pipeline run failed; cancelling the {len(not_done)} still "
                    f"running: {[futures[f] for f in not_done]}"
                )
                for future in not_done:
                    resource_name = futures[future]
                    try:
                        # A run that ended on its own before the cancel landed keeps
                        # its own outcome; only runs this call cancelled are excused.
                        if VertexAIService.cancel_pipeline_run(resource_name):
                            cancelled_runs.add(resource_name)
                    except Exception as e:
                        logger.warning(f"Could not cancel {resource_name}: {e!r}")
                        cancel_failures[resource_name] = e
                # Every wait must return before the executor can shut down. The
                # cancelled waits return once Vertex AI reports the runs as
                # cancelled; a run whose cancel failed is awaited as before.
                done, _ = wait(futures, return_when=ALL_COMPLETED)
            failures = {
                futures[future]: future.exception()
                for future in done
                if future.exception() is not None
                and futures[future] not in cancelled_runs
            }
            for future in done:
                if future.exception() is None:
                    logger.info(
                        f"Pipeline run {futures[future]} completed successfully."
                    )

        if failures or cancel_failures:
            lines = [f"  {name}: {error}" for name, error in failures.items()]
            lines.extend(
                f"  cancel of {name} failed: {error}"
                for name, error in cancel_failures.items()
            )
            raise RuntimeError(
                f"{len(failures)} of {len(resource_names)} pipeline runs failed"
                + (
                    f" and {len(cancel_failures)} cancel(s) failed"
                    if cancel_failures
                    else ""
                )
                + ":\n"
                + "\n".join(lines)
            )
        logger.info(f"All {len(resource_names)} pipeline runs completed successfully.")
