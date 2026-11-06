"""Cancel the GCP jobs a GiGL pipeline run launched.

GiGL components launch Vertex AI CustomJobs, Dataflow jobs, and Dataproc clusters
from inside their own containers.
Vertex AI does not know about those children, so cancelling the pipeline run only
kills the component container.
This module finds the children by the names GiGL gives them and cancels them.

Usage, to cancel a pipeline run and its children:
    python -m gigl.src.common.job_cleanup \
        --pipeline_run=projects/my-project/locations/us-central1/pipelineJobs/my-run
"""

import argparse
from typing import Final

from google.cloud import aiplatform

from gigl.common import Uri, UriFactory
from gigl.common.logger import Logger
from gigl.common.services.dataflow import DataflowService
from gigl.common.services.dataproc import DataprocService
from gigl.common.services.vertex_ai import PipelineRunStoppedError, VertexAIService
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.utils.dataflow import get_sanitized_dataflow_job_name
from gigl.src.common.utils.spark_job_manager import (
    build_split_generator_cluster_name,
    build_subgraph_sampler_cluster_name,
)
from gigl.src.common.vertex_ai_launcher import build_vertex_ai_job_name

logger = Logger()

# A cancel takes a few minutes to reach CANCELLED. Wait longer than that, but do not
# use the 36 hour default of wait_for_run_completion.
_PIPELINE_CANCEL_TIMEOUT_S: Final[int] = 30 * 60


def cancel_child_jobs(job_name: str, resource_config_uri: Uri) -> None:
    """Cancel every Vertex AI CustomJob, Dataflow job, and Dataproc cluster of a GiGL job.

    Each kind and each child is attempted even if an earlier one raised, so one
    failing call does not leave the other children running.
    Finding nothing to cancel is not an error.

    Example:
        >>> cancel_child_jobs(
        ...     job_name="my_job",
        ...     resource_config_uri=GcsUri("gs://bucket/resource_config.yaml"),
        ... )

    Args:
        job_name: The GiGL applied task identifier the children were named after.
        resource_config_uri: Resource config that gives the project and region.

    Raises:
        RuntimeError: If any client raised. The message names each failed step.
    """
    resource_config = get_resource_config(resource_config_uri=resource_config_uri)
    project = resource_config.project
    region = resource_config.region
    failures: list[str] = []

    try:
        vertex_ai_service = VertexAIService(
            project=project,
            location=region,
            service_account=resource_config.service_account_email,
            staging_bucket=resource_config.temp_assets_regional_bucket_path.uri,
        )
        cancelled_jobs = vertex_ai_service.cancel_custom_jobs_with_name(
            [
                build_vertex_ai_job_name(job_name, GiGLComponents.Trainer),
                build_vertex_ai_job_name(job_name, GiGLComponents.Inferencer),
            ]
        )
        logger.info(f"Cancelled Vertex AI custom jobs: {cancelled_jobs or 'none'}")
    except Exception as e:
        failures.append(f"custom jobs: {e!r}")

    try:
        dataflow_prefix = get_sanitized_dataflow_job_name(f"gigl-{job_name}-")
        cancelled_dataflow_jobs = DataflowService(
            project=project, region=region
        ).cancel_jobs_with_name_prefix(dataflow_prefix)
        logger.info(f"Cancelled Dataflow jobs: {cancelled_dataflow_jobs or 'none'}")
    except Exception as e:
        failures.append(f"Dataflow jobs: {e!r}")

    try:
        dataproc_service = DataprocService(project_id=project, region=region)
        cluster_names = [
            build_subgraph_sampler_cluster_name(job_name),
            build_split_generator_cluster_name(job_name),
        ]
        deleted_clusters: list[str] = []
        for cluster_name in cluster_names:
            try:
                if dataproc_service.delete_cluster_if_exists(cluster_name):
                    deleted_clusters.append(cluster_name)
            except Exception as e:
                failures.append(f"Dataproc cluster {cluster_name}: {e!r}")
        logger.info(f"Deleted Dataproc clusters: {deleted_clusters or 'none'}")
    except Exception as e:
        failures.append(f"Dataproc clusters: {e!r}")

    if failures:
        raise RuntimeError(
            f"Cleanup for job {job_name} failed for: " + "; ".join(failures)
        )


def cancel_pipeline_run_and_child_jobs(pipeline_run: str) -> None:
    """Cancel a GiGL pipeline run, wait for it to stop, then cancel its child jobs.

    The job name and the resource config come from the run's ``job_name`` and
    ``resource_config_uri`` parameters.

    The wait makes sure no component launches a new child after the cleanup.

    A run that already stopped is not cancelled again, but its children are still
    cleaned up.

    Example:
        >>> cancel_pipeline_run_and_child_jobs(
        ...     "projects/p/locations/us-central1/pipelineJobs/my-run"
        ... )

    Args:
        pipeline_run: Full resource name of the pipeline run.

    Raises:
        RuntimeError: If the run does not stop in time, or a child cleanup failed.
    """
    run = aiplatform.PipelineJob.get(pipeline_run)
    parameters = run.gca_resource.runtime_config.parameter_values
    job_name = parameters["job_name"]
    resource_config_uri = UriFactory.create_uri(parameters["resource_config_uri"])

    VertexAIService.cancel_pipeline_run(pipeline_run)
    try:
        VertexAIService.wait_for_run_completion(
            resource_name=pipeline_run, timeout=_PIPELINE_CANCEL_TIMEOUT_S
        )
    except PipelineRunStoppedError as e:
        # A cancelled or failed run is the expected end of the wait.
        logger.info(f"Pipeline run {pipeline_run} stopped: {e}")

    cancel_child_jobs(job_name=job_name, resource_config_uri=resource_config_uri)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cancel a GiGL pipeline run and the jobs it launched."
    )
    parser.add_argument(
        "--pipeline_run",
        required=True,
        help="Full resource name, projects/<p>/locations/<l>/pipelineJobs/<id>.",
    )
    args = parser.parse_args()
    cancel_pipeline_run_and_child_jobs(args.pipeline_run)
