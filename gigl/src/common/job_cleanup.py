"""Cancel the GCP jobs a GiGL pipeline run launched.

GiGL components launch Vertex AI CustomJobs, Dataflow jobs, and Dataproc clusters
from inside their own containers.
Vertex AI does not know about those children, so cancelling the pipeline run only
kills the component container.
This module finds the children by the names GiGL gives them and cancels them.
"""

from gigl.common import Uri
from gigl.common.logger import Logger
from gigl.common.services.dataflow import DataflowService
from gigl.common.services.dataproc import DataprocService
from gigl.common.services.vertex_ai import VertexAIService
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.utils.dataflow import get_sanitized_dataflow_job_name
from gigl.src.common.utils.spark_job_manager import (
    build_split_generator_cluster_name,
    build_subgraph_sampler_cluster_name,
)
from gigl.src.common.vertex_ai_launcher import build_vertex_ai_job_name

logger = Logger()


def cancel_child_jobs(job_name: str, resource_config_uri: Uri) -> None:
    """Cancel every Vertex AI CustomJob, Dataflow job, and Dataproc cluster of a GiGL job.

    Each kind is attempted even if an earlier kind raised, so one failing client
    does not leave the other children running.
    Finding nothing to cancel is not an error.

    Example:
        >>> cancel_child_jobs(  # doctest: +SKIP
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
        cancelled_jobs = vertex_ai_service.cancel_custom_jobs_named(
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
            if dataproc_service.delete_cluster_if_exists(cluster_name):
                deleted_clusters.append(cluster_name)
        logger.info(f"Deleted Dataproc clusters: {deleted_clusters or 'none'}")
    except Exception as e:
        failures.append(f"Dataproc clusters: {e!r}")

    if failures:
        raise RuntimeError(
            f"Cleanup for job {job_name} failed for: " + "; ".join(failures)
        )
