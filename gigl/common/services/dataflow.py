"""Cancel Dataflow jobs by name through the ``google-cloud-dataflow-client`` library."""

from google.cloud import dataflow_v1beta3 as dataflow

from gigl.common.logger import Logger

logger = Logger()


class DataflowService:
    """Lists and cancels Dataflow jobs in one project and region.

    Example:
        >>> service = DataflowService(project="p", region="us-central1")
        >>> service.cancel_jobs_with_name_prefix("gigl-my-job-")
        ['2026-10-28_09_00_00-123']
    """

    def __init__(self, project: str, region: str) -> None:
        """
        Args:
            project: GCP project that owns the jobs.
            region: Dataflow regional endpoint the jobs run in.
        """
        self._project = project
        self._region = region
        self._jobs_client = dataflow.JobsV1Beta3Client()

    def cancel_jobs_with_name_prefix(self, name_prefix: str) -> list[str]:
        """Request a cancel for every active job whose name starts with ``name_prefix``.

        The match is a plain prefix.
        Build the prefix with
        :func:`gigl.src.common.utils.dataflow.get_sanitized_dataflow_job_name` and end
        it with ``-`` so the full GiGL job name must match.

        Args:
            name_prefix: Sanitized Dataflow job name prefix.

        Returns:
            Ids of the jobs a cancel was requested for.
        """
        active_jobs = self._jobs_client.list_jobs(
            request=dataflow.ListJobsRequest(
                project_id=self._project,
                location=self._region,
                filter=dataflow.ListJobsRequest.Filter.ACTIVE,
            )
        )
        cancelled: list[str] = []
        for job in active_jobs:
            if not job.name.startswith(name_prefix):
                continue
            logger.info(f"Cancelling Dataflow job {job.id} ({job.name}).")
            self._jobs_client.update_job(
                request=dataflow.UpdateJobRequest(
                    project_id=self._project,
                    location=self._region,
                    job_id=job.id,
                    job=dataflow.Job(
                        requested_state=dataflow.JobState.JOB_STATE_CANCELLED
                    ),
                )
            )
            cancelled.append(job.id)
        return cancelled
