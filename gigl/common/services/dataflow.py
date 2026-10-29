"""Cancel Dataflow jobs by name.

Uses the apitools Dataflow client that ships inside ``apache-beam[gcp]``, so no
extra dependency is needed in the ``src-cpu`` image.
"""

from typing import Optional

from apache_beam.internal.gcp.auth import get_service_credentials
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.runners.dataflow.internal.clients import dataflow

from gigl.common.logger import Logger

logger = Logger()

_DATAFLOW_ENDPOINT = "https://dataflow.googleapis.com/"


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
        credentials = get_service_credentials(PipelineOptions())
        self._client = dataflow.DataflowV1b3(
            url=_DATAFLOW_ENDPOINT, credentials=credentials, get_credentials=False
        )

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
        cancelled: list[str] = []
        for job in self._list_active_jobs():
            if not job.name.startswith(name_prefix):
                continue
            logger.info(f"Cancelling Dataflow job {job.id} ({job.name}).")
            self._client.projects_locations_jobs.Update(
                dataflow.DataflowProjectsLocationsJobsUpdateRequest(
                    projectId=self._project,
                    location=self._region,
                    jobId=job.id,
                    job=dataflow.Job(
                        requestedState=dataflow.Job.RequestedStateValueValuesEnum.JOB_STATE_CANCELLED
                    ),
                )
            )
            cancelled.append(job.id)
        return cancelled

    def _list_active_jobs(self) -> list[dataflow.Job]:
        jobs: list[dataflow.Job] = []
        page_token: Optional[str] = None
        while True:
            response = self._client.projects_locations_jobs.List(
                dataflow.DataflowProjectsLocationsJobsListRequest(
                    projectId=self._project,
                    location=self._region,
                    filter=dataflow.DataflowProjectsLocationsJobsListRequest.FilterValueValuesEnum.ACTIVE,
                    pageToken=page_token,
                )
            )
            jobs.extend(response.jobs or [])
            page_token = response.nextPageToken
            if not page_token:
                return jobs
