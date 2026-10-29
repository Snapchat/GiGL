from unittest.mock import patch

from google.cloud import dataflow_v1beta3 as dataflow

from gigl.common.services.dataflow import DataflowService
from tests.test_assets.test_case import TestCase


class _FakeJobsClient:
    """Stands in for ``dataflow_v1beta3.JobsV1Beta3Client`` over a fixed job list."""

    def __init__(self, jobs: list[dataflow.Job]) -> None:
        self._jobs = jobs
        self.list_requests: list[dataflow.ListJobsRequest] = []
        self.update_requests: list[dataflow.UpdateJobRequest] = []

    def list_jobs(self, request: dataflow.ListJobsRequest) -> list[dataflow.Job]:
        self.list_requests.append(request)
        return list(self._jobs)

    def update_job(self, request: dataflow.UpdateJobRequest) -> dataflow.Job:
        self.update_requests.append(request)
        return request.job


class CancelJobsWithNamePrefixTest(TestCase):
    def test_cancels_active_jobs_with_the_prefix_only(self) -> None:
        jobs_client = _FakeJobsClient(
            [
                dataflow.Job(id="1", name="gigl-my-job-node-feature-prep-x"),
                dataflow.Job(id="2", name="gigl-my-job-infer"),
                dataflow.Job(id="3", name="gigl-my-job-bar-infer"),
                dataflow.Job(id="4", name="gigl-other-infer"),
            ]
        )
        with patch(
            "gigl.common.services.dataflow.dataflow.JobsV1Beta3Client",
            return_value=jobs_client,
        ):
            service = DataflowService(project="p", region="us-central1")
            cancelled = service.cancel_jobs_with_name_prefix("gigl-my-job-")

        self.assertEqual(cancelled, ["1", "2", "3"])
        (list_request,) = jobs_client.list_requests
        self.assertEqual(list_request.filter, dataflow.ListJobsRequest.Filter.ACTIVE)
        self.assertEqual(
            (list_request.project_id, list_request.location), ("p", "us-central1")
        )
        self.assertEqual(
            [r.job_id for r in jobs_client.update_requests], ["1", "2", "3"]
        )
        for request in jobs_client.update_requests:
            self.assertEqual(
                (request.project_id, request.location), ("p", "us-central1")
            )
            self.assertEqual(
                request.job.requested_state, dataflow.JobState.JOB_STATE_CANCELLED
            )
