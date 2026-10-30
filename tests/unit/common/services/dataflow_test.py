from unittest.mock import patch

from google.api_core.exceptions import FailedPrecondition
from google.cloud import dataflow_v1beta3 as dataflow

from gigl.common.services.dataflow import _CANCEL_ATTEMPTS, DataflowService
from tests.test_assets.test_case import TestCase


class _FakeJobsClient:
    """Stands in for ``dataflow_v1beta3.JobsV1Beta3Client`` over a fixed job list.

    ``rejections`` is how many update calls fail with FailedPrecondition before one
    is accepted, as Dataflow does for a job that has no backend assigned yet.
    """

    def __init__(self, jobs: list[dataflow.Job], rejections: int = 0) -> None:
        self._jobs = jobs
        self._rejections = rejections
        self.list_requests: list[dataflow.ListJobsRequest] = []
        self.update_requests: list[dataflow.UpdateJobRequest] = []

    def list_jobs(self, request: dataflow.ListJobsRequest) -> list[dataflow.Job]:
        self.list_requests.append(request)
        return list(self._jobs)

    def update_job(self, request: dataflow.UpdateJobRequest) -> dataflow.Job:
        self.update_requests.append(request)
        if self._rejections > 0:
            self._rejections -= 1
            raise FailedPrecondition("Job has no backend assigned.")
        return request.job


def _service_over(jobs_client: _FakeJobsClient) -> DataflowService:
    with patch(
        "gigl.common.services.dataflow.dataflow.JobsV1Beta3Client",
        return_value=jobs_client,
    ):
        return DataflowService(project="p", region="us-central1")


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
        cancelled = _service_over(jobs_client).cancel_jobs_with_name_prefix(
            "gigl-my-job-"
        )

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

    def test_retries_a_cancel_the_job_is_too_new_to_accept(self) -> None:
        jobs_client = _FakeJobsClient(
            [dataflow.Job(id="1", name="gigl-a-x")], rejections=2
        )

        with patch("gigl.common.services.dataflow.time.sleep") as sleep:
            cancelled = _service_over(jobs_client).cancel_jobs_with_name_prefix(
                "gigl-a-"
            )

        self.assertEqual(cancelled, ["1"])
        self.assertEqual(len(jobs_client.update_requests), 3)
        self.assertEqual(sleep.call_count, 2)

    def test_gives_up_when_the_cancel_is_never_accepted(self) -> None:
        jobs_client = _FakeJobsClient(
            [dataflow.Job(id="1", name="gigl-a-x")], rejections=100
        )

        with (
            patch("gigl.common.services.dataflow.time.sleep"),
            self.assertRaises(FailedPrecondition),
        ):
            _service_over(jobs_client).cancel_jobs_with_name_prefix("gigl-a-")

        self.assertEqual(len(jobs_client.update_requests), _CANCEL_ATTEMPTS)
