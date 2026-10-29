from typing import Optional
from unittest.mock import patch

from apache_beam.runners.dataflow.internal.clients import dataflow

from gigl.common.services.dataflow import DataflowService
from tests.test_assets.test_case import TestCase

_ACTIVE = dataflow.DataflowProjectsLocationsJobsListRequest.FilterValueValuesEnum.ACTIVE
_CANCELLED = dataflow.Job.RequestedStateValueValuesEnum.JOB_STATE_CANCELLED


def _job(job_id: str, name: str) -> dataflow.Job:
    return dataflow.Job(id=job_id, name=name)


class _FakeJobsService:
    """Stands in for ``DataflowV1b3.projects_locations_jobs``.

    ``pages`` is a list of job lists; each List call returns the next page and a
    page token until the last page.
    """

    def __init__(self, pages: list[list[dataflow.Job]]) -> None:
        self._pages = pages
        self.list_requests: list[dataflow.DataflowProjectsLocationsJobsListRequest] = []
        self.update_requests: list[
            dataflow.DataflowProjectsLocationsJobsUpdateRequest
        ] = []

    def List(
        self, request: dataflow.DataflowProjectsLocationsJobsListRequest
    ) -> dataflow.ListJobsResponse:
        self.list_requests.append(request)
        page_index = int(request.pageToken) if request.pageToken else 0
        next_token: Optional[str] = (
            str(page_index + 1) if page_index + 1 < len(self._pages) else None
        )
        return dataflow.ListJobsResponse(
            jobs=self._pages[page_index], nextPageToken=next_token
        )

    def Update(
        self, request: dataflow.DataflowProjectsLocationsJobsUpdateRequest
    ) -> dataflow.Job:
        self.update_requests.append(request)
        return request.job


class CancelJobsWithNamePrefixTest(TestCase):
    def _service(
        self, pages: list[list[dataflow.Job]]
    ) -> tuple[DataflowService, _FakeJobsService]:
        jobs_service = _FakeJobsService(pages)
        with (
            patch("gigl.common.services.dataflow.get_service_credentials"),
            patch(
                "gigl.common.services.dataflow.dataflow.DataflowV1b3"
            ) as client_class,
        ):
            client_class.return_value.projects_locations_jobs = jobs_service
            service = DataflowService(project="p", region="us-central1")
        return service, jobs_service

    def test_cancels_active_jobs_with_the_prefix_only(self) -> None:
        service, jobs_service = self._service(
            [
                [
                    _job("1", "gigl-my-job-node-feature-prep-x"),
                    _job("2", "gigl-my-job-infer"),
                    _job("3", "gigl-my-job-bar-infer"),
                    _job("4", "gigl-other-infer"),
                ]
            ]
        )

        cancelled = service.cancel_jobs_with_name_prefix("gigl-my-job-")

        self.assertEqual(cancelled, ["1", "2", "3"])
        (list_request,) = jobs_service.list_requests
        self.assertEqual(list_request.filter, _ACTIVE)
        self.assertEqual(
            (list_request.projectId, list_request.location), ("p", "us-central1")
        )
        self.assertEqual(
            [r.jobId for r in jobs_service.update_requests], ["1", "2", "3"]
        )
        for request in jobs_service.update_requests:
            self.assertEqual(
                (request.projectId, request.location), ("p", "us-central1")
            )
            self.assertEqual(request.job.requestedState, _CANCELLED)

    def test_follows_page_tokens(self) -> None:
        service, jobs_service = self._service(
            [[_job("1", "gigl-a-x")], [_job("2", "gigl-a-y")]]
        )

        cancelled = service.cancel_jobs_with_name_prefix("gigl-a-")

        self.assertEqual(cancelled, ["1", "2"])
        self.assertEqual(len(jobs_service.list_requests), 2)
        self.assertEqual(jobs_service.list_requests[1].pageToken, "1")
