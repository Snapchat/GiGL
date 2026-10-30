from unittest.mock import patch

from google.cloud.devtools import cloudbuild_v1

from gigl.common.services.cloud_build import CloudBuildService
from tests.test_assets.test_case import TestCase


class _FakeCloudBuildClient:
    """Stands in for ``cloudbuild_v1.CloudBuildClient`` over a fixed build list."""

    def __init__(self, builds: list[cloudbuild_v1.Build]) -> None:
        self._builds = builds
        self.list_requests: list[cloudbuild_v1.ListBuildsRequest] = []
        self.cancel_requests: list[cloudbuild_v1.CancelBuildRequest] = []

    def list_builds(
        self, request: cloudbuild_v1.ListBuildsRequest
    ) -> list[cloudbuild_v1.Build]:
        self.list_requests.append(request)
        return list(self._builds)

    def cancel_build(
        self, request: cloudbuild_v1.CancelBuildRequest
    ) -> cloudbuild_v1.Build:
        self.cancel_requests.append(request)
        return cloudbuild_v1.Build(id=request.id)


def _service_over(client: _FakeCloudBuildClient) -> CloudBuildService:
    with patch(
        "gigl.common.services.cloud_build.cloudbuild_v1.CloudBuildClient",
        return_value=client,
    ):
        return CloudBuildService(project="p")


class CancelBuildsWithTagTest(TestCase):
    def test_cancels_every_listed_build_and_returns_their_ids(self) -> None:
        client = _FakeCloudBuildClient(
            [cloudbuild_v1.Build(id="b1"), cloudbuild_v1.Build(id="b2")]
        )
        cancelled = _service_over(client).cancel_builds_with_tag("gigl-ci-run-1")

        self.assertEqual(cancelled, ["b1", "b2"])
        (list_request,) = client.list_requests
        self.assertEqual(list_request.project_id, "p")
        self.assertEqual(
            list_request.filter,
            'tags="gigl-ci-run-1" AND (status=WORKING OR status=QUEUED)',
        )
        self.assertEqual(
            [(r.project_id, r.id) for r in client.cancel_requests],
            [("p", "b1"), ("p", "b2")],
        )

    def test_returns_nothing_when_no_build_matches(self) -> None:
        client = _FakeCloudBuildClient([])
        self.assertEqual(_service_over(client).cancel_builds_with_tag("t"), [])
        self.assertEqual(client.cancel_requests, [])
