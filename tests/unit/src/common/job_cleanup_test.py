from typing import Sequence
from unittest.mock import patch

from google.api_core.exceptions import PermissionDenied

import gigl.src.common.constants.local_fs as local_fs_constants
from gigl.common import LocalUri
from gigl.src.common.job_cleanup import cancel_child_jobs
from tests.test_assets.test_case import TestCase

_RESOURCE_CONFIG_URI = LocalUri.join(
    local_fs_constants.get_project_root_directory(),
    "deployment",
    "configs",
    "unittest_resource_config.yaml",
)


class _FailingVertexAIService:
    def __init__(self, **_: object) -> None:
        pass

    def cancel_custom_jobs_named(self, display_names: Sequence[str]) -> list[str]:
        raise PermissionDenied("custom job cancel denied")


class _FailingDataflowService:
    def __init__(self, **_: object) -> None:
        pass

    def cancel_jobs_with_name_prefix(self, name_prefix: str) -> list[str]:
        raise RuntimeError("dataflow unreachable")


class _RecordingDataprocService:
    deleted: list[str] = []

    def __init__(self, **_: object) -> None:
        pass

    def delete_cluster_if_exists(self, cluster_name: str) -> bool:
        _RecordingDataprocService.deleted.append(cluster_name)
        return True


class CancelChildJobsTest(TestCase):
    def test_every_child_kind_is_attempted_and_each_failure_is_reported(
        self,
    ) -> None:
        """The Vertex and Dataflow clients fail; the Dataproc clusters are still deleted,
        and the single error names both failed steps."""
        _RecordingDataprocService.deleted = []
        with (
            patch(
                "gigl.src.common.job_cleanup.VertexAIService", _FailingVertexAIService
            ),
            patch(
                "gigl.src.common.job_cleanup.DataflowService", _FailingDataflowService
            ),
            patch(
                "gigl.src.common.job_cleanup.DataprocService",
                _RecordingDataprocService,
            ),
            self.assertRaises(RuntimeError) as raised,
        ):
            cancel_child_jobs(
                job_name="My_Job", resource_config_uri=_RESOURCE_CONFIG_URI
            )

        message = str(raised.exception)
        self.assertIn("custom jobs: PermissionDenied", message)
        self.assertIn("Dataflow jobs: RuntimeError", message)
        self.assertNotIn("Dataproc", message)
        self.assertEqual(
            _RecordingDataprocService.deleted, ["sgs-My-Job", "split-My-Job"]
        )
