from typing import Sequence
from unittest.mock import MagicMock, patch

from google.api_core.exceptions import PermissionDenied
from google.cloud import aiplatform

import gigl.src.common.constants.local_fs as local_fs_constants
from gigl.common import LocalUri
from gigl.common.services.vertex_ai import PipelineRunStoppedError
from gigl.src.common.job_cleanup import (
    cancel_child_jobs,
    cancel_pipeline_run_and_child_jobs,
)
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

    def cancel_custom_jobs_with_name(self, display_names: Sequence[str]) -> list[str]:
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


class _FirstClusterFailsDataprocService:
    deleted: list[str] = []

    def __init__(self, **_: object) -> None:
        pass

    def delete_cluster_if_exists(self, cluster_name: str) -> bool:
        if cluster_name.startswith("sgs-"):
            raise PermissionDenied("delete denied")
        _FirstClusterFailsDataprocService.deleted.append(cluster_name)
        return True


class _NothingToCancelVertexAIService:
    def __init__(self, **_: object) -> None:
        pass

    def cancel_custom_jobs_with_name(self, display_names: Sequence[str]) -> list[str]:
        return []


class _NothingToCancelDataflowService:
    def __init__(self, **_: object) -> None:
        pass

    def cancel_jobs_with_name_prefix(self, name_prefix: str) -> list[str]:
        return []


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

    def test_a_failed_cluster_delete_does_not_stop_the_other_delete(self) -> None:
        _FirstClusterFailsDataprocService.deleted = []
        with (
            patch(
                "gigl.src.common.job_cleanup.VertexAIService",
                _NothingToCancelVertexAIService,
            ),
            patch(
                "gigl.src.common.job_cleanup.DataflowService",
                _NothingToCancelDataflowService,
            ),
            patch(
                "gigl.src.common.job_cleanup.DataprocService",
                _FirstClusterFailsDataprocService,
            ),
            self.assertRaises(RuntimeError) as raised,
        ):
            cancel_child_jobs(
                job_name="My_Job", resource_config_uri=_RESOURCE_CONFIG_URI
            )

        self.assertIn("Dataproc cluster sgs-My-Job", str(raised.exception))
        self.assertEqual(_FirstClusterFailsDataprocService.deleted, ["split-My-Job"])


class CancelPipelineRunAndChildJobsTest(TestCase):
    _PIPELINE_RUN = "projects/p/locations/us-central1/pipelineJobs/my-run"

    def _run_with_parameters(self) -> MagicMock:
        run = MagicMock()
        run.gca_resource.runtime_config.parameter_values = {
            "job_name": "my_job",
            "resource_config_uri": "gs://bucket/resource_config.yaml",
        }
        return run

    def test_cancels_the_run_waits_then_cancels_the_children(self) -> None:
        calls: list[str] = []

        def record_wait(resource_name: str, timeout: float) -> None:
            calls.append("wait")
            raise PipelineRunStoppedError(
                "stopped",
                state=aiplatform.gapic.PipelineState.PIPELINE_STATE_CANCELLED,
            )

        with (
            patch.object(
                aiplatform.PipelineJob,
                "get",
                return_value=self._run_with_parameters(),
            ),
            patch(
                "gigl.src.common.job_cleanup.VertexAIService.cancel_pipeline_run",
                side_effect=lambda resource_name: calls.append("cancel") or True,
            ),
            patch(
                "gigl.src.common.job_cleanup.VertexAIService.wait_for_run_completion",
                side_effect=record_wait,
            ),
            patch(
                "gigl.src.common.job_cleanup.cancel_child_jobs",
                side_effect=lambda **kwargs: calls.append("children"),
            ) as cancel_children,
        ):
            cancel_pipeline_run_and_child_jobs(self._PIPELINE_RUN)

        self.assertEqual(calls, ["cancel", "wait", "children"])
        self.assertEqual(cancel_children.call_args.kwargs["job_name"], "my_job")
        self.assertEqual(
            cancel_children.call_args.kwargs["resource_config_uri"].uri,
            "gs://bucket/resource_config.yaml",
        )

    def test_a_run_that_does_not_stop_leaves_the_children_alone(self) -> None:
        with (
            patch.object(
                aiplatform.PipelineJob,
                "get",
                return_value=self._run_with_parameters(),
            ),
            patch(
                "gigl.src.common.job_cleanup.VertexAIService.cancel_pipeline_run",
                return_value=True,
            ),
            patch(
                "gigl.src.common.job_cleanup.VertexAIService.wait_for_run_completion",
                side_effect=RuntimeError("timed out"),
            ),
            patch("gigl.src.common.job_cleanup.cancel_child_jobs") as cancel_children,
            self.assertRaises(RuntimeError),
        ):
            cancel_pipeline_run_and_child_jobs(self._PIPELINE_RUN)

        cancel_children.assert_not_called()
