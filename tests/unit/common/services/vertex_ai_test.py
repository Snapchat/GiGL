from typing import cast
from unittest.mock import patch

from google.cloud import aiplatform
from google.cloud.aiplatform_v1.types import job_state

from gigl.common.services.vertex_ai import (
    VertexAIService,
    _wait_for_job_to_leave_queue,
)
from tests.test_assets.test_case import TestCase

JobState = job_state.JobState
PipelineState = aiplatform.gapic.PipelineState


class _FakeCustomJob:
    """Stands in for ``aiplatform.CustomJob``.

    Returns the scripted states in order and then repeats the last one.
    """

    def __init__(self, states: list[JobState]) -> None:
        self._states = list(states)
        self.cancel_calls = 0
        self.resource_name = "projects/p/locations/us-central1/customJobs/1"
        self.location = "us-central1"

    @property
    def state(self) -> JobState:
        if len(self._states) > 1:
            return self._states.pop(0)
        return self._states[0]

    def cancel(self) -> None:
        self.cancel_calls += 1


class WaitForJobToLeaveQueueTest(TestCase):
    def test_returns_once_the_job_starts_running(self) -> None:
        job = _FakeCustomJob(
            [
                JobState.JOB_STATE_QUEUED,
                JobState.JOB_STATE_PENDING,
                JobState.JOB_STATE_RUNNING,
            ]
        )

        _wait_for_job_to_leave_queue(
            cast(aiplatform.CustomJob, job), queue_timeout_s=5, poll_interval_s=0.01
        )

        self.assertEqual(job.cancel_calls, 0)
        # The scripted states were consumed down to the running state.
        self.assertEqual(job.state, JobState.JOB_STATE_RUNNING)

    def test_returns_when_the_job_already_finished(self) -> None:
        job = _FakeCustomJob([JobState.JOB_STATE_SUCCEEDED])

        _wait_for_job_to_leave_queue(
            cast(aiplatform.CustomJob, job), queue_timeout_s=5, poll_interval_s=0.01
        )

        self.assertEqual(job.cancel_calls, 0)

    def test_cancels_and_raises_when_the_job_stays_queued(self) -> None:
        job = _FakeCustomJob([JobState.JOB_STATE_QUEUED])

        with self.assertRaisesRegex(RuntimeError, "still queued"):
            _wait_for_job_to_leave_queue(
                cast(aiplatform.CustomJob, job),
                queue_timeout_s=0.05,
                poll_interval_s=0.01,
            )

        self.assertEqual(job.cancel_calls, 1)


class _FakePipelineJob:
    """Stands in for ``aiplatform.PipelineJob`` with a fixed state."""

    def __init__(self, resource_name: str, state: PipelineState) -> None:
        self.resource_name = resource_name
        self.state = state
        self.cancel_calls = 0

    def cancel(self) -> None:
        self.cancel_calls += 1


class CancelPipelineRunTest(TestCase):
    def _cancel(self, state: PipelineState) -> tuple[bool, _FakePipelineJob]:
        run = _FakePipelineJob("projects/p/locations/l/pipelineJobs/r", state)
        with patch.object(aiplatform.PipelineJob, "get", return_value=run) as get:
            cancelled = VertexAIService.cancel_pipeline_run(run.resource_name)
        get.assert_called_once_with(run.resource_name)
        return cancelled, run

    def test_cancels_a_running_run(self) -> None:
        cancelled, run = self._cancel(PipelineState.PIPELINE_STATE_RUNNING)

        self.assertTrue(cancelled)
        self.assertEqual(run.cancel_calls, 1)

    def test_skips_a_run_already_cancelling(self) -> None:
        cancelled, run = self._cancel(PipelineState.PIPELINE_STATE_CANCELLING)

        self.assertFalse(cancelled)
        self.assertEqual(run.cancel_calls, 0)

    def test_skips_a_finished_run(self) -> None:
        cancelled, run = self._cancel(PipelineState.PIPELINE_STATE_SUCCEEDED)

        self.assertFalse(cancelled)
        self.assertEqual(run.cancel_calls, 0)


class CancelCustomJobsNamedTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        init_patcher = patch.object(aiplatform, "init")
        init_patcher.start()
        self.addCleanup(init_patcher.stop)
        self.service = VertexAIService(
            project="p",
            location="us-central1",
            service_account="sa",
            staging_bucket="gs://b",
        )

    def test_cancels_only_active_jobs(self) -> None:
        running = _FakeCustomJob([JobState.JOB_STATE_RUNNING])
        running.resource_name = "projects/p/locations/us-central1/customJobs/running"
        finished = _FakeCustomJob([JobState.JOB_STATE_SUCCEEDED])
        finished.resource_name = "projects/p/locations/us-central1/customJobs/finished"
        jobs_by_filter = {
            'display_name="gigl_train_x"': [running],
            'display_name="gigl_infer_x"': [finished],
        }

        def fake_list(filter: str, project: str, location: str) -> list[_FakeCustomJob]:
            self.assertEqual((project, location), ("p", "us-central1"))
            return jobs_by_filter[filter]

        with patch.object(aiplatform.CustomJob, "list", side_effect=fake_list):
            cancelled = self.service.cancel_custom_jobs_named(
                ["gigl_train_x", "gigl_infer_x"]
            )

        self.assertEqual(cancelled, [running.resource_name])
        self.assertEqual(running.cancel_calls, 1)
        self.assertEqual(finished.cancel_calls, 0)

    def test_returns_empty_when_nothing_matches(self) -> None:
        with patch.object(aiplatform.CustomJob, "list", return_value=[]):
            cancelled = self.service.cancel_custom_jobs_named(["gigl_train_x"])

        self.assertEqual(cancelled, [])
