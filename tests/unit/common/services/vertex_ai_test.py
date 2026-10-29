from typing import Optional, cast
from unittest.mock import patch

from google.api_core.exceptions import FailedPrecondition
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

    def __init__(
        self,
        resource_name: str,
        state: PipelineState,
        cancel_error: Optional[Exception] = None,
    ) -> None:
        self.resource_name = resource_name
        self.state = state
        self.cancel_calls = 0
        self.cancel_error = cancel_error

    def cancel(self) -> None:
        self.cancel_calls += 1
        if self.cancel_error is not None:
            raise self.cancel_error


class CancelPipelineRunTest(TestCase):
    def _cancel(self, state: PipelineState) -> tuple[bool, _FakePipelineJob]:
        run = _FakePipelineJob("projects/p/locations/l/pipelineJobs/r", state)
        with patch.object(aiplatform.PipelineJob, "get", return_value=run):
            cancelled = VertexAIService.cancel_pipeline_run(run.resource_name)
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

    def test_returns_false_when_the_run_ends_before_the_cancel_lands(self) -> None:
        # The run was RUNNING at the state check but reached FAILED before the cancel
        # request arrived; Vertex AI then rejects the cancel with FailedPrecondition.
        run = _FakePipelineJob(
            "projects/p/locations/l/pipelineJobs/r",
            PipelineState.PIPELINE_STATE_RUNNING,
            cancel_error=FailedPrecondition(
                'The PipelineJob "r" is in state JOB_STATE_FAILED and cannot be canceled.'
            ),
        )
        with patch.object(aiplatform.PipelineJob, "get", return_value=run):
            cancelled = VertexAIService.cancel_pipeline_run(run.resource_name)

        self.assertFalse(cancelled)
        self.assertEqual(run.cancel_calls, 1)


class CancelPipelineRunsWithLabelTest(TestCase):
    def test_cancels_only_the_active_labelled_runs(self) -> None:
        running = _FakePipelineJob(
            "projects/p/locations/l/pipelineJobs/running",
            PipelineState.PIPELINE_STATE_RUNNING,
        )
        pending = _FakePipelineJob(
            "projects/p/locations/l/pipelineJobs/pending",
            PipelineState.PIPELINE_STATE_PENDING,
        )
        failed = _FakePipelineJob(
            "projects/p/locations/l/pipelineJobs/failed",
            PipelineState.PIPELINE_STATE_FAILED,
        )
        runs_by_name = {run.resource_name: run for run in (running, pending, failed)}

        def fake_list(
            filter: str, project: str, location: str
        ) -> list[_FakePipelineJob]:
            self.assertEqual(filter, 'labels.github_run_id="123"')
            self.assertEqual((project, location), ("p", "l"))
            return [running, pending, failed]

        def fake_get(resource_name: str) -> _FakePipelineJob:
            return runs_by_name[resource_name]

        with (
            patch.object(aiplatform.PipelineJob, "list", side_effect=fake_list),
            patch.object(aiplatform.PipelineJob, "get", side_effect=fake_get),
        ):
            cancelled = VertexAIService.cancel_pipeline_runs_with_label(
                project="p",
                location="l",
                label_key="github_run_id",
                label_value="123",
            )

        self.assertEqual(cancelled, [running.resource_name, pending.resource_name])
        self.assertEqual(
            (running.cancel_calls, pending.cancel_calls, failed.cancel_calls),
            (1, 1, 0),
        )


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
