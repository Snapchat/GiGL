from typing import cast

from google.cloud import aiplatform
from google.cloud.aiplatform_v1.types import job_state

from gigl.common.services.vertex_ai import _wait_for_job_to_leave_queue
from tests.test_assets.test_case import TestCase

JobState = job_state.JobState


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
