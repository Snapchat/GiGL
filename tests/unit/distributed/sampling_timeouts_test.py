import multiprocessing as std_mp
import os
from unittest import mock

from absl.testing import absltest

from gigl.distributed.constants import (
    DEFAULT_SAMPLING_RPC_TIMEOUT_SECONDS,
    DEFAULT_SAMPLING_STEADY_STATE_RPC_TIMEOUT_SECONDS,
    DEFAULT_SAMPLING_WORKER_INIT_TIMEOUT_SECONDS,
    SAMPLING_RPC_TIMEOUT_ENV,
    SAMPLING_STEADY_STATE_RPC_TIMEOUT_ENV,
    SAMPLING_WORKER_INIT_TIMEOUT_ENV,
    sampling_rpc_timeout_seconds,
    sampling_steady_state_rpc_timeout_seconds,
    sampling_worker_init_timeout_seconds,
)
from gigl.distributed.dist_sampling_producer import (
    DistSamplingProducer,
    _narrow_rpc_timeout_after_init,
)
from tests.test_assets.test_case import TestCase

_ALL_ENVS = (
    SAMPLING_RPC_TIMEOUT_ENV,
    SAMPLING_STEADY_STATE_RPC_TIMEOUT_ENV,
    SAMPLING_WORKER_INIT_TIMEOUT_ENV,
)


class _ExitedWorker:
    """Stands in for a sampling worker process that died before the barrier."""

    def __init__(self, exitcode: int) -> None:
        self.exitcode = exitcode

    def is_alive(self) -> bool:
        return False

    def join(self, timeout: float | None = None) -> None:
        pass

    def terminate(self) -> None:
        pass


class SamplingTimeoutConstantsTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._saved = {name: os.environ.pop(name, None) for name in _ALL_ENVS}

    def tearDown(self) -> None:
        for name, value in self._saved.items():
            os.environ.pop(name, None)
            if value is not None:
                os.environ[name] = value
        super().tearDown()

    def test_defaults_apply_when_the_environment_is_unset(self) -> None:
        self.assertEqual(
            sampling_rpc_timeout_seconds(), DEFAULT_SAMPLING_RPC_TIMEOUT_SECONDS
        )
        self.assertEqual(
            sampling_steady_state_rpc_timeout_seconds(),
            DEFAULT_SAMPLING_STEADY_STATE_RPC_TIMEOUT_SECONDS,
        )
        self.assertEqual(
            sampling_worker_init_timeout_seconds(),
            DEFAULT_SAMPLING_WORKER_INIT_TIMEOUT_SECONDS,
        )

    def test_environment_overrides_are_read(self) -> None:
        os.environ[SAMPLING_RPC_TIMEOUT_ENV] = "1200"
        os.environ[SAMPLING_STEADY_STATE_RPC_TIMEOUT_ENV] = "90"
        os.environ[SAMPLING_WORKER_INIT_TIMEOUT_ENV] = "2400"

        self.assertEqual(sampling_rpc_timeout_seconds(), 1200)
        self.assertEqual(sampling_steady_state_rpc_timeout_seconds(), 90)
        self.assertEqual(sampling_worker_init_timeout_seconds(), 2400)

    def test_invalid_values_fall_back_to_the_default(self) -> None:
        for bad in ("not-a-number", "0", "-5", ""):
            os.environ[SAMPLING_RPC_TIMEOUT_ENV] = bad
            self.assertEqual(
                sampling_rpc_timeout_seconds(),
                DEFAULT_SAMPLING_RPC_TIMEOUT_SECONDS,
                msg=f"for value {bad!r}",
            )


class SteadyStateNarrowingTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._saved = os.environ.pop(SAMPLING_STEADY_STATE_RPC_TIMEOUT_ENV, None)

    def tearDown(self) -> None:
        os.environ.pop(SAMPLING_STEADY_STATE_RPC_TIMEOUT_ENV, None)
        if self._saved is not None:
            os.environ[SAMPLING_STEADY_STATE_RPC_TIMEOUT_ENV] = self._saved
        super().tearDown()

    def test_a_mismatched_bringup_timeout_is_narrowed_to_steady_state(self) -> None:
        with mock.patch("torch.distributed.rpc._set_rpc_timeout") as set_rpc_timeout:
            _narrow_rpc_timeout_after_init(rank=0, bringup_rpc_timeout=3600)

        set_rpc_timeout.assert_called_once_with(
            float(DEFAULT_SAMPLING_STEADY_STATE_RPC_TIMEOUT_SECONDS)
        )

    def test_equal_timeouts_leave_the_agent_untouched(self) -> None:
        with mock.patch("torch.distributed.rpc._set_rpc_timeout") as set_rpc_timeout:
            _narrow_rpc_timeout_after_init(
                rank=0,
                bringup_rpc_timeout=DEFAULT_SAMPLING_STEADY_STATE_RPC_TIMEOUT_SECONDS,
            )

        set_rpc_timeout.assert_not_called()

    def test_an_overridden_steady_state_value_is_applied(self) -> None:
        os.environ[SAMPLING_STEADY_STATE_RPC_TIMEOUT_ENV] = "90"
        with mock.patch("torch.distributed.rpc._set_rpc_timeout") as set_rpc_timeout:
            _narrow_rpc_timeout_after_init(rank=0, bringup_rpc_timeout=600)

        set_rpc_timeout.assert_called_once_with(90.0)


class WorkerInitBarrierTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._saved_timeout = os.environ.pop(SAMPLING_WORKER_INIT_TIMEOUT_ENV, None)

    def tearDown(self) -> None:
        os.environ.pop(SAMPLING_WORKER_INIT_TIMEOUT_ENV, None)
        if self._saved_timeout is not None:
            os.environ[SAMPLING_WORKER_INIT_TIMEOUT_ENV] = self._saved_timeout
        super().tearDown()

    def _producer_with_workers(self, workers: list) -> DistSamplingProducer:
        producer = DistSamplingProducer.__new__(DistSamplingProducer)
        producer._workers = workers
        # The timeout path calls GLT's shutdown(), which reads these
        producer._shutdown = False
        producer._task_queues = []
        return producer

    def test_a_worker_that_never_arrives_raises_instead_of_hanging(self) -> None:
        os.environ[SAMPLING_WORKER_INIT_TIMEOUT_ENV] = "1"
        producer = self._producer_with_workers([_ExitedWorker(exitcode=1)])
        # 2 parties and only this process waiting = the dead worker's slot never fills
        barrier = std_mp.get_context("spawn").Barrier(2)

        with self.assertRaisesRegex(RuntimeError, r"rank -> exitcode.*\{0: 1\}"):
            producer._wait_for_workers_at_barrier(barrier)

    def test_a_full_barrier_passes_without_raising(self) -> None:
        os.environ[SAMPLING_WORKER_INIT_TIMEOUT_ENV] = "5"
        producer = self._producer_with_workers([])
        barrier = std_mp.get_context("spawn").Barrier(1)

        producer._wait_for_workers_at_barrier(barrier)


if __name__ == "__main__":
    absltest.main()
