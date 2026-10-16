import threading
import time
from unittest import mock

import torch.multiprocessing as mp
from absl.testing import absltest
from graphlearn_torch.distributed.dist_sampling_producer import MpCommand

from gigl.distributed import dist_sampling_producer
from gigl.distributed.dist_sampling_producer import DistSamplingProducer
from gigl.distributed.graph_store import shared_dist_sampling_producer
from gigl.distributed.graph_store.shared_dist_sampling_producer import (
    SharedDistSamplingBackend,
    SharedMpCommand,
)
from tests.test_assets.test_case import TestCase

_NUM_WORKERS = 4
_GRACE_PERIOD_SECONDS = 1.0


def _exit_on_stop(task_queue, ready) -> None:
    ready.set()
    command, _ = task_queue.get()
    assert command in (MpCommand.STOP, SharedMpCommand.STOP)


def _ignore_stop(task_queue, ready) -> None:
    ready.set()
    # Like a sampling worker blocked sending to a full channel: it never reads the queue.
    time.sleep(600)


def _producer_with_workers(target) -> DistSamplingProducer:
    # Only the worker bookkeeping that ``shutdown`` touches is populated.
    context = mp.get_context("spawn")
    producer = DistSamplingProducer.__new__(DistSamplingProducer)
    producer._shutdown = False
    producer._task_queues = [context.Queue(1) for _ in range(_NUM_WORKERS)]
    ready_events = [context.Event() for _ in range(_NUM_WORKERS)]
    producer._workers = [
        context.Process(target=target, args=(task_queue, ready), daemon=True)
        for task_queue, ready in zip(producer._task_queues, ready_events)
    ]
    for worker in producer._workers:
        worker.start()
    # Spawned workers take seconds to import; shutdown's grace period must not include that.
    for ready in ready_events:
        assert ready.wait(timeout=120)
    return producer


def _shared_backend_with_workers(target) -> SharedDistSamplingBackend:
    backend = SharedDistSamplingBackend.__new__(SharedDistSamplingBackend)
    producer = _producer_with_workers(target)
    backend._shutdown = False
    backend._lock = threading.RLock()
    backend._task_queues = producer._task_queues
    backend._workers = producer._workers
    backend._event_queue = None
    return backend


class DistSamplingProducerShutdownTest(TestCase):
    def test_workers_that_stop_exit_without_being_terminated(self) -> None:
        producer = _producer_with_workers(_exit_on_stop)
        producer.shutdown()
        producer.shutdown()  # a second call is a no-op
        for worker in producer._workers:
            worker.join(timeout=30)
        self.assertEqual([w.exitcode for w in producer._workers], [0] * _NUM_WORKERS)

    def test_unresponsive_workers_share_one_grace_period(self) -> None:
        producer = _producer_with_workers(_ignore_stop)
        with mock.patch.object(
            dist_sampling_producer, "MP_STATUS_CHECK_INTERVAL", _GRACE_PERIOD_SECONDS
        ):
            start = time.monotonic()
            producer.shutdown()
            elapsed = time.monotonic() - start
        for worker in producer._workers:
            worker.join(timeout=30)
        self.assertFalse(any(w.is_alive() for w in producer._workers))
        # A per-worker wait would take _NUM_WORKERS * _GRACE_PERIOD_SECONDS.
        self.assertLess(elapsed, 2 * _GRACE_PERIOD_SECONDS)

    def test_shared_backend_unresponsive_workers_share_one_grace_period(self) -> None:
        backend = _shared_backend_with_workers(_ignore_stop)
        with mock.patch.object(
            shared_dist_sampling_producer,
            "MP_STATUS_CHECK_INTERVAL",
            _GRACE_PERIOD_SECONDS,
        ):
            start = time.monotonic()
            backend.shutdown()
            elapsed = time.monotonic() - start
        for worker in backend._workers:
            worker.join(timeout=30)
        self.assertFalse(any(w.is_alive() for w in backend._workers))
        self.assertLess(elapsed, 2 * _GRACE_PERIOD_SECONDS)


if __name__ == "__main__":
    absltest.main()
