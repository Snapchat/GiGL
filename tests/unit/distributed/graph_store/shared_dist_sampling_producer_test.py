import contextlib
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from typing import cast
from unittest.mock import MagicMock, patch

import torch
import torch.multiprocessing as mp
from graphlearn_torch.channel import QueueTimeoutError
from graphlearn_torch.sampler import NodeSamplerInput, SamplingConfig, SamplingType

from gigl.distributed.graph_store.shared_dist_sampling_producer import (
    EPOCH_DONE_EVENT,
    ActiveEpochState,
    CommandPayload,
    RegisterInputCmd,
    SharedDistSamplingBackend,
    SharedMpCommand,
    StartEpochCmd,
    _compute_num_batches,
    _compute_worker_seeds_ranges,
    _epoch_batch_indices,
    _shared_sampling_worker_loop,
)
from gigl.distributed.sampler_options import KHopNeighborSamplerOptions
from tests.test_assets.test_case import TestCase


def _make_sampling_config(*, shuffle: bool = False) -> SamplingConfig:
    return SamplingConfig(
        sampling_type=SamplingType.NODE,
        num_neighbors=[2],
        batch_size=2,
        shuffle=shuffle,
        drop_last=False,
        with_edge=True,
        collect_features=True,
        with_neg=False,
        with_weight=False,
        edge_dir="out",
        seed=1234,
    )


class _FakeProcess:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.daemon = False

    def start(self) -> None:
        return None

    def join(self, timeout: float | None = None) -> None:
        return None

    def is_alive(self) -> bool:
        return False

    def terminate(self) -> None:
        return None


class _FakeMpContext:
    def Barrier(self, parties: int) -> MagicMock:
        return MagicMock(wait=MagicMock())

    def Queue(self, maxsize: int = 0) -> MagicMock:
        return MagicMock()

    def Process(self, *args: object, **kwargs: object) -> _FakeProcess:
        return _FakeProcess(*args, **kwargs)


class _FakeOutputChannel:
    def __init__(self) -> None:
        self._messages: list[object] = []
        self.drained_event = threading.Event()

    def send(self, msg: object) -> None:
        self._messages.append(msg)

    def recv(self, timeout_ms: int | None = None, **_: object) -> object:
        # Match graphlearn_torch.channel.shm_channel.ShmChannel semantics:
        # timeout_ms=0 (or None) blocks indefinitely; positive timeout_ms
        # raises QueueTimeoutError when the queue is empty.
        if not self._messages:
            if timeout_ms is None or timeout_ms <= 0:
                raise AssertionError(
                    "_FakeOutputChannel.recv called with blocking timeout on "
                    "empty channel — production code is about to deadlock."
                )
            raise QueueTimeoutError("Timeout: Queue is empty.")
        self.drained_event.set()
        return self._messages.pop(0)


class _DeferredFakeSampler:
    def __init__(self, channel: _FakeOutputChannel) -> None:
        self.channel = channel
        self.sample_called = threading.Event()
        self.wait_all_called = threading.Event()
        self.callback_returned = threading.Event()
        self.callbacks: list[Callable[[object], None]] = []

    def start_loop(self) -> None:
        return None

    def wait_all(self) -> None:
        self.wait_all_called.set()
        if self.callbacks and not self.callback_returned.wait(timeout=0.2):
            raise TimeoutError("wait_all called before sampler callback returned")

    def shutdown_loop(self) -> None:
        return None

    def sample_from_nodes(
        self, _sampler_input: object, callback: Callable[[object], None]
    ) -> None:
        self.channel.send({"seed": torch.tensor([1], dtype=torch.long)})
        self.callbacks.append(callback)
        self.sample_called.set()


class DistSamplingProducerTest(TestCase):
    def test_compute_num_batches(self) -> None:
        self.assertEqual(_compute_num_batches(0, 2, False), 0)
        self.assertEqual(_compute_num_batches(1, 2, True), 0)
        self.assertEqual(_compute_num_batches(1, 2, False), 1)
        self.assertEqual(_compute_num_batches(5, 2, False), 3)
        self.assertEqual(_compute_num_batches(5, 2, True), 2)

    def test_epoch_batch_indices(self) -> None:
        active_state = ActiveEpochState(
            channel_id=0,
            epoch=0,
            input_len=6,
            batch_size=2,
            drop_last=False,
            seeds_index=torch.arange(6),
            total_batches=3,
            submitted_batches=1,
            cancelled=False,
        )
        result = _epoch_batch_indices(active_state)
        assert result is not None
        self.assert_tensor_equality(result, torch.tensor([2, 3]))

    def test_compute_worker_seeds_ranges(self) -> None:
        self.assertEqual(
            _compute_worker_seeds_ranges(input_len=7, batch_size=2, num_workers=3),
            [(0, 2), (2, 4), (4, 7)],
        )

    @patch("gigl.distributed.graph_store.shared_dist_sampling_producer.get_context")
    @patch("gigl.distributed.graph_store.shared_dist_sampling_producer.mp.get_context")
    def test_init_backend_prepares_worker_options(
        self,
        mock_get_mp_context: MagicMock,
        mock_get_context: MagicMock,
    ) -> None:
        worker_options = MagicMock()
        worker_options.num_workers = 2
        worker_options.worker_concurrency = 1
        mock_get_context.return_value = MagicMock(
            is_server=MagicMock(return_value=True)
        )
        mock_get_mp_context.return_value = _FakeMpContext()
        backend = SharedDistSamplingBackend(
            data=MagicMock(),
            worker_options=worker_options,
            sampling_config=_make_sampling_config(),
            sampler_options=KHopNeighborSamplerOptions(num_neighbors=[2]),
            degree_tensors=None,
        )

        backend.init_backend()

        worker_options._assign_worker_devices.assert_called_once()
        worker_options._set_worker_ranks.assert_called_once_with(
            mock_get_context.return_value
        )
        self.assertEqual(len(backend._task_queues), 2)
        self.assertEqual(len(backend._workers), 2)
        self.assertTrue(backend._initialized)

    def test_start_new_epoch_sampling_shuffle_refreshes_per_epoch(self) -> None:
        worker_options = MagicMock()
        worker_options.num_workers = 2
        worker_options.worker_concurrency = 1
        backend = SharedDistSamplingBackend(
            data=MagicMock(),
            worker_options=worker_options,
            sampling_config=_make_sampling_config(shuffle=True),
            sampler_options=KHopNeighborSamplerOptions(num_neighbors=[2]),
            degree_tensors=None,
        )
        backend._initialized = True
        recorded: list[tuple[int, SharedMpCommand, object]] = []

        def _record_command(
            worker_rank: int,
            command: SharedMpCommand,
            payload: CommandPayload,
        ) -> None:
            recorded.append((worker_rank, command, payload))

        backend._enqueue_worker_command = _record_command  # type: ignore[method-assign]  # ty: ignore[invalid-assignment]

        channel = MagicMock()
        input_tensor = torch.arange(6, dtype=torch.long)
        backend.register_input(
            channel_id=1,
            worker_key="loader_a_compute_rank_0",
            sampler_input=NodeSamplerInput(node=input_tensor.clone()),
            sampling_config=_make_sampling_config(shuffle=True),
            channel=channel,
        )
        backend.register_input(
            channel_id=2,
            worker_key="loader_b_compute_rank_0",
            sampler_input=NodeSamplerInput(node=input_tensor.clone()),
            sampling_config=_make_sampling_config(shuffle=True),
            channel=channel,
        )

        def _collect_epoch_indices(channel_id: int, epoch: int) -> torch.Tensor:
            recorded.clear()
            backend.start_new_epoch_sampling(channel_id, epoch)
            worker_payloads = {
                worker_rank: cast(StartEpochCmd, payload).seeds_index
                for worker_rank, command, payload in recorded
                if command == SharedMpCommand.START_EPOCH
            }
            assert all(
                seed_index is not None for seed_index in worker_payloads.values()
            )
            return torch.cat(
                [
                    cast(torch.Tensor, worker_payloads[worker_rank])
                    for worker_rank in sorted(worker_payloads)
                ]
            )

        channel_1_epoch_0 = _collect_epoch_indices(1, 0)
        channel_2_epoch_0 = _collect_epoch_indices(2, 0)
        channel_1_epoch_1 = _collect_epoch_indices(1, 1)

        self.assert_tensor_equality(channel_1_epoch_0, channel_2_epoch_0)
        self.assertNotEqual(
            channel_1_epoch_0.tolist(),
            channel_1_epoch_1.tolist(),
        )

    def test_describe_channel_reports_completed_workers(self) -> None:
        worker_options = MagicMock()
        worker_options.num_workers = 2
        worker_options.worker_concurrency = 1
        backend = SharedDistSamplingBackend(
            data=MagicMock(),
            worker_options=worker_options,
            sampling_config=_make_sampling_config(),
            sampler_options=KHopNeighborSamplerOptions(num_neighbors=[2]),
            degree_tensors=None,
        )
        backend._initialized = True
        backend._event_queue = cast(mp.Queue, queue.Queue())
        backend._channel_input_sizes[1] = [4, 2]
        backend._channel_epoch[1] = 3
        cast(queue.Queue, backend._event_queue).put((EPOCH_DONE_EVENT, 1, 3, 0))

        description = backend.describe_channel(1)

        self.assertEqual(description["epoch"], 3)
        self.assertEqual(description["input_sizes"], [4, 2])
        self.assertEqual(description["completed_workers"], 1)

    def test_unregister_input_is_fire_and_forget(self) -> None:
        worker_options = MagicMock()
        worker_options.num_workers = 2
        worker_options.worker_concurrency = 1
        backend = SharedDistSamplingBackend(
            data=MagicMock(),
            worker_options=worker_options,
            sampling_config=_make_sampling_config(),
            sampler_options=KHopNeighborSamplerOptions(num_neighbors=[2]),
            degree_tensors=None,
        )
        backend._initialized = True
        backend._event_queue = cast(mp.Queue, queue.Queue())
        backend._channel_sampling_config[1] = _make_sampling_config()
        backend._channel_input_sizes[1] = [2, 2]
        backend._channel_worker_seeds_ranges[1] = [(0, 2), (2, 4)]
        backend._channel_shuffle_generators[1] = None
        backend._channel_epoch[1] = 0

        commands: list[tuple[int, SharedMpCommand, object]] = []

        def enqueue_worker_command(
            worker_rank: int,
            command: SharedMpCommand,
            payload: object,
        ) -> None:
            commands.append((worker_rank, command, payload))

        backend._enqueue_worker_command = enqueue_worker_command  # type: ignore[method-assign]  # ty: ignore[invalid-assignment]

        # Returns without any worker acknowledgement: there is no sync wait.
        backend.unregister_input(1)

        self.assertEqual(
            commands,
            [
                (0, SharedMpCommand.UNREGISTER_INPUT, 1),
                (1, SharedMpCommand.UNREGISTER_INPUT, 1),
            ],
        )
        self.assertNotIn(1, backend._channel_sampling_config)

    @patch("gigl.distributed.graph_store.shared_dist_sampling_producer.shutdown_rpc")
    @patch("gigl.distributed.graph_store.shared_dist_sampling_producer.init_rpc")
    @patch(
        "gigl.distributed.graph_store.shared_dist_sampling_producer.init_worker_group"
    )
    @patch(
        "gigl.distributed.graph_store.shared_dist_sampling_producer._set_worker_signal_handlers"
    )
    @patch(
        "gigl.distributed.graph_store.shared_dist_sampling_producer.torch.set_num_threads"
    )
    @patch(
        "gigl.distributed.graph_store.shared_dist_sampling_producer.create_dist_sampler"
    )
    def test_worker_unregister_drains_buffered_output_and_waits_for_completion(
        self,
        mock_create_dist_sampler: MagicMock,
        _mock_set_num_threads: MagicMock,
        _mock_signal_handlers: MagicMock,
        _mock_init_worker_group: MagicMock,
        _mock_init_rpc: MagicMock,
        _mock_shutdown_rpc: MagicMock,
    ) -> None:
        worker_options = MagicMock()
        worker_options.worker_world_size = 1
        worker_options.worker_ranks = [0]
        worker_options.use_all2all = False
        worker_options.num_rpc_threads = 1
        worker_options.worker_devices = [torch.device("cpu")]
        worker_options.master_addr = "127.0.0.1"
        worker_options.master_port = 12345
        worker_options.rpc_timeout = 30
        worker_options.worker_concurrency = 2
        output_channel = _FakeOutputChannel()
        fake_sampler = _DeferredFakeSampler(output_channel)
        mock_create_dist_sampler.return_value = fake_sampler
        task_queue: queue.Queue[tuple[SharedMpCommand, object]] = queue.Queue()
        event_queue: queue.Queue[tuple[object, ...]] = queue.Queue()
        barrier = MagicMock(wait=MagicMock())
        data = MagicMock(num_partitions=1)
        sampling_config = _make_sampling_config()
        channel_id = 7

        worker_thread = threading.Thread(
            target=_shared_sampling_worker_loop,
            args=(
                0,
                data,
                worker_options,
                task_queue,
                event_queue,
                barrier,
                KHopNeighborSamplerOptions(num_neighbors=[2]),
                None,
            ),
        )
        worker_thread.start()
        task_queue.put(
            (
                SharedMpCommand.REGISTER_INPUT,
                RegisterInputCmd(
                    channel_id=channel_id,
                    worker_key="loader_a_compute_rank_0",
                    sampler_input=NodeSamplerInput(node=torch.arange(2)),
                    sampling_config=sampling_config,
                    channel=output_channel,
                ),
            )
        )
        task_queue.put(
            (
                SharedMpCommand.START_EPOCH,
                StartEpochCmd(
                    channel_id=channel_id,
                    epoch=0,
                    seeds_index=torch.arange(2),
                ),
            )
        )
        self.assertTrue(fake_sampler.sample_called.wait(timeout=5.0))

        task_queue.put((SharedMpCommand.UNREGISTER_INPUT, channel_id))
        self.assertTrue(output_channel.drained_event.wait(timeout=5.0))
        self.assertTrue(event_queue.empty())

        callback = fake_sampler.callbacks[0]
        callback_errors: list[BaseException] = []

        def run_callback() -> None:
            try:
                callback(None)
            except BaseException as exc:
                callback_errors.append(exc)
            finally:
                fake_sampler.callback_returned.set()

        callback_thread = threading.Thread(target=run_callback)
        callback_thread.start()
        callback_thread.join(timeout=5.0)
        self.assertFalse(callback_thread.is_alive())
        self.assertEqual(callback_errors, [])
        self.assertTrue(fake_sampler.wait_all_called.wait(timeout=5.0))

        epoch_done_event = event_queue.get(timeout=5.0)
        self.assertEqual(epoch_done_event[0], EPOCH_DONE_EVENT)
        self.assertTrue(event_queue.empty())

        task_queue.put((SharedMpCommand.STOP, None))
        worker_thread.join(timeout=5.0)
        self.assertFalse(worker_thread.is_alive())


class _BoundedBlockingChannel:
    """Bounded channel fake whose sends block when the consumer stops draining.

    Positive receive timeouts match ``ShmChannel``; a blocking receive on an
    empty test channel asserts instead of hanging the suite.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._buffer: deque[object] = deque()
        self._cond = threading.Condition()
        self.total_sent = 0
        self.total_received = 0

    def send(self, msg: object) -> None:
        with self._cond:
            while len(self._buffer) >= self._capacity:
                self._cond.wait()
            self._buffer.append(msg)
            self.total_sent += 1
            self._cond.notify_all()

    def recv(self, timeout_ms: int | None = None, **_: object) -> object:
        with self._cond:
            if not self._buffer:
                if timeout_ms is None or timeout_ms <= 0:
                    raise AssertionError(
                        "_BoundedBlockingChannel.recv called with blocking timeout "
                        "on empty channel -- production code is about to deadlock."
                    )
                deadline = time.monotonic() + timeout_ms / 1000.0
                while not self._buffer:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise QueueTimeoutError("Timeout: Queue is empty.")
                    self._cond.wait(remaining)
            msg = self._buffer.popleft()
            self.total_received += 1
            self._cond.notify_all()
            return msg


class _GltOrderFakeSampler:
    """Reproduce GLT's semaphore, send, callback, and release ordering.

    ``sample_from_nodes`` acquires the semaphore on the scheduler thread. The
    worker thread then sends, invokes the callback, and releases the semaphore.
    This callback-before-release order is load-bearing because completion can
    re-enqueue a channel just before GLT makes the slot available.
    """

    def __init__(
        self, output_channel: _BoundedBlockingChannel, concurrency: int
    ) -> None:
        self._channel = output_channel
        self._sem = threading.BoundedSemaphore(concurrency)
        self._threads: list[threading.Thread] = []
        self._threads_lock = threading.Lock()
        self.submit_count = 0

    def start_loop(self) -> None: ...

    def wait_all(self) -> None:
        with self._threads_lock:
            threads = list(self._threads)
        for thread in threads:
            thread.join()

    def shutdown_loop(self) -> None:
        self.wait_all()

    def sample_from_nodes(
        self, _sampler_input: object, callback: Callable[[object], None]
    ) -> None:
        with self._threads_lock:
            self.submit_count += 1
        self._sem.acquire()

        def _coroutine() -> None:
            try:
                self._channel.send({"seed": torch.tensor([1], dtype=torch.long)})
                callback(None)
            finally:
                self._sem.release()

        thread = threading.Thread(target=_coroutine, daemon=True)
        with self._threads_lock:
            self._threads.append(thread)
        thread.start()


class _CountingTaskQueue:
    """Count blocking and non-blocking gets to detect park-only busy-spinning."""

    def __init__(self) -> None:
        self._queue: queue.Queue[tuple[SharedMpCommand, object]] = queue.Queue()
        self._lock = threading.Lock()
        self.blocking_get_calls = 0
        self.nowait_get_calls = 0

    def put(self, item: tuple[SharedMpCommand, object]) -> None:
        self._queue.put(item)

    def get_nowait(self) -> tuple[SharedMpCommand, object]:
        with self._lock:
            self.nowait_get_calls += 1
        return self._queue.get_nowait()

    def get(self, timeout: float | None = None) -> tuple[SharedMpCommand, object]:
        with self._lock:
            self.blocking_get_calls += 1
        return self._queue.get(timeout=timeout)


class StallFixWorkerLoopTest(TestCase):
    """Exercise scheduler liveness with paused and active consumers.

    A paused consumer fills its channel and holds sampler slots; the shared
    scheduler must continue serving active channels and commands.
    """

    @staticmethod
    def _make_worker_options(worker_concurrency: int) -> MagicMock:
        worker_options = MagicMock()
        worker_options.worker_world_size = 1
        worker_options.worker_ranks = [0]
        worker_options.use_all2all = False
        worker_options.num_rpc_threads = 1
        worker_options.worker_devices = [torch.device("cpu")]
        worker_options.master_addr = "127.0.0.1"
        worker_options.master_port = 12345
        worker_options.rpc_timeout = 30
        worker_options.worker_concurrency = worker_concurrency
        return worker_options

    @staticmethod
    @contextlib.contextmanager
    def _draining(*channels: _BoundedBlockingChannel) -> Iterator[None]:
        """Drain channels during teardown so blocked sends cannot hang the worker."""
        stop = threading.Event()

        def _drain() -> None:
            while not stop.is_set():
                for channel in channels:
                    try:
                        channel.recv(timeout_ms=20)
                    except QueueTimeoutError:
                        continue

        thread = threading.Thread(target=_drain, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=5.0)

    def _run_paused_and_active(
        self,
        *,
        mock_create_dist_sampler: MagicMock,
    ) -> None:
        worker_concurrency = 2
        # B never drains and saturates; A drains continuously. The scheduler must
        # keep A runnable while B is parked.
        channel_a = _BoundedBlockingChannel(capacity=4)
        channel_b = _BoundedBlockingChannel(capacity=worker_concurrency)

        mock_create_dist_sampler.side_effect = lambda **kwargs: _GltOrderFakeSampler(
            kwargs["channel"], worker_concurrency
        )

        worker_options = self._make_worker_options(worker_concurrency)
        task_queue: queue.Queue[tuple[SharedMpCommand, object]] = queue.Queue()
        event_queue: queue.Queue[tuple[object, ...]] = queue.Queue()
        barrier = MagicMock(wait=MagicMock())
        data = MagicMock(num_partitions=1)
        sampling_config = _make_sampling_config()

        channel_a_id, channel_b_id = 1, 2
        active_batches, paused_batches = 6, 20  # batch_size 2 -> 12 vs 40 seeds

        stop_consumer = threading.Event()

        def _consume_active() -> None:
            while not stop_consumer.is_set():
                try:
                    channel_a.recv(timeout_ms=20)
                except QueueTimeoutError:
                    continue

        consumer_thread = threading.Thread(target=_consume_active, daemon=True)
        consumer_thread.start()

        for channel_id, channel, node_len in (
            (channel_a_id, channel_a, active_batches * 2),
            (channel_b_id, channel_b, paused_batches * 2),
        ):
            task_queue.put(
                (
                    SharedMpCommand.REGISTER_INPUT,
                    RegisterInputCmd(
                        channel_id=channel_id,
                        worker_key=f"loader_compute_rank_{channel_id}",
                        sampler_input=NodeSamplerInput(node=torch.arange(node_len)),
                        sampling_config=sampling_config,
                        channel=channel,
                    ),
                )
            )
        # Start the paused channel first so it begins saturating.
        task_queue.put(
            (
                SharedMpCommand.START_EPOCH,
                StartEpochCmd(
                    channel_id=channel_b_id,
                    epoch=0,
                    seeds_index=torch.arange(paused_batches * 2),
                ),
            )
        )
        task_queue.put(
            (
                SharedMpCommand.START_EPOCH,
                StartEpochCmd(
                    channel_id=channel_a_id,
                    epoch=0,
                    seeds_index=torch.arange(active_batches * 2),
                ),
            )
        )

        worker_thread = threading.Thread(
            target=_shared_sampling_worker_loop,
            args=(
                0,
                data,
                worker_options,
                task_queue,
                event_queue,
                barrier,
                KHopNeighborSamplerOptions(num_neighbors=[2]),
                None,
            ),
        )
        worker_thread.start()
        try:
            # A must complete while B remains parked behind its paused consumer.
            done_epoch_0 = event_queue.get(timeout=10.0)
            self.assertEqual(done_epoch_0, (EPOCH_DONE_EVENT, channel_a_id, 0, 0))

            # Starting another A epoch after B saturates verifies command
            # draining remains live.
            task_queue.put(
                (
                    SharedMpCommand.START_EPOCH,
                    StartEpochCmd(
                        channel_id=channel_a_id,
                        epoch=1,
                        seeds_index=torch.arange(active_batches * 2),
                    ),
                )
            )
            done_epoch_1 = event_queue.get(timeout=10.0)
            self.assertEqual(done_epoch_1, (EPOCH_DONE_EVENT, channel_a_id, 1, 0))

            self.assertEqual(channel_b.total_received, 0)
            self.assertGreaterEqual(channel_a.total_received, active_batches)

            task_queue.put((SharedMpCommand.UNREGISTER_INPUT, channel_b_id))
        finally:
            # Release B's blocked sends before waiting for STOP.
            with self._draining(channel_b):
                task_queue.put((SharedMpCommand.STOP, None))
                worker_thread.join(timeout=10.0)
            stop_consumer.set()
            consumer_thread.join(timeout=5.0)

        self.assertFalse(worker_thread.is_alive())

    @patch("gigl.distributed.graph_store.shared_dist_sampling_producer.shutdown_rpc")
    @patch("gigl.distributed.graph_store.shared_dist_sampling_producer.init_rpc")
    @patch(
        "gigl.distributed.graph_store.shared_dist_sampling_producer.init_worker_group"
    )
    @patch(
        "gigl.distributed.graph_store.shared_dist_sampling_producer._set_worker_signal_handlers"
    )
    @patch(
        "gigl.distributed.graph_store.shared_dist_sampling_producer.torch.set_num_threads"
    )
    @patch(
        "gigl.distributed.graph_store.shared_dist_sampling_producer.create_dist_sampler"
    )
    def test_paused_consumer_does_not_stall_active_channel_unweighted(
        self,
        mock_create_dist_sampler: MagicMock,
        _mock_set_num_threads: MagicMock,
        _mock_signal_handlers: MagicMock,
        _mock_init_worker_group: MagicMock,
        _mock_init_rpc: MagicMock,
        _mock_shutdown_rpc: MagicMock,
    ) -> None:
        self._run_paused_and_active(
            mock_create_dist_sampler=mock_create_dist_sampler,
        )

    @patch("gigl.distributed.graph_store.shared_dist_sampling_producer.shutdown_rpc")
    @patch("gigl.distributed.graph_store.shared_dist_sampling_producer.init_rpc")
    @patch(
        "gigl.distributed.graph_store.shared_dist_sampling_producer.init_worker_group"
    )
    @patch(
        "gigl.distributed.graph_store.shared_dist_sampling_producer._set_worker_signal_handlers"
    )
    @patch(
        "gigl.distributed.graph_store.shared_dist_sampling_producer.torch.set_num_threads"
    )
    @patch(
        "gigl.distributed.graph_store.shared_dist_sampling_producer.create_dist_sampler"
    )
    def test_unregister_while_parked_tears_down_cleanly(
        self,
        mock_create_dist_sampler: MagicMock,
        _mock_set_num_threads: MagicMock,
        _mock_signal_handlers: MagicMock,
        _mock_init_worker_group: MagicMock,
        _mock_init_rpc: MagicMock,
        _mock_shutdown_rpc: MagicMock,
    ) -> None:
        worker_concurrency = 2
        channel_b = _BoundedBlockingChannel(capacity=worker_concurrency)
        created_samplers: list[_GltOrderFakeSampler] = []

        def _make_sampler(**kwargs: object) -> _GltOrderFakeSampler:
            sampler = _GltOrderFakeSampler(
                cast(_BoundedBlockingChannel, kwargs["channel"]), worker_concurrency
            )
            created_samplers.append(sampler)
            return sampler

        mock_create_dist_sampler.side_effect = _make_sampler

        worker_options = self._make_worker_options(worker_concurrency)
        task_queue = _CountingTaskQueue()
        event_queue: queue.Queue[tuple[object, ...]] = queue.Queue()
        barrier = MagicMock(wait=MagicMock())
        data = MagicMock(num_partitions=1)
        sampling_config = _make_sampling_config()
        channel_b_id, paused_batches = 5, 20

        task_queue.put(
            (
                SharedMpCommand.REGISTER_INPUT,
                RegisterInputCmd(
                    channel_id=channel_b_id,
                    worker_key="loader_compute_rank_5",
                    sampler_input=NodeSamplerInput(
                        node=torch.arange(paused_batches * 2)
                    ),
                    sampling_config=sampling_config,
                    channel=channel_b,
                ),
            )
        )
        task_queue.put(
            (
                SharedMpCommand.START_EPOCH,
                StartEpochCmd(
                    channel_id=channel_b_id,
                    epoch=0,
                    seeds_index=torch.arange(paused_batches * 2),
                ),
            )
        )

        worker_thread = threading.Thread(
            target=_shared_sampling_worker_loop,
            args=(
                0,
                data,
                worker_options,
                task_queue,
                event_queue,
                barrier,
                KHopNeighborSamplerOptions(num_neighbors=[2]),
                None,
            ),
        )
        worker_thread.start()
        try:
            # B parks after capacity completed batches fill the buffer and
            # worker_concurrency additional sends remain in flight.
            expected_parked_submits = worker_concurrency + channel_b._capacity
            deadline = time.monotonic() + 10.0
            sampler: _GltOrderFakeSampler | None = None
            while time.monotonic() < deadline:
                if created_samplers:
                    sampler = created_samplers[0]
                    if sampler.submit_count >= expected_parked_submits:
                        break
                time.sleep(0.01)
            self.assertIsNotNone(sampler)
            assert sampler is not None
            self.assertGreaterEqual(sampler.submit_count, expected_parked_submits)

            # A park-only scheduler should enter Phase 3 rather than submit or
            # spin. Blocking gets prove the idle wait runs; excessive
            # non-blocking gets would indicate a busy loop.
            blocking_gets_before = task_queue.blocking_get_calls
            nowait_gets_before = task_queue.nowait_get_calls
            time.sleep(0.3)
            blocking_gets = task_queue.blocking_get_calls - blocking_gets_before
            nowait_gets = task_queue.nowait_get_calls - nowait_gets_before

            self.assertEqual(sampler.submit_count, expected_parked_submits)
            self.assertEqual(channel_b.total_received, 0)
            self.assertGreaterEqual(blocking_gets, 1)
            self.assertLess(nowait_gets, 100)

            task_queue.put((SharedMpCommand.UNREGISTER_INPUT, channel_b_id))
        finally:
            # Release blocked sends before waiting for STOP.
            with self._draining(channel_b):
                task_queue.put((SharedMpCommand.STOP, None))
                worker_thread.join(timeout=10.0)

        self.assertFalse(worker_thread.is_alive())
        for thread in sampler._threads:
            self.assertFalse(thread.is_alive())

    @patch("gigl.distributed.graph_store.shared_dist_sampling_producer.shutdown_rpc")
    @patch("gigl.distributed.graph_store.shared_dist_sampling_producer.init_rpc")
    @patch(
        "gigl.distributed.graph_store.shared_dist_sampling_producer.init_worker_group"
    )
    @patch(
        "gigl.distributed.graph_store.shared_dist_sampling_producer._set_worker_signal_handlers"
    )
    @patch(
        "gigl.distributed.graph_store.shared_dist_sampling_producer.torch.set_num_threads"
    )
    @patch(
        "gigl.distributed.graph_store.shared_dist_sampling_producer.create_dist_sampler"
    )
    def test_parked_channel_resumes_and_completes_when_consumer_drains(
        self,
        mock_create_dist_sampler: MagicMock,
        _mock_set_num_threads: MagicMock,
        _mock_signal_handlers: MagicMock,
        _mock_init_worker_group: MagicMock,
        _mock_init_rpc: MagicMock,
        _mock_shutdown_rpc: MagicMock,
    ) -> None:
        """Verify completion re-enqueues a parked channel after consumption resumes."""
        worker_concurrency = 2
        channel = _BoundedBlockingChannel(capacity=worker_concurrency)
        created_samplers: list[_GltOrderFakeSampler] = []

        def _make_sampler(**kwargs: object) -> _GltOrderFakeSampler:
            sampler = _GltOrderFakeSampler(
                cast(_BoundedBlockingChannel, kwargs["channel"]), worker_concurrency
            )
            created_samplers.append(sampler)
            return sampler

        mock_create_dist_sampler.side_effect = _make_sampler

        worker_options = self._make_worker_options(worker_concurrency)
        task_queue: queue.Queue[tuple[SharedMpCommand, object]] = queue.Queue()
        event_queue: queue.Queue[tuple[object, ...]] = queue.Queue()
        barrier = MagicMock(wait=MagicMock())
        data = MagicMock(num_partitions=1)
        sampling_config = _make_sampling_config()
        channel_id, total_batches = 7, 12  # batch_size 2 -> 24 seeds

        task_queue.put(
            (
                SharedMpCommand.REGISTER_INPUT,
                RegisterInputCmd(
                    channel_id=channel_id,
                    worker_key="loader_compute_rank_7",
                    sampler_input=NodeSamplerInput(
                        node=torch.arange(total_batches * 2)
                    ),
                    sampling_config=sampling_config,
                    channel=channel,
                ),
            )
        )
        task_queue.put(
            (
                SharedMpCommand.START_EPOCH,
                StartEpochCmd(
                    channel_id=channel_id,
                    epoch=0,
                    seeds_index=torch.arange(total_batches * 2),
                ),
            )
        )

        # Pause the consumer until the sampler reaches its in-flight cap.
        resume = threading.Event()
        stop_consumer = threading.Event()

        def _consume() -> None:
            resume.wait()
            while not stop_consumer.is_set():
                try:
                    channel.recv(timeout_ms=20)
                except QueueTimeoutError:
                    continue

        consumer_thread = threading.Thread(target=_consume, daemon=True)
        consumer_thread.start()

        worker_thread = threading.Thread(
            target=_shared_sampling_worker_loop,
            args=(
                0,
                data,
                worker_options,
                task_queue,
                event_queue,
                barrier,
                KHopNeighborSamplerOptions(num_neighbors=[2]),
                None,
            ),
        )
        worker_thread.start()
        try:
            # The channel parks after capacity completed batches fill the buffer
            # and worker_concurrency additional sends remain in flight.
            expected_parked_submits = worker_concurrency + channel._capacity
            deadline = time.monotonic() + 10.0
            sampler: _GltOrderFakeSampler | None = None
            while time.monotonic() < deadline:
                if created_samplers:
                    sampler = created_samplers[0]
                    if sampler.submit_count >= expected_parked_submits:
                        break
                time.sleep(0.01)
            self.assertIsNotNone(sampler)
            assert sampler is not None

            # Verify the channel is parked, not merely progressing slowly.
            time.sleep(0.2)
            self.assertEqual(sampler.submit_count, expected_parked_submits)
            self.assertEqual(channel.total_received, 0)
            self.assertLess(sampler.submit_count, total_batches)

            # Draining completes a send, whose callback re-enqueues the channel.
            resume.set()
            done = event_queue.get(timeout=10.0)
            self.assertEqual(done, (EPOCH_DONE_EVENT, channel_id, 0, 0))

            self.assertEqual(sampler.submit_count, total_batches)
            # EPOCH_DONE fires when the last batch is buffered; the consumer
            # drains that tail later, so poll rather than race it.
            drain_deadline = time.monotonic() + 5.0
            while (
                channel.total_received < total_batches
                and time.monotonic() < drain_deadline
            ):
                time.sleep(0.01)
            self.assertEqual(channel.total_received, total_batches)
        finally:
            with self._draining(channel):
                task_queue.put((SharedMpCommand.STOP, None))
                worker_thread.join(timeout=10.0)
            stop_consumer.set()
            resume.set()  # unblock the consumer if we failed before resuming
            consumer_thread.join(timeout=5.0)

        self.assertFalse(worker_thread.is_alive())
