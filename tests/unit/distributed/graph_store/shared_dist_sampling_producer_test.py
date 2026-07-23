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
    """Output channel with GLT ``ShmChannel``-like bounded, blocking semantics.

    ``send`` blocks while the buffer is full: a paused consumer never drains, so
    its channel saturates and the sampler coroutines wedge in ``send`` -- exactly
    the condition that must park a channel instead of blocking the shared
    scheduler thread.

    ``recv`` mirrors ``ShmChannel``: a positive ``timeout_ms`` raises
    ``QueueTimeoutError`` on an empty channel, while a non-positive/None timeout
    on an empty channel is a would-be production deadlock and asserts.
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
    """Sampler that reproduces GLT ``ConcurrentEventLoop`` submit/complete order.

    ``sample_from_nodes`` acquires a per-channel ``BoundedSemaphore`` on the
    CALLING (scheduler) thread -- the exact point where GLT's ``add_task`` blocks
    once ``worker_concurrency`` coroutines are already pending -- and only then
    hands the "coroutine" to a background thread that (1) sends the result on the
    bounded channel (blocking while the consumer is paused), (2) fires the
    completion callback, and (3) releases the semaphore, in that order.

    Firing the callback BEFORE the release mirrors ``event_loop.py`` ``on_done``
    (``result``/``callback`` then ``self._sem.release()``), so
    ``completed_batches`` reflects a freed slot at the same instant the real code
    would.  The semaphore acquire on the scheduler thread means that if the
    scheduler ever over-submits past the in-flight cap (i.e. the fix regresses),
    the whole scheduler thread wedges here -- which the tests detect as a stall.
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
        # Acquired on the scheduler thread: blocks here iff the scheduler
        # over-submits past the in-flight cap (the regression this fix prevents).
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
    """Task queue that counts blocking vs non-blocking gets.

    Lets a test distinguish a scheduler idling in Phase 3 (``get`` with a
    timeout, ~one per ``SCHEDULER_TICK_SECS``) from one busy-spinning through the
    phases (``get_nowait`` many times per tick, zero blocking ``get`` calls) --
    the signature of a regression that wrongly reports progress on a park-only
    pump cycle and pins the core at 100% CPU instead of engaging the idle tick.
    """

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
    """GLT-order-faithful regressions for the wait-free scheduler stall fix.

    Each drives the real ``_shared_sampling_worker_loop`` with a paused consumer
    (its channel saturates and its coroutines wedge in ``send``, holding the
    per-channel sampler semaphore) alongside an active consumer, and asserts the
    active channel keeps progressing and commands keep draining -- i.e. the
    single scheduler thread is never parked on the saturated channel.

    Without the in-flight cap the scheduler blocks in ``sample_from_nodes`` on the
    saturated channel's exhausted semaphore, so these ``event_queue.get`` calls
    time out (the pre-fix cross-rank deadlock).
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
        """Continuously drain the given channels for the duration of the block.

        Releases any sends wedged on a full channel so the worker can always
        reach Phase 1 and drain ``STOP``.  Without this, a reverted in-flight cap
        would leave the non-daemon worker blocked in the sampler semaphore, so a
        red test would hang the whole suite instead of failing cleanly.
        """
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
        # channel_a's consumer drains continuously so its coroutines never wedge;
        # channel_b has NO consumer, so it saturates at capacity and its
        # coroutines block in send -- the paused-consumer condition.  Under plain
        # round-robin the scheduler must rotate past channel_b's saturated channel
        # instead of parking the shared thread on it.
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

        # create_dist_sampler picks up each channel by the ``channel`` kwarg, so
        # channel_a's sampler sends to channel_a and channel_b's to channel_b.
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
            # Active channel completes epoch 0 despite channel B's paused
            # consumer: the scheduler rotated past B's saturated channel instead
            # of parking on it.  (Pre-fix: deadlock -> this get times out.)
            done_epoch_0 = event_queue.get(timeout=10.0)
            self.assertEqual(done_epoch_0, (EPOCH_DONE_EVENT, channel_a_id, 0, 0))

            # Commands keep draining while B stays wedged: start a SECOND active
            # epoch AFTER B has saturated and confirm it also completes (proves
            # Phase-1 command draining + Phase-2 pumping never parked).
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

            # The paused channel never streamed to a consumer while the active
            # channel drained two full epochs -- asymmetric progress, no
            # head-of-line blocking.
            self.assertEqual(channel_b.total_received, 0)
            self.assertGreaterEqual(channel_a.total_received, active_batches)

            # Unregister the wedged (parked) channel: draining its buffered output
            # unblocks the stuck sends so cleanup finalizes without hanging.
            task_queue.put((SharedMpCommand.UNREGISTER_INPUT, channel_b_id))
        finally:
            # Drain the paused channel while stopping so the worker can always
            # reach Phase 1 to process STOP -- see ``_draining``.
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
            # The channel parks after submitting exactly capacity (completed +
            # buffered) plus worker_concurrency (wedged in-flight) batches, and no
            # more -- a busy-spinning or non-parking scheduler would keep climbing
            # toward all 20 batches (or wedge the whole scheduler on the exhausted
            # semaphore before reaching this count).
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

            # Give the scheduler ample time to (incorrectly) submit more or
            # busy-spin; parked means it idles in Phase 3 instead.
            blocking_gets_before = task_queue.blocking_get_calls
            nowait_gets_before = task_queue.nowait_get_calls
            time.sleep(0.3)
            blocking_gets = task_queue.blocking_get_calls - blocking_gets_before
            nowait_gets = task_queue.nowait_get_calls - nowait_gets_before

            # It stays put -- no submits past the cap.
            self.assertEqual(sampler.submit_count, expected_parked_submits)
            self.assertEqual(channel_b.total_received, 0)
            # F3 event-wake: Phase 3 now waits on the completion ``wake_event``, NOT on
            # ``task_queue``, so the scheduler issues ZERO blocking ``task_queue.get``
            # calls while parked.  Still NOT busy-spinning: each idle iteration does one
            # Phase-1 ``get_nowait()`` then blocks in ``wake_event.wait()`` for a full
            # tick, so over ~0.3s / 0.05s tick only a handful of ``get_nowait`` calls
            # accrue -- a regression that wrongly set ``made_progress`` on a park-only
            # cycle would skip the wait and burn thousands of ``get_nowait`` at 100% CPU.
            self.assertEqual(blocking_gets, 0)
            self.assertGreaterEqual(nowait_gets, 1)
            self.assertLess(nowait_gets, 100)

            # Unregister while parked: must drain buffered output, unblock the
            # wedged sends, and finalize cleanup without hanging.
            task_queue.put((SharedMpCommand.UNREGISTER_INPUT, channel_b_id))
        finally:
            # Drain the wedged channel while stopping so the worker can always
            # reach Phase 1 to process STOP -- see ``_draining``.
            with self._draining(channel_b):
                task_queue.put((SharedMpCommand.STOP, None))
                worker_thread.join(timeout=10.0)

        self.assertFalse(worker_thread.is_alive())
        # Every wedged coroutine was released during teardown.
        for thread in sampler._threads:
            self.assertFalse(thread.is_alive())

    @patch(
        "gigl.distributed.graph_store.shared_dist_sampling_producer.SCHEDULER_TICK_SECS",
        30.0,
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
    def test_completion_event_wakes_parked_channel_without_tick(
        self,
        mock_create_dist_sampler: MagicMock,
        _mock_set_num_threads: MagicMock,
        _mock_signal_handlers: MagicMock,
        _mock_init_worker_group: MagicMock,
        _mock_init_rpc: MagicMock,
        _mock_shutdown_rpc: MagicMock,
    ) -> None:
        """A batch completion wakes the parked scheduler immediately via the F3
        ``wake_event`` -- it does NOT wait out the Phase-3 fallback tick.

        ``SCHEDULER_TICK_SECS`` is patched to 30s so the fallback tick is
        effectively disabled: the ONLY way the parked channel's freed slot is
        refilled within the test timeout is the event-wake.  This makes the wake
        path load-bearing (non-vacuous).  Without the fix -- i.e. no
        ``wake_event.set()`` in ``_on_batch_done`` and Phase 3 sleeping on the
        tick -- the second submit would not issue for 30s and the
        ``submit_signals.get(timeout=5.0)`` below would time out (this is exactly
        the ~25ms-mean freed-slot idle the fix removes, amplified to 30s here).

        ``worker_concurrency=1`` so a single in-flight batch parks the channel;
        the sampler withholds completion callbacks so parking is fully
        controlled by the test.  ``_CountingTaskQueue`` confirms Phase 3 waits on
        the event rather than issuing any blocking ``task_queue.get``.
        """
        worker_concurrency = 1
        channel_id = 3

        class _WithheldSampler:
            """Sampler that records each submit and withholds its callback.

            Withholding the callback keeps the single in-flight batch pending, so
            at ``worker_concurrency=1`` the channel parks after one submit and
            only the test can drive completion (and thus the wake).
            """

            def __init__(self) -> None:
                self.callbacks: list[Callable[[object], None]] = []
                self.submit_signals: queue.Queue[int] = queue.Queue()
                self._lock = threading.Lock()

            def start_loop(self) -> None: ...

            def wait_all(self) -> None: ...

            def shutdown_loop(self) -> None: ...

            def sample_from_nodes(
                self, _sampler_input: object, callback: Callable[[object], None]
            ) -> None:
                with self._lock:
                    self.callbacks.append(callback)
                    index = len(self.callbacks)
                self.submit_signals.put(index)

        sampler = _WithheldSampler()
        mock_create_dist_sampler.side_effect = lambda **_: sampler

        worker_options = self._make_worker_options(worker_concurrency)
        task_queue = _CountingTaskQueue()
        event_queue: queue.Queue[tuple[object, ...]] = queue.Queue()
        barrier = MagicMock(wait=MagicMock())
        data = MagicMock(num_partitions=1)
        sampling_config = _make_sampling_config()

        # 3 batches (6 seeds / batch_size 2): enough to submit, park, then resume.
        task_queue.put(
            (
                SharedMpCommand.REGISTER_INPUT,
                RegisterInputCmd(
                    channel_id=channel_id,
                    worker_key="loader_compute_rank_3",
                    sampler_input=NodeSamplerInput(node=torch.arange(6)),
                    sampling_config=sampling_config,
                    channel=MagicMock(),
                ),
            )
        )
        task_queue.put(
            (
                SharedMpCommand.START_EPOCH,
                StartEpochCmd(
                    channel_id=channel_id,
                    epoch=0,
                    seeds_index=torch.arange(6),
                ),
            )
        )

        worker_thread = threading.Thread(
            target=_shared_sampling_worker_loop,
            args=(
                0,
                data,
                worker_options,
                cast(mp.Queue, task_queue),
                cast(mp.Queue, event_queue),
                barrier,
                KHopNeighborSamplerOptions(num_neighbors=[2]),
                None,
            ),
        )
        worker_thread.start()
        try:
            # First batch submits, then the channel parks (in-flight == cap with
            # the callback withheld) and the scheduler enters Phase 3, blocking
            # on wake_event with the (patched 30s) fallback tick.
            first_submit = sampler.submit_signals.get(timeout=10.0)
            self.assertEqual(first_submit, 1)

            # Parked: no further submit issues on its own.  With the 30s tick
            # this also confirms the scheduler is genuinely blocked in the wait,
            # not spinning through ticks.
            time.sleep(0.2)
            self.assertTrue(sampler.submit_signals.empty())
            self.assertEqual(len(sampler.callbacks), 1)
            # Phase 3 has NOT touched task_queue -- it waits on the event.
            self.assertEqual(task_queue.blocking_get_calls, 0)

            # Fire the withheld completion from a separate thread (mirroring the
            # sampler worker thread): _on_batch_done re-enqueues the channel and
            # wake_event.set()s, waking Phase 3.
            first_callback = sampler.callbacks[0]
            wake_thread = threading.Thread(target=lambda: first_callback(None))
            wake_thread.start()
            wake_thread.join(timeout=5.0)
            self.assertFalse(wake_thread.is_alive())

            # LOAD-BEARING: the freed slot is refilled PROMPTLY via the event --
            # within 5s, far under the 30s fallback tick.  Without
            # wake_event.set() the scheduler would sleep the full 30s and this
            # get would time out.
            second_submit = sampler.submit_signals.get(timeout=5.0)
            self.assertEqual(second_submit, 2)

            # The wake came purely from the event: still zero blocking
            # task_queue.get calls.
            self.assertEqual(task_queue.blocking_get_calls, 0)
        finally:
            # The channel re-parked after the second submit (callback 2 withheld),
            # so STOP alone would sit unseen until the 30s tick.  Fire the second
            # completion to set wake_event -> Phase 3 wakes -> Phase 1 drains STOP.
            task_queue.put((SharedMpCommand.STOP, None))
            if len(sampler.callbacks) >= 2:
                sampler.callbacks[1](None)
            worker_thread.join(timeout=10.0)

        self.assertFalse(worker_thread.is_alive())
        # Phase 3 never blocked on the task queue across the whole run.
        self.assertEqual(task_queue.blocking_get_calls, 0)
