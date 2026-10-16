"""Runs exporter work in a background worker process, for `EmbeddingExporter(in_background=True)` and friends.

Per-row conversion and Avro encoding are pure Python, so on a thread they compete with the caller's loop for the GIL.
Here the caller only snapshots each batch and a small feeder thread copies it into shared memory; a child process
builds the real exporter and does the encoding and uploads. The child sees the same calls in the same order as the
synchronous exporter, so the output files are identical.

The feeder thread exists so that CUDA tensors never block the caller: the device-to-host copy is queued into a pinned
buffer and the feeder waits for it, instead of the caller waiting for the forward pass to finish.

Nothing here blocks without a timeout. Every wait polls in short intervals and checks that the worker is alive and
making progress, and the worker checks that its parent is alive. A worker that dies, or that makes no progress for
`stall_timeout_s` while work is pending, is killed and reported as a `BackgroundExportError`; there is no fallback to
synchronous export.
"""

import atexit
import multiprocessing as mp
import os
import pickle
import queue
import signal
import threading
import time
import weakref
from multiprocessing import shared_memory
from typing import Any, Optional

import torch

_ADD, _FLUSH, _STOP = "add", "flush", "stop"
_POLL_SECONDS = 0.2
_PARENT_CHECK_SECONDS = 1.0
_KILL_GRACE_SECONDS = 5.0
# Extra time the parent gives the worker's own per-command timeout to report before declaring a stall.
_STALL_GRACE_SECONDS = 5.0

DEFAULT_STALL_TIMEOUT_SECONDS = 300.0
DEFAULT_STARTUP_TIMEOUT_SECONDS = 180.0


class BackgroundExportError(RuntimeError):
    """The background exporter failed: its worker died, or could not be reached."""


class BackgroundExportStalledError(BackgroundExportError):
    """The background worker made no progress for too long while work was pending. The worker was killed."""


def _view(
    shm: shared_memory.SharedMemory, offset: int, shape: tuple, dtype: torch.dtype
):
    numel = 1
    for dim in shape:
        numel *= dim
    nbytes = numel * torch.empty(0, dtype=dtype).element_size()
    return (
        torch.frombuffer(shm.buf, dtype=dtype, count=numel, offset=offset).reshape(
            shape
        ),
        nbytes,
    )


def _picklable(error: BaseException) -> BaseException:
    try:
        pickle.loads(pickle.dumps(error))
        return error
    except Exception:
        return RuntimeError(f"{type(error).__name__}: {error}")


def _unlink_quietly(name: str) -> None:
    try:
        segment = shared_memory.SharedMemory(name=name)
        segment.close()
        segment.unlink()
    except Exception:
        pass


def _watch_parent(parent_pid: int, attached_names: set[str]) -> None:
    """Worker-side thread: exits the worker, freeing its shared memory, once the parent is gone.

    Runs on its own thread so that it works even when the main thread is stuck in an exporter call.
    """
    while os.getppid() == parent_pid:
        time.sleep(_PARENT_CHECK_SECONDS)
    for name in list(attached_names):
        _unlink_quietly(name)
    os._exit(1)


def _worker_main(
    exporter_cls,
    exporter_kwargs,
    commands,
    free_slots,
    events,
    progress,
    command_timeout_s: float,
    parent_pid: int,
) -> None:
    """Child process: owns the real exporter. Replies on `events` and returns each slot to `free_slots`."""
    # The worker must never block on exit because a queue it wrote to has unread data.
    events.cancel_join_thread()
    free_slots.cancel_join_thread()
    attached: dict[int, tuple[str, shared_memory.SharedMemory]] = {}
    attached_names: set[str] = set()
    threading.Thread(
        target=_watch_parent,
        args=(parent_pid, attached_names),
        daemon=True,
        name="parent-watch",
    ).start()

    def _on_alarm(signum, frame):
        raise TimeoutError(
            f"Exporter call did not finish within {command_timeout_s:.0f} seconds."
        )

    signal.signal(signal.SIGALRM, _on_alarm)

    error: Optional[BaseException] = None
    exporter: Any = None
    try:
        exporter = exporter_cls(**exporter_kwargs)
        exporter.__enter__()
        events.put(("ready", None))
    except BaseException as e:
        error = e
        events.put(("error", _picklable(e)))
    while True:
        try:
            command = commands.get(timeout=_POLL_SECONDS)
        except queue.Empty:
            continue
        kind = command[0]
        signal.setitimer(signal.ITIMER_REAL, command_timeout_s)
        try:
            if kind == _ADD:
                _, slot, name, method, ids_spec, values_spec, record_type = command
                try:
                    if error is None:
                        if slot not in attached or attached[slot][0] != name:
                            if slot in attached:
                                attached[slot][1].close()
                            shm = shared_memory.SharedMemory(name=name)
                            attached[slot] = (name, shm)
                            attached_names.add(name)
                        shm = attached[slot][1]
                        ids, offset = _view(shm, 0, *ids_spec)
                        values, _ = _view(shm, offset, *values_spec)
                        getattr(exporter, method)(ids, values, record_type)
                        del ids, values
                finally:
                    free_slots.put(slot)
            elif kind == _FLUSH:
                if error is None:
                    exporter.flush_records()  # type: ignore
                events.put(("done", None))
            else:
                if error is None:
                    exporter.__exit__(None, None, None)  # type: ignore
                events.put(("done", None))
                return
        except BaseException as e:
            if error is None:
                error = e
                events.put(("error", _picklable(e)))
            if kind != _ADD:
                events.put(("done", None))
            if kind == _STOP:
                return
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            with progress.get_lock():
                progress.value += 1


class _ProcessProxy:
    """Parent-side handle to the worker process. Safe to use from the caller and feeder threads."""

    def __init__(
        self,
        exporter_cls,
        exporter_kwargs: dict[str, Any],
        num_slots: int,
        stall_timeout_s: float,
        startup_timeout_s: float,
    ):
        ctx = mp.get_context("spawn")
        self._stall_timeout_s = stall_timeout_s
        self._startup_timeout_s = startup_timeout_s
        self._ready = False
        self._commands = ctx.Queue()
        self._free_slots = ctx.Queue()
        self._events = ctx.Queue()
        # The parent must never block on exit because a dead worker left unread data in a queue.
        self._commands.cancel_join_thread()
        self._free_slots.cancel_join_thread()
        self._events.cancel_join_thread()
        for slot in range(num_slots):
            self._free_slots.put(slot)
        self._progress = ctx.Value("Q", 0)  # Commands the worker has finished.
        self._submitted = 0  # Commands sent to the worker.
        self._seen_progress = 0
        self._last_progress_at = time.monotonic()
        self._segments: dict[int, shared_memory.SharedMemory] = {}
        self._error: Optional[BaseException] = None
        self._tokens: list[str] = []  # `ready` / `done` replies not yet consumed.
        self._lock = threading.RLock()
        self._dead = False
        self._process = ctx.Process(
            target=_worker_main,
            args=(
                exporter_cls,
                exporter_kwargs,
                self._commands,
                self._free_slots,
                self._events,
                self._progress,
                stall_timeout_s,
                os.getpid(),
            ),
            daemon=True,
            name="background-exporter",
        )
        self._process.start()

    def poll_error(self) -> None:
        """Raises the worker's first error, or a `BackgroundExportError` if it died or stalled."""
        with self._lock:
            while self._error is None:
                try:
                    kind, payload = self._events.get_nowait()
                except queue.Empty:
                    break
                if kind == "error":
                    self._error = payload
                else:
                    self._note_token(kind)
            if self._error is None and not self._dead:
                self._check_liveness()
            if self._error is not None:
                raise self._error

    def _note_token(self, kind: str) -> None:
        self._tokens.append(kind)
        if kind == "ready":
            self._ready = True
            self._last_progress_at = time.monotonic()

    def _check_liveness(self) -> None:
        if not self._process.is_alive():
            self._fail(
                BackgroundExportError(
                    f"Background exporter process exited unexpectedly (exit code {self._process.exitcode})."
                )
            )
            return
        if not self._ready:
            return  # Start-up is bounded separately by `wait_ready`.
        done = self._progress.value
        now = time.monotonic()
        pending = self._submitted - done
        if done != self._seen_progress or pending <= 0:
            self._seen_progress = done
            self._last_progress_at = now
        elif (
            now - self._last_progress_at > self._stall_timeout_s + _STALL_GRACE_SECONDS
        ):
            self._fail(
                BackgroundExportStalledError(
                    f"Background exporter made no progress for {now - self._last_progress_at:.0f}s "
                    f"(limit {self._stall_timeout_s:.0f}s) with {pending} command(s) pending; "
                    f"{self._submitted} sent, {done} finished. The worker was killed."
                )
            )

    def _fail(self, error: BaseException) -> None:
        self._error = error
        self.kill()

    def kill(self) -> None:
        """Terminates the worker, escalating to SIGKILL, and frees shared memory. Idempotent."""
        with self._lock:
            self._dead = True
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(_KILL_GRACE_SECONDS)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(_KILL_GRACE_SECONDS)
            for segment in self._segments.values():
                try:
                    segment.close()
                    segment.unlink()
                except Exception:
                    pass
            self._segments.clear()

    def _put_command(self, command: tuple) -> None:
        with self._lock:
            if self._submitted == self._progress.value:
                self._last_progress_at = (
                    time.monotonic()
                )  # The worker was idle; start the clock now.
            self._submitted += 1
            self._commands.put(command)

    def send(
        self, method: str, ids: torch.Tensor, values: torch.Tensor, record_type: str
    ) -> None:
        while True:
            self.poll_error()
            try:
                # Blocks while every slot is in flight, which bounds shared memory.
                slot = self._free_slots.get(timeout=_POLL_SECONDS)
                break
            except queue.Empty:
                pass
        need = ids.numel() * ids.element_size() + values.numel() * values.element_size()
        with self._lock:
            if self._dead:
                self.poll_error()
            segment = self._segments.get(slot)
            if segment is None or segment.size < need:
                if segment is not None:
                    segment.close()
                    segment.unlink()
                segment = self._segments[slot] = shared_memory.SharedMemory(
                    create=True, size=need
                )
            ids_view, offset = _view(segment, 0, tuple(ids.shape), ids.dtype)
            ids_view.copy_(ids)
            values_view, _ = _view(segment, offset, tuple(values.shape), values.dtype)
            values_view.copy_(values)
            del ids_view, values_view
            self._put_command(
                (
                    _ADD,
                    slot,
                    segment.name,
                    method,
                    (tuple(ids.shape), ids.dtype),
                    (tuple(values.shape), values.dtype),
                    record_type,
                )
            )

    def _wait_for(self, token: str, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while True:
            with self._lock:
                if token in self._tokens:
                    self._tokens.remove(token)
                    break
            if time.monotonic() > deadline:
                self._fail(
                    BackgroundExportStalledError(
                        f"Background exporter did not reply '{token}' within {timeout_s:.0f}s. The worker was killed."
                    )
                )
                self.poll_error()
            try:
                kind, payload = self._events.get(timeout=_POLL_SECONDS)
            except queue.Empty:
                self.poll_error()
                continue
            with self._lock:
                if kind == "error":
                    self._error = self._error or payload
                else:
                    self._note_token(kind)
            self.poll_error()

    def wait_ready(self) -> None:
        self._wait_for("ready", self._startup_timeout_s)

    def flush(self) -> None:
        self.poll_error()
        self._put_command((_FLUSH,))
        self._wait_for("done", self._stall_timeout_s + _STALL_GRACE_SECONDS)

    def stop(self) -> None:
        """Asks the worker to finish and exit. Always leaves the worker dead and shared memory freed."""
        try:
            if not self._dead:
                self._put_command((_STOP,))
                self._wait_for("done", self._stall_timeout_s + _STALL_GRACE_SECONDS)
        finally:
            self._process.join(_KILL_GRACE_SECONDS)
            self.kill()
        if self._error is not None:
            raise self._error


_runners: "weakref.WeakSet[BackgroundRunner]" = weakref.WeakSet()


@atexit.register
def _abort_runners_at_exit() -> None:
    for runner in list(_runners):
        runner._abort()


class BackgroundRunner:
    """Feeds batches to a worker process that runs `exporter_cls(**exporter_kwargs)`.

    Args:
        exporter_cls: Exporter class the worker instantiates (with background disabled).
        exporter_kwargs: Constructor arguments for it. Must be picklable.
        max_pending_batches: Batches in flight. Once reached, `submit` blocks until the worker catches up.
        stall_timeout_s: How long the worker may go without finishing anything while work is pending, and how long a
            single exporter call (such as an upload) may take, before the worker is killed and a
            `BackgroundExportStalledError` is raised.
        startup_timeout_s: How long the worker may take to import and construct the exporter.
    """

    def __init__(
        self,
        exporter_cls,
        exporter_kwargs: dict[str, Any],
        max_pending_batches: int = 8,
        stall_timeout_s: float = DEFAULT_STALL_TIMEOUT_SECONDS,
        startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
    ):
        if max_pending_batches < 1:
            raise ValueError(
                f"max_pending_batches must be positive, but got {max_pending_batches}"
            )
        if stall_timeout_s <= 0:
            raise ValueError(
                f"stall_timeout_s must be positive, but got {stall_timeout_s}"
            )
        self._exporter_cls = exporter_cls
        self._exporter_kwargs = exporter_kwargs
        self._max_pending_batches = max_pending_batches
        self._stall_timeout_s = stall_timeout_s
        self._startup_timeout_s = startup_timeout_s
        self._feeder_error: Optional[BaseException] = None
        self._closing = threading.Event()
        self._proxy, self._queue, self._feeder = self._launch()
        self._running = True
        _runners.add(self)

    def _launch(self) -> tuple[_ProcessProxy, "queue.Queue[Any]", threading.Thread]:
        self._feeder_error = None
        self._closing = threading.Event()
        proxy = _ProcessProxy(
            self._exporter_cls,
            self._exporter_kwargs,
            self._max_pending_batches,
            self._stall_timeout_s,
            self._startup_timeout_s,
        )
        batches: "queue.Queue[Any]" = queue.Queue(maxsize=self._max_pending_batches)
        feeder = threading.Thread(
            target=self._feed,
            args=(proxy, batches, self._closing),
            name="background-exporter-feeder",
            daemon=True,
        )
        feeder.start()
        return proxy, batches, feeder

    def submit(
        self,
        method: str,
        id_batch: torch.Tensor,
        value_batch: torch.Tensor,
        record_type: str,
    ) -> None:
        """Queues a batch. Tensors may be on the GPU and may be modified as soon as this returns."""
        if not self._running:
            self._proxy, self._queue, self._feeder = self._launch()
            self._running = True
        self._raise_error()
        ids, values = _snapshot(id_batch), _snapshot(value_batch)
        copied: Optional[torch.cuda.Event] = None
        if id_batch.is_cuda or value_batch.is_cuda:
            # Device-to-host copies are asynchronous; the feeder waits on this before reading.
            copied = torch.cuda.Event()
            copied.record()
        item = (method, ids, values, copied, record_type)
        while True:
            try:
                self._queue.put(item, timeout=_POLL_SECONDS)
                return
            except queue.Full:
                self._raise_error()

    def flush_records(self) -> None:
        """Waits until every queued batch is exported, then flushes the worker's exporter."""
        if not self._running:
            return
        while self._queue.unfinished_tasks:
            self._raise_error()
            time.sleep(0.005)
        self._raise_error()
        self._proxy.flush()

    def close(self) -> None:
        """Stops the worker after it finishes queued batches. Does not flush. A later `submit` starts a new one.

        Raises the worker's error if it has not been raised yet. The worker is always gone when this returns.
        """
        if not self._running:
            return
        self._running = False
        self._closing.set()
        self._feeder.join(self._stall_timeout_s + _STALL_GRACE_SECONDS)
        self._proxy.stop()
        if self._feeder_error is not None:
            raise self._feeder_error

    def _abort(self) -> None:
        self._closing.set()
        self._proxy.kill()

    def _raise_error(self) -> None:
        if self._feeder_error is not None:
            raise self._feeder_error
        self._proxy.poll_error()
        if self._running and not self._feeder.is_alive():
            raise BackgroundExportError(
                "Background exporter feeder thread exited unexpectedly."
            )

    def _feed(
        self, proxy: _ProcessProxy, batches: queue.Queue, closing: threading.Event
    ) -> None:
        try:
            proxy.wait_ready()
        except BaseException as e:
            self._feeder_error = e
        while True:
            try:
                item = batches.get(timeout=_POLL_SECONDS)
            except queue.Empty:
                if closing.is_set():
                    return  # Closing and nothing left to feed.
                continue
            try:
                if self._feeder_error is not None:
                    continue  # Drop work after a failure so the queue still drains.
                method, ids, values, copied, record_type = item
                if copied is not None:
                    self._wait_for_copy(copied)
                proxy.send(method, ids, values, record_type)
            except BaseException as e:  # Surfaced on the caller's thread.
                self._feeder_error = e
            finally:
                batches.task_done()

    def _wait_for_copy(self, copied: "torch.cuda.Event") -> None:
        deadline = time.monotonic() + self._stall_timeout_s
        while not copied.query():
            if time.monotonic() > deadline:
                raise BackgroundExportStalledError(
                    f"A device-to-host copy did not finish within {self._stall_timeout_s:.0f}s."
                )
            time.sleep(0.001)


def _snapshot(tensor: torch.Tensor) -> torch.Tensor:
    """Returns a host tensor the caller can no longer affect.

    CUDA tensors are copied asynchronously into a fresh pinned buffer; CPU tensors are cloned.
    """
    if tensor.is_cuda:
        host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
        host.copy_(tensor, non_blocking=True)
        return host
    return tensor.detach().clone()
