"""Runs exporter work in a background worker process, for `EmbeddingExporter(in_background=True)` and friends.

Per-row conversion and Avro encoding are pure Python, so on a thread they compete with the caller's loop for the GIL.
Here the caller only snapshots each batch and a small feeder thread copies it into shared memory; a child process
builds the real exporter and does the encoding and uploads. The child sees the same calls in the same order as the
synchronous exporter, so the output files are identical.

The feeder thread exists so that CUDA tensors never block the caller: the device-to-host copy is queued into a pinned
buffer and the feeder waits for it, instead of the caller waiting for the forward pass to finish.
"""

import multiprocessing as mp
import pickle
import queue
import threading
from multiprocessing import shared_memory
from typing import Any, Optional

import torch

_ADD, _FLUSH, _STOP = "add", "flush", "stop"
_POLL_SECONDS = 0.5


def _view(
    shm: shared_memory.SharedMemory, offset: int, shape: tuple, dtype: torch.dtype
):
    numel = 1
    for dim in shape:
        numel *= dim
    nbytes = numel * torch.empty(0, dtype=dtype).element_size()
    return torch.frombuffer(shm.buf, dtype=dtype, count=numel, offset=offset).reshape(
        shape
    ), nbytes


def _picklable(error: BaseException) -> BaseException:
    try:
        pickle.loads(pickle.dumps(error))
        return error
    except Exception:
        return RuntimeError(f"{type(error).__name__}: {error}")


def _worker_main(exporter_cls, exporter_kwargs, commands, free_slots, events) -> None:
    """Child process: owns the real exporter. Replies on `events` and returns each slot to `free_slots`."""
    attached: dict[int, tuple[str, shared_memory.SharedMemory]] = {}
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
        command = commands.get()
        kind = command[0]
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


class _ProcessProxy:
    """Parent-side handle to the worker process. Used only from the feeder thread, apart from `flush` / `stop`."""

    def __init__(self, exporter_cls, exporter_kwargs: dict[str, Any], num_slots: int):
        ctx = mp.get_context("spawn")
        self._commands = ctx.Queue()
        self._free_slots = ctx.Queue()
        self._events = ctx.Queue()
        for slot in range(num_slots):
            self._free_slots.put(slot)
        self._segments: dict[int, shared_memory.SharedMemory] = {}
        self._error: Optional[BaseException] = None
        self._tokens: list[str] = []  # `ready` / `done` replies not yet consumed.
        self._process = ctx.Process(
            target=_worker_main,
            args=(
                exporter_cls,
                exporter_kwargs,
                self._commands,
                self._free_slots,
                self._events,
            ),
            daemon=True,
            name="background-exporter",
        )
        self._process.start()

    def poll_error(self) -> None:
        """Raises the worker's first error, or an error if the worker died."""
        while self._error is None:
            try:
                kind, payload = self._events.get_nowait()
            except queue.Empty:
                break
            if kind == "error":
                self._error = payload
            else:
                self._tokens.append(kind)
        if self._error is not None:
            raise self._error
        if not self._process.is_alive():
            raise RuntimeError(
                f"Background exporter process exited unexpectedly (exit code {self._process.exitcode})."
            )

    def send(
        self, method: str, ids: torch.Tensor, values: torch.Tensor, record_type: str
    ) -> None:
        while True:
            try:
                # Blocks while every slot is in flight, which bounds shared memory.
                slot = self._free_slots.get(timeout=_POLL_SECONDS)
                break
            except queue.Empty:
                self.poll_error()
        need = ids.numel() * ids.element_size() + values.numel() * values.element_size()
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
        self._commands.put(
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

    def _wait_for(self, token: str) -> None:
        while token not in self._tokens and self._error is None:
            try:
                kind, payload = self._events.get(timeout=_POLL_SECONDS)
            except queue.Empty:
                if not self._process.is_alive():
                    self.poll_error()
                continue
            if kind == "error":
                self._error = self._error or payload
            else:
                self._tokens.append(kind)
        if self._error is not None:
            raise self._error
        self._tokens.remove(token)

    def wait_ready(self) -> None:
        self._wait_for("ready")

    def flush(self) -> None:
        self._commands.put((_FLUSH,))
        self._wait_for("done")

    def stop(self) -> None:
        try:
            self._commands.put((_STOP,))
            self._wait_for("done")
        finally:
            self._process.join(timeout=30)
            if self._process.is_alive():
                self._process.terminate()
            for segment in self._segments.values():
                segment.close()
                segment.unlink()
            self._segments.clear()


class BackgroundRunner:
    """Feeds batches to a worker process that runs `exporter_cls(**exporter_kwargs)`.

    Args:
        exporter_cls: Exporter class the worker instantiates (with background disabled).
        exporter_kwargs: Constructor arguments for it. Must be picklable.
        max_pending_batches: Batches in flight. Once reached, `submit` blocks until the worker catches up.
    """

    def __init__(
        self,
        exporter_cls,
        exporter_kwargs: dict[str, Any],
        max_pending_batches: int = 8,
    ):
        if max_pending_batches < 1:
            raise ValueError(
                f"max_pending_batches must be positive, but got {max_pending_batches}"
            )
        self._exporter_cls = exporter_cls
        self._exporter_kwargs = exporter_kwargs
        self._max_pending_batches = max_pending_batches
        self._feeder_error: Optional[BaseException] = None
        self._proxy, self._queue, self._feeder = self._launch()
        self._running = True

    def _launch(self) -> tuple[_ProcessProxy, "queue.Queue[Any]", threading.Thread]:
        self._feeder_error = None
        proxy = _ProcessProxy(
            self._exporter_cls, self._exporter_kwargs, self._max_pending_batches
        )
        batches: "queue.Queue[Any]" = queue.Queue(maxsize=self._max_pending_batches)
        feeder = threading.Thread(
            target=self._feed,
            args=(proxy, batches),
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
        self._queue.put((method, ids, values, copied, record_type))

    def flush_records(self) -> None:
        """Waits until every queued batch is exported, then flushes the worker's exporter."""
        if not self._running:
            return
        self._queue.join()
        self._raise_error()
        self._proxy.flush()

    def close(self) -> None:
        """Stops the worker after it finishes queued batches. Does not flush. A later `submit` starts a new one.

        Raises the worker's error if it has not been raised yet.
        """
        if not self._running:
            return
        self._queue.put(None)
        self._feeder.join()
        self._running = False
        self._proxy.stop()

    def _raise_error(self) -> None:
        if self._feeder_error is not None:
            raise self._feeder_error
        self._proxy.poll_error()

    def _feed(self, proxy: _ProcessProxy, batches: queue.Queue) -> None:
        try:
            proxy.wait_ready()
        except BaseException as e:
            self._feeder_error = e
        while True:
            item = batches.get()
            try:
                if item is None:
                    return
                if self._feeder_error is not None:
                    continue  # Drop work after a failure so the queue still drains.
                method, ids, values, copied, record_type = item
                if copied is not None:
                    copied.synchronize()
                proxy.send(method, ids, values, record_type)
            except BaseException as e:  # Surfaced on the caller's thread.
                self._feeder_error = e
            finally:
                batches.task_done()


def _snapshot(tensor: torch.Tensor) -> torch.Tensor:
    """Returns a host tensor the caller can no longer affect.

    CUDA tensors are copied asynchronously into a fresh pinned buffer; CPU tensors are cloned.
    """
    if tensor.is_cuda:
        host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
        host.copy_(tensor, non_blocking=True)
        return host
    return tensor.detach().clone()
