"""Opt-in, aggregated timing counters for the GLT sampling pipeline.

Set ``GIGL_SAMPLING_STATS_EVERY_N_BATCHES=<N>`` on the process that builds a loader. Every
sampling worker it spawns inherits the variable and logs one ``GLT_SAMPLING_STATS`` line per
``N`` batches it produces; the loader logs one ``GLT_LOADER_STATS`` line per ``N`` batches it
consumes. Each line reports per-batch means over its window together with the window's wall
time and CPU time, so throughput and saturation can be read off the line without per-batch
log spam. Unset or ``0`` keeps every hook on its zero-cost path.
"""

import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Callable, Final, Optional

import torch
from graphlearn_torch.distributed.dist_feature import DistFeature

from gigl.common.logger import Logger

logger = Logger()

SAMPLING_STATS_EVERY_N_BATCHES_ENV: Final[str] = "GIGL_SAMPLING_STATS_EVERY_N_BATCHES"


def _non_negative_int_from_env(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        parsed = int(raw)
    except ValueError:
        logger.warning(f"{name}={raw!r} is not an integer; ignoring it")
        return None
    if parsed < 0:
        logger.warning(f"{name}={parsed} is negative; ignoring it")
        return None
    return parsed


def sampling_stats_every_n_batches() -> int:
    """Window length for sampling statistics, in batches; ``0`` means disabled."""
    return _non_negative_int_from_env(SAMPLING_STATS_EVERY_N_BATCHES_ENV) or 0


def tensor_map_bytes(tensors: dict) -> int:
    """Total storage bytes of the tensors in a ``SampleMessage``-like dict."""
    return sum(
        value.numel() * value.element_size()
        for value in tensors.values()
        if isinstance(value, torch.Tensor)
    )


class SamplingStats:
    """Thread-safe windowed accumulator that emits one summary line every ``every_n_batches``.

    ``add`` accumulates a float per key; ``timer`` adds elapsed milliseconds. ``batch_done``
    advances the window and, at the boundary, logs and resets it. Keys are reported as per-batch
    means, so a value added once per batch reads as its mean and a value added once per RPC
    reads as the per-batch total.

    Example:
        >>> stats = SamplingStats("GLT_SAMPLING_STATS", every_n_batches=2, scope="worker=0")
        >>> stats.add("rpc_count", 3)
        >>> stats.batch_done() is None
        True
        >>> stats.add("rpc_count", 5)
        >>> line = stats.batch_done()
        >>> "rpc_count=4.00" in line and "batches=2" in line
        True
    """

    def __init__(
        self,
        prefix: str,
        every_n_batches: int,
        scope: str,
        log_fn: Callable[[str], None] = logger.info,
    ) -> None:
        if every_n_batches <= 0:
            raise ValueError(f"every_n_batches must be positive, got {every_n_batches}")
        self._prefix = prefix
        self._every_n_batches = every_n_batches
        self._scope = scope
        self._log_fn = log_fn
        self._lock = threading.Lock()
        self._values: dict[str, float] = {}
        self._batches = 0
        # The reporting thread's CPU clock is only meaningful on that thread, so the first
        # window's baseline is taken at its first ``batch_done`` rather than here.
        self._window_thread_cpu: Optional[float] = None
        self._reset_window_clocks()

    def _reset_window_clocks(self) -> None:
        self._window_start = time.perf_counter()
        times = os.times()
        self._window_proc_cpu = times.user + times.system

    def add(self, key: str, value: float) -> None:
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + value

    @contextmanager
    def timer(self, key: str) -> Iterator[None]:
        """Adds the elapsed wall time of the block to ``key`` in milliseconds."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(key, (time.perf_counter() - start) * 1e3)

    def batch_done(self) -> Optional[str]:
        """Counts one finished batch; at a window boundary logs the report and resets.

        Call this from the thread whose CPU time should be reported as ``loop_cpu_s``; the
        very first window's ``loop_cpu_s`` excludes the batch that establishes that baseline.

        Returns:
            The report line when a window closed, else ``None``.
        """
        with self._lock:
            if self._window_thread_cpu is None:
                self._window_thread_cpu = time.thread_time()
            self._batches += 1
            if self._batches < self._every_n_batches:
                return None
            line = self._format_report_locked()
            self._values.clear()
            self._batches = 0
            self._reset_window_clocks()
            self._window_thread_cpu = time.thread_time()
        self._log_fn(line)
        return line

    def _format_report_locked(self) -> str:
        batches = self._batches
        wall_s = time.perf_counter() - self._window_start
        times = os.times()
        proc_cpu_s = times.user + times.system - self._window_proc_cpu
        assert self._window_thread_cpu is not None
        loop_cpu_s = time.thread_time() - self._window_thread_cpu
        fields = [
            f"{self._prefix} {self._scope} batches={batches}",
            f"wall_s={wall_s:.2f}",
            f"batches_per_s={batches / wall_s if wall_s > 0 else 0.0:.3f}",
            f"proc_cpu_s={proc_cpu_s:.2f}",
            f"loop_cpu_s={loop_cpu_s:.2f}",
        ]
        fields.extend(
            f"{key}={total / batches:.2f}"
            for key, total in sorted(self._values.items())
        )
        return " ".join(fields)


def instrument_dist_feature(
    feature: DistFeature, stats: SamplingStats, name: str
) -> None:
    """Times a GLT ``DistFeature`` instance's remote issue, RPC wait, local gather, and stitch.

    GLT builds its ``DistFeature`` objects inside ``DistNeighborSampler.__init__`` and registers
    an RPC callee for each (a collective), so they cannot be swapped for a subclass afterwards.
    The three methods are wrapped on the instance instead; the wrappers add to ``stats`` under
    ``<name>_remote_issue_ms``, ``<name>_rpc_wait_ms``, ``<name>_rpc_count``, ``<name>_remote_mb``,
    ``<name>_local_ms`` and ``<name>_stitch_ms``.

    Args:
        feature: The ``DistFeature`` to instrument.
        stats: Accumulator to report into.
        name: Key prefix, e.g. ``nfeat``.
    """
    remote_get = feature._remote_selecting_get
    local_get = feature._local_selecting_get
    stitch = feature._stitch

    def timed_remote_get(ids: torch.Tensor, input_type=None) -> torch.futures.Future:
        issue_start = time.perf_counter()
        fut = remote_get(ids, input_type)
        stats.add(f"{name}_remote_issue_ms", (time.perf_counter() - issue_start) * 1e3)

        def on_done(done: torch.futures.Future) -> None:
            stats.add(f"{name}_rpc_wait_ms", (time.perf_counter() - issue_start) * 1e3)
            try:
                partials = done.wait()
            except Exception:
                return
            stats.add(f"{name}_rpc_count", len(partials))
            stats.add(
                f"{name}_remote_mb",
                sum(feat.numel() * feat.element_size() for feat, _ in partials) / 2**20,
            )

        fut.add_done_callback(on_done)
        return fut

    def timed_local_get(ids: torch.Tensor, input_type=None):
        with stats.timer(f"{name}_local_ms"):
            return local_get(ids, input_type)

    def timed_stitch(ids: torch.Tensor, local, remotes) -> torch.Tensor:
        with stats.timer(f"{name}_stitch_ms"):
            return stitch(ids, local, remotes)

    setattr(feature, "_remote_selecting_get", timed_remote_get)
    setattr(feature, "_local_selecting_get", timed_local_get)
    setattr(feature, "_stitch", timed_stitch)


_CALLEE_INSTRUMENTED_ATTR: Final[str] = "_gigl_sampling_stats_instrumented"


def instrument_rpc_callees(stats: SamplingStats) -> None:
    """Times the RPC callees this process serves to other sampling workers.

    GLT dispatches incoming sampling and feature-lookup requests through the process-global
    ``_rpc_callee_pool`` in ``graphlearn_torch.distributed.rpc``; each entry's ``call`` runs on
    an RPC worker thread. Wrapping ``call`` on each instance records ``srv_sample_calls``,
    ``srv_sample_ms``, ``srv_sample_seeds`` and ``srv_feat_calls``, ``srv_feat_ms``,
    ``srv_feat_mb``. Callees already instrumented (a second sampler in the same process) are left
    alone, so the first sampler's ``stats`` receives the served-request counters.

    Args:
        stats: Accumulator to report into.
    """
    from graphlearn_torch.distributed import rpc as glt_rpc
    from graphlearn_torch.distributed.dist_feature import RpcFeatureLookupCallee
    from graphlearn_torch.distributed.dist_neighbor_sampler import RpcSamplingCallee

    for callee in glt_rpc._rpc_callee_pool.values():
        if getattr(callee, _CALLEE_INSTRUMENTED_ATTR, False):
            continue
        if isinstance(callee, RpcSamplingCallee):
            prefix = "srv_sample"
        elif isinstance(callee, RpcFeatureLookupCallee):
            prefix = "srv_feat"
        else:
            continue
        original_call = callee.call

        def timed_call(*args, _original=original_call, _prefix=prefix, **kwargs):
            start = time.perf_counter()
            result = _original(*args, **kwargs)
            stats.add(f"{_prefix}_ms", (time.perf_counter() - start) * 1e3)
            stats.add(f"{_prefix}_calls", 1)
            if _prefix == "srv_feat":
                stats.add("srv_feat_mb", result.numel() * result.element_size() / 2**20)
            else:
                stats.add("srv_sample_seeds", args[0].numel())
            return result

        setattr(callee, "call", timed_call)
        setattr(callee, _CALLEE_INSTRUMENTED_ATTR, True)
