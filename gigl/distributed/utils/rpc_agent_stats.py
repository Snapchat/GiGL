"""Torch RPC agent counters, sampled on a background thread.

Used by the loader's sampling statistics to tell an RPC thread pool that is saturated or
waiting on the GIL from one that is idle.
"""

import threading
from typing import Callable, Optional

import torch.distributed.rpc as rpc
from torch.distributed.rpc.api import _get_current_rpc_agent

from gigl.common.logger import Logger

logger = Logger()


class RpcAgentSampler:
    """Samples the torch RPC agent's metrics every ``interval_s`` on a daemon thread.

    The reporter from ``make_reporter`` summarizes the samples taken since its previous call
    (or since ``start``) and then starts a new window. It is safe to call from
    ``SamplingStats``'s window close because it only reads sampler state.

    Args:
        interval_s: Seconds between samples.
    """

    def __init__(self, interval_s: float = 2.0) -> None:
        self._interval_s = interval_s
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._reset_window()

    def _reset_window(self) -> None:
        self._idle_threads: list[float] = []
        self._pool_size = 0.0
        self._server_active: list[float] = []
        self._client_active: list[float] = []
        self._gil_wait_us: list[float] = []

    def start(self) -> None:
        rpc.enable_gil_profiling(True)
        self._thread = threading.Thread(
            target=self._run, name="rpc-agent-sampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            try:
                self._sample()
            except RuntimeError:
                # The agent has shut down; there is nothing left to sample.
                return
            except Exception:
                logger.exception("RPC agent sample failed")

    def _sample(self) -> None:
        metrics = _get_current_rpc_agent().get_metrics()
        with self._lock:
            self._idle_threads.append(float(metrics["agent.num_idle_threads"]))
            self._pool_size = float(metrics["agent.thread_pool_size"])
            self._server_active.append(float(metrics["agent.server_active_calls"]))
            self._client_active.append(float(metrics["agent.client_active_calls"]))
            self._gil_wait_us.append(float(metrics["agent.gil_average_wait_time_us"]))

    def make_reporter(self) -> Callable[[], dict[str, float]]:
        """Returns a ``SamplingStats`` reporter summarizing the samples since its last call.

        A window with no samples yet (shorter than ``interval_s``) reads the agent once on demand.
        """
        return self._report

    def _report(self) -> dict[str, float]:
        with self._lock:
            has_samples = bool(self._idle_threads)
        if not has_samples:
            # A window shorter than ``interval_s`` still reports one reading of the agent.
            try:
                self._sample()
            except Exception:
                logger.debug("No RPC agent reading for an empty window", exc_info=True)
        with self._lock:
            report: dict[str, float] = {"rpc_samples": float(len(self._idle_threads))}
            if self._idle_threads:
                report["rpc_idle_threads_mean"] = sum(self._idle_threads) / len(
                    self._idle_threads
                )
                report["rpc_idle_threads_min"] = min(self._idle_threads)
                report["rpc_pool_size"] = self._pool_size
                report["rpc_server_active_mean"] = sum(self._server_active) / len(
                    self._server_active
                )
                report["rpc_server_active_max"] = max(self._server_active)
                report["rpc_client_active_mean"] = sum(self._client_active) / len(
                    self._client_active
                )
                report["rpc_client_active_max"] = max(self._client_active)
                report["rpc_gil_wait_us_mean"] = sum(self._gil_wait_us) / len(
                    self._gil_wait_us
                )
            self._reset_window()
            return report
