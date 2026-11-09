"""Per-thread CPU time and scheduler run delay of the current process, grouped by thread role.

Used by the loader's sampling statistics to tell which kind of thread (RPC pool, TensorPipe I/O,
the sampler's asyncio loop, torch intra-op workers) is saturating the CPU or waiting for a core.
Everything here reads Linux ``/proc/self/task``; on hosts without it the reported values are 0.
"""

import os
from typing import Callable, Optional

from gigl.common.logger import Logger

logger = Logger()

_ROLES = ("rpc_pool", "tp_uv", "tp_other", "loop", "omp", "other")
_TASK_DIR = "/proc/self/task"
_RPC_POOL_COMM = "pt_thread_pool"
_TP_UV_COMM = "TP_UV_loop"
_NS = 1e9


def _list_tids() -> set[int]:
    try:
        return {int(name) for name in os.listdir(_TASK_DIR)}
    except OSError:
        return set()


def _read(path: str) -> Optional[str]:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


class ThreadRoles:
    """Classifies the process's threads by role and reports per-role CPU and run delay.

    The RPC worker pool is identified by the thread ids that appear while ``init_rpc`` runs and
    are named like the pool: torch's intra-op threads inherit the name of the thread that created
    them, so threads named like the RPC pool cannot be told apart by name alone. Other threads
    created during ``init_rpc`` (TensorPipe reactors and transports, the agent timeout thread,
    OpenMP teams) are reported as ``tp_other``.

    The reporter from ``make_reporter`` returns, for each role, the CPU seconds and run delay
    seconds accrued since the previous window and the number of threads seen in the scan. Creating
    the reporter takes a baseline scan, so the first window starts at that point rather than at
    thread creation.
    """

    def __init__(self) -> None:
        self._tids_before_rpc: set[int] = set()
        self._rpc_pool_tids: set[int] = set()
        self._tp_other_tids: set[int] = set()
        self._loop_tid: Optional[int] = None
        self._clk_tck = float(os.sysconf("SC_CLK_TCK"))
        self._last_cpu_s: dict[int, float] = {}
        self._last_rundelay_s: dict[int, float] = {}

    def snapshot_before_rpc(self) -> None:
        """Records the threads that exist before ``init_rpc``."""
        self._tids_before_rpc = _list_tids()

    def snapshot_after_rpc(self) -> None:
        """Records the threads created since ``snapshot_before_rpc``.

        Those named like the RPC pool are the pool; the rest, except TensorPipe's libuv loops
        which are recognized by name, are ``tp_other``.
        """
        for tid in _list_tids() - self._tids_before_rpc:
            comm = _read(f"{_TASK_DIR}/{tid}/comm")
            if comm is None:
                continue
            comm = comm.strip()
            if comm == _RPC_POOL_COMM:
                self._rpc_pool_tids.add(tid)
            elif comm != _TP_UV_COMM:
                self._tp_other_tids.add(tid)

    def set_loop_tid(self, tid: int) -> None:
        """Sets the thread id of the sampler's asyncio loop thread."""
        self._loop_tid = tid

    def classify(self, tid: int, comm: str) -> str:
        """Returns the role of a thread: rpc_pool, tp_uv, tp_other, loop, omp or other."""
        if tid in self._rpc_pool_tids:
            return "rpc_pool"
        if comm == _TP_UV_COMM:
            return "tp_uv"
        if tid == self._loop_tid:
            return "loop"
        if tid in self._tp_other_tids:
            return "tp_other"
        if comm == _RPC_POOL_COMM:
            return "omp"
        return "other"

    def _cpu_seconds(self, tid: int) -> Optional[float]:
        text = _read(f"{_TASK_DIR}/{tid}/stat")
        if text is None:
            return None
        # The comm field may contain spaces, so the numeric fields start after the last ")".
        fields = text.rsplit(")", 1)[1].split()
        # utime and stime are fields 14 and 15 overall; fields[0] here is field 3.
        return (int(fields[11]) + int(fields[12])) / self._clk_tck

    def _run_delay_seconds(self, tid: int) -> Optional[float]:
        text = _read(f"{_TASK_DIR}/{tid}/schedstat")
        if text is None:
            return None
        return int(text.split()[1]) / _NS

    def read_window(self) -> dict[str, float]:
        """Returns per-role CPU, run delay and thread counts since the previous call.

        Also returns, per role, the largest CPU and run delay seconds accrued by any single
        thread of the role (``cpu_<role>_max_thread_s``, ``rundelay_<role>_max_thread_s``).
        """
        report: dict[str, float] = {}
        for role in _ROLES:
            report[f"cpu_{role}_s"] = 0.0
            report[f"rundelay_{role}_s"] = 0.0
            report[f"threads_{role}"] = 0.0
            report[f"cpu_{role}_max_thread_s"] = 0.0
            report[f"rundelay_{role}_max_thread_s"] = 0.0
        for tid in _list_tids():
            comm = _read(f"{_TASK_DIR}/{tid}/comm")
            cpu_s = self._cpu_seconds(tid)
            if comm is None or cpu_s is None:
                continue
            rundelay_s = self._run_delay_seconds(tid)
            if rundelay_s is None:
                rundelay_s = 0.0
            role = self.classify(tid, comm.strip())
            thread_cpu_s = cpu_s - self._last_cpu_s.get(tid, 0.0)
            thread_rundelay_s = rundelay_s - self._last_rundelay_s.get(tid, 0.0)
            report[f"cpu_{role}_s"] += thread_cpu_s
            report[f"rundelay_{role}_s"] += thread_rundelay_s
            report[f"cpu_{role}_max_thread_s"] = max(
                report[f"cpu_{role}_max_thread_s"], thread_cpu_s
            )
            report[f"rundelay_{role}_max_thread_s"] = max(
                report[f"rundelay_{role}_max_thread_s"], thread_rundelay_s
            )
            report[f"threads_{role}"] += 1.0
            self._last_cpu_s[tid] = cpu_s
            self._last_rundelay_s[tid] = rundelay_s
        return report

    def make_reporter(self) -> Callable[[], dict[str, float]]:
        """Returns a ``SamplingStats`` reporter covering the window since its last call."""
        self._read_window_guarded()
        return self._read_window_guarded

    def _read_window_guarded(self) -> dict[str, float]:
        try:
            return self.read_window()
        except Exception:
            logger.exception("Thread statistics scan failed")
            return {}
