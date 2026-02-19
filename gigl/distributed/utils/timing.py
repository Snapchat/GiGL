"""Lightweight timing instrumentation for profiling distributed graph store operations.

Usage:
    from gigl.distributed.utils.timing import TimingStats

    stats = TimingStats.get_instance()

    # As a context manager:
    with stats.track("my_operation"):
        do_something()

    # Manual recording:
    stats.record("my_operation", duration_seconds)

    # Print report (sorted by total time descending):
    stats.report(prefix="[Rank 0] ")
"""

import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Generator

from gigl.common.logger import Logger

logger = Logger()


class TimingStats:
    """Thread-safe, per-process timing statistics collector.

    Uses a singleton pattern so that all code paths within a single process
    accumulate into the same instance. Each spawned subprocess gets its own
    instance automatically.
    """

    _instance: "TimingStats | None" = None
    _instance_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "TimingStats":
        """Return (or create) the process-local singleton."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = TimingStats()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Discard the current singleton (useful between test runs)."""
        with cls._instance_lock:
            cls._instance = None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._durations: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def track(self, name: str) -> Generator[None, None, None]:
        """Context manager that records wall-clock time for *name*."""
        start = time.monotonic()
        try:
            yield
        finally:
            duration = time.monotonic() - start
            with self._lock:
                self._durations[name].append(duration)

    def record(self, name: str, duration: float) -> None:
        """Manually record a *duration* (seconds) for *name*."""
        with self._lock:
            self._durations[name].append(duration)

    def report(self, prefix: str = "") -> str:
        """Log and return a human-readable summary sorted by total time (descending).

        Args:
            prefix: Prepended to the header line (e.g. "[Rank 0] ").

        Returns:
            The full report string.
        """
        with self._lock:
            if not self._durations:
                msg = f"{prefix}No timing data collected."
                logger.info(msg)
                return msg

            lines: list[str] = []
            lines.append(f"{prefix}===== TIMING STATS (pid={os.getpid()}) =====")

            sorted_stats = sorted(
                self._durations.items(),
                key=lambda item: sum(item[1]),
                reverse=True,
            )

            for name, durations in sorted_stats:
                total = sum(durations)
                count = len(durations)
                avg = total / count
                min_d = min(durations)
                max_d = max(durations)
                lines.append(
                    f"  {name}: total={total:.3f}s, count={count}, "
                    f"avg={avg:.3f}s, min={min_d:.3f}s, max={max_d:.3f}s"
                )

            lines.append(f"{prefix}{'=' * 42}")
            report_str = "\n".join(lines)
            logger.info(report_str)
            return report_str
