"""Lightweight benchmarking utilities for timing method calls.

Provides a process-local, thread-safe timer singleton and a class decorator
to wrap selected methods with timing instrumentation.

Usage::

    from gigl.distributed.benchmark import benchmark_methods, BenchmarkTimer

    @benchmark_methods("get_node_ids", "get_ablp_input")
    class MyClass:
        def get_node_ids(self): ...
        def get_ablp_input(self): ...

    # After test runs:
    BenchmarkTimer.instance().print_stats("My Test Results")
"""

import functools
import os
import threading
import time
from typing import Callable, Optional


class _TimingStats:
    """Accumulated timing statistics for a single method."""

    __slots__ = ("call_count", "total_time", "min_time", "max_time")

    def __init__(self) -> None:
        self.call_count: int = 0
        self.total_time: float = 0.0
        self.min_time: float = float("inf")
        self.max_time: float = 0.0

    @property
    def avg_time(self) -> float:
        return self.total_time / self.call_count if self.call_count > 0 else 0.0

    def record(self, elapsed: float) -> None:
        self.call_count += 1
        self.total_time += elapsed
        if elapsed < self.min_time:
            self.min_time = elapsed
        if elapsed > self.max_time:
            self.max_time = elapsed


class BenchmarkTimer:
    """Process-local, thread-safe singleton that collects method timing statistics.

    Each spawned process gets its own independent instance via the singleton pattern.
    All ``record`` and ``print_stats`` calls are guarded by a threading lock so that
    RPC handler threads on the server side don't corrupt the data.
    """

    _instance: Optional["BenchmarkTimer"] = None
    _creation_lock = threading.Lock()

    def __init__(self) -> None:
        self._stats: dict[str, _TimingStats] = {}
        self._lock = threading.Lock()

    @classmethod
    def instance(cls) -> "BenchmarkTimer":
        """Return the process-local singleton, creating it on first access."""
        if cls._instance is None:
            with cls._creation_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def record(self, name: str, elapsed: float) -> None:
        """Record a single timing measurement for *name*."""
        with self._lock:
            if name not in self._stats:
                self._stats[name] = _TimingStats()
            self._stats[name].record(elapsed)

    def reset(self) -> None:
        """Clear all collected statistics."""
        with self._lock:
            self._stats.clear()

    def print_stats(self, title: str = "Benchmark Timing Stats") -> None:
        """Pretty-print collected timing stats as an ASCII table."""
        with self._lock:
            snapshot = dict(self._stats)

        if not snapshot:
            print(
                f"\n{'=' * 80}\n"
                f"  {title} (PID {os.getpid()}) -- no data collected\n"
                f"{'=' * 80}\n",
                flush=True,
            )
            return

        sorted_entries = sorted(snapshot.items())

        method_width = max(len(name) for name in snapshot)
        method_width = max(method_width, len("Method"))

        def _fmt(seconds: float) -> str:
            """Format a duration for display, choosing appropriate units."""
            if seconds == float("inf"):
                return "n/a"
            if seconds < 0.001:
                return f"{seconds * 1_000_000:.1f}us"
            elif seconds < 1.0:
                return f"{seconds * 1_000:.3f}ms"
            else:
                return f"{seconds:.4f}s"

        header = (
            f"  {'Method':<{method_width}}  {'Calls':>7}  {'Total':>12}  "
            f"{'Avg':>12}  {'Min':>12}  {'Max':>12}"
        )
        rule = "=" * len(header)
        dash = "-" * len(header)

        lines: list[str] = [
            "",
            rule,
            f"  {title} (PID {os.getpid()})",
            rule,
            header,
            dash,
        ]

        for name, stats in sorted_entries:
            lines.append(
                f"  {name:<{method_width}}  {stats.call_count:>7}  "
                f"{_fmt(stats.total_time):>12}  "
                f"{_fmt(stats.avg_time):>12}  "
                f"{_fmt(stats.min_time):>12}  "
                f"{_fmt(stats.max_time):>12}"
            )

        lines.append(rule)
        lines.append("")
        print("\n".join(lines), flush=True)


def benchmark(func: Callable) -> Callable:
    """Decorator that times a function/method and records statistics."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        cls_name = ""
        if args and hasattr(args[0], "__class__"):
            cls_name = args[0].__class__.__name__ + "."
        name = f"{cls_name}{func.__name__}"

        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            BenchmarkTimer.instance().record(name, elapsed)

    return wrapper


def benchmark_methods(*method_names: str) -> Callable:
    """Class decorator that wraps the listed methods with benchmark timing.

    Example::

        @benchmark_methods("forward", "backward")
        class MyModel:
            def forward(self, x): ...
            def backward(self, grad): ...
    """

    def class_decorator(cls):  # type: ignore[no-untyped-def]
        for name in method_names:
            original = getattr(cls, name, None)
            if original is not None and callable(original):
                setattr(cls, name, benchmark(original))
        return cls

    return class_decorator
