import ctypes
import os
import threading
import time
from unittest import mock

from gigl.distributed.utils.thread_stats import ThreadRoles
from tests.test_assets.test_case import TestCase

_RPC_POOL_COMM = "pt_thread_pool"
_TASK = "/proc/self/task"


def _name_current_thread(name: str) -> None:
    ctypes.CDLL("libc.so.6").prctl(15, name.encode(), 0, 0, 0)


class _NamedThread:
    """Starts a thread that names itself, reports its TID, then waits to be released."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._ready = threading.Event()
        self._release = threading.Event()
        self.tid = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait()

    def _run(self) -> None:
        _name_current_thread(self._name)
        self.tid = threading.get_native_id()
        self._ready.set()
        self._release.wait()

    def stop(self) -> None:
        self._release.set()
        self._thread.join()


class ThreadRolesTest(TestCase):
    def test_rpc_pool_is_the_set_difference_across_rpc_init(self) -> None:
        roles = ThreadRoles()
        roles.snapshot_before_rpc()
        threads = [_NamedThread(_RPC_POOL_COMM) for _ in range(2)]
        self.addCleanup(lambda: [t.stop() for t in threads])
        roles.snapshot_after_rpc()
        for thread in threads:
            self.assertEqual(roles.classify(thread.tid, _RPC_POOL_COMM), "rpc_pool")

    def test_tensorpipe_io_threads_are_not_rpc_pool(self) -> None:
        roles = ThreadRoles()
        roles.snapshot_before_rpc()
        thread = _NamedThread("TP_UV_loop")
        self.addCleanup(thread.stop)
        roles.snapshot_after_rpc()
        self.assertEqual(roles.classify(thread.tid, "TP_UV_loop"), "tp_uv")

    def test_other_threads_created_across_rpc_init_are_tp_other(self) -> None:
        roles = ThreadRoles()
        roles.snapshot_before_rpc()
        thread = _NamedThread("TP_reactor")
        self.addCleanup(thread.stop)
        roles.snapshot_after_rpc()
        self.assertEqual(roles.classify(thread.tid, "TP_reactor"), "tp_other")

    def test_same_named_thread_created_after_init_is_omp(self) -> None:
        roles = ThreadRoles()
        roles.snapshot_before_rpc()
        roles.snapshot_after_rpc()
        thread = _NamedThread(_RPC_POOL_COMM)
        self.addCleanup(thread.stop)
        self.assertEqual(roles.classify(thread.tid, _RPC_POOL_COMM), "omp")

    def test_loop_and_other_threads(self) -> None:
        roles = ThreadRoles()
        roles.set_loop_tid(1234)
        self.assertEqual(roles.classify(1234, "python"), "loop")
        self.assertEqual(roles.classify(99999, "python"), "other")

    def test_busy_thread_accrues_cpu_in_its_role(self) -> None:
        roles = ThreadRoles()
        started = threading.Event()
        done = threading.Event()
        go = threading.Event()
        release = threading.Event()
        tids: list[int] = []

        def spin() -> None:
            tids.append(threading.get_native_id())
            started.set()
            go.wait()
            deadline = time.perf_counter() + 0.3
            while time.perf_counter() < deadline:
                pass
            done.set()
            release.wait()

        thread = threading.Thread(target=spin, daemon=True)
        thread.start()
        started.wait()
        roles.set_loop_tid(tids[0])
        roles.read_window()
        go.set()
        done.wait()
        window = roles.read_window()
        release.set()
        thread.join()
        self.assertGreaterEqual(window["cpu_loop_s"], 0.1)
        self.assertEqual(window["threads_loop"], 1.0)
        for role in ("rpc_pool", "tp_uv", "tp_other", "loop", "omp", "other"):
            self.assertIn(f"cpu_{role}_s", window)
            self.assertIn(f"rundelay_{role}_s", window)
            self.assertIn(f"threads_{role}", window)

    def test_second_window_reports_only_new_cpu(self) -> None:
        roles = ThreadRoles()
        roles.set_loop_tid(threading.get_native_id())
        end = time.perf_counter() + 0.3
        while time.perf_counter() < end:
            pass
        first = roles.read_window()
        second = roles.read_window()
        self.assertGreaterEqual(first["cpu_loop_s"], 0.2)
        self.assertLess(second["cpu_loop_s"], 0.1)

    def test_exited_thread_mid_scan_is_skipped(self) -> None:
        roles = ThreadRoles()
        dead_tid = 4194000
        listed = [str(tid) for tid in os.listdir(_TASK)] + [str(dead_tid)]

        def listdir(path: str) -> list[str]:
            return listed

        with mock.patch("gigl.distributed.utils.thread_stats.os.listdir", listdir):
            window = roles.read_window()
        total_threads = sum(v for k, v in window.items() if k.startswith("threads_"))
        self.assertNotIn(dead_tid, roles._last_cpu_s)
        self.assertEqual(total_threads, len(listed) - 1)

    def test_first_reporter_window_excludes_cpu_before_creation(self) -> None:
        roles = ThreadRoles()
        roles.set_loop_tid(threading.get_native_id())
        end = time.perf_counter() + 0.3
        while time.perf_counter() < end:
            pass
        report = roles.make_reporter()()
        self.assertLess(report["cpu_loop_s"], 0.05)

    def test_priming_failure_does_not_raise(self) -> None:
        roles = ThreadRoles()
        with mock.patch.object(roles, "read_window", side_effect=ValueError("bad")):
            roles.make_reporter()

    def test_reporter_swallows_read_window_failures(self) -> None:
        roles = ThreadRoles()
        reporter = roles.make_reporter()
        with mock.patch.object(roles, "read_window", side_effect=ValueError("bad")):
            self.assertEqual(reporter(), {})
