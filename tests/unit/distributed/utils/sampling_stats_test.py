import os
import re
import threading
import time
from typing import TypeVar
from unittest import mock

import torch
from graphlearn_torch.distributed.dist_feature import (
    DistFeature,
    RpcFeatureLookupCallee,
)
from graphlearn_torch.distributed.dist_neighbor_sampler import RpcSamplingCallee

from gigl.distributed.utils.fused_feature import FusedDistFeature
from gigl.distributed.utils.sampling_stats import (
    SAMPLING_STATS_EVERY_N_BATCHES_ENV,
    SAMPLING_WORKER_NUM_RPC_THREADS_ENV,
    SamplingStats,
    instrument_rpc_callees,
    sampling_stats_every_n_batches,
    sampling_worker_num_rpc_threads_override,
    tensor_map_bytes,
)
from tests.test_assets.test_case import TestCase


def _fields(line: str) -> dict[str, str]:
    return dict(re.findall(r"(\S+)=(\S+)", line))


def _window_line(stats: SamplingStats) -> str:
    line = stats.batch_done()
    assert line is not None
    return line


class SamplingStatsTest(TestCase):
    def test_window_reports_per_batch_means_then_resets(self) -> None:
        lines: list[str] = []
        stats = SamplingStats(
            "GLT_SAMPLING_STATS",
            every_n_batches=2,
            scope="worker=3",
            log_fn=lines.append,
        )

        stats.add("rpc_count", 3)
        with stats.timer("send_ms"):
            pass
        self.assertIsNone(stats.batch_done())
        self.assertEqual(lines, [])

        stats.add("rpc_count", 5)
        line = _window_line(stats)
        self.assertEqual(lines, [line])
        self.assertTrue(line.startswith("GLT_SAMPLING_STATS worker=3 batches=2 "))
        fields = _fields(line)
        self.assertEqual(fields["rpc_count"], "4.00")
        # The timer fired in one batch of two, so its per-batch mean is half its total.
        self.assertGreaterEqual(float(fields["send_ms"]), 0.0)
        for key in ("wall_s", "batches_per_s", "proc_cpu_s", "loop_cpu_s"):
            self.assertIn(key, fields)

        # The next window starts empty: keys from the previous window are gone.
        stats.add("other", 1)
        stats.batch_done()
        second = _window_line(stats)
        self.assertNotIn("rpc_count", _fields(second))
        self.assertEqual(_fields(second)["other"], "0.50")

    def test_reporter_values_are_appended_unscaled_at_window_close(self) -> None:
        lines: list[str] = []
        stats = SamplingStats("P", every_n_batches=2, scope="s", log_fn=lines.append)
        stats.register_reporter(lambda: {"gauge_x": 7.0})
        stats.add("k", 4.0)
        stats.batch_done()
        stats.add("k", 2.0)
        line = _window_line(stats)
        self.assertEqual(lines, [line])
        self.assertEqual(stats.last_report, line)
        fields = _fields(line)
        self.assertEqual(fields["k"], "3.00")
        self.assertEqual(fields["gauge_x"], "7.00")
        self.assertLessEqual(float(fields["t_start"]), float(fields["t_end"]))

    def test_last_report_is_none_until_a_window_closes(self) -> None:
        stats = SamplingStats("P", every_n_batches=2, scope="s", log_fn=lambda _: None)
        self.assertIsNone(stats.last_report)
        stats.batch_done()
        self.assertIsNone(stats.last_report)

    def test_concurrent_adds_are_not_lost(self) -> None:
        stats = SamplingStats("P", every_n_batches=1, scope="s", log_fn=lambda _: None)

        def worker() -> None:
            for _ in range(1000):
                stats.add("calls", 1)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(_fields(_window_line(stats))["calls"], "8000.00")

    def test_rejects_non_positive_window(self) -> None:
        with self.assertRaises(ValueError):
            SamplingStats("P", every_n_batches=0, scope="s")

    def test_tensor_map_bytes_counts_tensor_storage_only(self) -> None:
        message = {
            "ids": torch.zeros(10, dtype=torch.int64),
            "nfeats": torch.zeros(10, 4, dtype=torch.float32),
            "#META.flag": "not a tensor",
        }
        self.assertEqual(tensor_map_bytes(message), 10 * 8 + 10 * 4 * 4)


class SamplingStatsEnvTest(TestCase):
    def test_unset_zero_and_invalid_values_disable_stats(self) -> None:
        for value in (None, "0", "", "abc", "-5"):
            env = {}
            if value is not None:
                env = {SAMPLING_STATS_EVERY_N_BATCHES_ENV: value}
            with mock.patch.dict(os.environ, env, clear=True):
                self.assertEqual(sampling_stats_every_n_batches(), 0, msg=repr(value))

    def test_positive_values_enable_stats_and_thread_overrides(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                SAMPLING_STATS_EVERY_N_BATCHES_ENV: "50",
                SAMPLING_WORKER_NUM_RPC_THREADS_ENV: "4",
            },
            clear=True,
        ):
            self.assertEqual(sampling_stats_every_n_batches(), 50)
            self.assertEqual(sampling_worker_num_rpc_threads_override(), 4)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(sampling_worker_num_rpc_threads_override())


_CalleeT = TypeVar("_CalleeT")


class _FakeSamplingCallee(RpcSamplingCallee):
    def __init__(self) -> None:
        pass

    def call(self, ids, num_nbr, etype, *unexpected):
        assert not unexpected
        return ids


class _FakeFeatureCallee(RpcFeatureLookupCallee):
    def __init__(self) -> None:
        pass

    def call(self, ids, input_type, *unexpected):
        assert not unexpected
        return torch.zeros(ids.numel(), 4)


class InstrumentRpcCalleesQueueTimeTest(TestCase):
    def _instrumented(self, callee: _CalleeT) -> tuple[SamplingStats, _CalleeT]:
        stats = SamplingStats(
            "GLT_SAMPLING_STATS", every_n_batches=1, scope="t", log_fn=lambda _: None
        )
        with mock.patch(
            "graphlearn_torch.distributed.rpc._rpc_callee_pool", {0: callee}
        ):
            instrument_rpc_callees(stats)
        return stats, callee

    def test_sampling_callee_records_queue_time_and_strips_timestamp(self) -> None:
        stats, callee = self._instrumented(_FakeSamplingCallee())
        ids = torch.arange(3)
        result = callee.call(ids, 2, None, time.time() - 0.05)
        self.assertIs(result, ids)
        fields = _fields(_window_line(stats))
        self.assertGreaterEqual(float(fields["srv_sample_queue_ms"]), 40.0)

    def test_sampling_callee_without_timestamp_records_no_queue_time(self) -> None:
        stats, callee = self._instrumented(_FakeSamplingCallee())
        callee.call(torch.arange(3), 2, None)
        self.assertNotIn("srv_sample_queue_ms", _fields(_window_line(stats)))

    def test_feature_callee_records_queue_time(self) -> None:
        stats, callee = self._instrumented(_FakeFeatureCallee())
        callee.call(torch.arange(3), None, time.time() - 0.05)
        fields = _fields(_window_line(stats))
        self.assertGreaterEqual(float(fields["srv_feat_queue_ms"]), 40.0)

    def test_feature_callee_without_timestamp_records_no_queue_time(self) -> None:
        stats, callee = self._instrumented(_FakeFeatureCallee())
        callee.call(torch.arange(3), None)
        self.assertNotIn("srv_feat_queue_ms", _fields(_window_line(stats)))


class FusedDistFeatureRemoteGetTest(TestCase):
    def test_uninstrumented_remote_get_is_glts(self) -> None:
        feature = FusedDistFeature.__new__(FusedDistFeature)
        ids = torch.arange(3)
        with mock.patch.object(
            DistFeature, "_remote_selecting_get", return_value="glt"
        ) as glt_get:
            self.assertEqual(feature._remote_selecting_get(ids, "user"), "glt")
        glt_get.assert_called_once_with(ids, "user")
