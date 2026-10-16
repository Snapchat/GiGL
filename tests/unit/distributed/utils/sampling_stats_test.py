import os
import re
import threading
from unittest import mock

import torch

from gigl.distributed.utils.sampling_stats import (
    SAMPLING_STATS_EVERY_N_BATCHES_ENV,
    SamplingStats,
    sampling_stats_every_n_batches,
    tensor_map_bytes,
)
from tests.test_assets.test_case import TestCase


def _fields(line: str) -> dict[str, str]:
    return dict(re.findall(r"(\S+)=(\S+)", line))


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
        line = stats.batch_done()
        assert line is not None
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
        second = stats.batch_done()
        assert second is not None
        self.assertNotIn("rpc_count", _fields(second))
        self.assertEqual(_fields(second)["other"], "0.50")

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
        line = stats.batch_done()
        assert line is not None
        self.assertEqual(_fields(line)["calls"], "8000.00")

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
            env = {} if value is None else {SAMPLING_STATS_EVERY_N_BATCHES_ENV: value}
            with mock.patch.dict(os.environ, env, clear=True):
                self.assertEqual(sampling_stats_every_n_batches(), 0, msg=repr(value))

    def test_positive_value_enables_stats(self) -> None:
        with mock.patch.dict(
            os.environ, {SAMPLING_STATS_EVERY_N_BATCHES_ENV: "50"}, clear=True
        ):
            self.assertEqual(sampling_stats_every_n_batches(), 50)
