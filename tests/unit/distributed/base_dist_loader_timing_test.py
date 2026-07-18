"""Unit tests for the generic, opt-in collate-vs-recv timing accumulator on
``BaseDistLoader.__next__``.

These tests do NOT spin up GLT distributed infrastructure. They construct a bare
object, attach the minimal attributes ``__next__`` reads, and drive the override
directly to assert it (a) preserves GLT control flow and (b) attributes wall time
to ``recv`` vs ``collate`` only when timing is enabled.
"""

import time
import types

from torch_geometric.data import Data, HeteroData

from gigl.distributed.base_dist_loader import BaseDistLoader
from tests.test_assets.test_case import TestCase


def _make_bare_loader(
    *,
    num_expected: int,
    recv_sleep_s: float,
    collate_sleep_s: float,
    collect_loader_timings: bool,
) -> BaseDistLoader:
    """Build a BaseDistLoader without running its __init__ and wire only the
    attributes ``__next__`` touches, with sleep-instrumented recv/collate."""
    loader = BaseDistLoader.__new__(BaseDistLoader)
    loader._num_recv = 0
    loader._num_expected = num_expected
    loader._with_channel = True
    loader._collect_loader_timings = collect_loader_timings
    loader._sync_cuda_for_timings = False
    loader._recv_time_s = 0.0
    loader._collate_time_s = 0.0
    loader._timed_batches = 0

    class _Channel:
        def recv(self) -> dict:
            time.sleep(recv_sleep_s)
            return {"payload": 1}

    loader._channel = _Channel()

    def _collate_fn(self, msg: dict) -> Data:
        time.sleep(collate_sleep_s)
        d = Data()
        d.batch_size = 1
        return d

    loader._collate_fn = types.MethodType(_collate_fn, loader)
    return loader


class TestBaseDistLoaderTiming(TestCase):
    def test_disabled_timing_is_noop_and_preserves_flow(self) -> None:
        loader = _make_bare_loader(
            num_expected=2,
            recv_sleep_s=0.01,
            collate_sleep_s=0.01,
            collect_loader_timings=False,
        )
        out1 = loader.__next__()
        out2 = loader.__next__()
        self.assertIsInstance(out1, (Data, HeteroData))
        self.assertIsInstance(out2, (Data, HeteroData))
        self.assertEqual(loader._num_recv, 2)
        # Counters untouched when disabled.
        self.assertEqual(loader._recv_time_s, 0.0)
        self.assertEqual(loader._collate_time_s, 0.0)
        self.assertEqual(loader._timed_batches, 0)
        # StopIteration after num_expected.
        with self.assertRaises(StopIteration):
            loader.__next__()

    def test_enabled_timing_attributes_recv_vs_collate(self) -> None:
        loader = _make_bare_loader(
            num_expected=3,
            recv_sleep_s=0.05,
            collate_sleep_s=0.01,
            collect_loader_timings=True,
        )
        for _ in range(3):
            loader.__next__()
        self.assertEqual(loader._timed_batches, 3)
        # recv slept ~5x longer than collate; assert the attribution ordering
        # holds with generous slack for scheduler jitter.
        self.assertGreater(loader._recv_time_s, loader._collate_time_s)
        self.assertGreater(loader._recv_time_s, 0.10)  # ~0.15s expected
        self.assertGreater(loader._collate_time_s, 0.0)
        self.assertEqual(loader._num_recv, 3)

    def test_reset_loader_timings_zeros_counters(self) -> None:
        loader = _make_bare_loader(
            num_expected=2,
            recv_sleep_s=0.01,
            collate_sleep_s=0.01,
            collect_loader_timings=True,
        )
        loader.__next__()
        self.assertGreater(loader._timed_batches, 0)
        loader.reset_loader_timings()
        self.assertEqual(loader._recv_time_s, 0.0)
        self.assertEqual(loader._collate_time_s, 0.0)
        self.assertEqual(loader._timed_batches, 0)
