import time

import torch

from gigl.distributed.base_dist_loader import BaseDistLoader
from tests.test_assets.test_case import TestCase


class _CollateHarness(BaseDistLoader):
    """Minimal BaseDistLoader subclass exercising the collate-timer accounting.

    Bypasses __init__ (which needs a real dataset/producer) and calls the REAL
    BaseDistLoader._collate_fn timing wrapper, but stubs the heavy parent
    ``graphlearn_torch`` collate by overriding the inner work via ``_collate_inner``.
    """

    def __init__(self, collate_s: float) -> None:
        self._collate_s = collate_s
        self.to_device = None  # skip the CUDA non-blocking-transfer branch
        self._non_blocking_transfers = False
        self._collect_loader_timings = False
        self._recv_time_s = 0.0
        self._collate_time_s = 0.0
        self._timed_batches = 0

    # The real BaseDistLoader._collate_fn calls super()._collate_fn(msg); we
    # intercept that by overriding the parent call site through a seam: the
    # timing wrapper added in Step 4 must call self._collate_inner(msg) for the
    # actual collation so tests can stub it. (Step 4 introduces _collate_inner.)
    def _collate_inner(self, msg):
        time.sleep(self._collate_s)
        return msg


class TestBaseDistLoaderCollateTimer(TestCase):
    def test_timer_off_by_default(self) -> None:
        loader = _CollateHarness(collate_s=0.0)
        for _ in range(3):
            loader._collate_fn({"stub": torch.zeros(1)})
        self.assertEqual(loader._timed_batches, 0)
        self.assertEqual(loader._collate_time_s, 0.0)

    def test_timer_accumulates_when_enabled(self) -> None:
        loader = _CollateHarness(collate_s=0.02)
        loader._collect_loader_timings = True
        loader.reset_loader_timings()
        for _ in range(4):
            loader._collate_fn({"stub": torch.zeros(1)})
        self.assertEqual(loader._timed_batches, 4)
        self.assertGreater(loader._collate_time_s, 0.06)  # ~4 * 0.02, loose bound

    def test_reset_zeroes_counters(self) -> None:
        loader = _CollateHarness(collate_s=0.0)
        loader._collate_time_s = 7.0
        loader._timed_batches = 9
        loader._recv_time_s = 5.0
        loader.reset_loader_timings()
        self.assertEqual(loader._collate_time_s, 0.0)
        self.assertEqual(loader._timed_batches, 0)
        self.assertEqual(loader._recv_time_s, 0.0)
