import multiprocessing
import time

from absl.testing import absltest

from gigl.distributed.utils.dist_sampler import join_workers_with_shared_deadline
from tests.test_assets.test_case import TestCase


class JoinWorkersWithSharedDeadlineTest(TestCase):
    def test_unresponsive_workers_share_one_deadline(self) -> None:
        # Fork, not spawn: a spawned child re-imports the test runner, which takes seconds.
        context = multiprocessing.get_context("fork")
        # Like sampling workers blocked on a full channel: they never exit on their own.
        workers = [
            context.Process(target=time.sleep, args=(600,), daemon=True)
            for _ in range(4)
        ]
        for worker in workers:
            worker.start()
            self.addCleanup(worker.join)
            self.addCleanup(worker.terminate)

        start = time.monotonic()
        join_workers_with_shared_deadline(workers, timeout=0.5)
        elapsed = time.monotonic() - start

        # A wait for each worker would take 4 * 0.5 = 2 seconds.
        self.assertGreaterEqual(elapsed, 0.5)
        self.assertLess(elapsed, 1.5)
        self.assertTrue(all(worker.is_alive() for worker in workers))


if __name__ == "__main__":
    absltest.main()
