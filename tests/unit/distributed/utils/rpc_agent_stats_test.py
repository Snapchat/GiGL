import math
import time
from typing import Any

import torch
import torch.multiprocessing as mp
from absl.testing import absltest
from graphlearn_torch.distributed import init_rpc, init_worker_group, shutdown_rpc
from torch.distributed import rpc

from gigl.distributed.utils.networking import get_free_port
from gigl.distributed.utils.rpc_agent_stats import RpcAgentSampler
from tests.test_assets.test_case import TestCase

_WORLD_SIZE = 2
_NUM_RPC_THREADS = 4


def _run_rpc_rank(rank: int, port: int, result_queue: Any) -> None:
    init_worker_group(world_size=_WORLD_SIZE, rank=rank, group_name="rpc_agent_test")
    init_rpc(
        master_addr="127.0.0.1",
        master_port=port,
        num_rpc_threads=_NUM_RPC_THREADS,
    )
    report: dict[str, float] = {}
    if rank == 0:
        sampler = RpcAgentSampler(interval_s=0.05)
        sampler.start()
        tensor = torch.ones(100_000)
        futures = [
            rpc.rpc_async("rpc_agent_test_1", torch.add, args=(tensor, tensor))
            for _ in range(20)
        ]
        time.sleep(0.5)
        for future in futures:
            future.wait()
        sampler.stop()
        report = sampler.make_reporter()()
    shutdown_rpc()
    if rank == 0:
        result_queue.put(report)


class RpcAgentSamplerTest(TestCase):
    def test_samples_a_live_rpc_agent(self) -> None:
        result_queue = mp.get_context("spawn").SimpleQueue()
        mp.spawn(
            fn=_run_rpc_rank,
            args=(get_free_port(), result_queue),
            nprocs=_WORLD_SIZE,
        )
        report = result_queue.get()

        self.assertEqual(report["rpc_pool_size"], _NUM_RPC_THREADS)
        self.assertGreaterEqual(report["rpc_samples"], 5)
        self.assertLessEqual(report["rpc_idle_threads_min"], report["rpc_pool_size"])

    def test_empty_window_reports_zero_samples_without_nans(self) -> None:
        reporter = RpcAgentSampler().make_reporter()

        for _ in range(2):
            report = reporter()
            self.assertEqual(report["rpc_samples"], 0.0)
            for key, value in report.items():
                self.assertFalse(math.isnan(value), msg=key)


if __name__ == "__main__":
    absltest.main()
