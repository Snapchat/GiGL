import os
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from graphlearn_torch.distributed import shutdown_rpc

from gigl.common.logger import Logger
from gigl.distributed.dist_context import DistributedContext
from gigl.distributed.distributed_neighborloader import DistNeighborLoader
from gigl.distributed.utils import get_free_port
from gigl.distributed.utils.sampling_stats import SAMPLING_STATS_EVERY_N_BATCHES_ENV
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    CORA_NODE_ANCHOR_MOCKED_DATASET_INFO,
)
from tests.test_assets.distributed.run_distributed_dataset import (
    run_distributed_dataset,
)
from tests.test_assets.test_case import TestCase

logger = Logger()

_WORLD_SIZE = 2


def _iterate_loader(
    rank: int, dataset_port: int, loader_port: int, result_dir: str
) -> None:
    """Iterates the loader once on this rank and writes its batch count and wall time."""
    dataset = run_distributed_dataset(
        rank=rank,
        world_size=_WORLD_SIZE,
        mocked_dataset_info=CORA_NODE_ANCHOR_MOCKED_DATASET_INFO,
        _port=dataset_port,
    )
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{loader_port}",
        rank=rank,
        world_size=_WORLD_SIZE,
    )
    loader = DistNeighborLoader(
        dataset=dataset,
        context=DistributedContext(
            main_worker_ip_address="127.0.0.1",
            global_rank=rank,
            global_world_size=_WORLD_SIZE,
        ),
        local_process_rank=0,
        local_process_world_size=1,
        num_neighbors=[2, 2],
        batch_size=64,
        pin_memory_device=torch.device("cpu"),
    )
    start = time.perf_counter()
    batches = 0
    for _ in loader:
        batches += 1
    wall = time.perf_counter() - start
    with open(os.path.join(result_dir, f"rank_{rank}.txt"), "w") as result:
        result.write(f"{batches} {wall}")
    shutdown_rpc()


def _timed_epoch(stats_every_n_batches: str) -> tuple[int, float]:
    """Returns the total batch count and the slowest rank's wall time for one epoch."""
    if stats_every_n_batches:
        os.environ[SAMPLING_STATS_EVERY_N_BATCHES_ENV] = stats_every_n_batches
    else:
        os.environ.pop(SAMPLING_STATS_EVERY_N_BATCHES_ENV, None)
    with tempfile.TemporaryDirectory() as result_dir:
        mp.spawn(
            fn=_iterate_loader,
            args=(get_free_port(), get_free_port(), result_dir),
            nprocs=_WORLD_SIZE,
        )
        batches = 0
        wall = 0.0
        for rank in range(_WORLD_SIZE):
            with open(os.path.join(result_dir, f"rank_{rank}.txt")) as result:
                rank_batches, rank_wall = result.read().split()
            batches += int(rank_batches)
            wall = max(wall, float(rank_wall))
    return batches, wall


class SamplingStatsOverheadTest(TestCase):
    def test_stats_on_iterates_same_batches_as_stats_off(self) -> None:
        self.addCleanup(os.environ.pop, SAMPLING_STATS_EVERY_N_BATCHES_ENV, None)
        batches_off, wall_off = _timed_epoch("")
        batches_on, wall_on = _timed_epoch("50")
        self.assertGreater(batches_off, 0)
        self.assertEqual(batches_on, batches_off)
        logger.info(
            f"sampling stats overhead: off={wall_off:.2f}s on={wall_on:.2f}s "
            f"ratio={wall_on / wall_off:.3f} batches={batches_on}"
        )
