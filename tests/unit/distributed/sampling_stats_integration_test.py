import os
import re
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from graphlearn_torch.distributed import shutdown_rpc

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

_WORLD_SIZE = 2
_SAMPLER_KEYS = (
    "hop_rpc_count",
    "hop_rpc_wait_ms",
    "nfeat_rpc_count",
    "nfeat_rpc_wait_ms",
    "nfeat_stitch_ms",
    "nfeat_local_ms",
    "send_ms",
    "srv_feat_calls",
    "srv_sample_calls",
    "induce_ms",
    "nodes_per_batch",
)


def _run_two_partition_loader(
    rank: int, dataset_port: int, loader_port: int, stderr_dir: str
) -> None:
    """Runs one compute rank and leaves its sampling workers' log lines in ``stderr_dir``.

    The sampling workers are child processes, so their stats lines are collected by pointing
    this process's stderr, which the children inherit, at a file.
    """
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
    stderr_fd = os.open(
        os.path.join(stderr_dir, f"rank_{rank}.log"), os.O_WRONLY | os.O_CREAT
    )
    os.dup2(stderr_fd, 2)
    loader = DistNeighborLoader(
        dataset=dataset,
        # Each rank plays one machine, which is what makes the two data partitions remote
        # to each other.
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
    for _ in loader:
        pass
    shutdown_rpc()


class SamplingStatsIntegrationTest(TestCase):
    def test_two_partition_sampling_reports_every_stage(self) -> None:
        os.environ[SAMPLING_STATS_EVERY_N_BATCHES_ENV] = "1"
        self.addCleanup(os.environ.pop, SAMPLING_STATS_EVERY_N_BATCHES_ENV)
        with tempfile.TemporaryDirectory() as stderr_dir:
            mp.spawn(
                fn=_run_two_partition_loader,
                args=(get_free_port(), get_free_port(), stderr_dir),
                nprocs=_WORLD_SIZE,
            )
            peak: dict[str, float] = {}
            for name in os.listdir(stderr_dir):
                with open(os.path.join(stderr_dir, name)) as log:
                    for line in log:
                        if "GLT_SAMPLING_STATS" not in line:
                            continue
                        for key, value in re.findall(r"(\S+)=(\S+)", line):
                            peak[key] = max(peak.get(key, 0.0), float(value))
        for key in _SAMPLER_KEYS:
            self.assertGreater(peak.get(key, 0.0), 0.0, msg=key)
