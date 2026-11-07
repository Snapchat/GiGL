import math
import os
import re
import tempfile
from collections.abc import Mapping

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from graphlearn_torch.distributed import shutdown_rpc

from gigl.distributed.dist_context import DistributedContext
from gigl.distributed.distributed_neighborloader import DistNeighborLoader
from gigl.distributed.utils import get_free_port
from gigl.distributed.utils.sampling_stats import (
    SAMPLING_SKIP_EDGE_FEATURES_ENV,
    SAMPLING_STATS_EVERY_N_BATCHES_ENV,
)
from gigl.src.mocking.lib.mocked_dataset_resources import MockedDatasetInfo
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    CORA_NODE_ANCHOR_MOCKED_DATASET_INFO,
    HETEROGENEOUS_TOY_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
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
    rank: int,
    dataset_port: int,
    loader_port: int,
    stderr_dir: str,
    mocked_dataset_info: MockedDatasetInfo,
) -> None:
    """Runs one compute rank and leaves its sampling workers' log lines in ``stderr_dir``.

    The sampling workers are child processes, so their stats lines are collected by pointing
    this process's stderr, which the children inherit, at a file.
    """
    dataset = run_distributed_dataset(
        rank=rank,
        world_size=_WORLD_SIZE,
        mocked_dataset_info=mocked_dataset_info,
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
    os.close(stderr_fd)
    input_nodes = None
    if isinstance(dataset.node_ids, Mapping):
        node_type = sorted(dataset.node_ids)[0]
        input_nodes = (node_type, dataset.node_ids[node_type])
    loader = DistNeighborLoader(
        dataset=dataset,
        input_nodes=input_nodes,
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
    assert len(loader.sampling_worker_pids()) > 0
    shutdown_rpc()


def _collect_stats_lines(mocked_dataset_info: MockedDatasetInfo) -> list[str]:
    """Runs the two-rank loader with statistics on and returns every sampler and loader stats line."""
    os.environ[SAMPLING_STATS_EVERY_N_BATCHES_ENV] = "1"
    lines: list[str] = []
    try:
        with tempfile.TemporaryDirectory() as stderr_dir:
            mp.spawn(
                fn=_run_two_partition_loader,
                args=(
                    get_free_port(),
                    get_free_port(),
                    stderr_dir,
                    mocked_dataset_info,
                ),
                nprocs=_WORLD_SIZE,
            )
            for name in os.listdir(stderr_dir):
                with open(os.path.join(stderr_dir, name)) as log:
                    lines.extend(
                        line
                        for line in log
                        if re.search("GLT_(SAMPLING|LOADER)_STATS", line)
                    )
    finally:
        os.environ.pop(SAMPLING_STATS_EVERY_N_BATCHES_ENV, None)
    return lines


def _numeric_fields(line: str) -> dict[str, float]:
    fields: dict[str, float] = {}
    for key, value in re.findall(r"(\S+)=(\S+)", line):
        try:
            fields[key] = float(value)
        except ValueError:
            continue
    return fields


class SamplingStatsIntegrationTest(TestCase):
    def test_two_partition_sampling_reports_every_stage(self) -> None:
        lines = _collect_stats_lines(CORA_NODE_ANCHOR_MOCKED_DATASET_INFO)
        peak: dict[str, float] = {}
        for line in lines:
            self.assertIn(" host=", line)
            for key, value in _numeric_fields(line).items():
                peak[key] = max(peak.get(key, 0.0), value)
        for key in _SAMPLER_KEYS:
            self.assertGreater(peak.get(key, 0.0), 0.0, msg=key)
        for key in ("cpu_rpc_pool_s", "cpu_loop_s", "threads_rpc_pool"):
            self.assertGreater(peak.get(key, 0.0), 0.0, msg=key)
        for key in ("rpc_pool_size", "rpc_samples"):
            self.assertGreater(peak.get(key, 0.0), 0.0, msg=key)
        for key in ("srv_sample_queue_ms", "srv_feat_queue_ms"):
            self.assertIn(key, peak)
            self.assertGreaterEqual(peak[key], 0.0, msg=key)
        self.assertIn("rundelay_rpc_pool_s", peak)
        self.assertGreaterEqual(peak["rundelay_rpc_pool_s"], 0.0)
        self.assertIn("reporters_ms", peak)
        self.assertIn("cpu_tp_other_s", peak)
        self.assertIn("channel_qsize", peak)

    def test_heterogeneous_sampling_reports_finite_values(self) -> None:
        lines = _collect_stats_lines(
            HETEROGENEOUS_TOY_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO
        )
        self.assertGreater(len(lines), 0)
        keys: set[str] = set()
        for line in lines:
            for key, value in _numeric_fields(line).items():
                self.assertTrue(math.isfinite(value), msg=f"{key}={value}")
                keys.add(key)
        for key in ("induce_ms", "nodes_per_batch", "hop_rpc_count"):
            self.assertIn(key, keys)

    def test_heterogeneous_sampling_without_edge_features_reports_finite_values(
        self,
    ) -> None:
        os.environ[SAMPLING_SKIP_EDGE_FEATURES_ENV] = "1"
        self.addCleanup(os.environ.pop, SAMPLING_SKIP_EDGE_FEATURES_ENV)
        lines = _collect_stats_lines(
            HETEROGENEOUS_TOY_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO
        )
        self.assertGreater(len(lines), 0)
        for line in lines:
            for key, value in _numeric_fields(line).items():
                self.assertTrue(math.isfinite(value), msg=f"{key}={value}")
