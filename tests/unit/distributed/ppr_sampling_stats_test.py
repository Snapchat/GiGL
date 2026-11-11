import os

import torch
import torch.multiprocessing as mp
from absl.testing import absltest
from graphlearn_torch.distributed import init_rpc, init_worker_group, shutdown_rpc
from graphlearn_torch.sampler import NodeSamplerInput

from gigl.distributed.dist_ppr_sampler import DistPPRNeighborSampler
from gigl.distributed.utils import get_free_port
from gigl.distributed.utils.sampling_stats import SAMPLING_STATS_EVERY_N_BATCHES_ENV
from tests.test_assets.distributed.test_dataset import (
    create_homogeneous_dataset,
)
from tests.test_assets.distributed.utils import create_test_process_group
from tests.test_assets.test_case import TestCase

_EDGE_INDEX = torch.tensor(
    [
        [0, 1, 0, 2, 1, 2, 1, 3, 2, 4],
        [1, 0, 2, 0, 2, 1, 3, 1, 4, 2],
    ]
)
_SEEDS = torch.tensor([0, 3])
_PPR_STAT_KEYS = (
    "induce_ms",
    "nodes_per_batch",
    "edges_per_batch",
    "ppr_state_init_ms",
    "ppr_drain_ms",
    "ppr_push_ms",
    "ppr_extract_ms",
    "ppr_iterations",
    "ppr_neighbor_fetch_count",
    "ppr_neighbor_fetch_issue_ms",
    "ppr_neighbor_fetch_wait_ms",
)


def _sample_batch(_: int, master_port: int, output: dict) -> None:
    """Samples one PPR batch with statistics off and one with them on, and records both."""
    create_test_process_group()
    dataset = create_homogeneous_dataset(edge_index=_EDGE_INDEX, edge_dir="out")
    init_worker_group(world_size=1, rank=0)
    init_rpc(master_addr="127.0.0.1", master_port=master_port, num_rpc_threads=2)
    for name, every_n_batches in (("off", "0"), ("on", "1")):
        os.environ[SAMPLING_STATS_EVERY_N_BATCHES_ENV] = every_n_batches
        sampler = DistPPRNeighborSampler(
            data=dataset,
            num_neighbors=[1],
            concurrency=1,
            degree_tensors=dataset.degree_tensor,
            eps=1e-4,
            max_ppr_nodes=4,
        )
        sampler.start_loop()
        result = sampler.run_task(
            coro=sampler._sample_from_nodes(NodeSamplerInput(node=_SEEDS))
        )
        report = None
        if sampler._stats is not None:
            sampler._stats.batch_done()
            report = sampler._stats.last_report
        output[name] = {
            "node": result.node.tolist(),
            "edge_index": result.metadata["edge_index"].tolist(),
            "edge_attr": result.metadata["edge_attr"].tolist(),
            "stats_is_none": sampler._stats is None,
            "report": report,
        }
        sampler.shutdown_loop()
    shutdown_rpc()


def _report_values(report: str) -> dict[str, float]:
    values = {}
    for field in report.split():
        key, separator, value = field.partition("=")
        if separator:
            try:
                values[key] = float(value)
            except ValueError:
                pass
    return values


class PPRSamplingStatsTest(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        output = mp.Manager().dict()
        mp.spawn(_sample_batch, args=(get_free_port(), output), nprocs=1)
        cls.off = dict(output["off"])
        cls.on = dict(output["on"])

    def test_stats_off_records_nothing(self) -> None:
        self.assertTrue(self.off["stats_is_none"])
        self.assertIsNone(self.off["report"])

    def test_stats_on_reports_ppr_keys(self) -> None:
        values = _report_values(self.on["report"])
        for key in _PPR_STAT_KEYS:
            self.assertIn(key, values)
            self.assertGreaterEqual(values[key], 0.0, msg=key)
        self.assertEqual(values["nodes_per_batch"], len(self.on["node"]))
        self.assertEqual(values["edges_per_batch"], len(self.on["edge_index"][0]))
        self.assertGreater(values["ppr_iterations"], 0)
        # Iterations whose frontier is already cached push without a fetch.
        self.assertGreater(values["ppr_neighbor_fetch_count"], 0)
        self.assertLessEqual(
            values["ppr_neighbor_fetch_count"], values["ppr_iterations"]
        )

    def test_stats_do_not_change_results(self) -> None:
        for key in ("node", "edge_index", "edge_attr"):
            self.assertEqual(self.off[key], self.on[key], msg=key)


if __name__ == "__main__":
    absltest.main()
