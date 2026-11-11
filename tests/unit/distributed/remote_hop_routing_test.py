import hashlib
import inspect
import os

import torch
import torch.multiprocessing as mp
from absl.testing import absltest
from graphlearn_torch.distributed import DistNeighborSampler as GLTDistNeighborSampler
from graphlearn_torch.distributed import (
    barrier,
    init_rpc,
    init_worker_group,
    shutdown_rpc,
)

from gigl.distributed.base_sampler import BaseDistNeighborSampler
from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.utils import get_free_port
from gigl.distributed.utils.partition_routing import GROUPED_ROUTING_ENV
from gigl.types.graph import GraphPartitionData, PartitionOutput
from tests.test_assets.test_case import TestCase

_NUM_NODES = 10

# sha256 of inspect.getsource of graphlearn_torch's DistNeighborSampler._sample_one_hop, which
# BaseDistNeighborSampler._sample_one_hop copies when grouped routing is on.
_GLT_SAMPLE_ONE_HOP_SHA256 = (
    "2158a4886c544e60380cc72e1011630dfd1f833ceb64bd50019f594f609fcb5d"
)


def _partition_dataset(rank: int) -> DistDataset:
    """One partition of a ring graph where node ``i`` has the single neighbor ``(i + 1) % 10``.

    Even nodes live on partition 0 and odd nodes on partition 1, each with its own out-edges.
    """
    sources = torch.arange(rank, _NUM_NODES, 2)
    partition_output = PartitionOutput(
        node_partition_book=torch.arange(_NUM_NODES) % 2,
        edge_partition_book=torch.arange(_NUM_NODES) % 2,
        partitioned_edge_index=GraphPartitionData(
            edge_index=torch.stack([sources, (sources + 1) % _NUM_NODES]),
            edge_ids=None,
        ),
        partitioned_node_features=None,
        partitioned_edge_features=None,
        partitioned_positive_labels=None,
        partitioned_negative_labels=None,
        partitioned_node_labels=None,
    )
    dataset = DistDataset(rank=rank, world_size=2, edge_dir="out")
    dataset.build(partition_output=partition_output)
    return dataset


def _sample_hop_on_partition(
    rank: int, master_port: int, output: dict, grouped: bool
) -> None:
    """Samples one hop for every node, half of which another partition owns, and records the neighbors."""
    os.environ[GROUPED_ROUTING_ENV] = "0"
    if grouped:
        os.environ[GROUPED_ROUTING_ENV] = "1"
    init_worker_group(world_size=2, rank=rank)
    init_rpc(master_addr="127.0.0.1", master_port=master_port, num_rpc_threads=4)
    sampler = BaseDistNeighborSampler(
        data=_partition_dataset(rank), num_neighbors=[1], concurrency=1
    )
    sampler.start_loop()
    barrier()
    srcs = torch.tensor([9, 0, 3, 4, 7, 2, 5, 8, 1, 6])
    result = sampler.run_task(coro=sampler._sample_one_hop(srcs, 1, None))
    barrier()
    output[rank] = (srcs.tolist(), result.nbr.tolist(), result.nbr_num.tolist())
    sampler.shutdown_loop()
    shutdown_rpc()


class RemoteHopRoutingTest(TestCase):
    def _assert_remote_hop_samples_the_ring(self, grouped: bool) -> None:
        output = mp.Manager().dict()
        mp.spawn(
            _sample_hop_on_partition,
            args=(get_free_port(), output, grouped),
            nprocs=2,
        )
        for rank in range(2):
            srcs, nbrs, nbr_num = output[rank]
            self.assertEqual(nbr_num, [1] * len(srcs))
            self.assertEqual(nbrs, [(src + 1) % _NUM_NODES for src in srcs])

    def test_remote_hop_without_grouped_routing(self) -> None:
        self._assert_remote_hop_samples_the_ring(grouped=False)

    def test_remote_hop_with_grouped_routing(self) -> None:
        self._assert_remote_hop_samples_the_ring(grouped=True)


class GltSampleOneHopDriftTest(TestCase):
    def test_glt_sample_one_hop_source_is_unchanged(self) -> None:
        source = inspect.getsource(GLTDistNeighborSampler._sample_one_hop)
        digest = hashlib.sha256(source.encode()).hexdigest()
        self.assertEqual(
            digest,
            _GLT_SAMPLE_ONE_HOP_SHA256,
            "graphlearn_torch's DistNeighborSampler._sample_one_hop changed. Re-diff it against "
            "gigl.distributed.base_sampler.BaseDistNeighborSampler._sample_one_hop and port the "
            f"change. After a deliberate GLT upgrade, set _GLT_SAMPLE_ONE_HOP_SHA256 to {digest}.",
        )


if __name__ == "__main__":
    absltest.main()
