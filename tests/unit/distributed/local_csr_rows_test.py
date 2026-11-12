"""Sampling on local-row topologies must match sampling on full-length ``indptr`` topologies.

Each test builds the same three-partition graph twice, once with ``local_csr_rows`` and once
without, samples it from every rank, and compares the outputs exactly.
"""

from types import SimpleNamespace
from typing import Any, Literal

import torch
import torch.multiprocessing as mp
from absl.testing import absltest
from graphlearn_torch.data import Graph
from graphlearn_torch.distributed import (
    barrier,
    init_rpc,
    init_worker_group,
    shutdown_rpc,
)
from graphlearn_torch.partition import RangePartitionBook
from graphlearn_torch.sampler import NeighborSampler, NodeSamplerInput
from parameterized import param, parameterized

from gigl.distributed.base_sampler import (
    BaseDistNeighborSampler,
    _translate_seeds_to_local_rows,
)
from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.dist_ppr_sampler import DistPPRNeighborSampler
from gigl.distributed.utils import get_free_port
from gigl.distributed.utils.degree import compute_and_broadcast_degree_tensor
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.types.graph import GraphPartitionData, PartitionOutput
from gigl.utils.csr import CompactTopology, get_row_offset
from tests.test_assets.test_case import TestCase

_USER = NodeType("user")
_FOLLOWS = EdgeType(_USER, Relation("follows"), _USER)
_LIKES = EdgeType(_USER, Relation("likes"), _USER)
# Uneven ranges. Nodes 27 to 29 have no edges, so the last rank ends in zero-degree rows.
_PARTITION_RANGES = [(0, 7), (7, 19), (19, 30)]
_NUM_NODES = 30
_NUM_PARTITIONS = 3
# Larger than any degree in the graph, so a fixed fanout returns every neighbor.
_LARGE_FANOUT = 64


def _edges(edge_type: EdgeType) -> torch.Tensor:
    """All edges of ``edge_type``, between nodes 0 to 26, with distinct (src, dst) pairs."""
    seed = 3
    if edge_type == _LIKES:
        seed = 5
    generator = torch.Generator().manual_seed(seed)
    pairs = torch.randint(0, 27, (2, 200), generator=generator)
    return torch.unique(pairs, dim=1)


def _partition_dataset(
    rank: int, edge_dir: Literal["in", "out"], local_csr_rows: bool
) -> DistDataset:
    """The partition of ``rank``: each edge goes to the owner of the node it is sampled from."""
    compressed = 0
    if edge_dir == "in":
        compressed = 1
    start, end = _PARTITION_RANGES[rank]
    partitioned_edge_index: dict[EdgeType, GraphPartitionData] = {}
    for edge_type in (_FOLLOWS, _LIKES):
        edges = _edges(edge_type)
        owned = (edges[compressed] >= start) & (edges[compressed] < end)
        partitioned_edge_index[edge_type] = GraphPartitionData(
            edge_index=edges[:, owned], edge_ids=None
        )
    partition_output = PartitionOutput(
        node_partition_book={
            _USER: RangePartitionBook(
                partition_ranges=_PARTITION_RANGES, partition_idx=rank
            )
        },
        edge_partition_book=None,
        partitioned_edge_index=partitioned_edge_index,
        partitioned_node_features=None,
        partitioned_edge_features=None,
        partitioned_positive_labels=None,
        partitioned_negative_labels=None,
        partitioned_node_labels=None,
    )
    dataset = DistDataset(rank=rank, world_size=_NUM_PARTITIONS, edge_dir=edge_dir)
    dataset.build(partition_output=partition_output, local_csr_rows=local_csr_rows)
    return dataset


def _to_lists(value: Any) -> Any:
    """Convert tensors, possibly nested in dicts and sampler outputs, to comparable lists."""
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _to_lists(item) for key, item in value.items()}
    if hasattr(value, "__dict__"):
        return {key: _to_lists(item) for key, item in vars(value).items()}
    return value


def _sample_on_partition(
    rank: int,
    rpc_port: int,
    process_group_port: int,
    edge_dir: Literal["in", "out"],
    local_csr_rows: bool,
    output: dict,
) -> None:
    """Sample every node from this rank, through routing and directly, and record the results."""
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{process_group_port}",
        rank=rank,
        world_size=_NUM_PARTITIONS,
    )
    init_worker_group(world_size=_NUM_PARTITIONS, rank=rank)
    init_rpc(master_addr="127.0.0.1", master_port=rpc_port, num_rpc_threads=4)
    dataset = _partition_dataset(rank, edge_dir, local_csr_rows)
    assert isinstance(dataset.graph, dict)
    start, _ = _PARTITION_RANGES[rank]
    expected_offset = 0
    if local_csr_rows:
        expected_offset = start
    for graph in dataset.graph.values():
        assert get_row_offset(graph.topo) == expected_offset

    degrees = compute_and_broadcast_degree_tensor(dataset.graph, edge_dir)
    sampler = BaseDistNeighborSampler(
        data=dataset, num_neighbors=[-1], with_edge=True, concurrency=1
    )
    ppr_sampler = DistPPRNeighborSampler(
        data=dataset,
        num_neighbors=[-1],
        concurrency=1,
        degree_tensors=degrees,
        max_ppr_nodes=10,
    )
    sampler.start_loop()
    ppr_sampler.start_loop()
    barrier()
    # Every id: below this rank's range, in it (first and last owned rows), and above it.
    seeds = torch.arange(_NUM_NODES)
    result: dict[str, Any] = {"degrees": _to_lists(degrees)}
    for edge_type in (_FOLLOWS, _LIKES):
        for fanout in (-1, _LARGE_FANOUT):
            routed = sampler.run_task(
                coro=sampler._sample_one_hop(seeds, fanout, edge_type)
            )
            direct = sampler.sampler.sample_one_hop(seeds, fanout, edge_type)
            key = f"{edge_type}/{fanout}"
            result[f"routed/{key}"] = _to_lists(routed)
            result[f"direct/{key}"] = _to_lists(direct)
    ppr_output = ppr_sampler.run_task(
        coro=ppr_sampler._sample_from_nodes(
            NodeSamplerInput(node=seeds, input_type=_USER)
        )
    )
    result["ppr"] = _to_lists(ppr_output)
    barrier()
    output[rank] = result
    sampler.shutdown_loop()
    ppr_sampler.shutdown_loop()
    shutdown_rpc()
    torch.distributed.destroy_process_group()


def _run(edge_dir: Literal["in", "out"], local_csr_rows: bool) -> dict[int, Any]:
    output = mp.Manager().dict()
    mp.spawn(
        _sample_on_partition,
        args=(get_free_port(), get_free_port(), edge_dir, local_csr_rows, output),
        nprocs=_NUM_PARTITIONS,
    )
    return dict(output)


class LocalCsrRowsSamplingTest(TestCase):
    @parameterized.expand([param("in", edge_dir="in"), param("out", edge_dir="out")])
    def test_local_rows_sample_exactly_as_full_rows(
        self, _name: str, edge_dir: Literal["in", "out"]
    ) -> None:
        full_rows = _run(edge_dir, local_csr_rows=False)
        local_rows = _run(edge_dir, local_csr_rows=True)
        for rank in range(_NUM_PARTITIONS):
            self.assertEqual(set(local_rows[rank]), set(full_rows[rank]))
            for key, expected in full_rows[rank].items():
                self.assertEqual(local_rows[rank][key], expected, f"rank {rank} {key}")
        # The graph must give the samplers real work: some neighbors from every partition.
        sampled = full_rows[0][f"routed/{_FOLLOWS}/-1"]
        self.assertGreater(len(sampled["nbr"]), _NUM_NODES)
        self.assertEqual(full_rows[0]["degrees"]["user"][27:], [0, 0, 0])


class LocalCsrRowsGuardTest(TestCase):
    """Operations that read ``indptr`` without ``sample_one_hop`` must refuse local rows."""

    def _dist_sampler(self, with_neg: bool) -> SimpleNamespace:
        """The attributes of a GLT distributed sampler that the translation reads."""
        topology = CompactTopology(
            torch.tensor([[7, 8], [0, 1]]),
            layout="CSR",
            num_rows=10,
            row_start=7,
            local_rows=True,
        )
        graph = Graph(topology, "CPU", None)
        return SimpleNamespace(
            dist_graph=SimpleNamespace(local_graph=graph),
            with_neg=with_neg,
            sampler=NeighborSampler(graph, [-1], device=torch.device("cpu")),
        )

    def test_negative_sampling_raises(self) -> None:
        with self.assertRaises(ValueError):
            _translate_seeds_to_local_rows(self._dist_sampler(with_neg=True))

    def test_subgraph_extraction_raises(self) -> None:
        dist_sampler = self._dist_sampler(with_neg=False)
        _translate_seeds_to_local_rows(dist_sampler)
        with self.assertRaises(ValueError):
            dist_sampler.sampler.subgraph_op

    def test_sampling_takes_global_ids(self) -> None:
        dist_sampler = self._dist_sampler(with_neg=False)
        _translate_seeds_to_local_rows(dist_sampler)
        output = dist_sampler.sampler.sample_one_hop(torch.tensor([0, 7, 8, 9, 10]), -1)
        self.assertEqual(output.nbr.tolist(), [0, 1])
        self.assertEqual(output.nbr_num.tolist(), [0, 1, 1, 0, 0])


if __name__ == "__main__":
    absltest.main()
