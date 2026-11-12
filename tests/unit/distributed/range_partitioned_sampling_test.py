"""Sampling a range-partitioned graph must match sampling the same graph without row offsets.

Under a range partition book each rank's topology stores only the rows of the ids it owns and
records its range start as a row offset. Under a tensor partition book every rank stores a row
for every id. Each test builds the same three-partition graph both ways, samples it from every
rank, and compares the outputs exactly.
"""

from typing import Any, Literal

import torch
import torch.multiprocessing as mp
from absl.testing import absltest
from graphlearn_torch.data import Graph, Topology
from graphlearn_torch.distributed import (
    barrier,
    init_rpc,
    init_worker_group,
    shutdown_rpc,
)
from graphlearn_torch.partition import RangePartitionBook
from graphlearn_torch.sampler import NodeSamplerInput
from parameterized import param, parameterized

from gigl.distributed.base_sampler import (
    BaseDistNeighborSampler,
    _GlobalIdNeighborSampler,
    _stores_only_owned_rows,
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
    rank: int, edge_dir: Literal["in", "out"], range_partitioned: bool
) -> DistDataset:
    """The partition of ``rank``: each edge goes to the owner of the node it is sampled from.

    Both kinds of partition book assign the same nodes to the same ranks.
    """
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
    if range_partitioned:
        node_partition_book = RangePartitionBook(
            partition_ranges=_PARTITION_RANGES, partition_idx=rank
        )
    else:
        node_partition_book = torch.zeros(_NUM_NODES, dtype=torch.int64)
        for partition_index, (range_start, range_end) in enumerate(_PARTITION_RANGES):
            node_partition_book[range_start:range_end] = partition_index
    partition_output = PartitionOutput(
        node_partition_book={_USER: node_partition_book},
        edge_partition_book=None,
        partitioned_edge_index=partitioned_edge_index,
        partitioned_node_features=None,
        partitioned_edge_features=None,
        partitioned_positive_labels=None,
        partitioned_negative_labels=None,
        partitioned_node_labels=None,
    )
    dataset = DistDataset(rank=rank, world_size=_NUM_PARTITIONS, edge_dir=edge_dir)
    # The samplers sample with edge ids, which the compact build stores only on request.
    dataset.build(partition_output=partition_output, retain_edge_ids=True)
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
    range_partitioned: bool,
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
    dataset = _partition_dataset(rank, edge_dir, range_partitioned)
    assert isinstance(dataset.graph, dict)
    start, _ = _PARTITION_RANGES[rank]
    expected_offset = 0
    if range_partitioned:
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


def _run(edge_dir: Literal["in", "out"], range_partitioned: bool) -> dict[int, Any]:
    output = mp.Manager().dict()
    mp.spawn(
        _sample_on_partition,
        args=(get_free_port(), get_free_port(), edge_dir, range_partitioned, output),
        nprocs=_NUM_PARTITIONS,
    )
    return dict(output)


class RangePartitionedSamplingTest(TestCase):
    @parameterized.expand([param("in", edge_dir="in"), param("out", edge_dir="out")])
    def test_row_offsets_sample_exactly_as_rows_for_every_id(
        self, _name: str, edge_dir: Literal["in", "out"]
    ) -> None:
        every_row = _run(edge_dir, range_partitioned=False)
        offset_rows = _run(edge_dir, range_partitioned=True)
        for rank in range(_NUM_PARTITIONS):
            self.assertEqual(set(offset_rows[rank]), set(every_row[rank]))
            for key, expected in every_row[rank].items():
                self.assertEqual(offset_rows[rank][key], expected, f"rank {rank} {key}")
        # The graph must give the samplers real work: some neighbors from every partition.
        sampled = every_row[0][f"routed/{_FOLLOWS}/-1"]
        self.assertGreater(len(sampled["nbr"]), _NUM_NODES)
        self.assertEqual(every_row[0]["degrees"]["user"][27:], [0, 0, 0])


class GlobalIdNeighborSamplerTest(TestCase):
    """``_GlobalIdNeighborSampler`` on topologies with and without a row offset."""

    def _sampler(
        self, topology: Topology, stores_only_owned_rows: bool, with_neg: bool = False
    ) -> _GlobalIdNeighborSampler:
        return _GlobalIdNeighborSampler(
            Graph(topology, "CPU", None),
            [-1],
            device=torch.device("cpu"),
            with_neg=with_neg,
            stores_only_owned_rows=stores_only_owned_rows,
        )

    def test_sampling_takes_global_ids_on_a_row_offset(self) -> None:
        """Ids 7 to 9 are stored as rows 0 to 2."""
        topology = CompactTopology(
            torch.tensor([[0, 1], [0, 1]]), layout="CSR", num_rows=3, row_offset=7
        )
        output = self._sampler(topology, stores_only_owned_rows=True).sample_one_hop(
            torch.tensor([0, 7, 8, 9, 10]), -1
        )
        self.assertEqual(output.nbr.tolist(), [0, 1])
        self.assertEqual(output.nbr_num.tolist(), [0, 1, 1, 0, 0])

    def test_glt_topology_samples_as_glt_does(self) -> None:
        """GLT's own build has no row offset; ids past its last row get no neighbors."""
        topology = Topology(edge_index=torch.tensor([[0, 1, 1], [2, 0, 2]]))
        seeds = torch.tensor([1, 0, 5])
        output = self._sampler(topology, stores_only_owned_rows=False).sample_one_hop(
            seeds, -1
        )
        self.assertEqual(output.nbr.tolist(), [0, 2, 2])
        self.assertEqual(output.nbr_num.tolist(), [2, 1, 0])

    def test_negative_sampling_raises_when_only_owned_rows_are_stored(self) -> None:
        topology = CompactTopology(
            torch.tensor([[0, 1], [0, 1]]), layout="CSR", row_offset=0
        )
        with self.assertRaises(ValueError):
            self._sampler(
                topology, stores_only_owned_rows=True, with_neg=True
            ).lazy_init_neg_sampler()

    def test_subgraph_extraction_raises_when_only_owned_rows_are_stored(self) -> None:
        topology = CompactTopology(
            torch.tensor([[0, 1], [0, 1]]), layout="CSR", row_offset=0
        )
        with self.assertRaises(ValueError):
            self._sampler(topology, stores_only_owned_rows=True).subgraph_op

    @parameterized.expand(
        [
            param("range_rank_0", rank=0, range_partitioned=True, expected=True),
            param("range_rank_1", rank=1, range_partitioned=True, expected=True),
            param("tensor_rank_1", rank=1, range_partitioned=False, expected=False),
        ]
    )
    def test_every_rank_of_a_range_partitioned_build_stores_only_owned_rows(
        self, _name: str, rank: int, range_partitioned: bool, expected: bool
    ) -> None:
        dataset = _partition_dataset(rank, "out", range_partitioned)
        self.assertEqual(_stores_only_owned_rows(dataset), expected)


if __name__ == "__main__":
    absltest.main()
