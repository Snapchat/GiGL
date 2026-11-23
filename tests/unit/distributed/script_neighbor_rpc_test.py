"""The TorchScript neighbor RPC must sample exactly what GLT's Python callee samples.

The multi-process tests build one three-partition graph with node features, homogeneous or heterogeneous, sample it
from every rank with ``GIGL_TORCHSCRIPT_NEIGHBOR_RPC`` off and then on, and compare one-hop neighbors, k-hop samples,
PPR tokens and scores, and the collated message with features, exactly.
"""

import os
from typing import Any, Literal, Optional
from unittest import mock

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

import gigl.distributed.base_sampler as base_sampler_module
from gigl.distributed.base_sampler import BaseDistNeighborSampler
from gigl.distributed.constants import (
    TORCHSCRIPT_NEIGHBOR_RPC_ENV,
    resolve_torchscript_neighbor_rpc,
)
from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.dist_neighbor_sampler import DistNeighborSampler
from gigl.distributed.dist_ppr_sampler import DistPPRNeighborSampler
from gigl.distributed.utils import get_free_port
from gigl.distributed.utils.degree import compute_and_broadcast_degree_tensor
from gigl.distributed.utils.script_neighbor_rpc import (
    edge_type_key,
    register_partition_csr,
)
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.types.graph import FeaturePartitionData, GraphPartitionData, PartitionOutput
from tests.test_assets.test_case import TestCase

_USER = NodeType("user")
_FOLLOWS = EdgeType(_USER, Relation("follows"), _USER)
_LIKES = EdgeType(_USER, Relation("likes"), _USER)
_PARTITION_RANGES = [(0, 37), (37, 101), (101, 150)]
_NUM_NODES = 150
_NUM_PARTITIONS = 3
# Larger than any degree in the graph, so the fanout returns every neighbor.
_LARGE_FANOUT = 1000
# Smaller than many degrees, so those lists are sub-sampled.
_SMALL_FANOUT = 2


def _edges(edge_type: Optional[EdgeType]) -> torch.Tensor:
    """Distinct (src, dst) pairs between nodes 0 to 139; nodes 140 to 149 have no edges."""
    seed = 11
    if edge_type == _LIKES:
        seed = 13
    generator = torch.Generator().manual_seed(seed)
    # Skewed sources give a spread of degrees, including some above _SMALL_FANOUT and some zero.
    sources = (torch.rand(3000, generator=generator) ** 2 * 140).long()
    destinations = torch.randint(0, 140, (3000,), generator=generator)
    return torch.unique(torch.stack([sources, destinations]), dim=1)


def _edge_types(heterogeneous: bool) -> list[Optional[EdgeType]]:
    if heterogeneous:
        return [_FOLLOWS, _LIKES]
    return [None]


def _partition_dataset(
    rank: int,
    edge_dir: Literal["in", "out"],
    range_partitioned: bool,
    heterogeneous: bool,
) -> DistDataset:
    """The partition of ``rank``; node ``i`` has feature row ``[i, -i, 0.5 * i]``."""
    compressed = 0
    if edge_dir == "in":
        compressed = 1
    start, end = _PARTITION_RANGES[rank]

    def owned_edges(edge_type: Optional[EdgeType]) -> GraphPartitionData:
        edges = _edges(edge_type)
        owned = (edges[compressed] >= start) & (edges[compressed] < end)
        edge_index = edges[:, owned]
        if range_partitioned:
            # int32 ids give the compact build int32 column ids; the tensor-book case keeps int64.
            edge_index = edge_index.to(torch.int32)
        return GraphPartitionData(edge_index=edge_index, edge_ids=None)

    node_partition_book: Any
    if range_partitioned:
        node_partition_book = RangePartitionBook(
            partition_ranges=_PARTITION_RANGES, partition_idx=rank
        )
    else:
        node_partition_book = torch.zeros(_NUM_NODES, dtype=torch.int64)
        for partition_index, (range_start, range_end) in enumerate(_PARTITION_RANGES):
            node_partition_book[range_start:range_end] = partition_index
    owned_nodes = torch.arange(start, end)
    features = FeaturePartitionData(
        feats=torch.stack(
            [owned_nodes.float(), -owned_nodes.float(), 0.5 * owned_nodes.float()],
            dim=1,
        ),
        ids=owned_nodes,
    )
    if heterogeneous:
        partition_output = PartitionOutput(
            node_partition_book={_USER: node_partition_book},
            edge_partition_book=None,
            partitioned_edge_index={
                _FOLLOWS: owned_edges(_FOLLOWS),
                _LIKES: owned_edges(_LIKES),
            },
            partitioned_node_features={_USER: features},
            partitioned_edge_features=None,
            partitioned_positive_labels=None,
            partitioned_negative_labels=None,
            partitioned_node_labels=None,
        )
    else:
        partition_output = PartitionOutput(
            node_partition_book=node_partition_book,
            edge_partition_book=None,
            partitioned_edge_index=owned_edges(None),
            partitioned_node_features=features,
            partitioned_edge_features=None,
            partitioned_positive_labels=None,
            partitioned_negative_labels=None,
            partitioned_node_labels=None,
        )
    dataset = DistDataset(rank=rank, world_size=_NUM_PARTITIONS, edge_dir=edge_dir)
    dataset.build(partition_output=partition_output, retain_edge_ids=True)
    return dataset


def _to_lists(value: Any) -> Any:
    """Convert tensors, possibly nested in dicts and sampler outputs, to comparable values."""
    if isinstance(value, torch.Tensor):
        return (str(value.dtype), value.tolist())
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
    heterogeneous: bool,
    with_edge: bool,
    flag_value: str,
    unsupported_rank: Optional[int],
    output: dict,
) -> None:
    """Samples every node from this rank with the TorchScript neighbor RPC on or off and records the results."""
    os.environ[TORCHSCRIPT_NEIGHBOR_RPC_ENV] = flag_value
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{process_group_port}",
        rank=rank,
        world_size=_NUM_PARTITIONS,
    )
    init_worker_group(world_size=_NUM_PARTITIONS, rank=rank)
    init_rpc(master_addr="127.0.0.1", master_port=rpc_port, num_rpc_threads=4)
    dataset = _partition_dataset(rank, edge_dir, range_partitioned, heterogeneous)
    degrees = compute_and_broadcast_degree_tensor(dataset.graph, edge_dir)
    input_type: Optional[NodeType] = None
    if heterogeneous:
        input_type = _USER

    # A rank that cannot serve the TorchScript path, as a weighted or GPU sampler cannot.
    register_patch = mock.patch.object(
        base_sampler_module, "register_partition_csr", return_value=False
    )
    if rank == unsupported_rank:
        register_patch.start()
    one_hop_sampler = BaseDistNeighborSampler(
        data=dataset, num_neighbors=[-1], with_edge=with_edge, concurrency=1
    )
    k_hop_sampler = DistNeighborSampler(
        data=dataset,
        num_neighbors=[_LARGE_FANOUT, _LARGE_FANOUT],
        with_edge=with_edge,
        collect_features=True,
        concurrency=1,
    )
    ppr_sampler = DistPPRNeighborSampler(
        data=dataset,
        num_neighbors=[-1],
        with_edge=with_edge,
        collect_features=True,
        concurrency=1,
        degree_tensors=degrees,
        max_ppr_nodes=12,
        num_neighbors_per_hop=_LARGE_FANOUT,
    )
    if rank == unsupported_rank:
        register_patch.stop()
    samplers = [one_hop_sampler, k_hop_sampler, ppr_sampler]
    for sampler in samplers:
        sampler.start_loop()
    barrier()

    sampling_callee_ids = {sampler.rpc_sample_callee_id for sampler in samplers}
    python_requests: list[int] = []
    script_requests: list[int] = []
    original_python_request = base_sampler_module.rpc_request_async
    original_script_request = base_sampler_module.send_script_neighbor_request

    def spy_python_request(worker_name, callee_id, args=None, kwargs=None):
        python_requests.append(callee_id)
        return original_python_request(worker_name, callee_id, args=args, kwargs=kwargs)

    def spy_script_request(worker_name, sampler_id, etype, ids, req_num):
        script_requests.append(sampler_id)
        return original_script_request(worker_name, sampler_id, etype, ids, req_num)

    seeds = torch.arange(_NUM_NODES)
    result: dict[str, Any] = {}
    with (
        mock.patch.object(base_sampler_module, "rpc_request_async", spy_python_request),
        mock.patch.object(
            base_sampler_module, "send_script_neighbor_request", spy_script_request
        ),
    ):
        for edge_type in _edge_types(heterogeneous):
            for fanout in (-1, _LARGE_FANOUT):
                routed = one_hop_sampler.run_task(
                    coro=one_hop_sampler._sample_one_hop(seeds, fanout, edge_type)
                )
                result[f"routed/{edge_type}/{fanout}"] = _to_lists(routed)
            subsampled = one_hop_sampler.run_task(
                coro=one_hop_sampler._sample_one_hop(seeds, _SMALL_FANOUT, edge_type)
            )
            result[f"subsampled/{edge_type}"] = _to_lists(subsampled)
        for name, sampler in (("khop", k_hop_sampler), ("ppr", ppr_sampler)):
            sampler_output = sampler.run_task(
                coro=sampler._sample_from_nodes(
                    NodeSamplerInput(node=seeds, input_type=input_type)
                )
            )
            result[f"{name}/output"] = _to_lists(sampler_output)
            message = sampler.run_task(coro=sampler._collate_fn(sampler_output))
            result[f"{name}/message"] = _to_lists(message)
    barrier()
    result["indices_dtypes"] = []
    if isinstance(dataset.graph, dict):
        for graph in dataset.graph.values():
            result["indices_dtypes"].append(str(graph.topo.indices.dtype))
    elif dataset.graph is not None:
        result["indices_dtypes"].append(str(dataset.graph.topo.indices.dtype))
    result["script_flags"] = [sampler._script_neighbor_rpc for sampler in samplers]
    result["python_neighbor_requests"] = sum(
        1 for callee_id in python_requests if callee_id in sampling_callee_ids
    )
    result["script_neighbor_requests"] = len(script_requests)
    output[rank] = result
    barrier()
    for sampler in samplers:
        sampler.shutdown_loop()
    shutdown_rpc()
    torch.distributed.destroy_process_group()


def _run(
    edge_dir: Literal["in", "out"],
    range_partitioned: bool,
    heterogeneous: bool,
    with_edge: bool,
    flag_value: str,
    unsupported_rank: Optional[int] = None,
) -> dict[int, Any]:
    output = mp.Manager().dict()
    mp.spawn(
        _sample_on_partition,
        args=(
            get_free_port(),
            get_free_port(),
            edge_dir,
            range_partitioned,
            heterogeneous,
            with_edge,
            flag_value,
            unsupported_rank,
            output,
        ),
        nprocs=_NUM_PARTITIONS,
    )
    return dict(output)


def _neighbor_sets(
    edge_type: Optional[EdgeType], edge_dir: Literal["in", "out"]
) -> dict[int, set[int]]:
    edges = _edges(edge_type)
    compressed, other = 0, 1
    if edge_dir == "in":
        compressed, other = 1, 0
    neighbors: dict[int, set[int]] = {node: set() for node in range(_NUM_NODES)}
    for source, destination in zip(edges[compressed].tolist(), edges[other].tolist()):
        neighbors[source].add(destination)
    return neighbors


class ScriptNeighborRpcParityTest(TestCase):
    def _assert_same_samples(
        self,
        python_result: dict[str, Any],
        script_result: dict[str, Any],
        rank: int,
        heterogeneous: bool,
        edge_dir: Literal["in", "out"],
    ) -> None:
        for result_key, expected in python_result.items():
            if result_key.startswith(("routed", "khop", "ppr")):
                self.assertEqual(
                    script_result[result_key], expected, f"rank {rank} {result_key}"
                )
        # Sub-sampled lists draw differently, but keep counts and draw from the row.
        for edge_type in _edge_types(heterogeneous):
            key = f"subsampled/{edge_type}"
            self.assertEqual(
                script_result[key]["nbr_num"], python_result[key]["nbr_num"]
            )
            neighbor_sets = _neighbor_sets(edge_type, edge_dir)
            _, counts = script_result[key]["nbr_num"]
            _, neighbors = script_result[key]["nbr"]
            position = 0
            for node, count in enumerate(counts):
                self.assertEqual(count, min(_SMALL_FANOUT, len(neighbor_sets[node])))
                for neighbor in neighbors[position : position + count]:
                    self.assertIn(neighbor, neighbor_sets[node])
                position += count

    @parameterized.expand(
        [
            param(
                "hetero_range_out_edges",
                edge_dir="out",
                range_partitioned=True,
                heterogeneous=True,
                with_edge=True,
            ),
            param(
                "hetero_range_in_no_edges",
                edge_dir="in",
                range_partitioned=True,
                heterogeneous=True,
                with_edge=False,
            ),
            param(
                "hetero_tensor_book_out",
                edge_dir="out",
                range_partitioned=False,
                heterogeneous=True,
                with_edge=False,
            ),
            param(
                "homo_range_out_edges",
                edge_dir="out",
                range_partitioned=True,
                heterogeneous=False,
                with_edge=True,
            ),
            param(
                "homo_tensor_book_in",
                edge_dir="in",
                range_partitioned=False,
                heterogeneous=False,
                with_edge=False,
            ),
        ]
    )
    def test_script_path_samples_exactly_as_the_python_path(
        self,
        _name: str,
        edge_dir: Literal["in", "out"],
        range_partitioned: bool,
        heterogeneous: bool,
        with_edge: bool,
    ) -> None:
        python_path = _run(
            edge_dir, range_partitioned, heterogeneous, with_edge, flag_value="off"
        )
        # Unset is on.
        script_path = _run(
            edge_dir, range_partitioned, heterogeneous, with_edge, flag_value=""
        )
        for rank in range(_NUM_PARTITIONS):
            python_result = python_path[rank]
            script_result = script_path[rank]
            # Which path served the requests.
            self.assertEqual(python_result["script_flags"], [False, False, False])
            self.assertEqual(script_result["script_flags"], [True, True, True])
            self.assertGreater(python_result["python_neighbor_requests"], 0)
            self.assertEqual(python_result["script_neighbor_requests"], 0)
            self.assertEqual(script_result["python_neighbor_requests"], 0)
            self.assertGreater(script_result["script_neighbor_requests"], 0)
            self._assert_same_samples(
                python_result, script_result, rank, heterogeneous, edge_dir
            )
        # The graph gives the samplers real work: remote neighbors and some sub-sampled lists.
        first_edge_type = _edge_types(heterogeneous)[0]
        _, all_neighbors = python_path[0][f"routed/{first_edge_type}/-1"]["nbr"]
        self.assertGreater(len(all_neighbors), 10 * _NUM_NODES)
        expected_dtype = "torch.int64"
        if range_partitioned:
            expected_dtype = "torch.int32"
        self.assertEqual(
            python_path[0]["indices_dtypes"],
            [expected_dtype] * len(_edge_types(heterogeneous)),
        )

    def test_one_rank_that_cannot_serve_turns_it_off_on_every_rank(self) -> None:
        python_path = _run(
            "out", True, heterogeneous=True, with_edge=True, flag_value="off"
        )
        fallback_path = _run(
            "out",
            True,
            heterogeneous=True,
            with_edge=True,
            flag_value="on",
            unsupported_rank=1,
        )
        for rank in range(_NUM_PARTITIONS):
            fallback_result = fallback_path[rank]
            self.assertEqual(fallback_result["script_flags"], [False, False, False])
            self.assertGreater(fallback_result["python_neighbor_requests"], 0)
            self.assertEqual(fallback_result["script_neighbor_requests"], 0)
            self._assert_same_samples(
                python_path[rank], fallback_result, rank, True, "out"
            )


class NativeLookupTest(TestCase):
    """The C++ lookup against GLT's semantics on hand-made CSRs, in one process."""

    def setUp(self) -> None:
        super().setUp()
        import gigl_core.rpc_neighbor_lookup as native

        self.native = native

    def tearDown(self) -> None:
        self.native.unregister_csr(7)
        super().tearDown()

    @parameterized.expand(
        [param("int32", dtype=torch.int32), param("int64", dtype=torch.int64)]
    )
    def test_rows_with_an_offset_copy_in_order(
        self, _name: str, dtype: torch.dtype
    ) -> None:
        # Rows 10, 11 and 12 of the global id space; row 11 is empty.
        indptr = torch.tensor([0, 2, 2, 5])
        indices = torch.tensor([7, 8, 1, 2, 3], dtype=dtype)
        edge_ids = torch.tensor([70, 80, 10, 20, 30])
        self.native.register_csr(7, "a|b|c", indptr, indices, edge_ids, 10, True)
        neighbors, counts, out_edge_ids = self.native.lookup_neighbors(
            7, "a|b|c", torch.tensor([12, 9, 10, 11, 13, 12]), 1000
        )
        self.assertEqual(counts.tolist(), [3, 0, 2, 0, 0, 3])
        self.assertEqual(neighbors.tolist(), [1, 2, 3, 7, 8, 1, 2, 3])
        self.assertEqual(out_edge_ids.tolist(), [10, 20, 30, 70, 80, 10, 20, 30])
        self.assertEqual(neighbors.dtype, torch.int64)

    def test_without_edge_ids_replies_with_an_empty_edge_tensor(self) -> None:
        self.native.register_csr(
            7, "", torch.tensor([0, 1]), torch.tensor([4]), None, 0, False
        )
        _, _, out_edge_ids = self.native.lookup_neighbors(7, "", torch.tensor([0]), -1)
        self.assertEqual(out_edge_ids.numel(), 0)

    def test_a_non_empty_sample_with_edges_needs_edge_ids(self) -> None:
        self.native.register_csr(
            7, "", torch.tensor([0, 1, 1]), torch.tensor([4]), None, 0, True
        )
        _, counts, _ = self.native.lookup_neighbors(7, "", torch.tensor([1]), -1)
        self.assertEqual(counts.tolist(), [0])
        with self.assertRaises(RuntimeError):
            self.native.lookup_neighbors(7, "", torch.tensor([0]), -1)

    def test_long_rows_are_sampled_with_replacement_from_the_row(self) -> None:
        self.native.register_csr(
            7, "", torch.tensor([0, 6]), torch.arange(100, 106), None, 0, False
        )
        neighbors, counts, _ = self.native.lookup_neighbors(
            7, "", torch.tensor([0, 0]), 4
        )
        self.assertEqual(counts.tolist(), [4, 4])
        self.assertTrue(all(100 <= value < 106 for value in neighbors.tolist()))

    def test_unregistered_sampler_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            self.native.lookup_neighbors(7, "", torch.tensor([0]), 1)

    def test_an_indptr_that_overruns_indices_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            self.native.register_csr(
                7, "", torch.tensor([0, 3]), torch.tensor([4]), None, 0, False
            )


class RegisterPartitionCsrTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.graph = Graph(
            Topology(edge_index=torch.tensor([[0, 0, 1], [1, 2, 2]]), layout="CSR"),
            mode="CPU",
        )

    def tearDown(self) -> None:
        import gigl_core.rpc_neighbor_lookup as native

        native.unregister_csr(7)
        super().tearDown()

    @parameterized.expand(
        [
            param("cpu", device="cpu", with_weight=False, expected=True),
            param("weighted", device="cpu", with_weight=True, expected=False),
            param("gpu_device", device="cuda", with_weight=False, expected=False),
        ]
    )
    def test_registers_only_what_the_lookup_supports(
        self, _name: str, device: str, with_weight: bool, expected: bool
    ) -> None:
        registered = register_partition_csr(
            7,
            self.graph,
            with_edge=False,
            with_weight=with_weight,
            device=torch.device(device),
            seed=None,
        )
        self.assertEqual(registered, expected)

    def test_a_non_cpu_graph_is_not_registered(self) -> None:
        self.graph.mode = "ZERO_COPY"
        self.assertFalse(
            register_partition_csr(
                7,
                {_FOLLOWS: self.graph},
                with_edge=False,
                with_weight=False,
                device=torch.device("cpu"),
                seed=None,
            )
        )


class FlagTest(TestCase):
    def test_on_unless_set_to_off(self) -> None:
        for value, expected in (("", True), ("on", True), ("OFF", False)):
            with mock.patch.dict(os.environ, {TORCHSCRIPT_NEIGHBOR_RPC_ENV: value}):
                self.assertEqual(resolve_torchscript_neighbor_rpc(), expected)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(resolve_torchscript_neighbor_rpc())
        with mock.patch.dict(os.environ, {TORCHSCRIPT_NEIGHBOR_RPC_ENV: "1"}):
            with self.assertRaises(ValueError):
                resolve_torchscript_neighbor_rpc()

    def test_edge_type_keys(self) -> None:
        self.assertEqual(edge_type_key(None), "")
        self.assertEqual(edge_type_key(_FOLLOWS), "user|follows|user")


if __name__ == "__main__":
    absltest.main()
