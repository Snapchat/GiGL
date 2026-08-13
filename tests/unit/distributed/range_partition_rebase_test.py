"""Tests for range-rebased (OffsetTopology) dataset builds and id translation.

Covers:
  1. ``DistDataset.build`` producing locally-sized topologies from range
     partition books, including the all-or-nothing validation for
     message-passing edge types and the per-type fallback for label edge types.
  2. The offset-aware ``id_select`` installed on the dataset.
  3. ForkingPickler round trips: ``Graph(OffsetTopology)`` and the full
     ``DistDataset`` ipc handle, which every spawned sampling worker relies on.
  4. End-to-end two-rank range-partitioned sampling with rank 1 owning a
     nonzero node range: local and remote one-hop sampling return correct
     global neighbor ids and edge ids, weighted sampling keeps its
     edge-to-weight pairing, and degrees aggregate globally.
"""

import pickle
from multiprocessing.reduction import ForkingPickler

import torch
import torch.multiprocessing as mp
from absl.testing import absltest
from graphlearn_torch.data import Graph, Topology
from graphlearn_torch.distributed import shutdown_rpc
from graphlearn_torch.partition import RangePartitionBook

from gigl.distributed.dist_context import DistributedContext
from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.distributed_neighborloader import DistNeighborLoader
from gigl.distributed.utils.topology import OffsetTopology
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.types.graph import (
    FeaturePartitionData,
    GraphPartitionData,
    PartitionOutput,
    message_passing_to_positive_label,
)
from tests.test_assets.distributed.utils import (
    assert_tensor_equality,
    get_process_group_init_method,
)
from tests.test_assets.test_case import TestCase

_USER = NodeType("user")
_STORY = NodeType("story")
_USER_TO_STORY = EdgeType(_USER, Relation("to"), _STORY)

# Three-partition range layout for the homogeneous tests; the dataset under
# test is rank 1, owning [4, 10).
_NODE_RANGES = [(0, 4), (4, 10), (10, 12)]
_RANK = 1
_WORLD_SIZE = 3
_LOWER, _UPPER = _NODE_RANGES[_RANK]

# Rank 1's partition: sources in [4, 10), destinations global.
_EDGE_INDEX = torch.tensor(
    [
        [4, 4, 5, 7, 9, 9],
        [0, 11, 3, 7, 2, 10],
    ]
)
_EDGE_IDS = torch.tensor([40, 41, 42, 43, 44, 45])


def _build_homogeneous_range_dataset() -> DistDataset:
    partition_output = PartitionOutput(
        node_partition_book=RangePartitionBook(_NODE_RANGES, _RANK),
        edge_partition_book=RangePartitionBook([(0, 40), (40, 46), (46, 50)], _RANK),
        partitioned_edge_index=GraphPartitionData(
            edge_index=_EDGE_INDEX.clone(),
            edge_ids=_EDGE_IDS.clone(),
        ),
        partitioned_node_features=None,
        partitioned_edge_features=None,
        partitioned_positive_labels=None,
        partitioned_negative_labels=None,
        partitioned_node_labels=None,
    )
    dataset = DistDataset(rank=_RANK, world_size=_WORLD_SIZE, edge_dir="out")
    dataset.build(partition_output=partition_output)
    return dataset


def _forking_pickler_roundtrip(obj):
    return pickle.loads(bytes(ForkingPickler.dumps(obj)))


class RangeRebasedBuildTest(TestCase):
    def test_build_produces_locally_sized_topology(self):
        dataset = _build_homogeneous_range_dataset()

        assert isinstance(dataset.graph, Graph)
        topology = dataset.graph.topo
        assert isinstance(topology, OffsetTopology)
        self.assertEqual(topology.offset, _LOWER)
        self.assertEqual(topology.indptr.numel(), _UPPER - _LOWER + 1)

        # Neighbor ids and edge ids stay global.
        row, col, edge_ids, _ = topology.to_coo()
        self.assertEqual(
            sorted(zip(row.tolist(), col.tolist(), edge_ids.tolist())),
            sorted(
                zip(
                    _EDGE_INDEX[0].tolist(), _EDGE_INDEX[1].tolist(), _EDGE_IDS.tolist()
                )
            ),
        )

    def test_build_raises_when_message_passing_edges_violate_partition_range(self):
        partition_output = PartitionOutput(
            node_partition_book=RangePartitionBook(_NODE_RANGES, _RANK),
            edge_partition_book=torch.zeros(6, dtype=torch.int64),
            partitioned_edge_index=GraphPartitionData(
                # Source id 2 is outside rank 1's range [4, 10).
                edge_index=torch.tensor([[4, 2], [0, 1]]),
                edge_ids=torch.tensor([0, 1]),
            ),
            partitioned_node_features=None,
            partitioned_edge_features=None,
            partitioned_positive_labels=None,
            partitioned_negative_labels=None,
            partitioned_node_labels=None,
        )
        dataset = DistDataset(rank=_RANK, world_size=_WORLD_SIZE, edge_dir="out")
        with self.assertRaises(ValueError):
            dataset.build(partition_output=partition_output)

    def test_build_with_empty_partition_produces_zero_indptr(self):
        partition_output = PartitionOutput(
            node_partition_book=RangePartitionBook(_NODE_RANGES, _RANK),
            edge_partition_book=torch.zeros(0, dtype=torch.int64),
            partitioned_edge_index=GraphPartitionData(
                edge_index=torch.empty(2, 0, dtype=torch.int64),
                edge_ids=torch.empty(0, dtype=torch.int64),
            ),
            partitioned_node_features=None,
            partitioned_edge_features=None,
            partitioned_positive_labels=None,
            partitioned_negative_labels=None,
            partitioned_node_labels=None,
        )
        dataset = DistDataset(rank=_RANK, world_size=_WORLD_SIZE, edge_dir="out")
        dataset.build(partition_output=partition_output)

        assert isinstance(dataset.graph, Graph)
        topology = dataset.graph.topo
        assert isinstance(topology, OffsetTopology)
        self.assert_tensor_equality(
            topology.indptr, torch.zeros(_UPPER - _LOWER + 1, dtype=torch.int64)
        )

    def test_label_edge_types_are_validated_per_type(self):
        """Aligned label edge types are rebased; misaligned ones fall back to a
        global topology instead of raising."""
        user_book = RangePartitionBook([(0, 4), (4, 8)], 1)
        story_book = RangePartitionBook([(0, 6), (6, 12)], 1)
        aligned_label_type = message_passing_to_positive_label(_USER_TO_STORY)
        misaligned_label_type = EdgeType(_USER, Relation("to_gigl_negative"), _STORY)

        partition_output = PartitionOutput(
            node_partition_book={_USER: user_book, _STORY: story_book},
            edge_partition_book={
                _USER_TO_STORY: torch.zeros(2, dtype=torch.int64),
                aligned_label_type: torch.zeros(2, dtype=torch.int64),
                misaligned_label_type: torch.zeros(2, dtype=torch.int64),
            },
            partitioned_edge_index={
                # edge_dir "in" compresses the destination (story) dimension;
                # rank 1 owns story ids [6, 12).
                _USER_TO_STORY: GraphPartitionData(
                    edge_index=torch.tensor([[0, 5], [6, 11]]),
                    edge_ids=torch.tensor([0, 1]),
                ),
                aligned_label_type: GraphPartitionData(
                    edge_index=torch.tensor([[1, 2], [7, 8]]),
                    edge_ids=torch.tensor([0, 1]),
                ),
                # Label edges whose destinations fall outside [6, 12).
                misaligned_label_type: GraphPartitionData(
                    edge_index=torch.tensor([[1, 2], [0, 5]]),
                    edge_ids=torch.tensor([0, 1]),
                ),
            },
            partitioned_node_features=None,
            partitioned_edge_features=None,
            partitioned_positive_labels=None,
            partitioned_negative_labels=None,
            partitioned_node_labels=None,
        )
        dataset = DistDataset(rank=1, world_size=2, edge_dir="in")
        dataset.build(partition_output=partition_output)

        assert isinstance(dataset.graph, dict)
        message_passing_topology = dataset.graph[_USER_TO_STORY].topo
        assert isinstance(message_passing_topology, OffsetTopology)
        self.assertEqual(message_passing_topology.offset, 6)
        self.assertEqual(message_passing_topology.indptr.numel(), 12 - 6 + 1)

        aligned_topology = dataset.graph[aligned_label_type].topo
        assert isinstance(aligned_topology, OffsetTopology)
        self.assertEqual(aligned_topology.offset, 6)

        misaligned_topology = dataset.graph[misaligned_label_type].topo
        self.assertIsInstance(misaligned_topology, Topology)
        self.assertNotIsInstance(misaligned_topology, OffsetTopology)
        # Global sizing: indptr spans [0, max compressed id].
        self.assertEqual(misaligned_topology.indptr.numel(), 5 + 2)

    def test_tensor_partition_book_keeps_global_topology(self):
        partition_output = PartitionOutput(
            node_partition_book=torch.zeros(12, dtype=torch.int64),
            edge_partition_book=torch.zeros(6, dtype=torch.int64),
            partitioned_edge_index=GraphPartitionData(
                edge_index=_EDGE_INDEX.clone(),
                edge_ids=_EDGE_IDS.clone(),
            ),
            partitioned_node_features=None,
            partitioned_edge_features=None,
            partitioned_positive_labels=None,
            partitioned_negative_labels=None,
            partitioned_node_labels=None,
        )
        dataset = DistDataset(rank=0, world_size=1, edge_dir="out")
        dataset.build(partition_output=partition_output)

        assert isinstance(dataset.graph, Graph)
        self.assertNotIsInstance(dataset.graph.topo, OffsetTopology)
        # Stock sizing from the global id ceiling: rows.max() + 2 entries.
        self.assertEqual(
            dataset.graph.topo.indptr.numel(), int(_EDGE_INDEX[0].max().item()) + 2
        )


class OffsetAwareIdSelectTest(TestCase):
    def test_id_select_translates_per_element_partition_lows(self):
        dataset = _build_homogeneous_range_dataset()
        book = RangePartitionBook(_NODE_RANGES, _RANK)
        srcs = torch.tensor([0, 3, 4, 9, 10, 11])

        partition_ids = book[srcs]
        # Per-partition masks, as GLT's sampler builds them.
        self.assert_tensor_equality(
            dataset.id_select(srcs, partition_ids == 0, book),
            torch.tensor([0, 3]),  # partition 0's lower bound is 0
        )
        self.assert_tensor_equality(
            dataset.id_select(srcs, partition_ids == 1, book),
            torch.tensor([0, 5]),  # rebased by lower bound 4
        )
        self.assert_tensor_equality(
            dataset.id_select(srcs, partition_ids == 2, book),
            torch.tensor([0, 1]),  # rebased by lower bound 10
        )

        # Per-element lows: a mask spanning partitions translates each id by
        # its own partition's lower bound.
        self.assert_tensor_equality(
            dataset.id_select(srcs, torch.ones_like(srcs, dtype=torch.bool), book),
            torch.tensor([0, 3, 0, 5, 0, 1]),
        )

    def test_id_select_is_plain_masked_select_for_global_topologies(self):
        partition_output = PartitionOutput(
            node_partition_book=torch.zeros(12, dtype=torch.int64),
            edge_partition_book=torch.zeros(6, dtype=torch.int64),
            partitioned_edge_index=GraphPartitionData(
                edge_index=_EDGE_INDEX.clone(),
                edge_ids=_EDGE_IDS.clone(),
            ),
            partitioned_node_features=None,
            partitioned_edge_features=None,
            partitioned_positive_labels=None,
            partitioned_negative_labels=None,
            partitioned_node_labels=None,
        )
        dataset = DistDataset(rank=0, world_size=1, edge_dir="out")
        dataset.build(partition_output=partition_output)

        srcs = torch.tensor([0, 3, 4, 9, 10, 11])
        mask = torch.tensor([True, False, True, False, True, False])
        self.assert_tensor_equality(
            dataset.id_select(srcs, mask, dataset.node_pb),
            torch.tensor([0, 4, 10]),
        )


class ForkingPicklerRoundTripTest(TestCase):
    def test_graph_with_offset_topology_roundtrips(self):
        topology = OffsetTopology(
            edge_index=_EDGE_INDEX.clone(),
            edge_ids=_EDGE_IDS.clone(),
            edge_weights=torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            layout="CSR",
            offset=_LOWER,
            num_nodes=_UPPER - _LOWER,
        )
        graph = Graph(topology, "CPU")

        rebuilt = _forking_pickler_roundtrip(graph)

        assert isinstance(rebuilt, Graph)
        rebuilt_topology = rebuilt.topo
        assert isinstance(rebuilt_topology, OffsetTopology)
        self.assertEqual(rebuilt_topology.offset, _LOWER)
        self.assert_tensor_equality(rebuilt_topology.indptr, topology.indptr)
        self.assert_tensor_equality(rebuilt_topology.indices, topology.indices)
        self.assert_tensor_equality(rebuilt_topology.edge_ids, topology.edge_ids)
        assert rebuilt_topology.edge_weights is not None
        self.assert_tensor_equality(
            rebuilt_topology.edge_weights, topology.edge_weights
        )
        # The Graph constructor moves the topology tensors to shared memory so
        # workers share storage rather than copying.
        self.assertTrue(rebuilt_topology.indptr.is_shared())

        # The rebuilt graph still reports the local node range and returns
        # global ids from to_coo().
        row, _, _, _ = rebuilt_topology.to_coo()
        self.assertEqual(int(row.min().item()), int(_EDGE_INDEX[0].min().item()))
        self.assertEqual(rebuilt.row_count, _UPPER - _LOWER)

    def test_dist_dataset_ipc_roundtrip_preserves_offset_aware_id_select(self):
        dataset = _build_homogeneous_range_dataset()

        rebuilt = _forking_pickler_roundtrip(dataset)

        assert isinstance(rebuilt, DistDataset)
        assert isinstance(rebuilt.graph, Graph)
        rebuilt_topology = rebuilt.graph.topo
        assert isinstance(rebuilt_topology, OffsetTopology)
        self.assertEqual(rebuilt_topology.offset, _LOWER)
        self.assertEqual(rebuilt_topology.indptr.numel(), _UPPER - _LOWER + 1)

        # The rebuilt dataset's id_select translates: the ipc handle does not
        # carry id_select, so it must be reinstalled by __init__ on rebuild.
        book = rebuilt.node_pb
        assert isinstance(book, RangePartitionBook)
        srcs = torch.tensor([0, 3, 4, 9, 10, 11])
        self.assert_tensor_equality(
            rebuilt.id_select(srcs, book[srcs] == 1, book),
            torch.tensor([0, 5]),
        )


# --- End-to-end two-rank range-partitioned sampling -------------------------

# 8 nodes over ranges [(0, 4), (4, 8)]: rank 1 owns a nonzero node range.
_E2E_NODE_RANGES = [(0, 4), (4, 8)]
_E2E_NUM_NODES = 8
# Out-neighbors per node. Degrees ([4, 2, 2, 4, 2, 2, 4, 2]) are non-uniform
# (misalignment is observable) and even (the degree assertion below survives
# the same-host over-counting division by 2).
_E2E_NEIGHBORS: dict[int, list[int]] = {
    node: (
        [(node + 1) % 8, (node + 5) % 8, (node + 2) % 8, (node + 3) % 8]
        if node % 3 == 0
        else [(node + 1) % 8, (node + 5) % 8]
    )
    for node in range(_E2E_NUM_NODES)
}
# Global edge ids are source-major in adjacency-list order.
_E2E_FIRST_EDGE_ID: dict[int, int] = {}
_edge_id_cursor = 0
for _node in range(_E2E_NUM_NODES):
    _E2E_FIRST_EDGE_ID[_node] = _edge_id_cursor
    _edge_id_cursor += len(_E2E_NEIGHBORS[_node])
_E2E_NUM_EDGES = _edge_id_cursor
_E2E_EDGE_RANGES = [
    (0, _E2E_FIRST_EDGE_ID[4]),
    (_E2E_FIRST_EDGE_ID[4], _E2E_NUM_EDGES),
]


def _expected_edge_ids_for_seed(seed: int) -> dict[int, int]:
    """Global (neighbor -> edge id) mapping for one seed's out-edges."""
    return {
        neighbor: _E2E_FIRST_EDGE_ID[seed] + position
        for position, neighbor in enumerate(_E2E_NEIGHBORS[seed])
    }


def _run_two_rank_range_partitioned_sampling(
    rank: int,
    world_size: int,
    init_method: str,
    weighted: bool,
) -> None:
    """Worker: builds this rank's rebased partition, then samples every global
    seed so both the local and the remote (RPC) one-hop branches are exercised
    against nonzero-offset topologies."""
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )
    lower, upper = _E2E_NODE_RANGES[rank]
    edges = [
        (source, destination)
        for source in range(lower, upper)
        for destination in _E2E_NEIGHBORS[source]
    ]
    edge_index = torch.tensor(
        [
            [source for source, _ in edges],
            [destination for _, destination in edges],
        ]
    )
    edge_ids = torch.arange(
        _E2E_FIRST_EDGE_ID[lower], _E2E_FIRST_EDGE_ID[lower] + len(edges)
    )
    if weighted:
        # Exactly one unit-weight out-edge per node — the first listed
        # neighbor. Weighted sampling with fanout 1 must always pick it; any
        # edge-to-weight misalignment surfaces as a zero-weight edge sampled.
        weights = torch.tensor(
            [
                1.0 if destination == (source + 1) % 8 else 0.0
                for source, destination in edges
            ]
        )
        edge_features = None
    else:
        weights = None
        # Edge features (the global edge id itself) so the loader samples with
        # with_edge=True and returns global edge ids per sampled edge.
        edge_features = FeaturePartitionData(
            feats=edge_ids.float().unsqueeze(1), ids=None
        )

    partition_output = PartitionOutput(
        node_partition_book=RangePartitionBook(_E2E_NODE_RANGES, rank),
        edge_partition_book=RangePartitionBook(_E2E_EDGE_RANGES, rank),
        partitioned_edge_index=GraphPartitionData(
            edge_index=edge_index,
            edge_ids=edge_ids,
            weights=weights,
        ),
        partitioned_node_features=None,
        partitioned_edge_features=edge_features,
        partitioned_positive_labels=None,
        partitioned_negative_labels=None,
        partitioned_node_labels=None,
    )
    dataset = DistDataset(rank=rank, world_size=world_size, edge_dir="out")
    dataset.build(partition_output=partition_output)

    graph = dataset.graph
    assert isinstance(graph, Graph)
    topology = graph.topo
    assert isinstance(topology, OffsetTopology), (
        f"Expected OffsetTopology, got {type(topology)}"
    )
    assert topology.offset == lower
    assert topology.indptr.numel() == upper - lower + 1

    # The unweighted fanout equals the maximum degree, so the sampler returns
    # every neighbor exactly once and the per-seed output is deterministic.
    max_degree = max(len(neighbors) for neighbors in _E2E_NEIGHBORS.values())
    loader = DistNeighborLoader(
        dataset=dataset,
        num_neighbors=[1] if weighted else [max_degree],
        # Every rank samples every global seed: seeds in [lower, upper) hit the
        # local branch, the rest go through the remote RPC branch.
        input_nodes=torch.arange(_E2E_NUM_NODES),
        batch_size=1,
        # Both partitions run on one host in this test; an explicit context
        # (one process per "node") makes the two ranks form a single sampling
        # worker group covering both partitions, rather than being treated as
        # two local shards of the same partition.
        context=DistributedContext(
            main_worker_ip_address="localhost",
            global_rank=rank,
            global_world_size=world_size,
        ),
        local_process_rank=0,
        local_process_world_size=1,
        with_weight=weighted,
        pin_memory_device=torch.device("cpu"),
    )
    batch_count = 0
    for datum in loader:
        seed = int(datum.batch.item())
        expected_neighbors = _E2E_NEIGHBORS[seed]
        if weighted:
            sampled_neighbors = set(datum.node.tolist()) - {seed}
            assert sampled_neighbors == {(seed + 1) % 8}, (
                f"Seed {seed}: weighted sampling must pick the unique "
                f"unit-weight neighbor {(seed + 1) % 8}, got {sampled_neighbors}"
            )
        else:
            assert set(datum.node.tolist()) == {seed} | set(expected_neighbors), (
                f"Seed {seed}: expected global neighbors {expected_neighbors}, "
                f"got nodes {datum.node.tolist()}"
            )
            expected_edge_ids = _expected_edge_ids_for_seed(seed)
            assert sorted(datum.edge.tolist()) == sorted(expected_edge_ids.values()), (
                f"Seed {seed}: expected global edge ids "
                f"{sorted(expected_edge_ids.values())}, got {sorted(datum.edge.tolist())}"
            )
            # Neighbor-to-edge-id pairing survives the rebase and the RPC.
            local_ends_a, local_ends_b = datum.edge_index[0], datum.edge_index[1]
            for position in range(datum.edge_index.size(1)):
                endpoint_a = int(datum.node[local_ends_a[position]].item())
                endpoint_b = int(datum.node[local_ends_b[position]].item())
                neighbor = endpoint_a if endpoint_b == seed else endpoint_b
                sampled_edge_id = int(datum.edge[position].item())
                assert expected_edge_ids[neighbor] == sampled_edge_id, (
                    f"Seed {seed}: neighbor {neighbor} paired with edge id "
                    f"{sampled_edge_id}, expected {expected_edge_ids[neighbor]}"
                )
        batch_count += 1
    assert batch_count == _E2E_NUM_NODES, (
        f"Expected {_E2E_NUM_NODES} batches, got {batch_count}"
    )

    torch.distributed.barrier()
    if not weighted:
        # Degrees are computed from the rebased topologies and must come back
        # globally indexed. Both ranks run on one host, so the over-counting
        # correction divides the all-reduced sum by 2; degrees are even so the
        # halved values stay exact and positionally comparable.
        expected_degrees = torch.tensor(
            [len(_E2E_NEIGHBORS[node]) // 2 for node in range(_E2E_NUM_NODES)],
            dtype=torch.int32,
        )
        degree_tensor = dataset.degree_tensor
        assert isinstance(degree_tensor, torch.Tensor)
        assert_tensor_equality(degree_tensor, expected_degrees)

    torch.distributed.barrier()
    shutdown_rpc()
    torch.distributed.destroy_process_group()


class TwoRankRangePartitionedSamplingTest(TestCase):
    def test_sampling_returns_global_neighbors_edge_ids_and_degrees(self):
        init_method = get_process_group_init_method()
        mp.spawn(
            fn=_run_two_rank_range_partitioned_sampling,
            args=(2, init_method, False),
            nprocs=2,
            join=True,
        )

    def test_weighted_sampling_keeps_edge_weight_pairing(self):
        init_method = get_process_group_init_method()
        mp.spawn(
            fn=_run_two_rank_range_partitioned_sampling,
            args=(2, init_method, True),
            nprocs=2,
            join=True,
        )


if __name__ == "__main__":
    absltest.main()
