"""Distributed integration tests for weighted edge sampling.

Covers two surfaces:
  1. DistPartitioner correctly partitions registered edge weights (weights land on
     the right rank and match the expected values).
  2. DistNeighborLoader with with_weight=True never traverses weight=0 edges —
     verified by encoding node type into features (hub=2.0, good=1.0, bad=0.0)
     and asserting no bad node appears in any sampled subgraph.
"""

from typing import MutableMapping

import torch
import torch.multiprocessing as mp
from absl.testing import absltest
from graphlearn_torch.distributed import shutdown_rpc
from torch.multiprocessing import Manager
from torch_geometric.data import Data, HeteroData

from gigl.distributed import DistPartitioner
from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.dist_range_partitioner import DistRangePartitioner
from gigl.distributed.distributed_neighborloader import DistNeighborLoader
from gigl.distributed.utils.networking import get_free_port
from gigl.types.graph import (
    FeaturePartitionData,
    GraphPartitionData,
    PartitionOutput,
)
from tests.test_assets.distributed.bipartite_weight_graph import (
    ITEM,
    ITEM_TO_USER,
    USER,
    USER_TO_ITEM,
    build_heterogeneous_bipartite_weight_graph,
    build_homogeneous_bipartite_weight_graph,
)
from tests.test_assets.distributed.constants import (
    MOCKED_NUM_PARTITIONS,
    MOCKED_U2U_EDGE_INDEX_ON_RANK_ONE,
    MOCKED_U2U_EDGE_INDEX_ON_RANK_ZERO,
    RANK_TO_MOCKED_GRAPH,
    USER_TO_ITEM_EDGE_TYPE,
    USER_TO_USER_EDGE_TYPE,
)
from tests.test_assets.distributed.run_distributed_partitioner import (
    InputDataStrategy,
    run_distributed_partitioner,
)
from tests.test_assets.distributed.utils import create_test_process_group
from tests.test_assets.test_case import TestCase

# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------


def _build_heterogeneous_bipartite_partial_weight_graph() -> tuple[
    PartitionOutput, int
]:
    """Same graph as _build_heterogeneous_bipartite_weight_graph but ITEM_TO_USER is unweighted.

    This validates that partial-weight heterogeneous graphs work correctly:
    the weighted U2I edge type still respects weights (bad items are unreachable)
    while the unweighted I2U edge type samples uniformly without crashing.
    """
    n_user = 10
    n_good_item = 40
    n_bad_item = 20
    n_item = n_good_item + n_bad_item

    user_ids = torch.arange(n_user)
    good_item_ids = torch.arange(n_good_item)
    bad_item_ids = torch.arange(n_good_item, n_item)

    u2gi_src = user_ids.repeat_interleave(n_good_item)
    u2gi_dst = good_item_ids.repeat(n_user)
    u2gi_w = torch.ones(n_user * n_good_item)

    u2bi_src = user_ids.repeat_interleave(n_bad_item)
    u2bi_dst = bad_item_ids.repeat(n_user)
    u2bi_w = torch.zeros(n_user * n_bad_item)

    gi2u_src = good_item_ids.repeat_interleave(n_user)
    gi2u_dst = user_ids.repeat(n_good_item)

    u2i_src = torch.cat([u2gi_src, u2bi_src])
    u2i_dst = torch.cat([u2gi_dst, u2bi_dst])
    u2i_w = torch.cat([u2gi_w, u2bi_w])
    n_u2i_edges = u2i_src.shape[0]

    user_feats = torch.full((n_user, 1), 2.0)
    item_feats = torch.cat(
        [
            torch.full((n_good_item, 1), 1.0),
            torch.full((n_bad_item, 1), 0.0),
        ]
    )

    partition_output = PartitionOutput(
        node_partition_book={
            USER: torch.zeros(n_user),
            ITEM: torch.zeros(n_item),
        },
        edge_partition_book={
            USER_TO_ITEM: torch.zeros(n_u2i_edges),
            ITEM_TO_USER: torch.zeros(gi2u_src.shape[0]),
        },
        partitioned_edge_index={
            USER_TO_ITEM: GraphPartitionData(
                edge_index=torch.stack([u2i_src, u2i_dst]),
                edge_ids=None,
                weights=u2i_w,
            ),
            ITEM_TO_USER: GraphPartitionData(
                edge_index=torch.stack([gi2u_src, gi2u_dst]),
                edge_ids=None,
                weights=None,  # unweighted — samples uniformly
            ),
        },
        partitioned_node_features={
            USER: FeaturePartitionData(feats=user_feats, ids=torch.arange(n_user)),
            ITEM: FeaturePartitionData(feats=item_feats, ids=torch.arange(n_item)),
        },
        partitioned_edge_features=None,
        partitioned_positive_labels=None,
        partitioned_negative_labels=None,
        partitioned_node_labels=None,
    )
    return partition_output, n_user


# ---------------------------------------------------------------------------
# Subprocess functions — must accept local_rank as first arg (mp.spawn)
# ---------------------------------------------------------------------------


def _run_weighted_sampling_correctness_homogeneous(
    _: int,
    dataset: DistDataset,
    n_hub: int,
) -> None:
    """Subprocess: verifies weight=0 edges are never traversed in homogeneous graph.

    Seeds are the hub nodes only.  Node features encode the type:
    hub=2.0, good=1.0, bad=0.0.  Any subgraph batch containing a bad node
    (feature==0.0) means a weight=0 edge was sampled — a test failure.
    """
    create_test_process_group()
    loader = DistNeighborLoader(
        dataset=dataset,
        input_nodes=torch.arange(n_hub),
        num_neighbors=[10, 5],
        with_weight=True,
        pin_memory_device=torch.device("cpu"),
    )
    count = 0
    for datum in loader:
        assert isinstance(datum, Data), f"Expected Data, got {type(datum)}"
        assert datum.x is not None, "Node features missing from sampled subgraph"
        # Bad nodes have feature 0.0; hub and good nodes have feature > 0.
        bad_mask = datum.x[:, 0] == 0.0
        assert not bad_mask.any(), (
            f"weight=0 edge was sampled: bad node(s) found in subgraph. "
            f"Features of bad nodes: {datum.x[bad_mask].squeeze().tolist()}"
        )
        count += 1
    assert count == n_hub, f"Expected {n_hub} batches (one per hub seed), got {count}"
    shutdown_rpc()


def _run_weighted_sampling_correctness_heterogeneous(
    _: int,
    dataset: DistDataset,
    n_user: int,
) -> None:
    """Subprocess: verifies weight=0 edges are never traversed in heterogeneous graph.

    Seeds are all user nodes.  Item features encode type: good=1.0, bad=0.0.
    Any batch containing a bad item node means a weight=0 edge was sampled.
    """
    create_test_process_group()
    node_ids = dataset.node_ids
    assert not isinstance(node_ids, torch.Tensor) and node_ids is not None, (
        "Expected heterogeneous dataset with dict node_ids"
    )
    loader = DistNeighborLoader(
        dataset=dataset,
        input_nodes=(USER, node_ids[USER]),
        num_neighbors=[10, 5],
        with_weight=True,
        pin_memory_device=torch.device("cpu"),
    )
    count = 0
    for datum in loader:
        assert isinstance(datum, HeteroData), f"Expected HeteroData, got {type(datum)}"
        if ITEM in datum.node_types:
            item_x = datum[ITEM].x
            assert item_x is not None, "Item features missing from sampled subgraph"
            bad_mask = item_x[:, 0] == 0.0
            assert not bad_mask.any(), (
                f"weight=0 edge was sampled: bad item node(s) found. "
                f"Features of bad items: {item_x[bad_mask].squeeze().tolist()}"
            )
        count += 1
    assert count == n_user, (
        f"Expected {n_user} batches (one per user seed), got {count}"
    )
    shutdown_rpc()


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class WeightedEdgePartitionerTestCase(TestCase):
    """Tests that DistPartitioner correctly partitions registered edge weights."""

    def setUp(self) -> None:
        self._master_ip_address = "localhost"

    def test_homogeneous_weights_partitioned_correctly(self) -> None:
        """Edge weights (= src_node_id / 10.0) land on the correct rank after partitioning.

        The mocked graph has edges with source nodes 0–3 on rank 0 and 4–7 on rank 1.
        Weights are set to src_node_id / 10.0, mirroring the existing edge-feature
        convention.  After partitioning by source node each rank should hold only its
        own weights, and each weight should equal the corresponding global edge ID * 0.1.
        """
        master_port = get_free_port()
        manager = Manager()
        output_dict: MutableMapping[int, PartitionOutput] = manager.dict()

        rank_to_edge_weights = {
            0: MOCKED_U2U_EDGE_INDEX_ON_RANK_ZERO[0].float() / 10.0,
            1: MOCKED_U2U_EDGE_INDEX_ON_RANK_ONE[0].float() / 10.0,
        }

        mp.spawn(
            run_distributed_partitioner,
            args=(
                output_dict,
                False,  # is_heterogeneous
                RANK_TO_MOCKED_GRAPH,
                True,  # should_assign_edges_by_src_node
                self._master_ip_address,
                master_port,
                InputDataStrategy.REGISTER_ALL_ENTITIES_SEPARATELY,
                DistPartitioner,
                rank_to_edge_weights,
            ),
            nprocs=MOCKED_NUM_PARTITIONS,
            join=True,
        )

        for rank, partition_output in output_dict.items():
            partitioned_edge_index = partition_output.partitioned_edge_index
            self.assertIsInstance(partitioned_edge_index, GraphPartitionData)
            assert isinstance(partitioned_edge_index, GraphPartitionData)

            weights = partitioned_edge_index.weights
            self.assertIsNotNone(
                weights,
                msg=f"Rank {rank}: expected weights in GraphPartitionData, got None",
            )
            assert weights is not None

            edge_ids = partitioned_edge_index.edge_ids
            self.assertIsNotNone(
                edge_ids,
                msg=f"Rank {rank}: edge_ids must be present when weights are registered",
            )
            assert edge_ids is not None

            self.assertEqual(
                weights.shape,
                edge_ids.shape,
                msg=f"Rank {rank}: weights and edge_ids must have the same length",
            )

            # weight for each edge == its global edge_id * 0.1 (i.e. src_node_id / 10.0).
            # Sort both so the comparison is order-independent.
            expected_weights = edge_ids.float() * 0.1
            torch.testing.assert_close(
                weights.sort().values,
                expected_weights.sort().values,
                msg=f"Rank {rank}: partitioned weights do not match expected src_node_id / 10.0",
            )

    def test_features_only_weights_are_none(self) -> None:
        """Features only (no weights): weights must be None, edge_ids and features present."""
        master_port = get_free_port()
        manager = Manager()
        output_dict: MutableMapping[int, PartitionOutput] = manager.dict()

        mp.spawn(
            run_distributed_partitioner,
            args=(
                output_dict,
                False,
                RANK_TO_MOCKED_GRAPH,
                True,
                self._master_ip_address,
                master_port,
                InputDataStrategy.REGISTER_ALL_ENTITIES_SEPARATELY,
                DistPartitioner,
                None,
            ),
            nprocs=MOCKED_NUM_PARTITIONS,
            join=True,
        )

        for rank, partition_output in output_dict.items():
            gpd = partition_output.partitioned_edge_index
            self.assertIsInstance(gpd, GraphPartitionData)
            assert isinstance(gpd, GraphPartitionData)
            self.assertIsNone(
                gpd.weights,
                msg=f"Rank {rank}: weights must be None when not registered",
            )
            self.assertIsNotNone(
                gpd.edge_ids,
                msg=f"Rank {rank}: edge_ids must be present when features are registered",
            )
            self.assertIsNotNone(
                partition_output.partitioned_edge_features,
                msg=f"Rank {rank}: expected features to be partitioned",
            )

    def test_weights_only_no_features_partitioned_correctly(self) -> None:
        """Weights without features: feature part is None, weights have correct values."""
        master_port = get_free_port()
        manager = Manager()
        output_dict: MutableMapping[int, PartitionOutput] = manager.dict()

        rank_to_edge_weights = {
            0: MOCKED_U2U_EDGE_INDEX_ON_RANK_ZERO[0].float() / 10.0,
            1: MOCKED_U2U_EDGE_INDEX_ON_RANK_ONE[0].float() / 10.0,
        }

        mp.spawn(
            run_distributed_partitioner,
            args=(
                output_dict,
                False,
                RANK_TO_MOCKED_GRAPH,
                True,
                self._master_ip_address,
                master_port,
                InputDataStrategy.REGISTER_EDGE_WEIGHTS_WITHOUT_EDGE_FEATURES,
                DistPartitioner,
                rank_to_edge_weights,
            ),
            nprocs=MOCKED_NUM_PARTITIONS,
            join=True,
        )

        for rank, partition_output in output_dict.items():
            gpd = partition_output.partitioned_edge_index
            self.assertIsInstance(gpd, GraphPartitionData)
            assert isinstance(gpd, GraphPartitionData)
            self.assertIsNone(
                partition_output.partitioned_edge_features,
                msg=f"Rank {rank}: features must be None when not registered",
            )

            weights = gpd.weights
            self.assertIsNotNone(
                weights, msg=f"Rank {rank}: expected weights to be partitioned"
            )
            assert weights is not None

            edge_ids = gpd.edge_ids
            self.assertIsNotNone(
                edge_ids,
                msg=f"Rank {rank}: edge_ids must be present when weights are registered",
            )
            assert edge_ids is not None

            self.assertEqual(weights.shape, edge_ids.shape)
            expected_weights = edge_ids.float() * 0.1
            torch.testing.assert_close(
                weights.sort().values,
                expected_weights.sort().values,
                msg=f"Rank {rank}: weights do not match src_node_id / 10.0",
            )

    def test_neither_features_nor_weights_gives_none_edge_ids(self) -> None:
        """No features, no weights: edge_ids and weights must both be None."""
        master_port = get_free_port()
        manager = Manager()
        output_dict: MutableMapping[int, PartitionOutput] = manager.dict()

        mp.spawn(
            run_distributed_partitioner,
            args=(
                output_dict,
                False,
                RANK_TO_MOCKED_GRAPH,
                True,
                self._master_ip_address,
                master_port,
                InputDataStrategy.REGISTER_MINIMAL_ENTITIES_SEPARATELY,
                DistPartitioner,
            ),
            nprocs=MOCKED_NUM_PARTITIONS,
            join=True,
        )

        for rank, partition_output in output_dict.items():
            gpd = partition_output.partitioned_edge_index
            self.assertIsInstance(gpd, GraphPartitionData)
            assert isinstance(gpd, GraphPartitionData)
            self.assertIsNone(
                gpd.weights,
                msg=f"Rank {rank}: weights must be None when not registered",
            )
            self.assertIsNone(
                gpd.edge_ids,
                msg=f"Rank {rank}: edge_ids must be None when neither features nor weights are registered",
            )
            self.assertIsNone(
                partition_output.partitioned_edge_features,
                msg=f"Rank {rank}: features must be None",
            )

    def test_features_and_weights_produce_consistent_edge_ids(self) -> None:
        """Both registered: GraphPartitionData.edge_ids must equal FeaturePartitionData.ids."""
        master_port = get_free_port()
        manager = Manager()
        output_dict: MutableMapping[int, PartitionOutput] = manager.dict()

        rank_to_edge_weights = {
            0: MOCKED_U2U_EDGE_INDEX_ON_RANK_ZERO[0].float() / 10.0,
            1: MOCKED_U2U_EDGE_INDEX_ON_RANK_ONE[0].float() / 10.0,
        }

        mp.spawn(
            run_distributed_partitioner,
            args=(
                output_dict,
                False,
                RANK_TO_MOCKED_GRAPH,
                True,
                self._master_ip_address,
                master_port,
                InputDataStrategy.REGISTER_ALL_ENTITIES_SEPARATELY,
                DistPartitioner,
                rank_to_edge_weights,
            ),
            nprocs=MOCKED_NUM_PARTITIONS,
            join=True,
        )

        for rank, partition_output in output_dict.items():
            gpd = partition_output.partitioned_edge_index
            assert isinstance(gpd, GraphPartitionData)
            feat_part = partition_output.partitioned_edge_features
            self.assertIsInstance(feat_part, FeaturePartitionData)
            assert isinstance(feat_part, FeaturePartitionData)

            self.assertIsNotNone(gpd.edge_ids)
            self.assertIsNotNone(gpd.weights)
            assert gpd.edge_ids is not None

            torch.testing.assert_close(
                gpd.edge_ids,
                feat_part.ids,
                msg=f"Rank {rank}: GraphPartitionData.edge_ids must equal FeaturePartitionData.ids",
            )

    def test_heterogeneous_partial_weights_by_edge_type(self) -> None:
        """Heterogeneous: edge type with weights has them; edge type without weights has None."""
        master_port = get_free_port()
        manager = Manager()
        output_dict: MutableMapping[int, PartitionOutput] = manager.dict()

        rank_to_edge_weights = {
            0: {
                USER_TO_USER_EDGE_TYPE: MOCKED_U2U_EDGE_INDEX_ON_RANK_ZERO[0].float()
                / 10.0
            },
            1: {
                USER_TO_USER_EDGE_TYPE: MOCKED_U2U_EDGE_INDEX_ON_RANK_ONE[0].float()
                / 10.0
            },
        }

        mp.spawn(
            run_distributed_partitioner,
            args=(
                output_dict,
                True,  # is_heterogeneous
                RANK_TO_MOCKED_GRAPH,
                True,
                self._master_ip_address,
                master_port,
                InputDataStrategy.REGISTER_ALL_ENTITIES_SEPARATELY,
                DistPartitioner,
                rank_to_edge_weights,
            ),
            nprocs=MOCKED_NUM_PARTITIONS,
            join=True,
        )

        for rank, partition_output in output_dict.items():
            partitioned_edge_index = partition_output.partitioned_edge_index
            self.assertIsInstance(partitioned_edge_index, dict)
            assert isinstance(partitioned_edge_index, dict)

            u2u_gpd = partitioned_edge_index[USER_TO_USER_EDGE_TYPE]
            u2i_gpd = partitioned_edge_index[USER_TO_ITEM_EDGE_TYPE]

            self.assertIsNotNone(
                u2u_gpd.weights,
                msg=f"Rank {rank}: U2U edge type should have weights",
            )
            self.assertIsNone(
                u2i_gpd.weights,
                msg=f"Rank {rank}: U2I edge type should not have weights",
            )
            # U2U edge_ids present (has both features and weights).
            self.assertIsNotNone(
                u2u_gpd.edge_ids,
                msg=f"Rank {rank}: U2U edge_ids must be present",
            )
            # U2I has neither features nor weights, so edge_ids is None.
            self.assertIsNone(
                u2i_gpd.edge_ids,
                msg=f"Rank {rank}: U2I edge_ids must be None",
            )

    def test_range_partitioner_homogeneous_weights_partitioned_correctly(self) -> None:
        """DistRangePartitioner: edge weights land on the correct rank after range-based partitioning.

        Mirrors test_homogeneous_weights_partitioned_correctly but uses DistRangePartitioner.
        With range-based partitioning, edge_ids are sequential per-rank (0..3 on rank 0,
        4..7 on rank 1), and the registered weights (src_node_id / 10.0) equal
        edge_id * 0.1 for this test graph.
        """
        master_port = get_free_port()
        manager = Manager()
        output_dict: MutableMapping[int, PartitionOutput] = manager.dict()

        rank_to_edge_weights = {
            0: MOCKED_U2U_EDGE_INDEX_ON_RANK_ZERO[0].float() / 10.0,
            1: MOCKED_U2U_EDGE_INDEX_ON_RANK_ONE[0].float() / 10.0,
        }

        mp.spawn(
            run_distributed_partitioner,
            args=(
                output_dict,
                False,  # is_heterogeneous
                RANK_TO_MOCKED_GRAPH,
                True,  # should_assign_edges_by_src_node
                self._master_ip_address,
                master_port,
                InputDataStrategy.REGISTER_ALL_ENTITIES_SEPARATELY,
                DistRangePartitioner,
                rank_to_edge_weights,
            ),
            nprocs=MOCKED_NUM_PARTITIONS,
            join=True,
        )

        for rank, partition_output in output_dict.items():
            partitioned_edge_index = partition_output.partitioned_edge_index
            self.assertIsInstance(partitioned_edge_index, GraphPartitionData)
            assert isinstance(partitioned_edge_index, GraphPartitionData)

            weights = partitioned_edge_index.weights
            self.assertIsNotNone(
                weights,
                msg=f"Rank {rank}: expected weights in GraphPartitionData, got None",
            )
            assert weights is not None

            edge_ids = partitioned_edge_index.edge_ids
            self.assertIsNotNone(
                edge_ids,
                msg=f"Rank {rank}: edge_ids must be present when features are registered",
            )
            assert edge_ids is not None

            self.assertEqual(
                weights.shape,
                edge_ids.shape,
                msg=f"Rank {rank}: weights and edge_ids must have the same length",
            )

            expected_weights = edge_ids.float() * 0.1
            torch.testing.assert_close(
                weights.sort().values,
                expected_weights.sort().values,
                msg=f"Rank {rank}: partitioned weights do not match expected src_node_id / 10.0",
            )


class DistributedWeightedSamplingTest(TestCase):
    """End-to-end correctness tests for DistNeighborLoader with with_weight=True.

    Each test builds a graph with two classes of neighbors:
      - "good" neighbors connected by weight=1 edges
      - "bad" neighbors connected by weight=0 edges

    Node features encode the class (good=1.0, bad=0.0).  After weighted sampling,
    no bad node should ever appear in a sampled subgraph — if it does, a weight=0
    edge was traversed, indicating a bug.

    Graph sizes are chosen so that the fanout is strictly smaller than the number
    of available good neighbors, ensuring the sampler actively makes choices
    (rather than returning all neighbors) and the test is non-trivial.
    """

    def setUp(self) -> None:
        super().setUp()
        self._world_size = 1

    def tearDown(self) -> None:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        super().tearDown()

    def test_weighted_sampling_never_traverses_zero_weight_edges_homogeneous(
        self,
    ) -> None:
        """Homogeneous: weight=0 edges to bad nodes are never traversed.

        Graph: 10 hub seeds, each with 50 good neighbors (weight=1) and 40 bad
        neighbors (weight=0).  Good nodes have 5 further weight=1 edges for 2nd-hop
        sampling.  Fanout [10, 5] samples fewer neighbors than available good ones,
        so the weighted sampler actively selects from the pool each hop.
        """
        partition_output, n_hub = build_homogeneous_bipartite_weight_graph()
        assert isinstance(partition_output.partitioned_edge_index, GraphPartitionData)
        expected_weights = partition_output.partitioned_edge_index.weights

        dataset = DistDataset(rank=0, world_size=1, edge_dir="out")
        dataset.build(partition_output=partition_output)

        self.assertTrue(dataset.has_edge_weights)
        self.assertIsNotNone(dataset.edge_weights)
        assert isinstance(dataset.edge_weights, torch.Tensor)
        torch.testing.assert_close(dataset.edge_weights, expected_weights)

        mp.spawn(
            fn=_run_weighted_sampling_correctness_homogeneous,
            args=(dataset, n_hub),
        )

    def test_weighted_sampling_never_traverses_zero_weight_edges_heterogeneous(
        self,
    ) -> None:
        """Heterogeneous: weight=0 user→item edges to bad items are never traversed.

        Graph: 10 user seeds, each with 40 good items (weight=1) and 20 bad items
        (weight=0).  Good items have weight=1 back-edges to all users for 2nd-hop.
        Fanout [10, 5] is smaller than the 40 available good items, so the sampler
        actively selects.
        """
        partition_output, n_user = build_heterogeneous_bipartite_weight_graph()
        dataset = DistDataset(rank=0, world_size=1, edge_dir="out")
        dataset.build(partition_output=partition_output)

        mp.spawn(
            fn=_run_weighted_sampling_correctness_heterogeneous,
            args=(dataset, n_user),
        )

    def test_weighted_sampling_partial_weights_heterogeneous(self) -> None:
        """Partial weights: weighted U2I respects weights; unweighted I2U samples uniformly.

        U2I is weighted (good items weight=1, bad items weight=0) — bad items must
        never appear.  I2U has no weights registered, so it uses uniform sampling.
        Verifies that mixing weighted and unweighted edge types in one heterogeneous
        graph does not crash and that weighted edges still behave correctly.
        """
        partition_output, n_user = _build_heterogeneous_bipartite_partial_weight_graph()
        dataset = DistDataset(rank=0, world_size=1, edge_dir="out")
        dataset.build(partition_output=partition_output)

        mp.spawn(
            fn=_run_weighted_sampling_correctness_heterogeneous,
            args=(dataset, n_user),
        )


if __name__ == "__main__":
    absltest.main()
