"""Distributed integration tests for weighted edge sampling.

Covers two surfaces:
  1. DistPartitioner correctly partitions registered edge weights (weights land on
     the right rank and match the expected values).
  2. DistNeighborLoader with with_weight=True never traverses weight=0 edges —
     verified by encoding node type into features (hub=2.0, good=1.0, bad=0.0)
     and asserting no bad node appears in any sampled subgraph.
"""

from collections.abc import Mapping
from typing import MutableMapping

import torch
import torch.multiprocessing as mp
from absl.testing import absltest
from graphlearn_torch.distributed import shutdown_rpc
from torch.multiprocessing import Manager
from torch_geometric.data import Data, HeteroData

from gigl.distributed import DistPartitioner
from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.distributed_neighborloader import DistNeighborLoader
from gigl.distributed.utils.networking import get_free_port
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.types.graph import (
    FeaturePartitionData,
    GraphPartitionData,
    PartitionOutput,
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

_USER = NodeType("user")
_ITEM = NodeType("item")
_USER_TO_ITEM = EdgeType(_USER, Relation("to"), _ITEM)
_ITEM_TO_USER = EdgeType(_ITEM, Relation("to"), _USER)


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------


def _build_homogeneous_bipartite_weight_graph() -> tuple[
    PartitionOutput, int, int, int
]:
    """Build a homogeneous graph with hub, good, and bad nodes.

    Graph structure:
      - 10 hub nodes (0..9): used as seed nodes; feature value = 2.0
      - 50 good nodes (10..59): reachable from hubs via weight=1 edges; feature = 1.0
      - 40 bad nodes (60..99): reachable from hubs via weight=0 edges; feature = 0.0
      - Each good node also has 5 outgoing weight=1 edges to nearby good nodes
        (ring topology, for 2nd-hop sampling).

    With weighted sampling only good nodes should ever appear as sampled
    neighbors — weight=0 edges to bad nodes must never be traversed.

    Returns:
        (partition_output, n_hub, n_good, n_bad)
    """
    n_hub = 10
    n_good = 50
    n_bad = 40
    n = n_hub + n_good + n_bad  # 100

    hub_ids = torch.arange(n_hub)
    good_ids = torch.arange(n_hub, n_hub + n_good)
    bad_ids = torch.arange(n_hub + n_good, n)

    # Hub → Good: weight=1
    hub_good_src = hub_ids.repeat_interleave(n_good)
    hub_good_dst = good_ids.repeat(n_hub)
    hub_good_w = torch.ones(n_hub * n_good)

    # Hub → Bad: weight=0
    hub_bad_src = hub_ids.repeat_interleave(n_bad)
    hub_bad_dst = bad_ids.repeat(n_hub)
    hub_bad_w = torch.zeros(n_hub * n_bad)

    # Good → Good: ring with 5 outgoing edges per node, weight=1 (2nd-hop targets)
    connections_per_good = 5
    good_src = good_ids.repeat_interleave(connections_per_good)
    # Row i of [connections_per_good, n_good].T gives neighbors of good_ids[i]
    good_dst = torch.stack(
        [torch.roll(good_ids, -j) for j in range(1, connections_per_good + 1)]
    ).T.reshape(-1)
    good_w = torch.ones(n_good * connections_per_good)

    edge_src = torch.cat([hub_good_src, hub_bad_src, good_src])
    edge_dst = torch.cat([hub_good_dst, hub_bad_dst, good_dst])
    weights = torch.cat([hub_good_w, hub_bad_w, good_w])
    edge_index = torch.stack([edge_src, edge_dst])
    n_edges = edge_src.shape[0]

    # Feature encodes node type: hub=2.0, good=1.0, bad=0.0
    node_feats = torch.cat(
        [
            torch.full((n_hub, 1), 2.0),
            torch.full((n_good, 1), 1.0),
            torch.full((n_bad, 1), 0.0),
        ]
    )

    partition_output = PartitionOutput(
        node_partition_book=torch.zeros(n),
        edge_partition_book=torch.zeros(n_edges),
        partitioned_edge_index=GraphPartitionData(
            edge_index=edge_index,
            edge_ids=None,
            weights=weights,
        ),
        partitioned_node_features=FeaturePartitionData(
            feats=node_feats,
            ids=torch.arange(n),
        ),
        partitioned_edge_features=None,
        partitioned_positive_labels=None,
        partitioned_negative_labels=None,
        partitioned_node_labels=None,
    )
    return partition_output, n_hub, n_good, n_bad


def _build_heterogeneous_bipartite_weight_graph() -> tuple[
    PartitionOutput, int, int, int
]:
    """Build a heterogeneous (user/item) graph with good and bad item nodes.

    Graph structure:
      - 10 user nodes (0..9): seed nodes; user feature = 2.0
      - 60 item nodes total:
          - Items 0..39: good, reachable from users via weight=1 edges; feature = 1.0
          - Items 40..59: bad, reachable from users via weight=0 edges; feature = 0.0
      - Good items also have weight=1 edges back to all users (for 2nd-hop).

    With weighted sampling only good item nodes should ever appear as sampled
    item neighbors.

    Returns:
        (partition_output, n_user, n_good_item, n_bad_item)
    """
    n_user = 10
    n_good_item = 40
    n_bad_item = 20
    n_item = n_good_item + n_bad_item  # 60

    user_ids = torch.arange(n_user)
    good_item_ids = torch.arange(n_good_item)
    bad_item_ids = torch.arange(n_good_item, n_item)

    # User → Good Item: weight=1
    u2gi_src = user_ids.repeat_interleave(n_good_item)
    u2gi_dst = good_item_ids.repeat(n_user)
    u2gi_w = torch.ones(n_user * n_good_item)

    # User → Bad Item: weight=0
    u2bi_src = user_ids.repeat_interleave(n_bad_item)
    u2bi_dst = bad_item_ids.repeat(n_user)
    u2bi_w = torch.zeros(n_user * n_bad_item)

    # Good Item → User: weight=1 (2nd-hop back to users)
    gi2u_src = good_item_ids.repeat_interleave(n_user)
    gi2u_dst = user_ids.repeat(n_good_item)
    gi2u_w = torch.ones(n_good_item * n_user)

    u2i_src = torch.cat([u2gi_src, u2bi_src])
    u2i_dst = torch.cat([u2gi_dst, u2bi_dst])
    u2i_w = torch.cat([u2gi_w, u2bi_w])
    n_u2i_edges = u2i_src.shape[0]

    user_feats = torch.full((n_user, 1), 2.0)
    # Item feature encodes type: good=1.0, bad=0.0
    item_feats = torch.cat(
        [
            torch.full((n_good_item, 1), 1.0),
            torch.full((n_bad_item, 1), 0.0),
        ]
    )

    partition_output = PartitionOutput(
        node_partition_book={
            _USER: torch.zeros(n_user),
            _ITEM: torch.zeros(n_item),
        },
        edge_partition_book={
            _USER_TO_ITEM: torch.zeros(n_u2i_edges),
            _ITEM_TO_USER: torch.zeros(gi2u_src.shape[0]),
        },
        partitioned_edge_index={
            _USER_TO_ITEM: GraphPartitionData(
                edge_index=torch.stack([u2i_src, u2i_dst]),
                edge_ids=None,
                weights=u2i_w,
            ),
            _ITEM_TO_USER: GraphPartitionData(
                edge_index=torch.stack([gi2u_src, gi2u_dst]),
                edge_ids=None,
                weights=gi2u_w,
            ),
        },
        partitioned_node_features={
            _USER: FeaturePartitionData(feats=user_feats, ids=torch.arange(n_user)),
            _ITEM: FeaturePartitionData(feats=item_feats, ids=torch.arange(n_item)),
        },
        partitioned_edge_features=None,
        partitioned_positive_labels=None,
        partitioned_negative_labels=None,
        partitioned_node_labels=None,
    )
    return partition_output, n_user, n_good_item, n_bad_item


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
    assert isinstance(dataset.node_ids, Mapping)
    loader = DistNeighborLoader(
        dataset=dataset,
        input_nodes=(_USER, dataset.node_ids[_USER]),
        num_neighbors=[10, 5],
        with_weight=True,
        pin_memory_device=torch.device("cpu"),
    )
    count = 0
    for datum in loader:
        assert isinstance(datum, HeteroData), f"Expected HeteroData, got {type(datum)}"
        if _ITEM in datum.node_types:
            item_x = datum[_ITEM].x
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
        partition_output, n_hub, _, _ = _build_homogeneous_bipartite_weight_graph()
        dataset = DistDataset(rank=0, world_size=1, edge_dir="out")
        dataset.build(partition_output=partition_output)

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
        partition_output, n_user, _, _ = _build_heterogeneous_bipartite_weight_graph()
        dataset = DistDataset(rank=0, world_size=1, edge_dir="out")
        dataset.build(partition_output=partition_output)

        mp.spawn(
            fn=_run_weighted_sampling_correctness_heterogeneous,
            args=(dataset, n_user),
        )


if __name__ == "__main__":
    absltest.main()
