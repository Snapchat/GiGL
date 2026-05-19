"""End-to-end correctness tests for PPR weighted sampling.

Verifies that DistNeighborLoader with PPRSamplerOptions and with_weight=True
never traverses weight=0 edges.  The test graph encodes node type in features
(hub=2.0, good=1.0, bad=0.0); any bad node in a sampled subgraph indicates
that a weight=0 edge contributed PPR residual — a test failure.

With weight-proportional residual distribution, a weight=0 edge contributes
zero to totalFetchedWeight and receives zero residual per push step.  Bad
nodes therefore accumulate a PPR score of exactly 0 and are excluded from
every top-k result.
"""

import torch
import torch.multiprocessing as mp
from absl.testing import absltest
from graphlearn_torch.distributed import shutdown_rpc
from torch_geometric.data import Data, HeteroData

from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.distributed_neighborloader import DistNeighborLoader
from gigl.distributed.sampler_options import PPRSamplerOptions
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.types.graph import (
    FeaturePartitionData,
    GraphPartitionData,
    PartitionOutput,
)
from tests.test_assets.distributed.utils import create_test_process_group
from tests.test_assets.test_case import TestCase

_USER = NodeType("user")
_ITEM = NodeType("item")
_USER_TO_ITEM = EdgeType(_USER, Relation("to"), _ITEM)
_ITEM_TO_USER = EdgeType(_ITEM, Relation("to"), _USER)

# PPR parameters used across all tests.
_PPR_ALPHA = 0.5
_PPR_EPS = 1e-4
_PPR_MAX_NODES = 60
_PPR_NUM_NBRS = 200


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------


def _build_homogeneous_bipartite_weight_graph() -> tuple[
    PartitionOutput, int, int, int
]:
    """Build a homogeneous graph with hub, good, and bad nodes.

    Graph structure:
      - 10 hub nodes (0..9): seed nodes; feature = 2.0
      - 50 good nodes (10..59): reachable via weight=1 edges; feature = 1.0
      - 40 bad nodes (60..99): reachable via weight=0 edges; feature = 0.0
      - Each good node has 5 outgoing weight=1 edges to nearby good nodes (ring).

    Returns:
        (partition_output, n_hub, n_good, n_bad)
    """
    n_hub = 10
    n_good = 50
    n_bad = 40
    n = n_hub + n_good + n_bad

    hub_ids = torch.arange(n_hub)
    good_ids = torch.arange(n_hub, n_hub + n_good)
    bad_ids = torch.arange(n_hub + n_good, n)

    hub_good_src = hub_ids.repeat_interleave(n_good)
    hub_good_dst = good_ids.repeat(n_hub)
    hub_good_w = torch.ones(n_hub * n_good)

    hub_bad_src = hub_ids.repeat_interleave(n_bad)
    hub_bad_dst = bad_ids.repeat(n_hub)
    hub_bad_w = torch.zeros(n_hub * n_bad)

    connections_per_good = 5
    good_src = good_ids.repeat_interleave(connections_per_good)
    good_dst = torch.stack(
        [torch.roll(good_ids, -j) for j in range(1, connections_per_good + 1)]
    ).T.reshape(-1)
    good_w = torch.ones(n_good * connections_per_good)

    edge_src = torch.cat([hub_good_src, hub_bad_src, good_src])
    edge_dst = torch.cat([hub_good_dst, hub_bad_dst, good_dst])
    weights = torch.cat([hub_good_w, hub_bad_w, good_w])
    edge_index = torch.stack([edge_src, edge_dst])
    n_edges = edge_src.shape[0]

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
      - 40 good item nodes (0..39): weight=1 from users; feature = 1.0
      - 20 bad item nodes (40..59): weight=0 from users; feature = 0.0
      - Good items also connect back to all users via weight=1 (2nd-hop).

    Returns:
        (partition_output, n_user, n_good_item, n_bad_item)
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
    gi2u_w = torch.ones(n_good_item * n_user)

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
# Subprocess functions
# ---------------------------------------------------------------------------


def _run_ppr_weighted_correctness_homogeneous(
    _: int,
    dataset: DistDataset,
    n_hub: int,
) -> None:
    """Subprocess: verifies weight=0 edges never contribute PPR residual (homogeneous).

    Seeds are hub nodes only.  Node features encode type: hub=2.0, good=1.0, bad=0.0.
    Any batch containing a bad node (feature==0.0) means a weight=0 edge contributed
    PPR residual — a test failure.
    """
    create_test_process_group()
    loader = DistNeighborLoader(
        dataset=dataset,
        input_nodes=torch.arange(n_hub),
        num_neighbors=[],
        sampler_options=PPRSamplerOptions(
            alpha=_PPR_ALPHA,
            eps=_PPR_EPS,
            max_ppr_nodes=_PPR_MAX_NODES,
            num_neighbors_per_hop=_PPR_NUM_NBRS,
        ),
        with_weight=True,
        pin_memory_device=torch.device("cpu"),
        batch_size=1,
    )
    count = 0
    for datum in loader:
        assert isinstance(datum, Data), f"Expected Data, got {type(datum)}"
        assert datum.x is not None, "Node features missing from PPR batch"
        bad_mask = datum.x[:, 0] == 0.0
        assert not bad_mask.any(), (
            f"weight=0 edge contributed PPR residual: bad node(s) found. "
            f"Features of bad nodes: {datum.x[bad_mask].squeeze().tolist()}"
        )
        count += 1
    assert count == n_hub, f"Expected {n_hub} batches, got {count}"
    shutdown_rpc()


def _run_ppr_weighted_correctness_heterogeneous(
    _: int,
    dataset: DistDataset,
    n_user: int,
) -> None:
    """Subprocess: verifies weight=0 edges never contribute PPR residual (heterogeneous).

    Seeds are user nodes.  Item features encode type: good=1.0, bad=0.0.
    Any batch containing a bad item node means a weight=0 edge contributed PPR residual.
    """
    create_test_process_group()
    node_ids = dataset.node_ids
    assert not isinstance(node_ids, torch.Tensor) and node_ids is not None, (
        "Expected heterogeneous dataset with dict node_ids"
    )
    loader = DistNeighborLoader(
        dataset=dataset,
        input_nodes=(_USER, node_ids[_USER]),
        num_neighbors=[],
        sampler_options=PPRSamplerOptions(
            alpha=_PPR_ALPHA,
            eps=_PPR_EPS,
            max_ppr_nodes=_PPR_MAX_NODES,
            num_neighbors_per_hop=_PPR_NUM_NBRS,
        ),
        with_weight=True,
        pin_memory_device=torch.device("cpu"),
        batch_size=1,
    )
    count = 0
    for datum in loader:
        assert isinstance(datum, HeteroData), f"Expected HeteroData, got {type(datum)}"
        if _ITEM in datum.node_types:
            item_x = datum[_ITEM].x
            assert item_x is not None, "Item features missing from PPR batch"
            bad_mask = item_x[:, 0] == 0.0
            assert not bad_mask.any(), (
                f"weight=0 edge contributed PPR residual: bad item(s) found. "
                f"Features of bad items: {item_x[bad_mask].squeeze().tolist()}"
            )
        count += 1
    assert count == n_user, f"Expected {n_user} batches, got {count}"
    shutdown_rpc()


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class PPRWeightedSamplingTest(TestCase):
    """End-to-end correctness tests for PPR sampling with weight_proportional_residuals.

    Each test builds a bipartite graph with "good" neighbors (weight=1) and "bad"
    neighbors (weight=0) reachable from seed nodes.  With weight-proportional PPR,
    bad nodes must never appear in any sampled subgraph because weight=0 edges
    contribute zero residual per push step.
    """

    def tearDown(self) -> None:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        super().tearDown()

    def test_ppr_weighted_never_traverses_zero_weight_edges_homogeneous(self) -> None:
        """Homogeneous: weight=0 edges to bad nodes never contribute PPR residual.

        Graph: 10 hub seeds, each connected to 50 good nodes (weight=1) and 40 bad
        nodes (weight=0).  Good nodes have 5 further weight=1 edges for deeper walks.
        PPR max_ppr_nodes=60 is larger than the number of good neighbors, so the
        sampler must actively filter: correct weighting excludes bad nodes entirely.
        """
        partition_output, n_hub, _, _ = _build_homogeneous_bipartite_weight_graph()
        dataset = DistDataset(rank=0, world_size=1, edge_dir="out")
        dataset.build(partition_output=partition_output)

        self.assertTrue(dataset.has_edge_weights)

        mp.spawn(
            fn=_run_ppr_weighted_correctness_homogeneous,
            args=(dataset, n_hub),
            nprocs=1,
        )

    def test_ppr_weighted_never_traverses_zero_weight_edges_heterogeneous(self) -> None:
        """Heterogeneous: weight=0 user→item edges to bad items never contribute PPR residual.

        Graph: 10 user seeds, each connected to 40 good items (weight=1) and 20 bad
        items (weight=0).  Good items connect back to all users via weight=1 (2nd-hop).
        PPR max_ppr_nodes=60 is larger than n_good, so correct weighting is required
        to exclude bad items.
        """
        partition_output, n_user, _, _ = _build_heterogeneous_bipartite_weight_graph()
        dataset = DistDataset(rank=0, world_size=1, edge_dir="out")
        dataset.build(partition_output=partition_output)

        self.assertTrue(dataset.has_edge_weights)

        mp.spawn(
            fn=_run_ppr_weighted_correctness_heterogeneous,
            args=(dataset, n_user),
            nprocs=1,
        )


if __name__ == "__main__":
    absltest.main()
