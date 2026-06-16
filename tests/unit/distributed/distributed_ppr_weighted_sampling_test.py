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

from dataclasses import replace

import torch
import torch.multiprocessing as mp
from absl.testing import absltest
from graphlearn_torch.distributed import shutdown_rpc
from torch_geometric.data import Data, HeteroData

from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.distributed_neighborloader import DistNeighborLoader
from gigl.distributed.sampler_options import PPRSamplerOptions
from gigl.types.graph import GraphPartitionData
from tests.test_assets.distributed.bipartite_weight_graph import (
    ITEM,
    USER,
    build_heterogeneous_bipartite_weight_graph,
    build_homogeneous_bipartite_weight_graph,
)
from tests.test_assets.distributed.utils import create_test_process_group
from tests.test_assets.test_case import TestCase

# PPR parameters used across all tests.
_PPR_ALPHA = 0.5
_PPR_EPS = 1e-4
_PPR_MAX_NODES = 60
_PPR_NUM_NBRS = 200


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
        input_nodes=(USER, node_ids[USER]),
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
        if ITEM in datum.node_types:
            item_x = datum[ITEM].x
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
        partition_output, n_hub = build_homogeneous_bipartite_weight_graph()
        dataset = DistDataset(rank=0, world_size=1, edge_dir="out")
        dataset.build(partition_output=partition_output)

        self.assertTrue(dataset.has_edge_weights)

        mp.spawn(
            fn=_run_ppr_weighted_correctness_homogeneous,
            args=(dataset, n_hub),
            nprocs=1,
        )

    def test_ppr_weighted_handles_non_contiguous_edge_ids_homogeneous(self) -> None:
        """Homogeneous: sampled global edge ids resolve to local edge weights.

        Partitioned weighted graphs store local weights compactly, while GLT
        sampling returns the graph edge ids.  This test gives every local edge a
        large non-contiguous edge id; indexing the local weight tensor directly
        with sampled edge ids would go out of bounds.
        """
        partition_output, n_hub = build_homogeneous_bipartite_weight_graph()
        partitioned_edge_index = partition_output.partitioned_edge_index
        self.assertIsInstance(partitioned_edge_index, GraphPartitionData)
        assert isinstance(partitioned_edge_index, GraphPartitionData)
        num_edges = partitioned_edge_index.edge_index.size(1)
        partition_output = replace(
            partition_output,
            partitioned_edge_index=replace(
                partitioned_edge_index,
                edge_ids=torch.arange(
                    10_000,
                    10_000 + num_edges,
                    dtype=torch.long,
                ),
            ),
        )

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
        partition_output, n_user = build_heterogeneous_bipartite_weight_graph()
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
