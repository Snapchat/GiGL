"""Unit tests for gigl.analytics.inspect."""

import torch
import torch.multiprocessing as mp
from absl.testing import absltest
from graphlearn_torch.distributed import shutdown_rpc
from torch_geometric.data import HeteroData

from gigl.analytics.inspect import HeteroDataSummary, summary
from gigl.distributed.dist_ablp_neighborloader import DistABLPLoader
from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.distributed_neighborloader import DistNeighborLoader
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.types.graph import (
    FeaturePartitionData,
    GraphPartitionData,
    PartitionOutput,
    message_passing_to_negative_label,
    message_passing_to_positive_label,
)
from tests.test_assets.distributed.utils import create_test_process_group
from tests.test_assets.test_case import TestCase

_USER = NodeType("user")
_STORY = NodeType("story")
_USER_TO_STORY = EdgeType(_USER, Relation("to"), _STORY)
_STORY_TO_USER = EdgeType(_STORY, Relation("to"), _USER)


def _build_synthetic_hetero_dataset() -> DistDataset:
    """Build a tiny in-memory heterogeneous DistDataset for inspector tests.

    Graph: 5 users, 5 stories.

        user → story            story → user
        (1 per user)            (uneven fanout)
        ─────────────           ──────────────
          u0 ──→ s0               s0 ──┬──→ u0
          u1 ──→ s1                    └──→ u1
          u2 ──→ s2               s1 ──────→ u2
          u3 ──→ s3               s2 ──┬──→ u3
          u4 ──→ s4                    └──→ u4
                                  s3   (no outgoing)
                                  s4   (no outgoing)
    """
    partition_output = PartitionOutput(
        node_partition_book={
            _USER: torch.zeros(5),
            _STORY: torch.zeros(5),
        },
        edge_partition_book={
            _USER_TO_STORY: torch.zeros(5),
            _STORY_TO_USER: torch.zeros(5),
        },
        partitioned_edge_index={
            _USER_TO_STORY: GraphPartitionData(
                edge_index=torch.tensor(
                    [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=torch.long
                ),
                edge_ids=None,
            ),
            _STORY_TO_USER: GraphPartitionData(
                edge_index=torch.tensor(
                    [[0, 0, 1, 2, 2], [0, 1, 2, 3, 4]], dtype=torch.long
                ),
                edge_ids=None,
            ),
        },
        partitioned_node_features={
            _USER: FeaturePartitionData(feats=torch.zeros(5, 2), ids=torch.arange(5)),
            _STORY: FeaturePartitionData(feats=torch.zeros(5, 2), ids=torch.arange(5)),
        },
        partitioned_edge_features=None,
        partitioned_positive_labels=None,
        partitioned_negative_labels=None,
        partitioned_node_labels=None,
    )
    dataset = DistDataset(rank=0, world_size=1, edge_dir="out")
    dataset.build(partition_output=partition_output)
    return dataset


def _run_inspect_summary_hetero(_, dataset: DistDataset) -> None:
    """Subprocess body: build a loader, take one batch, validate summary().

    All 5 users are seeds. Tracing each seed's 2-hop walk through the
    synthetic graph (see ``_build_synthetic_hetero_dataset``):

        seed │ hop 1     │ hop 2          │ hop1 count │ hop2 count
        ─────┼───────────┼────────────────┼────────────┼───────────
         u0  │ → {s0}    │ → {u0, u1}     │     1      │     2
         u1  │ → {s1}    │ → {u2}         │     1      │     1
         u2  │ → {s2}    │ → {u3, u4}     │     1      │     2
         u3  │ → {s3}    │ → ∅            │     1      │     0
         u4  │ → {s4}    │ → ∅            │     1      │     0

        hop 1 counts = [1, 1, 1, 1, 1] → min=1 med=1 avg=1.0 max=1
        hop 2 counts = [2, 1, 2, 0, 0] → min=0 med=1 avg=1.0 max=2

    GiGL's patch_fanout_for_sampling rejects -1 despite the loader docstring
    claiming "all neighbors" is supported, so we pass a fanout that exceeds
    every per-node out-degree in the synthetic graph (saturates to all).
    """
    create_test_process_group()
    assert isinstance(dataset.node_ids, dict)
    loader = DistNeighborLoader(
        dataset=dataset,
        input_nodes=(_USER, dataset.node_ids[_USER]),  # ty: ignore[invalid-argument-type] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
        num_neighbors=[100, 100],
        batch_size=5,
        pin_memory_device=torch.device("cpu"),
    )

    batch = next(iter(loader))
    assert isinstance(batch, HeteroData)

    result = summary(batch)

    assert isinstance(result, HeteroDataSummary)
    assert result.seeds == 5
    assert len(result.per_hop) == 2

    hop1, hop2 = result.per_hop
    assert hop1 == HeteroDataSummary.HopStats(min=1, med=1, avg=1.0, max=1)
    assert hop2 == HeteroDataSummary.HopStats(min=0, med=1, avg=1.0, max=2)

    assert str(result) == (
        "seeds=5 hop1(min=1 med=1 avg=1.0 max=1) hop2(min=0 med=1 avg=1.0 max=2)"
    )

    shutdown_rpc()


def _build_synthetic_ablp_dataset() -> DistDataset:
    """Build a tiny in-memory ABLP DistDataset with positive/negative labels.

    Anchors: u0, u1 (2 users). Supervision edge type: user → story.

        message-passing edges          label edges (stripped from batch.edge_types,
        ─────────────────────          surfaced as batch.y_positive / y_negative)
        user → story                   ─────────────────────────────────────────
          u0 ──→ s0                    positive labels (user → story):
          u1 ──→ s1                      u0 ──→ s2
        story → user                     u1 ──→ s3
          s0 ──→ u0                    negative labels (user → story):
          s1 ──┬──→ u0                   u0 ──→ s4
               └──→ u1                   u1 ──→ s5
    """
    positive_et = message_passing_to_positive_label(_USER_TO_STORY)
    negative_et = message_passing_to_negative_label(_USER_TO_STORY)

    edge_index = {
        _USER_TO_STORY: torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
        _STORY_TO_USER: torch.tensor([[0, 1, 1], [0, 0, 1]], dtype=torch.long),
        positive_et: torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        negative_et: torch.tensor([[0, 1], [4, 5]], dtype=torch.long),
    }

    partition_output = PartitionOutput(
        node_partition_book={
            _USER: torch.zeros(2),
            _STORY: torch.zeros(6),
        },
        edge_partition_book={
            et: torch.zeros(int(ei.max().item()) + 1) for et, ei in edge_index.items()
        },
        partitioned_edge_index={
            et: GraphPartitionData(edge_index=ei, edge_ids=torch.arange(ei.size(1)))
            for et, ei in edge_index.items()
        },
        partitioned_node_features={
            _USER: FeaturePartitionData(feats=torch.zeros(2, 2), ids=torch.arange(2)),
            _STORY: FeaturePartitionData(feats=torch.zeros(6, 2), ids=torch.arange(6)),
        },
        partitioned_edge_features=None,
        partitioned_positive_labels=None,
        partitioned_negative_labels=None,
        partitioned_node_labels=None,
    )
    dataset = DistDataset(rank=0, world_size=1, edge_dir="out")
    dataset.build(partition_output=partition_output)
    return dataset


def _run_inspect_summary_ablp(_, dataset: DistDataset) -> None:
    """Subprocess body: build an ABLP loader, take one batch, validate summary().

    Anchors: u0, u1 — both seeded.

      seed │ hop 1     │ hop 2          │ hop1 count │ hop2 count
      ─────┼───────────┼────────────────┼────────────┼───────────
       u0  │ → {s0}    │ → {u0}         │     1      │     1
       u1  │ → {s1}    │ → {u0, u1}     │     1      │     2

      hop 1 counts = [1, 1] → min=1 med=1 avg=1.0 max=1
      hop 2 counts = [1, 2] → min=1 med=1 avg=1.5 max=2

    Label stories (s2, s3, s4, s5) are added to the sampling frontier
    internally — they have no outgoing message-passing edges in this graph,
    so they do not contribute to the per-anchor counts. The batch carries
    ``y_positive`` and ``y_negative`` dicts; label edge types are stripped
    from ``batch.edge_types`` by the loader.
    """
    create_test_process_group()
    assert isinstance(dataset.node_ids, dict)
    loader = DistABLPLoader(
        dataset=dataset,
        num_neighbors=[100, 100],
        input_nodes=(_USER, torch.tensor([0, 1])),
        supervision_edge_type=_USER_TO_STORY,
        batch_size=2,
        pin_memory_device=torch.device("cpu"),
    )

    batch = next(iter(loader))
    assert isinstance(batch, HeteroData)

    # Verify we are exercising the ABLP path: positive/negative label dicts
    # must be attached, and label edge types must NOT appear in edge_types
    # (they get stripped by the loader).
    assert hasattr(batch, "y_positive")
    assert hasattr(batch, "y_negative")
    assert isinstance(batch.y_positive, dict) and len(batch.y_positive) > 0
    assert isinstance(batch.y_negative, dict) and len(batch.y_negative) > 0
    positive_et = message_passing_to_positive_label(_USER_TO_STORY)
    negative_et = message_passing_to_negative_label(_USER_TO_STORY)
    assert positive_et not in batch.edge_types
    assert negative_et not in batch.edge_types

    result = summary(batch)

    assert isinstance(result, HeteroDataSummary)
    assert result.seeds == 2
    hop1, hop2 = result.per_hop
    assert hop1 == HeteroDataSummary.HopStats(min=1, med=1, avg=1.0, max=1)
    assert hop2 == HeteroDataSummary.HopStats(min=1, med=1, avg=1.5, max=2)
    assert str(result) == (
        "seeds=2 hop1(min=1 med=1 avg=1.0 max=1) hop2(min=1 med=1 avg=1.5 max=2)"
    )

    shutdown_rpc()


class SummaryIntegrationTest(TestCase):
    """End-to-end tests against real loader output (DistNeighborLoader + DistABLPLoader)."""

    def test_summary_with_dist_neighbor_loader(self):
        dataset = _build_synthetic_hetero_dataset()
        mp.spawn(fn=_run_inspect_summary_hetero, args=(dataset,))

    def test_summary_with_dist_ablp_loader(self):
        dataset = _build_synthetic_ablp_dataset()
        mp.spawn(fn=_run_inspect_summary_ablp, args=(dataset,))


class SummaryValidationTest(TestCase):
    """Hand-rolled HeteroData inputs that exercise both strict-contract
    guardrails and per-seed stat correctness.

    Hand-rolling lets us pre-compute the exact expected fanout from the
    edge_index without going through a sampler.
    """

    def test_no_batch_size_raises(self):
        """No node type has batch_size > 0 → can't pick a seed type."""
        data = HeteroData(
            {
                _USER: {"x": torch.zeros((3, 1))},
                _STORY: {"x": torch.zeros((2, 1))},
            }
        )
        with self.assertRaises(ValueError):
            summary(data)

    def test_multiple_batch_sizes_raises(self):
        """Two node types both have batch_size > 0 → ambiguous seed type."""
        data = HeteroData(
            {
                _USER: {"x": torch.zeros((3, 1)), "batch_size": 2},
                _STORY: {"x": torch.zeros((4, 1)), "batch_size": 3},
            }
        )
        with self.assertRaises(ValueError):
            summary(data)

    def test_missing_num_sampled_nodes_raises(self):
        """data.num_sampled_nodes absent → can't infer hop count."""
        data = HeteroData(
            {
                _USER: {"x": torch.zeros((3, 1)), "batch_size": 3},
                _STORY: {"x": torch.zeros((2, 1))},
                _USER_TO_STORY: {
                    "edge_index": torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
                },
            }
        )
        data.num_sampled_edges = {_USER_TO_STORY: torch.tensor([2])}
        with self.assertRaises(ValueError):
            summary(data)

    def test_missing_num_sampled_edges_raises(self):
        """data.num_sampled_edges absent → can't slice edges per hop."""
        data = HeteroData(
            {
                _USER: {"x": torch.zeros((3, 1)), "batch_size": 3},
                _STORY: {"x": torch.zeros((2, 1))},
                _USER_TO_STORY: {
                    "edge_index": torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
                },
            }
        )
        data.num_sampled_nodes = {_USER: torch.tensor([3, 2])}
        with self.assertRaises(ValueError):
            summary(data)

    def test_two_hop_uneven_fanout(self):
        """Seeds: u0, u1, u2.

        user → story          story → user
          u0 ──→ s0             s0 ──┬──→ u3
          u1 ──┬──→ s0               └──→ u4
               └──→ s1          s1 ──────→ u0
          u2 ──→ s2             s2 ──────→ u3

        seed │ hop 1        │ hop 2              │ counts
        ─────┼──────────────┼────────────────────┼─────────
         u0  │ → {s0}       │ → {u3, u4}         │  1, 2
         u1  │ → {s0, s1}   │ → {u3, u4, u0}     │  2, 3
         u2  │ → {s2}       │ → {u3}             │  1, 1

        hop1 counts = [1, 2, 1] → min=1 med=1 avg=1.3 max=2
        hop2 counts = [2, 3, 1] → min=1 med=2 avg=2.0 max=3
        """
        data = HeteroData(
            {
                _USER: {"x": torch.zeros((5, 1)), "batch_size": 3},
                _STORY: {"x": torch.zeros((3, 1))},
                _USER_TO_STORY: {
                    "edge_index": torch.tensor(
                        [[0, 1, 1, 2], [0, 0, 1, 2]], dtype=torch.long
                    ),
                },
                _STORY_TO_USER: {
                    "edge_index": torch.tensor(
                        [[0, 0, 1, 2], [3, 4, 0, 3]], dtype=torch.long
                    ),
                },
            }
        )
        data.num_sampled_nodes = {
            _USER: torch.tensor([3, 0, 2]),
            _STORY: torch.tensor([0, 3, 0]),
        }
        data.num_sampled_edges = {
            _USER_TO_STORY: torch.tensor([4, 0]),
            _STORY_TO_USER: torch.tensor([0, 4]),
        }

        result = summary(data)

        self.assertEqual(result.seeds, 3)
        self.assertEqual(len(result.per_hop), 2)

        hop1, hop2 = result.per_hop
        self.assertEqual(hop1.min, 1)
        self.assertEqual(hop1.med, 1)
        self.assertEqual(hop1.max, 2)
        self.assertAlmostEqual(hop1.avg, 4 / 3, places=5)

        self.assertEqual(hop2.min, 1)
        self.assertEqual(hop2.med, 2)
        self.assertEqual(hop2.max, 3)
        self.assertAlmostEqual(hop2.avg, 2.0, places=5)

        self.assertEqual(
            str(result),
            "seeds=3 hop1(min=1 med=1 avg=1.3 max=2) hop2(min=1 med=2 avg=2.0 max=3)",
        )

    def test_zero_edges_all_hops(self):
        """Seeds: u0, u1, u2 — no edges anywhere.

        user → story          story → user
          (empty)               (empty)

        seed │ hop 1 │ hop 2 │ counts
        ─────┼───────┼───────┼─────────
         u0  │ → ∅   │ → ∅   │  0, 0
         u1  │ → ∅   │ → ∅   │  0, 0
         u2  │ → ∅   │ → ∅   │  0, 0

        hop1 counts = [0, 0, 0] → min=0 med=0 avg=0.0 max=0
        hop2 counts = [0, 0, 0] → min=0 med=0 avg=0.0 max=0
        """
        data = HeteroData(
            {
                _USER: {"x": torch.zeros((3, 1)), "batch_size": 3},
                _STORY: {"x": torch.zeros((1, 1))},
                _USER_TO_STORY: {
                    "edge_index": torch.empty((2, 0), dtype=torch.long),
                },
                _STORY_TO_USER: {
                    "edge_index": torch.empty((2, 0), dtype=torch.long),
                },
            }
        )
        data.num_sampled_nodes = {
            _USER: torch.tensor([3, 0, 0]),
            _STORY: torch.tensor([0, 0, 0]),
        }
        data.num_sampled_edges = {
            _USER_TO_STORY: torch.tensor([0, 0]),
            _STORY_TO_USER: torch.tensor([0, 0]),
        }

        result = summary(data)

        self.assertEqual(result.seeds, 3)
        self.assertEqual(len(result.per_hop), 2)
        for hop_stats in result.per_hop:
            self.assertEqual(hop_stats.min, 0)
            self.assertEqual(hop_stats.med, 0)
            self.assertEqual(hop_stats.max, 0)
            self.assertEqual(hop_stats.avg, 0.0)
        self.assertEqual(
            str(result),
            "seeds=3 hop1(min=0 med=0 avg=0.0 max=0) hop2(min=0 med=0 avg=0.0 max=0)",
        )
