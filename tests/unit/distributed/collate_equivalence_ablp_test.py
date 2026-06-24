"""End-to-end equivalence test for DistABLPLoader across collate implementations.

Verifies that all collate implementations in ``COLLATE_IMPLS`` produce identical
batches for a homogeneous labeled dataset constructed from a synthetic graph.

Covers:
- Positive-only labels
- Positive + negative labels
- Positive + negative labels with per-anchor label cap
- Guaranteed-empty anchor (positive label outside 2-hop subgraph)
- Mutation guard: asserts the harness detects deliberate batch divergence
"""

import os

import torch
import torch.multiprocessing as mp
from absl.testing import absltest
from graphlearn_torch.distributed import shutdown_rpc
from parameterized import param, parameterized

from gigl.distributed.dist_ablp_neighborloader import DistABLPLoader
from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.utils.neighborloader import COLLATE_IMPL_ENV_VAR
from gigl.types.graph import (
    DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    GraphPartitionData,
    PartitionOutput,
    message_passing_to_negative_label,
    message_passing_to_positive_label,
    to_heterogeneous_node,
)
from tests.test_assets.distributed.collate_equivalence import (
    COLLATE_IMPLS,
    assert_impls_equivalent,
)
from tests.test_assets.distributed.utils import create_test_process_group
from tests.test_assets.test_case import TestCase

_POSITIVE_EDGE_TYPE = message_passing_to_positive_label(DEFAULT_HOMOGENEOUS_EDGE_TYPE)
_NEGATIVE_EDGE_TYPE = message_passing_to_negative_label(DEFAULT_HOMOGENEOUS_EDGE_TYPE)


def _build_homogeneous_ablp_dataset(
    labeled_edges: dict,
    max_labels_per_anchor_node,
) -> DistDataset:
    """Build a tiny homogeneous ABLP DistDataset for the equivalence cases.

    The message-passing edges are fixed; ``labeled_edges`` adds the positive/negative
    supervision edge types. Mirrors dist_ablp_neighborloader_test.py:511-545. Built in
    the parent (picklable) and passed to the child via ``mp.spawn`` args.

    Args:
        labeled_edges: Mapping of label edge type -> ``[2, E]`` edge tensor.
        max_labels_per_anchor_node: Optional per-anchor label cap (None = no cap).

    Returns:
        A built ``DistDataset`` with ``edge_dir="out"``.
    """
    edge_index = {
        DEFAULT_HOMOGENEOUS_EDGE_TYPE: torch.tensor(
            [[10, 10, 11, 11, 15, 15, 16, 16], [11, 12, 13, 17, 13, 14, 12, 14]]
        ),
    }
    edge_index.update(labeled_edges)
    partition_output = PartitionOutput(
        node_partition_book=to_heterogeneous_node(torch.zeros(18)),
        edge_partition_book={
            e_type: torch.zeros(int(e_idx.max().item() + 1))
            for e_type, e_idx in edge_index.items()
        },
        partitioned_edge_index={
            etype: GraphPartitionData(
                edge_index=idx, edge_ids=torch.arange(idx.size(1))
            )
            for etype, idx in edge_index.items()
        },
        partitioned_edge_features=None,
        partitioned_node_features=None,
        partitioned_negative_labels=None,
        partitioned_positive_labels=None,
        partitioned_node_labels=None,
    )
    dataset = DistDataset(
        rank=0,
        world_size=1,
        edge_dir="out",
        max_labels_per_anchor_node=max_labels_per_anchor_node,
    )
    dataset.build(partition_output=partition_output)
    return dataset


def _run_ablp_homogeneous_equivalence(
    _,
    dataset: DistDataset,
    input_nodes: torch.Tensor,
    batch_size: int,
) -> None:
    create_test_process_group()

    def make_loader():
        return DistABLPLoader(
            dataset=dataset,
            num_neighbors=[2, 2],
            input_nodes=input_nodes,
            batch_size=batch_size,
            pin_memory_device=torch.device("cpu"),
        )

    assert_impls_equivalent(make_loader, impls=COLLATE_IMPLS)
    shutdown_rpc()


class _PerturbingLoader:
    """Wraps a real loader and corrupts every emitted batch's ``node`` map.

    Used only by the mutation-guard test to simulate a divergent collate impl
    without depending on B's or C's internals. Mutating ``node`` (and thus the
    global ids the labels/coo resolve through) is exactly the class of bug the
    equivalence helper must catch.
    """

    def __init__(self, inner) -> None:
        self._inner = inner

    def __iter__(self):
        for batch in self._inner:
            batch.node = batch.node + 1  # global-id corruption
            yield batch


def _run_ablp_mutation_guard(
    _,
    dataset: DistDataset,
    input_nodes: torch.Tensor,
    batch_size: int,
) -> None:
    create_test_process_group()

    def make_loader():
        loader = DistABLPLoader(
            dataset=dataset,
            num_neighbors=[2, 2],
            input_nodes=input_nodes,
            batch_size=batch_size,
            pin_memory_device=torch.device("cpu"),
        )
        # Baseline ("python") is untouched; every non-baseline impl is perturbed,
        # so the captured streams MUST differ and assert_impls_equivalent MUST raise.
        if os.environ.get(COLLATE_IMPL_ENV_VAR) == "python":
            return loader
        return _PerturbingLoader(loader)

    raised = False
    try:
        assert_impls_equivalent(make_loader, impls=COLLATE_IMPLS)
    except AssertionError:
        raised = True
    assert raised, (
        "Mutation guard FAILED: assert_impls_equivalent did not detect a "
        "deliberately corrupted non-baseline batch stream. The harness is "
        "vacuous — it would pass even when impls diverge."
    )
    shutdown_rpc()


class CollateEquivalenceABLPTest(TestCase):
    def tearDown(self):
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        super().tearDown()

    @parameterized.expand(
        [
            param(
                "positive_and_negative",
                labeled_edges={
                    _POSITIVE_EDGE_TYPE: torch.tensor([[10, 15], [15, 16]]),
                    _NEGATIVE_EDGE_TYPE: torch.tensor(
                        [[10, 10, 11, 15], [13, 16, 14, 17]]
                    ),
                },
                max_labels_per_anchor_node=None,
            ),
            param(
                "positive_only",
                labeled_edges={
                    _POSITIVE_EDGE_TYPE: torch.tensor([[10, 15], [15, 16]]),
                },
                max_labels_per_anchor_node=None,
            ),
            param(
                "positive_and_negative_label_cap",
                labeled_edges={
                    _POSITIVE_EDGE_TYPE: torch.tensor([[10, 15], [15, 16]]),
                    _NEGATIVE_EDGE_TYPE: torch.tensor(
                        [[10, 10, 11, 15], [13, 16, 14, 17]]
                    ),
                },
                max_labels_per_anchor_node=1,
            ),
            param(
                # GUARANTEED-EMPTY-ANCHOR (the #1 ragged trap). Anchor 16's positive
                # label is node 13, but from 16 the 2-hop subgraph is {16,12,14}
                # (16->12, 16->14; 12 and 14 have no out-edges in the graph below),
                # so node 13 is NEVER sampled -> y_positive[local(16)] is an EMPTY
                # long tensor. An impl that drops empty-anchor keys diverges here.
                # Anchor 10 keeps a non-empty label, so the batch mixes empty + full.
                "positive_with_guaranteed_empty_anchor",
                labeled_edges={
                    _POSITIVE_EDGE_TYPE: torch.tensor([[10, 16], [15, 13]]),
                },
                max_labels_per_anchor_node=None,
                input_nodes=torch.tensor([10, 16]),
            ),
        ]
    )
    def test_ablp_homogeneous_equivalence(
        self, _, labeled_edges, max_labels_per_anchor_node, input_nodes=None
    ) -> None:
        dataset = _build_homogeneous_ablp_dataset(
            labeled_edges, max_labels_per_anchor_node
        )
        if input_nodes is None:
            input_nodes = torch.tensor([10, 15])
        mp.spawn(
            fn=_run_ablp_homogeneous_equivalence,
            args=(dataset, input_nodes, 2),
        )

    def test_ablp_equivalence_harness_detects_divergence(self) -> None:
        dataset = _build_homogeneous_ablp_dataset(
            labeled_edges={
                _POSITIVE_EDGE_TYPE: torch.tensor([[10, 15], [15, 16]]),
            },
            max_labels_per_anchor_node=None,
        )
        mp.spawn(
            fn=_run_ablp_mutation_guard,
            args=(dataset, torch.tensor([10, 15]), 2),
        )


if __name__ == "__main__":
    absltest.main()
