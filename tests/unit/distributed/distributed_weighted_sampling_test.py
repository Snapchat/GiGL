"""Distributed integration tests for weighted edge sampling.

Covers two surfaces:
  1. DistPartitioner correctly partitions registered edge weights (weights land on
     the right rank and match the expected values).
  2. DistNeighborLoader runs end-to-end without errors when with_weight=True is set
     and the dataset carries edge weights — both homogeneous and heterogeneous.
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
)
from tests.test_assets.distributed.run_distributed_partitioner import (
    InputDataStrategy,
    run_distributed_partitioner,
)
from tests.test_assets.distributed.utils import create_test_process_group
from tests.test_assets.test_case import TestCase

_USER = NodeType("user")
_STORY = NodeType("story")
_USER_TO_STORY = EdgeType(_USER, Relation("to"), _STORY)
_STORY_TO_USER = EdgeType(_STORY, Relation("to"), _USER)


# ---------------------------------------------------------------------------
# Subprocess functions — must accept local_rank as first arg (mp.spawn)
# ---------------------------------------------------------------------------


def _run_distributed_weighted_neighbor_loader_homogeneous(
    _: int,
    dataset: DistDataset,
    expected_data_count: int,
) -> None:
    """Subprocess: iterates a weighted homogeneous loader and checks batch count and type."""
    create_test_process_group()
    loader = DistNeighborLoader(
        dataset=dataset,
        num_neighbors=[2, 2],
        with_weight=True,
        pin_memory_device=torch.device("cpu"),
    )
    count = 0
    for datum in loader:
        assert isinstance(datum, Data), (
            f"Subgraph should be Data for homogeneous datasets, got {type(datum)}"
        )
        count += 1
    assert count == expected_data_count, (
        f"Expected {expected_data_count} batches, got {count}"
    )
    shutdown_rpc()


def _run_distributed_weighted_neighbor_loader_heterogeneous(
    _: int,
    dataset: DistDataset,
    expected_data_count: int,
) -> None:
    """Subprocess: iterates a weighted heterogeneous loader and checks batch count and type."""
    create_test_process_group()
    assert isinstance(dataset.node_ids, Mapping)
    loader = DistNeighborLoader(
        dataset=dataset,
        input_nodes=(_USER, dataset.node_ids[_USER]),
        num_neighbors=[2, 2],
        with_weight=True,
        pin_memory_device=torch.device("cpu"),
    )
    count = 0
    for datum in loader:
        assert isinstance(datum, HeteroData), (
            f"Subgraph should be HeteroData for heterogeneous datasets, got {type(datum)}"
        )
        count += 1
    assert count == expected_data_count, (
        f"Expected {expected_data_count} batches, got {count}"
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


class DistributedWeightedSamplingTest(TestCase):
    """End-to-end dataloading tests for DistNeighborLoader with with_weight=True."""

    def setUp(self) -> None:
        super().setUp()
        self._world_size = 1

    def tearDown(self) -> None:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        super().tearDown()

    def test_distributed_neighbor_loader_with_weights_homogeneous(self) -> None:
        """Homogeneous loader with with_weight=True iterates all nodes without error."""
        n = 5
        partition_output = PartitionOutput(
            node_partition_book=torch.zeros(n),
            edge_partition_book=torch.zeros(n),
            partitioned_edge_index=GraphPartitionData(
                edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]),
                edge_ids=None,
                weights=torch.ones(n, dtype=torch.float32),
            ),
            partitioned_node_features=FeaturePartitionData(
                feats=torch.zeros(n, 2), ids=torch.arange(n)
            ),
            partitioned_edge_features=None,
            partitioned_positive_labels=None,
            partitioned_negative_labels=None,
            partitioned_node_labels=None,
        )
        dataset = DistDataset(rank=0, world_size=1, edge_dir="out")
        dataset.build(partition_output=partition_output)

        mp.spawn(
            fn=_run_distributed_weighted_neighbor_loader_homogeneous,
            args=(dataset, n),
        )

    def test_distributed_neighbor_loader_with_weights_heterogeneous(self) -> None:
        """Heterogeneous loader with with_weight=True iterates all seed nodes without error."""
        n = 5
        partition_output = PartitionOutput(
            node_partition_book={
                _USER: torch.zeros(n),
                _STORY: torch.zeros(n),
            },
            edge_partition_book={
                _USER_TO_STORY: torch.zeros(n),
                _STORY_TO_USER: torch.zeros(n),
            },
            partitioned_edge_index={
                _USER_TO_STORY: GraphPartitionData(
                    edge_index=torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]),
                    edge_ids=None,
                    weights=torch.ones(n, dtype=torch.float32),
                ),
                _STORY_TO_USER: GraphPartitionData(
                    edge_index=torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]),
                    edge_ids=None,
                    weights=torch.ones(n, dtype=torch.float32),
                ),
            },
            partitioned_node_features={
                _USER: FeaturePartitionData(
                    feats=torch.zeros(n, 2), ids=torch.arange(n)
                ),
                _STORY: FeaturePartitionData(
                    feats=torch.zeros(n, 2), ids=torch.arange(n)
                ),
            },
            partitioned_edge_features=None,
            partitioned_positive_labels=None,
            partitioned_negative_labels=None,
            partitioned_node_labels=None,
        )
        dataset = DistDataset(rank=0, world_size=1, edge_dir="out")
        dataset.build(partition_output=partition_output)

        mp.spawn(
            fn=_run_distributed_weighted_neighbor_loader_heterogeneous,
            args=(dataset, n),
        )


if __name__ == "__main__":
    absltest.main()
