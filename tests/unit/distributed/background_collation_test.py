import torch
import torch.multiprocessing as mp
from absl.testing import absltest
from graphlearn_torch.distributed import shutdown_rpc
from torch_geometric.data import Data

from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.distributed_neighborloader import DistNeighborLoader
from gigl.types.graph import FeaturePartitionData, GraphPartitionData, PartitionOutput
from gigl.utils.iterator import InfiniteIterator
from tests.test_assets.distributed.utils import create_test_process_group
from tests.test_assets.test_case import TestCase

# GLT requires subclasses of DistNeighborLoader to be run in a separate process.
# Otherwise, we may run into segmentation fault or other memory issues.
# Calling these functions in separate processes also allows us to use shutdown_rpc()
# to ensure cleanup of ports, providing stronger guarantees of isolation between tests.


# We require each of these functions to accept local_rank as the first argument
# since we use mp.spawn with `nprocs=1`.


def _run_background_collation_batch_count(
    _: int,
    dataset: DistDataset,
    expected_data_count: int,
    queue_size: int,
) -> None:
    """Verifies that a loader with background collation produces the correct batch count."""
    create_test_process_group()
    loader = DistNeighborLoader(
        dataset=dataset,
        num_neighbors=[2, 2],
        pin_memory_device=torch.device("cpu"),
        background_collation_queue_size=queue_size,
    )

    count = 0
    for datum in loader:
        assert isinstance(datum, Data)
        count += 1

    assert (
        count == expected_data_count
    ), f"Expected {expected_data_count} batches, but got {count}."

    shutdown_rpc()


def _run_background_collation_multiple_epochs(
    _: int,
    dataset: DistDataset,
    expected_data_count: int,
) -> None:
    """Verifies thread restart across epoch boundaries via __iter__."""
    create_test_process_group()
    loader = DistNeighborLoader(
        dataset=dataset,
        num_neighbors=[2, 2],
        pin_memory_device=torch.device("cpu"),
        background_collation_queue_size=2,
    )

    for epoch in range(3):
        count = 0
        for datum in loader:
            assert isinstance(datum, Data)
            count += 1
        assert (
            count == expected_data_count
        ), f"Epoch {epoch}: expected {expected_data_count} batches, got {count}."

    shutdown_rpc()


def _run_background_collation_early_break(
    _: int,
    dataset: DistDataset,
    expected_data_count: int,
) -> None:
    """Verifies that breaking mid-epoch and re-iterating works correctly."""
    create_test_process_group()
    loader = DistNeighborLoader(
        dataset=dataset,
        num_neighbors=[2, 2],
        pin_memory_device=torch.device("cpu"),
        background_collation_queue_size=2,
    )

    # Partial iteration — break after 5 batches
    count = 0
    for datum in loader:
        assert isinstance(datum, Data)
        count += 1
        if count >= 5:
            break

    assert count == 5, f"Expected 5 batches in partial epoch, got {count}."

    # Full second epoch should still produce the correct count
    count = 0
    for datum in loader:
        assert isinstance(datum, Data)
        count += 1
    assert (
        count == expected_data_count
    ), f"Expected {expected_data_count} batches in full epoch, got {count}."

    shutdown_rpc()


def _run_background_collation_shutdown(
    _: int,
    dataset: DistDataset,
) -> None:
    """Verifies that partial iteration then shutdown() terminates cleanly."""
    create_test_process_group()
    loader = DistNeighborLoader(
        dataset=dataset,
        num_neighbors=[2, 2],
        pin_memory_device=torch.device("cpu"),
        background_collation_queue_size=2,
    )

    # Partial iteration
    count = 0
    for datum in loader:
        assert isinstance(datum, Data)
        count += 1
        if count >= 3:
            break

    loader.shutdown()
    shutdown_rpc()


def _run_background_collation_queue_size_one(
    _: int,
    dataset: DistDataset,
    expected_data_count: int,
) -> None:
    """Verifies that the minimum pipeline depth (queue_size=1) works correctly."""
    create_test_process_group()
    loader = DistNeighborLoader(
        dataset=dataset,
        num_neighbors=[2, 2],
        pin_memory_device=torch.device("cpu"),
        background_collation_queue_size=1,
    )

    count = 0
    for datum in loader:
        assert isinstance(datum, Data)
        count += 1

    assert (
        count == expected_data_count
    ), f"Expected {expected_data_count} batches, but got {count}."

    shutdown_rpc()


def _run_background_collation_infinite_iterator(
    _: int,
    dataset: DistDataset,
    max_num_batches: int,
) -> None:
    """Verifies that background collation works with InfiniteIterator."""
    create_test_process_group()
    loader = DistNeighborLoader(
        dataset=dataset,
        num_neighbors=[2, 2],
        pin_memory_device=torch.device("cpu"),
        background_collation_queue_size=2,
    )

    infinite_loader: InfiniteIterator = InfiniteIterator(loader)

    count = 0
    for datum in infinite_loader:
        assert isinstance(datum, Data)
        count += 1
        if count == max_num_batches:
            break

    assert count == max_num_batches, f"Expected {max_num_batches} batches, got {count}."

    shutdown_rpc()


class TestBackgroundCollation(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._world_size = 1

    def tearDown(self) -> None:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        super().tearDown()

    def test_produces_same_batch_count(self) -> None:
        """Loader with background collation produces correct number of batches."""
        expected_data_count = 18
        partition_output = PartitionOutput(
            node_partition_book=torch.zeros(18),
            edge_partition_book=None,
            partitioned_edge_index=GraphPartitionData(
                edge_index=torch.tensor([[10], [15]]), edge_ids=None
            ),
            partitioned_edge_features=None,
            partitioned_node_features=None,
            partitioned_negative_labels=None,
            partitioned_positive_labels=None,
            partitioned_node_labels=None,
        )
        dataset = DistDataset(rank=0, world_size=1, edge_dir="out")
        dataset.build(partition_output=partition_output)

        mp.spawn(
            fn=_run_background_collation_batch_count,
            args=(dataset, expected_data_count, 4),
        )

    def test_multiple_epochs(self) -> None:
        """Thread restart across epoch boundaries via __iter__."""
        expected_data_count = 18
        partition_output = PartitionOutput(
            node_partition_book=torch.zeros(18),
            edge_partition_book=None,
            partitioned_edge_index=GraphPartitionData(
                edge_index=torch.tensor([[10], [15]]), edge_ids=None
            ),
            partitioned_edge_features=None,
            partitioned_node_features=None,
            partitioned_negative_labels=None,
            partitioned_positive_labels=None,
            partitioned_node_labels=None,
        )
        dataset = DistDataset(rank=0, world_size=1, edge_dir="out")
        dataset.build(partition_output=partition_output)

        mp.spawn(
            fn=_run_background_collation_multiple_epochs,
            args=(dataset, expected_data_count),
        )

    def test_early_break(self) -> None:
        """Break mid-epoch, then iterate again — no hang or corruption."""
        expected_data_count = 18
        partition_output = PartitionOutput(
            node_partition_book=torch.zeros(18),
            edge_partition_book=None,
            partitioned_edge_index=GraphPartitionData(
                edge_index=torch.tensor([[10], [15]]), edge_ids=None
            ),
            partitioned_edge_features=None,
            partitioned_node_features=None,
            partitioned_negative_labels=None,
            partitioned_positive_labels=None,
            partitioned_node_labels=None,
        )
        dataset = DistDataset(rank=0, world_size=1, edge_dir="out")
        dataset.build(partition_output=partition_output)

        mp.spawn(
            fn=_run_background_collation_early_break,
            args=(dataset, expected_data_count),
        )

    def test_shutdown(self) -> None:
        """Partial iteration then shutdown() — clean termination."""
        partition_output = PartitionOutput(
            node_partition_book=torch.zeros(18),
            edge_partition_book=None,
            partitioned_edge_index=GraphPartitionData(
                edge_index=torch.tensor([[10], [15]]), edge_ids=None
            ),
            partitioned_edge_features=None,
            partitioned_node_features=None,
            partitioned_negative_labels=None,
            partitioned_positive_labels=None,
            partitioned_node_labels=None,
        )
        dataset = DistDataset(rank=0, world_size=1, edge_dir="out")
        dataset.build(partition_output=partition_output)

        mp.spawn(
            fn=_run_background_collation_shutdown,
            args=(dataset,),
        )

    def test_invalid_queue_size(self) -> None:
        """background_collation_queue_size=0 raises ValueError."""
        partition_output = PartitionOutput(
            node_partition_book=torch.zeros(5),
            edge_partition_book=torch.zeros(5),
            partitioned_edge_index=GraphPartitionData(
                edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]),
                edge_ids=None,
            ),
            partitioned_node_features=FeaturePartitionData(
                feats=torch.zeros(10, 2), ids=torch.arange(10)
            ),
            partitioned_edge_features=None,
            partitioned_positive_labels=None,
            partitioned_negative_labels=None,
            partitioned_node_labels=None,
        )
        dataset = DistDataset(rank=0, world_size=1, edge_dir="in")
        dataset.build(partition_output=partition_output)

        create_test_process_group()
        with self.assertRaises(ValueError):
            DistNeighborLoader(
                dataset=dataset,
                num_neighbors=[2, 2],
                pin_memory_device=torch.device("cpu"),
                background_collation_queue_size=0,
            )

    def test_queue_size_one(self) -> None:
        """Minimum pipeline depth works correctly."""
        expected_data_count = 18
        partition_output = PartitionOutput(
            node_partition_book=torch.zeros(18),
            edge_partition_book=None,
            partitioned_edge_index=GraphPartitionData(
                edge_index=torch.tensor([[10], [15]]), edge_ids=None
            ),
            partitioned_edge_features=None,
            partitioned_node_features=None,
            partitioned_negative_labels=None,
            partitioned_positive_labels=None,
            partitioned_node_labels=None,
        )
        dataset = DistDataset(rank=0, world_size=1, edge_dir="out")
        dataset.build(partition_output=partition_output)

        mp.spawn(
            fn=_run_background_collation_queue_size_one,
            args=(dataset, expected_data_count),
        )

    def test_infinite_iterator_with_background_collation(self) -> None:
        """Background collation works correctly with InfiniteIterator."""
        partition_output = PartitionOutput(
            node_partition_book=torch.zeros(18),
            edge_partition_book=None,
            partitioned_edge_index=GraphPartitionData(
                edge_index=torch.tensor([[10], [15]]), edge_ids=None
            ),
            partitioned_edge_features=None,
            partitioned_node_features=None,
            partitioned_negative_labels=None,
            partitioned_positive_labels=None,
            partitioned_node_labels=None,
        )
        dataset = DistDataset(rank=0, world_size=1, edge_dir="out")
        dataset.build(partition_output=partition_output)

        # Iterate across more than one epoch
        max_num_batches = 18 * 2

        mp.spawn(
            fn=_run_background_collation_infinite_iterator,
            args=(dataset, max_num_batches),
        )


if __name__ == "__main__":
    absltest.main()
