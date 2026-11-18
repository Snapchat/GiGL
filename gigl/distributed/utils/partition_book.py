from typing import Union

import torch
from graphlearn_torch.partition import PartitionBook, RangePartitionBook
from graphlearn_torch.partition.partition_book import OffsetId2Index


class EmptyRangeAllowedPartitionBook(RangePartitionBook):
    """A RangePartitionBook that accepts empty partition ranges.

    GLT's RangePartitionBook requires start < end for every range, which fails when a rank owns no entities
    (e.g. fewer nodes than ranks, or a rank with no edges of an edge type).
    Lookups, `id2index`, and `id_filter` already behave correctly for empty ranges.
    Ranges must still satisfy start <= end and be contiguous.

    Args:
        partition_ranges (list[tuple[int, int]]): Start and end (exclusive) of each partition range.
        partition_idx (int): Index of the current partition.

    Raises:
        ValueError: If any range has start > end, or the ranges are not contiguous.
    """

    def __init__(
        self, partition_ranges: list[tuple[int, int]], partition_idx: int
    ) -> None:
        # Intentionally does not call RangePartitionBook.__init__, which rejects empty ranges.
        if not all(start <= end for start, end in partition_ranges):
            raise ValueError("All partition ranges must have start <= end")
        if not all(
            previous[1] == current[0]
            for previous, current in zip(partition_ranges[:-1], partition_ranges[1:])
        ):
            raise ValueError("Partition ranges must be continuous")

        self.partition_bounds = torch.tensor(
            [end for _, end in partition_ranges], dtype=torch.long
        )
        self.partition_idx = partition_idx
        self._id2index = OffsetId2Index(partition_ranges[partition_idx][0])


def _check_partition_book(partition_book: torch.Tensor) -> None:
    """
    Checks if the partition book is valid.
    Args:
        partition_book (torch.Tensor): Partition book to check.
    Raises:
        ValueError: If the partition book is not a 1D tensor.
    """
    if partition_book.dim() != 1:
        raise ValueError("Partition book must be a 1D tensor.")


def _get_ids_from_range_partition_book(
    range_partition_book: PartitionBook, rank: int
) -> torch.Tensor:
    """
    This function is very similar to RangePartitionBook.id_filter(). However, we re-implement this here, since the usage-pattern for that is a bit strange
    i.e. range_partition_book.id_filter(node_pb=range_partition_book, partition_idx=rank).
    """
    assert isinstance(range_partition_book, RangePartitionBook)
    start_node_id = range_partition_book.partition_bounds[rank - 1] if rank > 0 else 0
    end_node_id = range_partition_book.partition_bounds[rank]
    return torch.arange(start_node_id, end_node_id, dtype=torch.int64)


def get_ids_on_rank(
    partition_book: Union[torch.Tensor, PartitionBook],
    rank: int,
) -> torch.Tensor:
    """
    Provided a tensor-based partition book or a range-based bartition book and a rank, returns all the ids that are stored on that rank.
    Args:
        partition_book (Union[torch.Tensor, PartitionBook]): Tensor or range-based partition book
        rank (int): Rank of current machine
    """
    if isinstance(partition_book, torch.Tensor):
        _check_partition_book(partition_book)
        return torch.nonzero(partition_book == rank).squeeze(dim=1)
    else:
        return _get_ids_from_range_partition_book(
            range_partition_book=partition_book, rank=rank
        )


def get_total_ids(partition_book: Union[torch.Tensor, PartitionBook]) -> int:
    """
    Returns the total number of ids (e.g. the total number of nodes) from a partition book.
    Args:
        partition_book (Union[torch.Tensor, PartitionBook]): Tensor or range-based partition book
    Returns:
        int: Total number of ids in the partition book
    """
    if isinstance(partition_book, torch.Tensor):
        _check_partition_book(partition_book)
        return int(partition_book.numel())
    elif isinstance(partition_book, RangePartitionBook):
        return int(
            partition_book.partition_bounds[-1].item()
        )  # Last bound is the total number of ids
    else:
        raise TypeError(
            f"Unsupported partition book type: {type(partition_book)}. "
            "Expected torch.Tensor or RangePartitionBook."
        )


def build_partition_book(
    num_entities: int, rank: int, world_size: int
) -> RangePartitionBook:
    """
    Builds a range-based partition book for a given number of entities, rank, and world size.

    The partition book is balanced, i.e. the difference between the number of entities in any two partitions is at most 1.

    Examples:
        num_entities = 10, world_size = 2, rank = 0
        -> RangePartitionBook(partition_ranges=[5, 10], partition_idx=0)

        num_entities = 7, world_size = 3, rank = 0
        -> RangePartitionBook(partition_ranges=[2, 4, 7], partition_idx=0)
    Args:
        num_entities (int): Number of entities
        rank (int): Rank of current machine
        world_size (int): Total number of machines
    Returns:
        RangePartitionBook: Range-based partition book
    """
    per_entity_num, remainder = divmod(num_entities, world_size)

    # We set `remainder` number of partitions to have at most one more item.

    start = 0
    partition_ranges: list[tuple[int, int]] = []
    for partition_index in range(world_size):
        if partition_index < remainder:
            end = start + per_entity_num + 1
        else:
            end = start + per_entity_num
        partition_ranges.append((start, end))
        start = end

    # Empty ranges are possible when num_entities < world_size.
    partition_book = EmptyRangeAllowedPartitionBook(
        partition_ranges=partition_ranges, partition_idx=rank
    )
    return partition_book
