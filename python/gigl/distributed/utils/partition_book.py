from typing import Union

import torch
from graphlearn_torch.partition import PartitionBook, RangePartitionBook


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
        return torch.nonzero(partition_book == rank).squeeze(dim=1)
    else:
        return _get_ids_from_range_partition_book(
            range_partition_book=partition_book, rank=rank
        )
