import unittest
from typing import Dict

import torch
from graphlearn_torch.partition import RangePartitionBook
from parameterized import param, parameterized

from gigl.distributed.utils.partition_book import get_ids_on_rank


class PartitionBookTest(unittest.TestCase):
    @parameterized.expand(
        [
            param(
                "Test getting ids for tensor-based partition book",
                partition_book=torch.Tensor([0, 1, 1, 0, 3, 3, 2, 0, 1, 1]),
                rank_to_expected_ids={
                    0: torch.Tensor([0, 3, 7]).to(torch.int64),
                    1: torch.Tensor([1, 2, 8, 9]).to(torch.int64),
                    2: torch.Tensor([6]).to(torch.int64),
                    3: torch.Tensor([4, 5]).to(torch.int64),
                },
            ),
            param(
                "Test getting ids for range-based partition book",
                partition_book=RangePartitionBook(
                    partition_ranges=[(0, 4), (4, 5), (5, 10), (10, 13)],
                    partition_idx=0,
                ),
                rank_to_expected_ids={
                    0: torch.Tensor([0, 1, 2, 3]).to(torch.int64),
                    1: torch.Tensor([4]).to(torch.int64),
                    2: torch.Tensor([5, 6, 7, 8, 9]).to(torch.int64),
                    3: torch.Tensor([10, 11, 12]).to(torch.int64),
                },
            ),
        ]
    )
    def test_getting_ids_on_rank(
        self,
        _,
        partition_book: torch.Tensor,
        rank_to_expected_ids: Dict[int, torch.Tensor],
    ):
        for rank, expected_ids in rank_to_expected_ids.items():
            output_ids = get_ids_on_rank(partition_book=partition_book, rank=rank)
            torch.testing.assert_close(actual=output_ids, expected=expected_ids)


if __name__ == "__main__":
    unittest.main()
