import torch
from absl.testing import absltest
from graphlearn_torch.partition import RangePartitionBook
from parameterized import param, parameterized

from gigl.distributed.utils.partition_book import (
    _check_partition_book,
    get_ids_on_rank,
    get_total_ids,
)
from tests.test_assets.test_case import TestCase


class PartitionBookTest(TestCase):
    @parameterized.expand(
        [
            param(
                "Test getting ids for tensor-based partition book",
                partition_book=torch.tensor([0, 1, 1, 0, 3, 3, 2, 0, 1, 1]),
                rank_to_expected_ids={
                    0: torch.tensor([0, 3, 7]).to(torch.int64),
                    1: torch.tensor([1, 2, 8, 9]).to(torch.int64),
                    2: torch.tensor([6]).to(torch.int64),
                    3: torch.tensor([4, 5]).to(torch.int64),
                },
            ),
            param(
                "Test getting ids for range-based partition book",
                partition_book=RangePartitionBook(
                    partition_ranges=[(0, 4), (4, 5), (5, 10), (10, 13)],
                    partition_idx=0,
                ),
                rank_to_expected_ids={
                    0: torch.tensor([0, 1, 2, 3]).to(torch.int64),
                    1: torch.tensor([4]).to(torch.int64),
                    2: torch.tensor([5, 6, 7, 8, 9]).to(torch.int64),
                    3: torch.tensor([10, 11, 12]).to(torch.int64),
                },
            ),
        ]
    )
    def test_getting_ids_on_rank(
        self,
        _,
        partition_book: torch.Tensor,
        rank_to_expected_ids: dict[int, torch.Tensor],
    ):
        for rank, expected_ids in rank_to_expected_ids.items():
            with self.subTest(rank=rank):
                output_ids = get_ids_on_rank(partition_book=partition_book, rank=rank)
                self.assert_tensor_equality(output_ids, expected_ids)

    @parameterized.expand(
        [
            param(
                "Test getting ids for tensor-based partition book",
                partition_book=torch.tensor([0, 1, 1, 0, 3, 3, 2, 0, 1, 1]),
                expected_num_nodes=10,
            ),
            param(
                "Test getting ids for range-based partition book",
                partition_book=RangePartitionBook(
                    partition_ranges=[(0, 4), (4, 5), (5, 10), (10, 13)],
                    partition_idx=0,
                ),
                expected_num_nodes=13,
            ),
        ]
    )
    def test_get_total_ids(
        self,
        _,
        partition_book: torch.Tensor,
        expected_num_nodes: int,
    ):
        self.assertEqual(get_total_ids(partition_book), expected_num_nodes)

    def test_check_partition_book(self):
        valid_partition_book = torch.tensor([0, 1, 1, 0, 3, 3, 2, 0, 1, 1])
        _check_partition_book(valid_partition_book)

    @parameterized.expand(
        [
            param(
                "Test invalid partition book with 2D tensor",
                partition_book=torch.tensor([[0, 1], [1, 0]]),
            ),
            param(
                "Test invalid partition book with unary tensor",
                partition_book=torch.tensor(1),
            ),
        ]
    )
    def test_check_partition_book_invalid(self, _, partition_book: torch.Tensor):
        with self.assertRaises(ValueError):
            _check_partition_book(partition_book)


if __name__ == "__main__":
    absltest.main()
