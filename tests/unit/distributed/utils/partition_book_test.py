import unittest

import torch
from graphlearn_torch.partition import RangePartitionBook
from parameterized import param, parameterized

from gigl.distributed.utils.partition_book import (
    build_partition_book,
    get_ids_on_rank,
    get_total_ids,
)
from tests.test_assets.distributed.utils import assert_tensor_equality


class TestGetIdsOnRank(unittest.TestCase):
    @parameterized.expand(
        [
            param(
                "tensor_partition_book",
                partition_book=torch.tensor([0, 1, 0, 1]),
                rank=0,
                expected=torch.tensor([0, 2]),
            ),
            param(
                "tensor_partition_book_rank_1",
                partition_book=torch.tensor([0, 1, 0, 1]),
                rank=1,
                expected=torch.tensor([1, 3]),
            ),
            param(
                "range_partition_book",
                partition_book=RangePartitionBook(
                    partition_ranges=[(0, 5), (5, 10)], partition_idx=0
                ),
                rank=0,
                expected=torch.arange(0, 5),
            ),
            param(
                "range_partition_book_rank_1",
                partition_book=RangePartitionBook(
                    partition_ranges=[(0, 5), (5, 10)], partition_idx=0
                ),
                rank=1,
                expected=torch.arange(5, 10),
            ),
        ]
    )
    def test_get_ids_on_rank(self, _, partition_book, rank, expected):
        assert_tensor_equality(get_ids_on_rank(partition_book, rank), expected)

    def test_invalid_tensor_partition_book(self):
        invalid_pb = torch.tensor([[0, 1], [0, 1]])  # 2D tensor
        with self.assertRaises(ValueError):
            get_ids_on_rank(invalid_pb, 0)


class TestGetTotalIds(unittest.TestCase):
    @parameterized.expand(
        [
            param(
                "tensor_partition_book",
                partition_book=torch.tensor([0, 1, 0, 1, 0]),
                expected=5,
            ),
            param(
                "range_partition_book",
                partition_book=RangePartitionBook(
                    partition_ranges=[(0, 5), (5, 10)], partition_idx=0
                ),
                expected=10,
            ),
        ]
    )
    def test_get_total_ids(self, _, partition_book, expected):
        self.assertEqual(get_total_ids(partition_book), expected)

    def test_invalid_tensor_partition_book(self):
        invalid_pb = torch.tensor([[0, 1], [0, 1]])  # 2D tensor
        with self.assertRaises(ValueError):
            get_total_ids(invalid_pb)


class TestBuildPartitionBook(unittest.TestCase):
    @parameterized.expand(
        [
            param(
                "divides_evenly",
                num_entities=10,
                world_size=2,
                expected_bounds=torch.tensor([5, 10]),
            ),
            param(
                "divides_unevenly",
                num_entities=7,
                world_size=3,
                expected_bounds=torch.tensor([3, 5, 7]),
            ),
        ]
    )
    def test_build_partition_book(self, _, num_entities, world_size, expected_bounds):
        pb = build_partition_book(
            num_entities=num_entities, rank=0, world_size=world_size
        )
        assert_tensor_equality(pb.partition_bounds, expected_bounds)


if __name__ == "__main__":
    unittest.main()
