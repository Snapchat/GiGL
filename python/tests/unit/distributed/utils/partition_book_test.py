import unittest

import torch
from graphlearn_torch.partition import RangePartitionBook

from gigl.distributed.utils.partition_book import (
    build_balanced_range_parition_book,
    get_ids_on_rank,
    get_total_ids,
)
from tests.test_assets.distributed.utils import assert_tensor_equality


class TestGetIdsOnRank(unittest.TestCase):
    def test_tensor_partition_book(self):
        # Nodes 0,2 on rank 0; nodes 1,3 on rank 1
        partition_book = torch.tensor([0, 1, 0, 1])
        assert_tensor_equality(get_ids_on_rank(partition_book, 0), torch.tensor([0, 2]))
        assert_tensor_equality(get_ids_on_rank(partition_book, 1), torch.tensor([1, 3]))

    def test_range_partition_book(self):
        # Nodes 0-4 on rank 0; nodes 5-9 on rank 1
        range_pb = RangePartitionBook(
            partition_ranges=[(0, 5), (5, 10)], partition_idx=0
        )
        assert_tensor_equality(get_ids_on_rank(range_pb, 0), torch.arange(0, 5))
        assert_tensor_equality(get_ids_on_rank(range_pb, 1), torch.arange(5, 10))

    def test_invalid_tensor_partition_book(self):
        invalid_pb = torch.tensor([[0, 1], [0, 1]])  # 2D tensor
        with self.assertRaises(ValueError):
            get_ids_on_rank(invalid_pb, 0)


class TestGetTotalIds(unittest.TestCase):
    def test_tensor_partition_book(self):
        partition_book = torch.tensor([0, 1, 0, 1, 0])
        self.assertEqual(get_total_ids(partition_book), 5)

    def test_range_partition_book(self):
        range_pb = RangePartitionBook(
            partition_ranges=[(0, 5), (5, 10)], partition_idx=0
        )
        self.assertEqual(get_total_ids(range_pb), 10)

    def test_invalid_tensor_partition_book(self):
        invalid_pb = torch.tensor([[0, 1], [0, 1]])  # 2D tensor
        with self.assertRaises(ValueError):
            get_total_ids(invalid_pb)


class TestBuildBalancedRangePartitionBook(unittest.TestCase):
    def test_divides_evenly(self):
        # 10 entities, 2 partitions -> 5 each
        pb = build_balanced_range_parition_book(num_entities=10, rank=0, world_size=2)
        assert_tensor_equality(pb.partition_bounds, torch.tensor([5, 10]))

    def test_divides_unevenly(self):
        # 7 entities, 3 partitions -> 3, 2, 2 (remainder distributed to first partitions)
        pb = build_balanced_range_parition_book(num_entities=7, rank=0, world_size=3)
        assert_tensor_equality(pb.partition_bounds, torch.tensor([3, 5, 7]))


if __name__ == "__main__":
    unittest.main()
