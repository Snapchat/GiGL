import torch
from absl.testing import absltest
from graphlearn_torch.partition import RangePartitionBook
from parameterized import param, parameterized

from gigl.distributed.utils.partition_book import (
    EmptyRangeAllowedPartitionBook,
    build_partition_book,
    get_ids_on_rank,
    get_total_ids,
)
from tests.test_assets.test_case import TestCase


class TestGetIdsOnRank(TestCase):
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
        self.assert_tensor_equality(get_ids_on_rank(partition_book, rank), expected)

    def test_invalid_tensor_partition_book(self):
        invalid_pb = torch.tensor([[0, 1], [0, 1]])  # 2D tensor
        with self.assertRaises(ValueError):
            get_ids_on_rank(invalid_pb, 0)


class TestGetTotalIds(TestCase):
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


class TestBuildPartitionBook(TestCase):
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
        self.assert_tensor_equality(pb.partition_bounds, expected_bounds)


class TestEmptyRangePartitionBook(TestCase):
    @parameterized.expand(
        [
            param(
                "fewer_entities_than_ranks",
                num_entities=1,
                world_size=3,
                expected_bounds=[1, 1, 1],
            ),
            param("no_entities", num_entities=0, world_size=2, expected_bounds=[0, 0]),
        ]
    )
    def test_build_partition_book_with_empty_ranges(
        self, _, num_entities: int, world_size: int, expected_bounds: list[int]
    ) -> None:
        for rank in range(world_size):
            partition_book = build_partition_book(
                num_entities=num_entities, rank=rank, world_size=world_size
            )
            self.assertIsInstance(partition_book, RangePartitionBook)
            self.assertEqual(partition_book.partition_bounds.tolist(), expected_bounds)
            if num_entities > 0:
                # The only entity (id 0) lives on rank 0.
                self.assert_tensor_equality(
                    partition_book[torch.arange(num_entities)],
                    torch.zeros(num_entities, dtype=torch.long),
                )
            for owner in range(world_size):
                expected_ids = (
                    torch.arange(num_entities) if owner == 0 else torch.empty(0)
                )
                self.assert_tensor_equality(
                    get_ids_on_rank(partition_book, owner),
                    expected_ids.to(torch.int64),
                )

    def test_lookup_skips_empty_range(self) -> None:
        partition_book = EmptyRangeAllowedPartitionBook(
            partition_ranges=[(0, 2), (2, 2), (2, 3)], partition_idx=2
        )
        self.assert_tensor_equality(
            partition_book[torch.tensor([0, 1, 2])], torch.tensor([0, 0, 2])
        )
        self.assertEqual(int(partition_book.id2index[torch.tensor([2])]), 0)

    @parameterized.expand(
        [
            param("start_after_end", ranges=[(0, 2), (3, 2)]),
            param("non_contiguous", ranges=[(0, 2), (3, 4)]),
            param("overlapping", ranges=[(0, 2), (1, 4)]),
        ]
    )
    def test_invalid_ranges_raise(self, _, ranges: list[tuple[int, int]]) -> None:
        with self.assertRaises(ValueError):
            EmptyRangeAllowedPartitionBook(partition_ranges=ranges, partition_idx=0)


if __name__ == "__main__":
    absltest.main()
