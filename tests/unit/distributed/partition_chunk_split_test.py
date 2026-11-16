from typing import Optional, Tuple

import torch
from absl.testing import absltest
from parameterized import param, parameterized

from gigl.distributed import DistPartitioner
from tests.test_assets.test_case import TestCase


class _RecordingPartitionManager:
    """Captures what would be sent to each rank, in place of the rpc exchange."""

    def __init__(self) -> None:
        self.chunk_res: list = []

    def process(self, chunk_res: list) -> None:
        self.chunk_res = chunk_res


def _mask_loop_split(
    input_data: Optional[Tuple[torch.Tensor, ...]],
    rank_indices: torch.Tensor,
    chunk_rank: torch.Tensor,
    world_size: int,
    generate_pb: bool,
) -> list:
    """Reference: one masking pass per destination rank."""
    input_indices = torch.arange(len(chunk_rank), dtype=torch.int64)
    result = []
    for rank in range(world_size):
        per_rank_indices = torch.masked_select(input_indices, chunk_rank == rank)
        data = None
        if input_data is not None:
            data = tuple(t[per_rank_indices] for t in input_data)
        pb_indices = None
        if generate_pb:
            pb_indices = rank_indices[per_rank_indices]
        result.append((data, pb_indices))
    return result


class PartitionChunkSplitTest(TestCase):
    def _assert_compact(self, tensor: torch.Tensor) -> None:
        """A piece must own exactly its bytes, as rpc sends a view's whole storage."""
        self.assertEqual(tensor.storage_offset(), 0)
        self.assertEqual(
            tensor.untyped_storage().nbytes(), tensor.numel() * tensor.element_size()
        )

    @parameterized.expand(
        [
            param("single_rank", world_size=1, num_items=50),
            param("few_ranks", world_size=3, num_items=1000),
            param("many_ranks_with_empty", world_size=60, num_items=40),
            param("int64_book_300_ranks", world_size=300, num_items=5000),
            param("empty_chunk", world_size=4, num_items=0),
            param("int32_ids", world_size=5, num_items=500, id_dtype=torch.int32),
            param("generate_pb", world_size=7, num_items=500, generate_pb=True),
            param("no_input_data", world_size=7, num_items=500, has_data=False),
            param("book_int64", world_size=7, num_items=500, book_dtype=torch.int64),
        ]
    )
    def test_split_matches_mask_loop(
        self,
        _,
        world_size: int,
        num_items: int,
        id_dtype: torch.dtype = torch.int64,
        generate_pb: bool = False,
        has_data: bool = True,
        book_dtype: torch.dtype = torch.uint8,
    ) -> None:
        torch.manual_seed(world_size * 1000 + num_items)
        if world_size > 255 and book_dtype == torch.uint8:
            book_dtype = torch.int64
        my_rank = world_size // 2
        src = torch.randint(0, 10_000, (num_items,)).to(id_dtype)
        dst = torch.randint(0, 10_000, (num_items,)).to(id_dtype)
        features = torch.randn(num_items, 6)
        edge_ids = torch.arange(num_items, dtype=torch.int64)
        # Distinct from the positions in the chunk, so mixing up the two would fail.
        pb_indices = torch.randperm(num_items) + 1_000_000
        input_data = None
        if has_data:
            input_data = (src, dst, features, edge_ids)
        chunk_rank = torch.randint(0, world_size, (num_items,)).to(book_dtype)
        # Leave some ranks without any item.
        chunk_rank[chunk_rank == (my_rank + 1) % world_size] = my_rank

        partitioner = object.__new__(DistPartitioner)
        partitioner._world_size = world_size
        partitioner._rank = my_rank
        partitioner._partition_mgr = _RecordingPartitionManager()
        partitioner._partition_single_chunk_data(
            input_data=input_data,
            rank_indices=pb_indices,
            partition_function=lambda _, __: chunk_rank,
            chunk_start_pos=0,
            chunk_end_pos=num_items,
            generate_pb=generate_pb,
        )

        expected = _mask_loop_split(
            input_data, pb_indices, chunk_rank.long(), world_size, generate_pb
        )
        actual = partitioner._partition_mgr.chunk_res
        self.assertEqual(len(actual), world_size)
        for (actual_data, actual_pb), (expected_data, expected_pb) in zip(
            actual, expected
        ):
            if expected_data is None:
                self.assertIsNone(actual_data)
            else:
                assert actual_data is not None
                for actual_tensor, expected_tensor in zip(actual_data, expected_data):
                    self.assert_tensor_equality(actual_tensor, expected_tensor)
                    self.assertEqual(actual_tensor.dtype, expected_tensor.dtype)
                    self._assert_compact(actual_tensor)
            if expected_pb is None:
                self.assertIsNone(actual_pb)
            else:
                assert actual_pb is not None
                self.assert_tensor_equality(actual_pb, expected_pb)
                self._assert_compact(actual_pb)


if __name__ == "__main__":
    absltest.main()
