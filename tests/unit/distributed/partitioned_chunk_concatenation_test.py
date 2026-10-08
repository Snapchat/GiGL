import torch
from absl.testing import absltest

from gigl.distributed.dist_partitioner import _concatenate_partitioned_chunks
from tests.test_assets.test_case import TestCase


def _random_chunk(
    num_rows: int, generator: torch.Generator
) -> tuple[torch.Tensor, ...]:
    # Fields follow the edge partitioning layout:
    #   0: src ids
    #   1: dst ids
    #   2: features
    #   3: packed uint8 features
    #   4: weights
    return (
        torch.randint(0, 1000, (num_rows,), dtype=torch.int64, generator=generator),
        torch.randint(0, 1000, (num_rows,), dtype=torch.int64, generator=generator),
        torch.rand((num_rows, 3), generator=generator),
        torch.randint(0, 256, (num_rows, 2), dtype=torch.uint8, generator=generator),
        torch.rand((num_rows,), dtype=torch.float64, generator=generator),
    )


class PartitionedChunkConcatenationTest(TestCase):
    def test_matches_cat_and_stack_and_consumes_chunks(self) -> None:
        generator = torch.Generator().manual_seed(0)
        # Includes zero-row chunks, which the partitioner produces when a chunk has no items for a rank.
        chunks = [_random_chunk(num_rows, generator) for num_rows in (5, 0, 7, 1, 0)]
        expected_edge_index = torch.stack(
            (torch.cat([c[0] for c in chunks]), torch.cat([c[1] for c in chunks])),
            dim=0,
        )
        expected_fields = [torch.cat([c[i] for c in chunks]) for i in (2, 3, 4)]

        edge_index, features, packed_features, weights = (
            _concatenate_partitioned_chunks(chunks, output_fields=[(0, 1), 2, 3, 4])
        )

        self.assertEqual(len(chunks), 0)
        for actual, expected in zip(
            (edge_index, features, packed_features, weights),
            (expected_edge_index, *expected_fields),
        ):
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertEqual(actual.shape, expected.shape)
            self.assertTrue(actual.is_contiguous())
            self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(edge_index.shape, (2, 13))

    def test_skips_legacy_empty_chunks_like_cat(self) -> None:
        # A rank with nothing to send can return 1-D empty tensors; torch.cat ignores those.
        chunks = [
            (torch.empty(0),),
            (torch.ones(2, 3),),
            (torch.empty(0),),
            (torch.full((1, 3), 2.0),),
        ]
        expected = torch.cat([c[0] for c in chunks])

        (actual,) = _concatenate_partitioned_chunks(chunks, output_fields=[0])

        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.equal(actual, expected))

    def test_all_zero_row_chunks_match_cat_and_stack(self) -> None:
        # A rank that receives nothing for a type gets only zero-row chunks, so the outputs are
        # shaped from the first chunk rather than from a chunk with rows.
        generator = torch.Generator().manual_seed(0)
        chunks = [_random_chunk(0, generator) for _ in range(2)]
        expected_edge_index = torch.stack(
            (torch.cat([c[0] for c in chunks]), torch.cat([c[1] for c in chunks])),
            dim=0,
        )
        expected_features = torch.cat([c[2] for c in chunks])

        edge_index, features = _concatenate_partitioned_chunks(
            chunks, output_fields=[(0, 1), 2]
        )

        self.assertEqual(edge_index.shape, expected_edge_index.shape)
        self.assertEqual(edge_index.dtype, expected_edge_index.dtype)
        self.assertEqual(features.shape, expected_features.shape)
        self.assertEqual(features.dtype, expected_features.dtype)

    def test_raises_on_mismatched_chunks(self) -> None:
        with self.assertRaises(ValueError):
            _concatenate_partitioned_chunks([], output_fields=[0])
        with self.assertRaises(ValueError):
            _concatenate_partitioned_chunks(
                [(torch.zeros(2, 3),), (torch.zeros(2, 3, dtype=torch.float64),)],
                output_fields=[0],
            )
        with self.assertRaises(ValueError):
            # torch.cat rejects this; a plain copy_ would broadcast the size-1 dim.
            _concatenate_partitioned_chunks(
                [(torch.zeros(2, 3),), (torch.zeros(2, 1),)], output_fields=[0]
            )


if __name__ == "__main__":
    absltest.main()
