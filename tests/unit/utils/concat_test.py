import os
from unittest import mock

import tensorflow as tf
import torch
from absl.testing import absltest

from gigl.utils.concat import concatenate_chunks
from gigl.utils.share_memory import share_memory
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


class ConcatenateChunksTest(TestCase):
    def test_matches_cat_and_stack_and_consumes_chunks(self) -> None:
        generator = torch.Generator().manual_seed(0)
        # Includes zero-row chunks, which the partitioner produces when a chunk has no items for a rank.
        chunks = [_random_chunk(num_rows, generator) for num_rows in (5, 0, 7, 1, 0)]
        expected = {
            (0, 1): torch.stack(
                (torch.cat([c[0] for c in chunks]), torch.cat([c[1] for c in chunks]))
            ),
            2: torch.cat([c[2] for c in chunks]),
            3: torch.cat([c[3] for c in chunks]),
            4: torch.cat([c[4] for c in chunks]),
        }

        # None entries stand for optional fields that are absent, and get no output.
        actual = concatenate_chunks(chunks, [(0, 1), 2, None, 3, 4])

        self.assertEqual(len(chunks), 0)
        self.assertEqual(actual.keys(), expected.keys())
        for field, expected_tensor in expected.items():
            self.assertEqual(actual[field].dtype, expected_tensor.dtype)
            self.assertTrue(actual[field].is_contiguous())
            self.assertTrue(torch.equal(actual[field], expected_tensor))
        self.assertEqual(actual[(0, 1)].shape, (2, 13))
        self.assertFalse(actual[2].is_shared())

    def test_skips_legacy_empty_chunks_like_cat(self) -> None:
        # A rank with nothing to send can return 1-D empty tensors; torch.cat ignores those.
        chunks = [
            (torch.empty(0),),
            (torch.ones(2, 3),),
            (torch.empty(0),),
            (torch.full((1, 3), 2.0),),
        ]
        expected = torch.cat([c[0] for c in chunks])

        actual = concatenate_chunks(chunks, [0])[0]

        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.equal(actual, expected))

    def test_all_zero_row_chunks_match_cat_and_stack(self) -> None:
        # A rank that receives nothing for a type gets only zero-row chunks, so the outputs are
        # shaped from the first chunk rather than from a chunk with rows.
        generator = torch.Generator().manual_seed(0)
        chunks = [_random_chunk(0, generator) for _ in range(2)]
        expected_edge_index = torch.stack(
            (torch.cat([c[0] for c in chunks]), torch.cat([c[1] for c in chunks]))
        )
        expected_features = torch.cat([c[2] for c in chunks])

        actual = concatenate_chunks(chunks, [(0, 1), 2])

        self.assertEqual(actual[(0, 1)].shape, expected_edge_index.shape)
        self.assertEqual(actual[(0, 1)].dtype, expected_edge_index.dtype)
        self.assertEqual(actual[2].shape, expected_features.shape)
        self.assertEqual(actual[2].dtype, expected_features.dtype)

    def test_converts_tensorflow_chunks_along_axis_1(self) -> None:
        # Edge ids load as (2, num_edges) TensorFlow batches, so they concatenate along axis 1.
        chunks = [
            (tf.constant([[0, 1], [10, 11]], dtype=tf.int64),),
            (tf.constant([[2], [12]], dtype=tf.int64),),
        ]

        actual = concatenate_chunks(
            chunks, [0], axis=1, convert=lambda t: torch.from_numpy(t.numpy())
        )[0]

        self.assertEqual(len(chunks), 0)
        self.assertEqual(actual.dtype, torch.int64)
        self.assertTrue(torch.equal(actual, torch.tensor([[0, 1, 2], [10, 11, 12]])))

    def test_preshare_allocates_in_shared_memory(self) -> None:
        chunks = [
            (torch.arange(12.0).reshape(3, 4),),
            (torch.arange(20.0).reshape(5, 4),),
        ]
        expected = torch.cat([c[0] for c in chunks])

        # Lower the preshare threshold so a small tensor takes the shared-memory path.
        with mock.patch.dict(os.environ, {"GIGL_TENSOR_SPILL_MIN_BYTES": "64"}):
            actual = concatenate_chunks(chunks, [0], preshare=True)[0]

        self.assertTrue(actual.is_shared())
        self.assertTrue(torch.equal(actual, expected))
        # Already shared, so sharing it again must not copy it to a new block.
        data_ptr = actual.data_ptr()
        share_memory(actual)
        self.assertEqual(actual.data_ptr(), data_ptr)

    def test_raises_on_invalid_chunks(self) -> None:
        with self.assertRaises(ValueError):
            concatenate_chunks([], [0])
        with self.assertRaises(ValueError):
            concatenate_chunks(
                [(torch.zeros(2, 3),), (torch.zeros(2, 3, dtype=torch.float64),)], [0]
            )
        with self.assertRaises(ValueError):
            # torch.cat rejects this; a plain copy_ would broadcast the size-1 dim.
            concatenate_chunks([(torch.zeros(2, 3),), (torch.zeros(2, 1),)], [0])


if __name__ == "__main__":
    absltest.main()
