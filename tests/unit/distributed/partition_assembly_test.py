import contextlib
import os
import tempfile
from typing import Iterator
from unittest import mock

import torch
from parameterized import param, parameterized

from gigl.distributed.dist_partitioner import DistPartitioner, _concat_chunks
from gigl.utils.share_memory import is_disk_backed
from tests.test_assets.test_case import TestCase


def _received_chunks(
    num_nodes: int, num_chunks: int, seed: int
) -> list[tuple[torch.Tensor, ...]]:
    # (features, labels, node ids) per chunk, with ids shuffled across chunks as the exchange delivers them.
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randperm(num_nodes, generator=generator)
    features = ids.float().unsqueeze(1).repeat(1, 3)
    labels = ids.unsqueeze(1)
    return [
        (feature_chunk, label_chunk, id_chunk)
        for feature_chunk, label_chunk, id_chunk in zip(
            features.chunk(num_chunks), labels.chunk(num_chunks), ids.chunk(num_chunks)
        )
    ]


class PartitionAssemblyTest(TestCase):
    def setUp(self) -> None:
        self._spill_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._spill_dir.cleanup)

    @contextlib.contextmanager
    def _spill_env(self, enabled: bool, min_bytes: int = 1) -> Iterator[None]:
        # Spilling is read from env vars, so also clear any spill dir set in the developer's environment.
        with mock.patch.dict(
            os.environ, {"GIGL_TENSOR_SPILL_MIN_BYTES": str(min_bytes)}
        ):
            if enabled:
                os.environ["GIGL_TENSOR_SPILL_DIR"] = self._spill_dir.name
            else:
                os.environ.pop("GIGL_TENSOR_SPILL_DIR", None)
            yield

    def test_concat_chunks_keeps_chunk_order(self) -> None:
        chunks = _received_chunks(num_nodes=1000, num_chunks=7, seed=0)
        expected = [torch.cat([chunk[field] for chunk in chunks]) for field in (0, 2)]

        assembled = _concat_chunks(chunks, [0, 2])

        self.assertEqual(chunks, [])
        for result, reference in zip(assembled, expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)

    @parameterized.expand(
        [
            param("in_ram", spill=False, block_bytes=1 << 30),
            param("on_disk_one_block", spill=True, block_bytes=1 << 30),
            param("on_disk_many_blocks", spill=True, block_bytes=40),
        ]
    )
    def test_assemble_node_chunks_sorts_by_node_id(
        self, _, spill: bool, block_bytes: int
    ) -> None:
        chunks = _received_chunks(num_nodes=1000, num_chunks=7, seed=1)

        with (
            self._spill_env(spill),
            mock.patch(
                "gigl.distributed.dist_partitioner._ORDERED_BLOCK_BYTES",
                block_bytes,
            ),
        ):
            features, labels, ids = DistPartitioner()._assemble_node_chunks(
                chunks, [0, 1, 2]
            )

        self.assertEqual(chunks, [])
        self.assertEqual(is_disk_backed(features), spill)
        torch.testing.assert_close(ids, torch.arange(1000), rtol=0, atol=0)
        torch.testing.assert_close(
            features, torch.arange(1000).float().unsqueeze(1).repeat(1, 3)
        )
        torch.testing.assert_close(labels, torch.arange(1000).unsqueeze(1))

    def test_results_are_disk_backed_when_spilling(self) -> None:
        with self._spill_env(True):
            tensor_results = _concat_chunks(
                _received_chunks(num_nodes=1000, num_chunks=7, seed=2),
                [0, 2],
            )
            range_results = DistPartitioner()._assemble_node_chunks(
                _received_chunks(num_nodes=1000, num_chunks=7, seed=2),
                [0, 2],
            )

        for result in tensor_results + range_results:
            self.assertTrue(is_disk_backed(result))
        torch.testing.assert_close(range_results[1], torch.arange(1000))

    def test_assemble_node_chunks_spills_when_only_some_fields_are_large(self) -> None:
        # Features (12000 B) spill; labels and ids (8000 B each) stay in RAM. Labels come first so
        # the first destination is in RAM while a later one is on disk.
        chunks = _received_chunks(num_nodes=1000, num_chunks=7, seed=3)

        with self._spill_env(True, min_bytes=10000):
            labels, features, ids = DistPartitioner()._assemble_node_chunks(
                chunks, [1, 0, 2]
            )

        self.assertTrue(is_disk_backed(features))
        self.assertFalse(is_disk_backed(labels))
        self.assertFalse(is_disk_backed(ids))
        torch.testing.assert_close(ids, torch.arange(1000), rtol=0, atol=0)
        torch.testing.assert_close(
            features, torch.arange(1000).float().unsqueeze(1).repeat(1, 3)
        )
        torch.testing.assert_close(labels, torch.arange(1000).unsqueeze(1))
