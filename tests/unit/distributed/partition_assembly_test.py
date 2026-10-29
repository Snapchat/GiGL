import os
import tempfile
from unittest import mock

import torch
from parameterized import param, parameterized

from gigl.distributed.dist_partitioner import DistPartitioner
from gigl.distributed.dist_range_partitioner import DistRangePartitioner
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

    def _spilling(self):
        return mock.patch.dict(
            os.environ,
            {
                "GIGL_TENSOR_SPILL_DIR": self._spill_dir.name,
                "GIGL_TENSOR_SPILL_MIN_BYTES": "1",
            },
        )

    def test_tensor_partitioner_keeps_chunk_order(self) -> None:
        chunks = _received_chunks(num_nodes=1000, num_chunks=7, seed=0)
        expected = [torch.cat([chunk[field] for chunk in chunks]) for field in (0, 2)]

        assembled = DistPartitioner._assemble_node_chunks(
            object.__new__(DistPartitioner), chunks, [0, 2]
        )

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
    def test_range_partitioner_sorts_by_node_id(
        self, _, spill: bool, block_bytes: int
    ) -> None:
        chunks = _received_chunks(num_nodes=1000, num_chunks=7, seed=1)

        with (
            self._spilling() if spill else mock.patch.dict(os.environ),
            mock.patch(
                "gigl.distributed.dist_range_partitioner._ORDERED_BLOCK_BYTES",
                block_bytes,
            ),
        ):
            features, labels, ids = DistRangePartitioner._assemble_node_chunks(
                object.__new__(DistRangePartitioner), chunks, [0, 1, 2]
            )

        self.assertEqual(chunks, [])
        self.assertEqual(is_disk_backed(features), spill)
        torch.testing.assert_close(ids, torch.arange(1000), rtol=0, atol=0)
        torch.testing.assert_close(
            features, torch.arange(1000).float().unsqueeze(1).repeat(1, 3)
        )
        torch.testing.assert_close(labels, torch.arange(1000).unsqueeze(1))

    def test_results_are_disk_backed_when_spilling(self) -> None:
        with self._spilling():
            tensor_results = DistPartitioner._assemble_node_chunks(
                object.__new__(DistPartitioner),
                _received_chunks(num_nodes=1000, num_chunks=7, seed=2),
                [0, 2],
            )
            range_results = DistRangePartitioner._assemble_node_chunks(
                object.__new__(DistRangePartitioner),
                _received_chunks(num_nodes=1000, num_chunks=7, seed=2),
                [0, 2],
            )

        for result in tensor_results + range_results:
            self.assertTrue(is_disk_backed(result))
        torch.testing.assert_close(range_results[1], torch.arange(1000))
