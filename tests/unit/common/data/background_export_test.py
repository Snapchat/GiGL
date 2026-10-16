import tempfile
import time
from pathlib import Path

import fastavro
import torch
from absl.testing import absltest

from gigl.common import LocalUri
from gigl.common.data.background_export import BackgroundRunner
from gigl.common.data.export import EmbeddingExporter, PredictionExporter
from tests.test_assets.test_case import TestCase

_NUM_BATCHES = 40
_BATCH_SIZE = 16
_DIM = 8
_SLOW_BATCH_SECONDS = 0.3


def _read_shards(directory: Path) -> list[bytes]:
    return [p.read_bytes() for p in sorted(directory.glob("*.avro"))]


def _read_records(directory: Path) -> list[dict]:
    records: list[dict] = []
    for p in sorted(directory.glob("*.avro")):
        with open(p, "rb") as f:
            records.extend(list(fastavro.reader(f)))
    return records


def _batches() -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(0)
    return [
        (
            torch.arange(i * _BATCH_SIZE, (i + 1) * _BATCH_SIZE, dtype=torch.int64),
            torch.rand(_BATCH_SIZE, _DIM, generator=generator),
        )
        for i in range(_NUM_BATCHES)
    ]


class _SlowExporter(EmbeddingExporter):
    def add_embedding(self, id_batch, embedding_batch, embedding_type) -> None:
        time.sleep(_SLOW_BATCH_SECONDS)
        super().add_embedding(id_batch, embedding_batch, embedding_type)


class TestBackgroundExporter(TestCase):
    def setUp(self):
        super().setUp()
        self._temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self._temp_dir.name)

    def tearDown(self):
        super().tearDown()
        self._temp_dir.cleanup()

    def test_embedding_output_matches_synchronous_exporter(self):
        sync_dir, bg_dir = self.root / "sync", self.root / "bg"
        # A small threshold produces several shards, exercising flushes in the worker.
        with EmbeddingExporter(
            LocalUri(str(sync_dir)), min_shard_size_threshold_bytes=4_000
        ) as exporter:
            for ids, embeddings in _batches():
                exporter.add_embedding(ids, embeddings, "user")
        with EmbeddingExporter(
            LocalUri(str(bg_dir)),
            min_shard_size_threshold_bytes=4_000,
            in_background=True,
        ) as exporter:
            for ids, embeddings in _batches():
                exporter.add_embedding(ids, embeddings, "user")
                # The caller may reuse its buffers as soon as `add_embedding` returns.
                ids.zero_()
                embeddings.zero_()
        exporter.close()

        sync_shards = _read_shards(sync_dir)
        self.assertGreater(len(sync_shards), 1)
        # Avro files embed a random sync marker, so compare decoded records and shard layout.
        self.assertEqual(_read_records(bg_dir), _read_records(sync_dir))
        self.assertEqual(
            [len(s) for s in _read_shards(bg_dir)], [len(s) for s in sync_shards]
        )
        self.assertEqual(
            [r["node_id"] for r in _read_records(bg_dir)],
            list(range(_NUM_BATCHES * _BATCH_SIZE)),
        )

    def test_prediction_output_matches_synchronous_exporter(self):
        sync_dir, bg_dir = self.root / "sync", self.root / "bg"
        with PredictionExporter(LocalUri(str(sync_dir))) as exporter:
            for ids, values in _batches():
                exporter.add_prediction(ids, values[:, 0], "user")
        with PredictionExporter(LocalUri(str(bg_dir)), in_background=True) as exporter:
            for ids, values in _batches():
                exporter.add_prediction(ids, values[:, 0], "user")
        exporter.close()
        self.assertEqual(_read_records(bg_dir), _read_records(sync_dir))

    def test_flush_waits_for_queued_batches_and_context_can_be_reentered(self):
        out = self.root / "out"
        batches = _batches()
        exporter = EmbeddingExporter(LocalUri(str(out)), in_background=True)
        with exporter:
            for ids, embeddings in batches[:10]:
                exporter.add_embedding(ids, embeddings, "user")
            exporter.flush_records()
            self.assertEqual(len(_read_records(out)), 10 * _BATCH_SIZE)
            for ids, embeddings in batches[10:20]:
                exporter.add_embedding(ids, embeddings, "user")
        self.assertEqual(len(list(out.glob("*.avro"))), 2)
        with exporter:
            for ids, embeddings in batches[20:30]:
                exporter.add_embedding(ids, embeddings, "user")
        self.assertEqual(len(list(out.glob("*.avro"))), 3)
        self.assertEqual(
            [r["node_id"] for r in _read_records(out)], list(range(30 * _BATCH_SIZE))
        )
        exporter.close()

    def test_worker_error_surfaces_on_next_add_flush_and_context_exit(self):
        ids, embeddings = _batches()[0]
        # An embedding type Avro cannot encode as a string fails inside the worker.
        with self.assertRaisesRegex(TypeError, "node_type"):
            with EmbeddingExporter(
                LocalUri(str(self.root / "a")), in_background=True
            ) as exporter:
                exporter.add_embedding(ids, embeddings, 123)  # type: ignore
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    exporter.add_embedding(ids, embeddings, "user")
                    time.sleep(0.1)
                self.fail("add_embedding never raised the worker's error")
        with self.assertRaisesRegex(TypeError, "node_type"):
            exporter.close()

        exporter = EmbeddingExporter(LocalUri(str(self.root / "b")), in_background=True)
        exporter.add_embedding(ids, embeddings, 123)  # type: ignore
        with self.assertRaisesRegex(TypeError, "node_type"):
            exporter.flush_records()
        with self.assertRaisesRegex(TypeError, "node_type"):
            exporter.close()

    def test_dead_worker_is_reported(self):
        ids, embeddings = _batches()[0]
        exporter = EmbeddingExporter(
            LocalUri(str(self.root / "out")), in_background=True
        )
        exporter.flush_records()
        assert exporter._background is not None
        process = exporter._background._proxy._process
        process.kill()
        process.join()
        with self.assertRaisesRegex(RuntimeError, "exited unexpectedly"):
            for _ in range(20):
                exporter.add_embedding(ids, embeddings, "user")
                time.sleep(0.1)
            exporter.flush_records()

    def test_producer_blocks_when_worker_is_behind(self):
        out = self.root / "out"
        runner = BackgroundRunner(
            _SlowExporter, dict(export_dir=LocalUri(str(out))), max_pending_batches=2
        )
        runner.flush_records()  # Wait for the worker to start.
        ids, embeddings = _batches()[0]
        num_batches = 12
        start = time.perf_counter()
        for _ in range(num_batches):
            runner.submit("add_embedding", ids, embeddings, "user")
        elapsed = time.perf_counter() - start
        # Two batches queue in the feeder, two occupy shared-memory slots, and one is in flight,
        # so the producer must wait out the remaining batches.
        self.assertGreater(elapsed, (num_batches - 7) * _SLOW_BATCH_SECONDS * 0.8)
        runner.flush_records()
        runner.close()
        self.assertEqual(len(_read_records(out)), num_batches * _BATCH_SIZE)

    def test_invalid_arguments(self):
        with self.assertRaisesRegex(ValueError, "max_pending_batches"):
            EmbeddingExporter(
                LocalUri(str(self.root / "out")),
                in_background=True,
                max_pending_batches=0,
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            EmbeddingExporter(
                LocalUri(str(self.root / "out")),
                min_shard_size_threshold_bytes=-1,
                in_background=True,
            )

    @absltest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_tensors_mutated_after_add_are_not_exported(self):
        out = self.root / "out"
        with EmbeddingExporter(LocalUri(str(out)), in_background=True) as exporter:
            for i in range(20):
                ids = torch.arange(i * 4, i * 4 + 4, device="cuda")
                embeddings = torch.full((4, _DIM), float(i), device="cuda")
                exporter.add_embedding(ids, embeddings, "user")
                ids.zero_()
                embeddings.fill_(-1.0)
        records = _read_records(out)
        self.assertEqual([r["node_id"] for r in records], list(range(80)))
        self.assertEqual(
            [r["emb"][0] for r in records], [float(i // 4) for i in range(80)]
        )


if __name__ == "__main__":
    absltest.main()
