import functools
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path
from typing import Any

import fastavro
import torch
from absl.testing import absltest

from gigl.common import LocalUri
from gigl.common.data.background_export import (
    BackgroundExportError,
    BackgroundExportStalledError,
    BackgroundRunner,
)
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


def _guard(seconds: int):
    """Fails the test, instead of hanging CI, if it runs longer than `seconds`."""

    def decorator(test):
        @functools.wraps(test)
        def wrapped(self, *args, **kwargs):
            def on_alarm(signum, frame):
                raise AssertionError(f"{test.__name__} hung for more than {seconds}s")

            previous = signal.signal(signal.SIGALRM, on_alarm)
            signal.alarm(seconds)
            try:
                return test(self, *args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous)

        return wrapped

    return decorator


class _HangingUploadExporter(EmbeddingExporter):
    """Blocks in an interruptible wait, like a hung network call."""

    def add_embedding(self, id_batch, embedding_batch, embedding_type) -> None:
        threading.Event().wait()


class _UnkillableUploadExporter(EmbeddingExporter):
    """Blocks with SIGALRM masked, so only the parent's watchdog can stop it."""

    def add_embedding(self, id_batch, embedding_batch, embedding_type) -> None:
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
        time.sleep(3600)


class _NeverStartsExporter(EmbeddingExporter):
    def __init__(self, *args: Any, **kwargs: Any):
        time.sleep(3600)


class _SlowExporter(EmbeddingExporter):
    def add_embedding(self, id_batch, embedding_batch, embedding_type) -> None:
        time.sleep(_SLOW_BATCH_SECONDS)
        super().add_embedding(id_batch, embedding_batch, embedding_type)


class _SteadyExporter(EmbeddingExporter):
    def add_embedding(self, id_batch, embedding_batch, embedding_type) -> None:
        time.sleep(1.0)
        super().add_embedding(id_batch, embedding_batch, embedding_type)


class TestBackgroundExporter(TestCase):
    def setUp(self):
        super().setUp()
        self._temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self._temp_dir.name)

    def tearDown(self):
        super().tearDown()
        self._temp_dir.cleanup()

    @_guard(240)
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

    @_guard(240)
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

    @_guard(240)
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

    @_guard(240)
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

    @_guard(240)
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
        with self.assertRaisesRegex(BackgroundExportError, "exited unexpectedly"):
            for _ in range(20):
                exporter.add_embedding(ids, embeddings, "user")
                time.sleep(0.1)
            exporter.flush_records()

    @_guard(240)
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

    @_guard(60)
    def test_hung_upload_becomes_an_error_and_the_worker_is_gone(self):
        runner = BackgroundRunner(
            _HangingUploadExporter,
            dict(export_dir=LocalUri(str(self.root / "out"))),
            stall_timeout_s=2,
        )
        ids, embeddings = _batches()[0]
        runner.submit("add_embedding", ids, embeddings, "user")
        start = time.perf_counter()
        with self.assertRaises(TimeoutError):
            runner.flush_records()
        self.assertLess(time.perf_counter() - start, 30)
        with self.assertRaises(TimeoutError):
            runner.close()
        self.assertFalse(runner._proxy._process.is_alive())

    @_guard(90)
    def test_uninterruptible_worker_is_killed_by_the_watchdog(self):
        runner = BackgroundRunner(
            _UnkillableUploadExporter,
            dict(export_dir=LocalUri(str(self.root / "out"))),
            stall_timeout_s=2,
        )
        process = runner._proxy._process
        ids, embeddings = _batches()[0]
        runner.submit("add_embedding", ids, embeddings, "user")
        start = time.perf_counter()
        with self.assertRaises(BackgroundExportStalledError):
            runner.flush_records()
        self.assertLess(time.perf_counter() - start, 40)
        self.assertFalse(process.is_alive())
        with self.assertRaises(BackgroundExportStalledError):
            runner.close()

    @_guard(90)
    def test_full_queue_with_a_stalled_worker_raises_instead_of_blocking(self):
        runner = BackgroundRunner(
            _UnkillableUploadExporter,
            dict(export_dir=LocalUri(str(self.root / "out"))),
            max_pending_batches=2,
            stall_timeout_s=2,
        )
        process = runner._proxy._process
        ids, embeddings = _batches()[0]
        start = time.perf_counter()
        with self.assertRaises(BackgroundExportStalledError):
            for _ in range(50):
                runner.submit("add_embedding", ids, embeddings, "user")
        self.assertLess(time.perf_counter() - start, 40)
        self.assertFalse(process.is_alive())
        with self.assertRaises(BackgroundExportStalledError):
            runner.close()

    @_guard(90)
    def test_worker_that_never_starts_raises(self):
        runner = BackgroundRunner(
            _NeverStartsExporter,
            dict(export_dir=LocalUri(str(self.root / "out"))),
            startup_timeout_s=10,
        )
        process = runner._proxy._process
        with self.assertRaises(BackgroundExportStalledError):
            runner.flush_records()
        self.assertFalse(process.is_alive())
        with self.assertRaises(BackgroundExportStalledError):
            runner.close()

    @_guard(120)
    def test_worker_exits_and_frees_shared_memory_when_parent_is_killed(self):
        script = textwrap.dedent(
            f"""
            import os, signal, torch
            from gigl.common import LocalUri
            from gigl.common.data.export import EmbeddingExporter
            exporter = EmbeddingExporter(LocalUri({str(self.root / "out")!r}), in_background=True)
            exporter.add_embedding(torch.arange(4), torch.rand(4, 8), "user")
            exporter.flush_records()
            proxy = exporter._background._proxy
            names = [s.name for s in proxy._segments.values()]
            print(proxy._process.pid, *names, flush=True)
            os.kill(os.getpid(), signal.SIGKILL)
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=100
        )
        pid, *names = result.stdout.split()
        self.assertTrue(names, result.stderr)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            alive = (
                os.path.exists(f"/proc/{pid}")
                and "zombie" not in open(f"/proc/{pid}/status").read()
            )
            leftovers = [n for n in names if os.path.exists(f"/dev/shm/{n}")]
            if not alive and not leftovers:
                break
            time.sleep(0.5)
        self.assertFalse(alive, "worker outlived its parent")
        self.assertEqual(leftovers, [])

    def test_invalid_arguments(self):
        with self.assertRaisesRegex(ValueError, "max_pending_batches"):
            EmbeddingExporter(
                LocalUri(str(self.root / "out")),
                in_background=True,
                max_pending_batches=0,
            )
        with self.assertRaisesRegex(ValueError, "stall_timeout_s"):
            EmbeddingExporter(
                LocalUri(str(self.root / "out")),
                in_background=True,
                background_stall_timeout_s=0,
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            EmbeddingExporter(
                LocalUri(str(self.root / "out")),
                min_shard_size_threshold_bytes=-1,
                in_background=True,
            )

    @_guard(240)
    def test_empty_batches_are_exported_like_synchronous(self):
        sync_dir, bg_dir = self.root / "sync", self.root / "bg"
        for directory, in_background in ((sync_dir, False), (bg_dir, True)):
            exporter = EmbeddingExporter(
                LocalUri(str(directory)), in_background=in_background
            )
            with exporter:
                exporter.add_embedding(
                    torch.empty(0, dtype=torch.int64), torch.empty(0, _DIM), "user"
                )
                exporter.add_embedding(torch.arange(4), torch.ones(4, _DIM), "user")
            exporter.close()
        self.assertEqual(_read_records(bg_dir), _read_records(sync_dir))
        self.assertEqual(len(_read_records(bg_dir)), 4)

    @_guard(240)
    def test_use_after_close_raises_instead_of_overwriting_shards(self):
        out = self.root / "out"
        exporter = EmbeddingExporter(LocalUri(str(out)), in_background=True)
        with exporter:
            exporter.add_embedding(torch.arange(4), torch.ones(4, _DIM), "user")
        exporter.close()
        with self.assertRaisesRegex(BackgroundExportError, "closed"):
            exporter.add_embedding(torch.arange(4), torch.ones(4, _DIM), "user")
        self.assertEqual([r["node_id"] for r in _read_records(out)], [0, 1, 2, 3])

    @_guard(240)
    def test_add_record_is_rejected_in_background_mode(self):
        exporter = EmbeddingExporter(
            LocalUri(str(self.root / "out")), in_background=True
        )
        with self.assertRaises(NotImplementedError):
            exporter.add_record([])
        exporter.close()

    @_guard(240)
    def test_flush_waits_for_a_slow_but_steady_worker(self):
        # Total time to drain the queue exceeds `stall_timeout_s`, but every command finishes well within it.
        runner = BackgroundRunner(
            _SteadyExporter,
            dict(export_dir=LocalUri(str(self.root / "out"))),
            max_pending_batches=8,
            stall_timeout_s=3,
        )
        for _ in range(8):
            runner.submit("add_embedding", torch.arange(4), torch.ones(4, _DIM), "user")
        runner.flush_records()
        runner.close()
        self.assertEqual(len(_read_records(self.root / "out")), 8 * 4)

    @absltest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    @_guard(240)
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
