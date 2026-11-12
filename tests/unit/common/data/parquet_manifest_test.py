import json
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from gigl.common import UriFactory
from gigl.common.data import parquet_manifest
from tests.test_assets.test_case import TestCase


def _write_parquet(path: Path, num_rows: int, row_group_size: int) -> None:
    table = pa.table({"node_id": pa.array(range(num_rows), type=pa.int64())})
    pq.write_table(table, path, row_group_size=row_group_size)


class ParquetManifestTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.root = Path(self._temp_dir.name)
        # Written out of order to check that listing sorts.
        _write_parquet(self.root / "part-00001-of-00002.parquet", 5, 5)
        _write_parquet(self.root / "part-00000-of-00002.parquet", 25, 10)
        (self.root / "notes.txt").write_text("not parquet")
        self.fs = pafs.LocalFileSystem()
        self.path = str(self.root)

    def test_write_manifest_records_row_groups_and_skips_other_files(self) -> None:
        manifest = parquet_manifest.write_manifest(UriFactory.create_uri(self.root))
        expected_files = {
            "part-00000-of-00002.parquet": [10, 10, 5],
            "part-00001-of-00002.parquet": [5],
        }
        self.assertEqual(manifest["version"], parquet_manifest.MANIFEST_VERSION)
        self.assertEqual(manifest["files"], expected_files)
        on_disk = json.loads(
            (self.root / parquet_manifest.MANIFEST_FILE_NAME).read_text()
        )
        self.assertEqual(on_disk, manifest)

    def test_list_parquet_files_is_sorted_and_excludes_manifest(self) -> None:
        parquet_manifest.write_manifest(UriFactory.create_uri(self.root))
        files = parquet_manifest.list_parquet_files(self.fs, self.path)
        self.assertEqual(
            files,
            [
                f"{self.path}/part-00000-of-00002.parquet",
                f"{self.path}/part-00001-of-00002.parquet",
            ],
        )

    def test_read_manifest_row_groups_follows_file_order(self) -> None:
        parquet_manifest.write_manifest(UriFactory.create_uri(self.root))
        files = parquet_manifest.list_parquet_files(self.fs, self.path)
        self.assertEqual(
            parquet_manifest.read_manifest_row_groups(self.fs, self.path, files),
            [[10, 10, 5], [5]],
        )
        self.assertEqual(
            parquet_manifest.read_manifest_row_groups(
                self.fs, self.path, list(reversed(files))
            ),
            [[5], [10, 10, 5]],
        )

    def test_read_manifest_row_groups_raises_when_manifest_is_missing(self) -> None:
        files = parquet_manifest.list_parquet_files(self.fs, self.path)
        with self.assertRaises(FileNotFoundError):
            parquet_manifest.read_manifest_row_groups(self.fs, self.path, files)

    def test_read_manifest_row_groups_raises_on_unknown_version(self) -> None:
        (self.root / parquet_manifest.MANIFEST_FILE_NAME).write_text(
            json.dumps({"version": 999, "files": {}})
        )
        with self.assertRaises(ValueError):
            parquet_manifest.read_manifest_row_groups(self.fs, self.path, [])

    def test_read_manifest_row_groups_raises_when_files_changed(self) -> None:
        parquet_manifest.write_manifest(UriFactory.create_uri(self.root))
        _write_parquet(self.root / "part-00002-of-00002.parquet", 3, 3)
        files = parquet_manifest.list_parquet_files(self.fs, self.path)
        with self.assertRaises(ValueError):
            parquet_manifest.read_manifest_row_groups(self.fs, self.path, files)
