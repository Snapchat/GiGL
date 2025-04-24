import tempfile
import unittest
from pathlib import Path
from typing import List, Optional

from parameterized import param, parameterized

import gigl.common.utils.local_fs as local_fs_utils
from gigl.common import LocalUri


class LocalFsUtilsTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        # Sets up the below file structure
        # temp_dir/
        #   ├── file1.txt
        #   ├── file2.txt
        #   ├── subdir/
        #   │   └── file3.txt
        #   └── subdir2/
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

        self.file1 = self.test_dir / "file1.txt"
        self.file1.touch()
        self.file2 = self.test_dir / "file2.txt"
        self.file2.touch()

        self.subdir = self.test_dir / "subdir"
        self.subdir.mkdir()
        self.file3 = self.subdir / "file3.txt"
        self.file3.touch()

        self.subdir2 = self.test_dir / "subdir2"
        self.subdir2.mkdir()

    def tearDown(self):
        super().tearDown()
        self.temp_dir.cleanup()

    @parameterized.expand(
        [
            param(
                test_name="no filters",
                regex=None,
                entity=None,
                expected_names=["file1.txt", "file2.txt", "subdir", "subdir2"],
            ),
            param(
                test_name="pattern",
                regex=r"\w+\d",
                entity=None,
                expected_names=["file1.txt", "file2.txt", "subdir2"],
            ),
            param(
                test_name="files",
                regex=None,
                entity=local_fs_utils.FileSystemEntity.FILE,
                expected_names=["file1.txt", "file2.txt"],
            ),
            param(
                test_name="dirs",
                regex=None,
                entity=local_fs_utils.FileSystemEntity.DIRECTORY,
                expected_names=["subdir", "subdir2"],
            ),
            param(
                test_name="pattern_and_files",
                regex=r"\w+\d",
                entity=local_fs_utils.FileSystemEntity.FILE,
                expected_names=["file1.txt", "file2.txt"],
            ),
            param(
                test_name="pattern_and_dirs",
                regex=r"\w+\d",
                entity=local_fs_utils.FileSystemEntity.DIRECTORY,
                expected_names=["subdir2"],
            ),
        ]
    )
    def test_list_at_path(
        self,
        test_name: str,
        regex: Optional[str],
        entity: Optional[local_fs_utils.FileSystemEntity],
        expected_names: List[str],
    ):
        del test_name  # unused.
        result = local_fs_utils.list_at_path(
            LocalUri(self.test_dir),
            regex=regex,
            file_system_entity=entity,
            names_only=False,
        )

        expected = {LocalUri(self.test_dir / x) for x in expected_names}
        self.assertEqual(set(result), expected)

    def test_list_at_path_name_only(self):
        result = local_fs_utils.list_at_path(LocalUri(self.test_dir), names_only=True)

        expected = {
            LocalUri(f) for f in ["file1.txt", "file2.txt", "subdir", "subdir2"]
        }
        self.assertEqual(set(result), expected)
