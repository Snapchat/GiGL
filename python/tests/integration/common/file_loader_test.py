import io
import os
import tempfile
import unittest
import uuid

import gigl.common.utils.local_fs as local_fs
from gigl.common import GcsUri, HttpUri, LocalUri, Uri
from gigl.common.utils.gcs import GcsUtils
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.utils.file_loader import FileLoader
from parameterized import param, parameterized


class FileLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.file_loader = FileLoader()
        self.gcs_utils = GcsUtils()
        test_uuid = str(uuid.uuid4())
        self.test_asset_directory: LocalUri = LocalUri.join(".test_assets", test_uuid)
        resource_config = get_resource_config()
        self.gcs_test_asset_directory: GcsUri = GcsUri.join(
            resource_config.shared_resource_config.common_compute_config.temp_assets_bucket,
            test_uuid,
        )

    def tearDown(self) -> None:
        self.gcs_utils.delete_files_in_bucket_dir(
            gcs_path=self.gcs_test_asset_directory
        )
        local_fs.remove_folder_if_exist(local_path=self.test_asset_directory)

    def test_local_temp_file(self):
        local_file_path_src: LocalUri = LocalUri.join(
            self.test_asset_directory, "test_local_temp_file.txt"
        )

        local_fs.remove_file_if_exist(local_path=local_file_path_src)

        # Create files and ensure they exist
        local_fs.create_empty_file_if_none_exists(local_path=local_file_path_src)
        self.assertTrue(local_fs.does_path_exist(local_file_path_src))
        with open(local_file_path_src.uri, "w") as f:
            f.write("Hello")

        temp_f = self.file_loader.load_to_temp_file(file_uri_src=local_file_path_src)
        with open(temp_f.name, "r") as f:
            msg = f.read()
        self.assertTrue(msg == "Hello")
        temp_f.close()

    def test_gcs_temp_file(self):
        local_file_path_src: LocalUri = LocalUri.join(
            self.test_asset_directory, "test_gcs_temp_file.txt"
        )
        gcs_file_path_src: GcsUri = GcsUri.join(
            self.gcs_test_asset_directory, "test_gcs_temp_file.txt"
        )

        local_fs.remove_file_if_exist(local_path=local_file_path_src)
        self.gcs_utils.delete_gcs_file_if_exist(gcs_path=gcs_file_path_src)

        # Create files and ensure they exist
        local_fs.create_empty_file_if_none_exists(local_path=local_file_path_src)
        self.assertTrue(local_fs.does_path_exist(local_file_path_src))
        with open(local_file_path_src.uri, "w") as f:
            f.write("Hello")
        self.gcs_utils.upload_files_to_gcs(
            local_file_path_to_gcs_path_map={local_file_path_src: gcs_file_path_src}
        )
        local_fs.remove_file_if_exist(local_path=local_file_path_src)
        self.assertTrue(self.gcs_utils.does_gcs_file_exist(gcs_file_path_src))

        temp_f = self.file_loader.load_to_temp_file(file_uri_src=gcs_file_path_src)
        with open(temp_f.name, "r") as f:
            msg = f.read()
        self.assertTrue(msg == "Hello")
        temp_f.close()
        self.gcs_utils.delete_gcs_file_if_exist(gcs_path=gcs_file_path_src)

    def test_local_to_local_file(self):
        local_file_path_src: LocalUri = LocalUri.join(
            self.test_asset_directory, "test_local_to_local_src.txt"
        )
        local_file_path_dst: LocalUri = LocalUri.join(
            self.test_asset_directory, "test_local_to_local_dst.txt"
        )

        local_fs.remove_file_if_exist(local_path=local_file_path_src)
        local_fs.remove_file_if_exist(local_path=local_file_path_dst)

        # Create files and ensure they exist
        local_fs.create_empty_file_if_none_exists(local_path=local_file_path_src)
        self.assertTrue(local_fs.does_path_exist(local_file_path_src))

        file_uri_map: dict[Uri, Uri] = {local_file_path_src: local_file_path_dst}
        self.file_loader.load_files(source_to_dest_file_uri_map=file_uri_map)
        self.assertTrue(local_fs.does_path_exist(local_file_path_dst))
        self.assertTrue(os.path.islink(local_file_path_dst.uri))

        self.file_loader.load_files(
            source_to_dest_file_uri_map=file_uri_map,
            should_create_symlinks_if_possible=False,
        )
        self.assertFalse(os.path.islink(local_file_path_dst.uri))

    def test_local_to_gcs_file(self):
        local_file_path_src: LocalUri = LocalUri.join(
            self.test_asset_directory, "test_local_to_gcs.txt"
        )
        gcs_file_path_dst: GcsUri = GcsUri.join(
            self.gcs_test_asset_directory, "test_local_to_gcs.txt"
        )

        local_fs.remove_file_if_exist(local_path=local_file_path_src)
        self.gcs_utils.delete_gcs_file_if_exist(gcs_path=gcs_file_path_dst)

        # Create files and ensure they exist
        local_fs.create_empty_file_if_none_exists(local_path=local_file_path_src)
        self.assertTrue(local_fs.does_path_exist(local_file_path_src))

        file_uri_map: dict[Uri, Uri] = {local_file_path_src: gcs_file_path_dst}
        self.file_loader.load_files(source_to_dest_file_uri_map=file_uri_map)
        self.assertTrue(self.gcs_utils.does_gcs_file_exist(gcs_path=gcs_file_path_dst))
        self.gcs_utils.delete_gcs_file_if_exist(gcs_path=gcs_file_path_dst)

    def test_http_to_local_file(self):
        http_file_path_src: HttpUri = HttpUri(
            "https://raw.githubusercontent.com/Snapchat/GiGL/refs/heads/main/LICENSE"
        )
        local_file_path_dst: LocalUri = LocalUri.join(
            self.test_asset_directory, "test_http_to_local.txt"
        )
        local_fs.remove_file_if_exist(local_path=local_file_path_dst)
        file_uri_map: dict[Uri, Uri] = {http_file_path_src: local_file_path_dst}
        self.file_loader.load_files(source_to_dest_file_uri_map=file_uri_map)
        self.assertTrue(local_fs.does_path_exist(local_file_path_dst))

    def test_gcs_to_local_file(self):
        local_file_path_src: LocalUri = LocalUri.join(
            self.test_asset_directory, "test_gcs_to_local.txt"
        )

        gcs_file_path_src: GcsUri = GcsUri.join(
            self.gcs_test_asset_directory, "test_gcs_to_local.txt"
        )

        local_file_path_dst: LocalUri = LocalUri.join(
            self.test_asset_directory, "test_gcs_to_local.txt"
        )

        local_fs.remove_file_if_exist(local_path=local_file_path_src)
        local_fs.remove_file_if_exist(local_path=local_file_path_dst)
        self.gcs_utils.delete_gcs_file_if_exist(gcs_path=gcs_file_path_src)

        # Create files and ensure they exist
        local_fs.create_empty_file_if_none_exists(local_file_path_src)
        self.assertTrue(local_fs.does_path_exist(local_file_path_src))
        self.gcs_utils.upload_files_to_gcs(
            local_file_path_to_gcs_path_map={local_file_path_src: gcs_file_path_src}
        )
        local_fs.remove_file_if_exist(local_path=local_file_path_src)
        self.assertTrue(self.gcs_utils.does_gcs_file_exist(gcs_file_path_src))

        file_uri_map: dict[Uri, Uri] = {gcs_file_path_src: local_file_path_dst}
        self.file_loader.load_files(source_to_dest_file_uri_map=file_uri_map)
        self.assertTrue(local_fs.does_path_exist(local_file_path_dst))
        self.gcs_utils.delete_gcs_file_if_exist(gcs_path=gcs_file_path_src)

    def test_gcs_to_gcs_file(self):
        gcs_file_path_src: GcsUri = GcsUri.join(
            self.gcs_test_asset_directory, "test_gcs_to_gcs_src.txt"
        )
        gcs_file_path_dst: GcsUri = GcsUri.join(
            self.gcs_test_asset_directory, "test_gcs_to_gcs_dst.txt"
        )
        # First upload the source file
        self.file_loader.load_from_filelike(
            gcs_file_path_src, io.BytesIO(b"Hello, world!")
        )
        self.file_loader.load_files(
            source_to_dest_file_uri_map={gcs_file_path_src: gcs_file_path_dst}
        )

        f = self.file_loader.load_to_temp_file(file_uri_src=gcs_file_path_dst)
        with open(f.name, "r") as file:
            read_text = file.read()
            self.assertEqual(read_text, "Hello, world!")

    def test_local_to_local_dir(self):
        local_files = ["a.txt", "b.txt", "c.txt", "d.txt"]
        local_src_dir: LocalUri = LocalUri.join(self.test_asset_directory, "src")
        local_dst_dir: LocalUri = LocalUri.join(self.test_asset_directory, "dst")

        local_file_paths_src: list[LocalUri] = [
            LocalUri.join(local_src_dir, file) for file in local_files
        ]
        local_file_paths_dst: list[LocalUri] = [
            LocalUri.join(local_dst_dir, file) for file in local_files
        ]

        local_fs.remove_folder_if_exist(local_path=local_src_dir)
        local_fs.remove_folder_if_exist(local_path=local_dst_dir)

        # Create files and ensure they exist
        for file in local_file_paths_src:
            local_fs.create_empty_file_if_none_exists(local_path=file)
            self.assertTrue(local_fs.does_path_exist(file))

        dir_uri_map: dict[Uri, Uri] = {local_src_dir: local_dst_dir}
        self.file_loader.load_directories(source_to_dest_directory_map=dir_uri_map)

        for file in local_file_paths_dst:
            self.assertTrue(local_fs.does_path_exist(file))

    def test_local_to_gcs_dir(self):
        local_files = ["a.txt", "b.txt", "c.txt", "d.txt"]
        local_src_dir: LocalUri = LocalUri.join(self.test_asset_directory, "src")
        gcs_dst_dir: GcsUri = GcsUri.join(
            self.gcs_test_asset_directory, self.test_asset_directory, "dst"
        )

        local_file_paths_src: list[LocalUri] = [
            LocalUri.join(local_src_dir, file) for file in local_files
        ]
        gcs_file_paths_dst: list[GcsUri] = [
            GcsUri.join(gcs_dst_dir, file) for file in local_files
        ]

        local_fs.remove_folder_if_exist(local_path=local_src_dir)
        self.gcs_utils.delete_files_in_bucket_dir(gcs_path=gcs_dst_dir)

        # Create files and ensure they exist
        for file in local_file_paths_src:
            local_fs.create_empty_file_if_none_exists(local_path=file)
            self.assertTrue(local_fs.does_path_exist(file))

        dir_uri_map: dict[Uri, Uri] = {local_src_dir: gcs_dst_dir}
        self.file_loader.load_directories(source_to_dest_directory_map=dir_uri_map)

        for gcs_file in gcs_file_paths_dst:
            self.assertTrue(self.gcs_utils.does_gcs_file_exist(gcs_path=gcs_file))
        self.gcs_utils.delete_files_in_bucket_dir(
            gcs_path=GcsUri.join(
                self.gcs_test_asset_directory, self.test_asset_directory
            )
        )

    def test_gcs_to_local_dir(self):
        local_files = ["a.txt", "b.txt", "c.txt", "d.txt"]
        local_src_dir: LocalUri = LocalUri.join(self.test_asset_directory, "src")
        gcs_src_dir: GcsUri = GcsUri.join(
            self.gcs_test_asset_directory, self.test_asset_directory, "src"
        )
        local_dst_dir: LocalUri = LocalUri.join(self.test_asset_directory, "dst")

        local_file_paths_src: list[LocalUri] = [
            LocalUri.join(local_src_dir, file) for file in local_files
        ]
        gcs_file_paths_src: list[GcsUri] = [
            GcsUri.join(gcs_src_dir, file) for file in local_files
        ]
        local_file_paths_dst: list[LocalUri] = [
            LocalUri.join(local_dst_dir, file) for file in local_files
        ]

        local_fs.remove_folder_if_exist(local_path=local_src_dir)
        local_fs.remove_folder_if_exist(local_path=local_dst_dir)
        self.gcs_utils.delete_files_in_bucket_dir(gcs_path=gcs_src_dir)

        # Create files and ensure they exist
        for local_file in local_file_paths_src:
            local_fs.create_empty_file_if_none_exists(local_file)
            self.assertTrue(local_fs.does_path_exist(local_file))

        local_file_path_to_gcs_path_map: dict[LocalUri, GcsUri] = {
            local_file_path_src: gcs_file_path_src
            for local_file_path_src, gcs_file_path_src in zip(
                local_file_paths_src, gcs_file_paths_src
            )
        }
        self.gcs_utils.upload_files_to_gcs(
            local_file_path_to_gcs_path_map=local_file_path_to_gcs_path_map
        )
        for gcs_file in gcs_file_paths_src:
            self.assertTrue(self.gcs_utils.does_gcs_file_exist(gcs_file))
        local_fs.remove_folder_if_exist(local_path=local_src_dir)

        dir_uri_map: dict[Uri, Uri] = {gcs_src_dir: local_dst_dir}
        self.file_loader.load_directories(source_to_dest_directory_map=dir_uri_map)

        for file in local_file_paths_dst:
            self.assertTrue(local_fs.does_path_exist(file))
        self.gcs_utils.delete_files_in_bucket_dir(
            gcs_path=GcsUri.join(
                self.gcs_test_asset_directory, self.test_asset_directory
            )
        )

    def test_gcs_to_gcs_dir(self):
        gcs_src_dir: GcsUri = GcsUri.join(
            self.gcs_test_asset_directory, self.test_asset_directory, "src"
        )
        gcs_dst_dir: GcsUri = GcsUri.join(
            self.gcs_test_asset_directory, self.test_asset_directory, "dst"
        )
        dir_uri_map: dict[Uri, Uri] = {gcs_src_dir: gcs_dst_dir}

        with self.assertRaises(TypeError):
            self.file_loader.load_directories(source_to_dest_directory_map=dir_uri_map)

    def test_can_file_loader_check_existance_and_delete_uris(self):
        tmp_local_file = LocalUri.join(self.test_asset_directory, "tmp_local_file.txt")
        tmp_gcs_file = GcsUri.join(self.gcs_test_asset_directory, "tmp_gcs_file.txt")
        http_file: HttpUri = HttpUri(
            "https://raw.githubusercontent.com/Snapchat/GiGL/refs/heads/main/LICENSE"
        )
        # Write to local file
        local_fs.create_empty_file_if_none_exists(local_path=tmp_local_file)
        # Copy the file to GCS
        self.gcs_utils.upload_files_to_gcs(
            local_file_path_to_gcs_path_map={tmp_local_file: tmp_gcs_file}
        )
        # Ensure both files exist
        file_loader = FileLoader()
        self.assertTrue(file_loader.does_uri_exist(uri=tmp_local_file))
        self.assertTrue(file_loader.does_uri_exist(uri=tmp_gcs_file))
        self.assertTrue(file_loader.does_uri_exist(uri=http_file))

        # Delete the supported files; we cannot delete the HTTP URIs
        file_loader.delete_files(uris=[tmp_local_file, tmp_gcs_file])
        # Ensure both files are deleted
        self.assertFalse(file_loader.does_uri_exist(uri=tmp_local_file))
        self.assertFalse(file_loader.does_uri_exist(uri=tmp_gcs_file))

    @parameterized.expand(
        [
            param(
                "local_bytesio_buffer",
                filelike=io.BytesIO(b"hello world"),
                expected_content=b"world",
                expected_mode="rb",
            ),
            param(
                "local_stringio_buffer",
                filelike=io.StringIO("hello world"),
                expected_content="world",
                expected_mode="r",
            ),
        ]
    )
    def test_load_from_filelike_local(
        self, _, filelike, expected_content, expected_mode
    ):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
            uri = LocalUri(tmp_path)
            loader = FileLoader()
            # Move pointer to middle, write from there, check only written from pointer
            filelike.seek(6)
            loader.load_from_filelike(uri, filelike)
            with open(
                tmp_path,
                expected_mode,
                encoding=None if expected_mode == "rb" else "utf-8",
            ) as f:
                result = f.read()
            self.assertEqual(result, expected_content)

    @parameterized.expand(
        [
            param(
                "gcs_bytesio_buffer",
                io.BytesIO(b"gcs test bytes"),
                "gcs test bytes",  # Note our tests just check that this is a string...
            ),
            param(
                "gcs_stringio_buffer",
                io.StringIO("gcs test string"),
                "gcs test string",
            ),
        ]
    )
    def test_load_from_filelike_gcs(self, _, filelike, expected_content):
        uri = GcsUri.join(
            self.gcs_test_asset_directory,
            f"upload_from_likelike/{uuid.uuid4().hex}/test_file.txt",
        )
        loader = FileLoader()
        loader.load_from_filelike(uri, filelike)

        gcs_utils = GcsUtils()
        # Read the content back from GCS
        content = gcs_utils.read_from_gcs(uri)
        self.assertEqual(content, expected_content)
