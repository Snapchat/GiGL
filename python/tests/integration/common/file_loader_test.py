import os
import unittest
import uuid
from typing import Dict, List

import gigl.common.utils.local_fs as local_fs
from gigl.common import GcsUri, LocalUri, Uri
from gigl.common.utils.gcs import GcsUtils
from gigl.src.common.utils.file_loader import FileLoader
from tests.test_assets.uri_constants import TEST_DATA_GCS_BUCKET


class FileLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.file_loader = FileLoader()
        self.gcs_utils = GcsUtils()
        test_uuid = str(uuid.uuid4())
        # TODO (svij): Refactor name and how the location of tests is derived
        # Also need to investigate whether or not we need to clean up these assets
        self.test_asset_directory: LocalUri = LocalUri.join(".test_assets", test_uuid)
        self.gcs_test_asset_directory: GcsUri = GcsUri.join(
            TEST_DATA_GCS_BUCKET, test_uuid
        )

    # def test_local_temp_file(self):
    #     local_file_path_src: LocalUri = LocalUri.join(
    #         self.test_asset_directory, "test_local_temp_file.txt"
    #     )

    #     local_fs.remove_file_if_exist(local_path=local_file_path_src)

    #     # Create files and ensure they exist
    #     local_fs.create_empty_file_if_none_exists(local_path=local_file_path_src)
    #     self.assertTrue(local_fs.does_path_exist(local_file_path_src))
    #     with open(local_file_path_src.uri, "w") as f:
    #         f.write("Hello")

    #     temp_f = self.file_loader.load_to_temp_file(file_uri_src=local_file_path_src)
    #     with open(temp_f.name, "r") as f:
    #         msg = f.read()
    #     self.assertTrue(msg == "Hello")
    #     temp_f.close()

    def test_gcs_temp_file(self):
        local_file_path_src: LocalUri = LocalUri.join(
            self.test_asset_directory, "test_gcs_temp_file.txt"
        )
        gcs_file_path_src: GcsUri = GcsUri.join(
            TEST_DATA_GCS_BUCKET, "test_gcs_temp_file.txt"
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

        file_uri_map: Dict[Uri, Uri] = {local_file_path_src: local_file_path_dst}
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
            TEST_DATA_GCS_BUCKET, "test_local_to_gcs.txt"
        )

        local_fs.remove_file_if_exist(local_path=local_file_path_src)
        self.gcs_utils.delete_gcs_file_if_exist(gcs_path=gcs_file_path_dst)

        # Create files and ensure they exist
        local_fs.create_empty_file_if_none_exists(local_path=local_file_path_src)
        self.assertTrue(local_fs.does_path_exist(local_file_path_src))

        file_uri_map: Dict[Uri, Uri] = {local_file_path_src: gcs_file_path_dst}
        self.file_loader.load_files(source_to_dest_file_uri_map=file_uri_map)
        self.assertTrue(self.gcs_utils.does_gcs_file_exist(gcs_path=gcs_file_path_dst))
        self.gcs_utils.delete_gcs_file_if_exist(gcs_path=gcs_file_path_dst)

    def test_gcs_to_local_file(self):
        local_file_path_src: LocalUri = LocalUri.join(
            self.test_asset_directory, "test_gcs_to_local.txt"
        )

        gcs_file_path_src: GcsUri = GcsUri.join(
            TEST_DATA_GCS_BUCKET, "test_gcs_to_local.txt"
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

        file_uri_map: Dict[Uri, Uri] = {gcs_file_path_src: local_file_path_dst}
        self.file_loader.load_files(source_to_dest_file_uri_map=file_uri_map)
        self.assertTrue(local_fs.does_path_exist(local_file_path_dst))
        self.gcs_utils.delete_gcs_file_if_exist(gcs_path=gcs_file_path_src)

    def test_gcs_to_gcs_file(self):
        gcs_file_path_src: GcsUri = GcsUri.join(
            TEST_DATA_GCS_BUCKET, "test_gcs_to_gcs_src.txt"
        )
        gcs_file_path_dst: GcsUri = GcsUri.join(
            TEST_DATA_GCS_BUCKET, "test_gcs_to_gcs_dst.txt"
        )
        with self.assertRaises(TypeError):
            self.file_loader.load_files(
                source_to_dest_file_uri_map={gcs_file_path_src: gcs_file_path_dst}
            )

    def test_local_to_local_dir(self):
        local_files = ["a.txt", "b.txt", "c.txt", "d.txt"]
        local_src_dir: LocalUri = LocalUri.join(self.test_asset_directory, "src")
        local_dst_dir: LocalUri = LocalUri.join(self.test_asset_directory, "dst")

        local_file_paths_src: List[LocalUri] = [
            LocalUri.join(local_src_dir, file) for file in local_files
        ]
        local_file_paths_dst: List[LocalUri] = [
            LocalUri.join(local_dst_dir, file) for file in local_files
        ]

        local_fs.remove_folder_if_exist(local_path=local_src_dir)
        local_fs.remove_folder_if_exist(local_path=local_dst_dir)

        # Create files and ensure they exist
        for file in local_file_paths_src:
            local_fs.create_empty_file_if_none_exists(local_path=file)
            self.assertTrue(local_fs.does_path_exist(file))

        dir_uri_map: Dict[Uri, Uri] = {local_src_dir: local_dst_dir}
        self.file_loader.load_directories(source_to_dest_directory_map=dir_uri_map)

        for file in local_file_paths_dst:
            self.assertTrue(local_fs.does_path_exist(file))

    def test_local_to_gcs_dir(self):
        local_files = ["a.txt", "b.txt", "c.txt", "d.txt"]
        local_src_dir: LocalUri = LocalUri.join(self.test_asset_directory, "src")
        gcs_dst_dir: GcsUri = GcsUri.join(
            TEST_DATA_GCS_BUCKET, self.test_asset_directory, "dst"
        )

        local_file_paths_src: List[LocalUri] = [
            LocalUri.join(local_src_dir, file) for file in local_files
        ]
        gcs_file_paths_dst: List[GcsUri] = [
            GcsUri.join(gcs_dst_dir, file) for file in local_files
        ]

        local_fs.remove_folder_if_exist(local_path=local_src_dir)
        self.gcs_utils.delete_files_in_bucket_dir(gcs_path=gcs_dst_dir)

        # Create files and ensure they exist
        for file in local_file_paths_src:
            local_fs.create_empty_file_if_none_exists(local_path=file)
            self.assertTrue(local_fs.does_path_exist(file))

        dir_uri_map: Dict[Uri, Uri] = {local_src_dir: gcs_dst_dir}
        self.file_loader.load_directories(source_to_dest_directory_map=dir_uri_map)

        for gcs_file in gcs_file_paths_dst:
            self.assertTrue(self.gcs_utils.does_gcs_file_exist(gcs_path=gcs_file))
        self.gcs_utils.delete_files_in_bucket_dir(
            gcs_path=GcsUri.join(TEST_DATA_GCS_BUCKET, self.test_asset_directory)
        )

    def test_gcs_to_local_dir(self):
        local_files = ["a.txt", "b.txt", "c.txt", "d.txt"]
        local_src_dir: LocalUri = LocalUri.join(self.test_asset_directory, "src")
        gcs_src_dir: GcsUri = GcsUri.join(
            TEST_DATA_GCS_BUCKET, self.test_asset_directory, "src"
        )
        local_dst_dir: LocalUri = LocalUri.join(self.test_asset_directory, "dst")

        local_file_paths_src: List[LocalUri] = [
            LocalUri.join(local_src_dir, file) for file in local_files
        ]
        gcs_file_paths_src: List[GcsUri] = [
            GcsUri.join(gcs_src_dir, file) for file in local_files
        ]
        local_file_paths_dst: List[LocalUri] = [
            LocalUri.join(local_dst_dir, file) for file in local_files
        ]

        local_fs.remove_folder_if_exist(local_path=local_src_dir)
        local_fs.remove_folder_if_exist(local_path=local_dst_dir)
        self.gcs_utils.delete_files_in_bucket_dir(gcs_path=gcs_src_dir)

        # Create files and ensure they exist
        for local_file in local_file_paths_src:
            local_fs.create_empty_file_if_none_exists(local_file)
            self.assertTrue(local_fs.does_path_exist(local_file))

        local_file_path_to_gcs_path_map: Dict[LocalUri, GcsUri] = {
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

        dir_uri_map: Dict[Uri, Uri] = {gcs_src_dir: local_dst_dir}
        self.file_loader.load_directories(source_to_dest_directory_map=dir_uri_map)

        for file in local_file_paths_dst:
            self.assertTrue(local_fs.does_path_exist(file))
        self.gcs_utils.delete_files_in_bucket_dir(
            gcs_path=GcsUri.join(TEST_DATA_GCS_BUCKET, self.test_asset_directory)
        )

    def test_gcs_to_gcs_dir(self):
        gcs_src_dir: GcsUri = GcsUri.join(
            TEST_DATA_GCS_BUCKET, self.test_asset_directory, "src"
        )
        gcs_dst_dir: GcsUri = GcsUri.join(
            TEST_DATA_GCS_BUCKET, self.test_asset_directory, "dst"
        )
        dir_uri_map: Dict[Uri, Uri] = {gcs_src_dir: gcs_dst_dir}

        with self.assertRaises(TypeError):
            self.file_loader.load_directories(source_to_dest_directory_map=dir_uri_map)

    def test_can_file_loader_check_existance_and_delete_uris(self):
        tmp_local_file = LocalUri.join(self.test_asset_directory, "tmp_local_file.txt")
        tmp_gcs_file = GcsUri.join(self.gcs_test_asset_directory, "tmp_gcs_file.txt")
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
        # Delete the files
        file_loader.delete_files(uris=[tmp_local_file, tmp_gcs_file])
        # Ensure both files are deleted
        self.assertFalse(file_loader.does_uri_exist(uri=tmp_local_file))
        self.assertFalse(file_loader.does_uri_exist(uri=tmp_gcs_file))

    def tearDown(self) -> None:
        pass
