import unittest
from uuid import uuid4

from gigl.common import GcsUri
from gigl.common.utils.gcs import GcsUtils
from gigl.env import dep_constants
from gigl.env.pipelines_config import get_resource_config


class GcsUtilsTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self._scratch_gcs_path = GcsUri(
            f"{get_resource_config().temp_assets_regional_bucket_path}/testing/gcs_utils/{uuid4().hex}/"
        )

    def tearDown(self):
        gcs_utils = GcsUtils()
        gcs_utils.delete_files_in_bucket_dir(self._scratch_gcs_path)

    def test_join_path(self):
        self.assertEquals(
            GcsUri.join("gs://bucket_name", "path", "file.txt"),
            GcsUri("gs://bucket_name/path/file.txt"),
        )
        with self.assertRaises(TypeError):
            GcsUri.join("no_prefix", "some_file.txt")
        with self.assertRaises(TypeError):
            GcsUri.join("back_slashes_in_name\\", "some_file.txt")

    def test_download_to_temp_file(self):
        gcs_utils = GcsUtils()
        f = gcs_utils.download_file_from_gcs_to_temp_file(
            GcsUri(
                f"gs://{dep_constants.GIGL_PUBLIC_BUCKET_NAME}/test_assets/sample_text.txt"
            )
        )
        with open(f.name) as file:
            read_text = file.read()
            self.assertEqual(read_text, "This is a test")

    def test_upload_from_string(self):
        gcs_utils = GcsUtils()
        gcs_path = GcsUri.join(
            self._scratch_gcs_path, "test_upload_from_string/sample_text.txt"
        )
        gcs_utils.upload_from_string(gcs_path, "This is a test")
        f = gcs_utils.download_file_from_gcs_to_temp_file(gcs_path)
        with open(f.name) as file:
            read_text = file.read()
            self.assertEqual(read_text, "This is a test")

    def test_upload_from_str_overwrites(self):
        gcs_utils = GcsUtils()
        gcs_path = GcsUri.join(
            self._scratch_gcs_path, "test_upload_from_str_overwrites/sample_text.txt"
        )
        gcs_utils.upload_from_string(gcs_path, "This is a test")
        f = gcs_utils.download_file_from_gcs_to_temp_file(gcs_path)
        with open(f.name) as file:
            read_text = file.read()
            self.assertEqual(read_text, "This is a test")

        # Overwrite the content
        gcs_utils.upload_from_string(gcs_path, "Updated test content")
        f = gcs_utils.download_file_from_gcs_to_temp_file(gcs_path)
        with open(f.name) as file:
            read_text = file.read()
            self.assertEqual(read_text, "Updated test content")

    def test_copy_from_gcs_to_gcs(self):
        gcs_utils = GcsUtils()
        source_gcs_path = GcsUri.join(
            self._scratch_gcs_path, "test_copy_from_gcs_to_gcs/sample_text.txt"
        )
        gcs_utils.upload_from_string(source_gcs_path, "This is a test")
        destination_gcs_path = GcsUri.join(
            self._scratch_gcs_path, "test_copy_from_gcs_to_gcs/copied_sample_text.txt"
        )

        gcs_utils.copy_gcs_blob(source_gcs_path, destination_gcs_path)
        f = gcs_utils.download_file_from_gcs_to_temp_file(destination_gcs_path)
        with open(f.name) as file:
            read_text = file.read()
            self.assertEqual(read_text, "This is a test")


if __name__ == "__main__":
    unittest.main()
