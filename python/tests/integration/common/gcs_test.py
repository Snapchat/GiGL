import unittest

from gigl.common import GcsUri
from gigl.common.utils.gcs import GcsUtils
from gigl.env import dep_constants


class GcsUtilsTest(unittest.TestCase):
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

    # TODO: (svij) Follow up PR to fix this test
    # def test_upload_from_string(self):
    #     # TODO, ensure cleanup and files dont conflict with each other if multiple tests simultaneously
    #     test_str = "This is a test"
    #     test_str_overwrite = "This is a test overwrite"

    #     gcs_utils = GcsUtils()
    #     gcs_path = GcsUri(
    #         f"gs://{dep_constants.GIGL_PUBLIC_BUCKET_NAME}/test_assets/test_upload_from_string.txt"
    #     )
    #     gcs_utils.upload_from_string(gcs_path, test_str)
    #     f = gcs_utils.download_file_from_gcs_to_temp_file(gcs_path)
    #     with open(f.name) as file:
    #         read_text = file.read()
    #         self.assertEqual(read_text, test_str)

    #     gcs_utils.upload_from_string(gcs_path, test_str_overwrite)
    #     f = gcs_utils.download_file_from_gcs_to_temp_file(gcs_path)
    #     with open(f.name) as file:
    #         read_text = file.read()
    #         self.assertEqual(read_text, test_str_overwrite)
