from io import BytesIO
from unittest.mock import MagicMock, patch

from absl.testing import absltest
from google.cloud.storage.blob import Blob
from google.cloud.storage.bucket import Bucket
from google.cloud.storage.client import Client

from gigl.common import GcsUri
from gigl.common.utils.gcs import DELETE_REQUEST_TIMEOUT_S, GcsUtils
from tests.test_assets.test_case import TestCase


class TestGcsUtils(TestCase):
    def _create_gcs_utils_with_get_bucket_blocked(self):
        mock_client = MagicMock(spec=Client)
        mock_client.project = "test-project"
        mock_bucket = MagicMock(spec=Bucket)
        mock_client.bucket.return_value = mock_bucket
        mock_client.get_bucket.side_effect = AssertionError(
            "get_bucket should not be used for object-only operations"
        )

        with patch("gigl.common.utils.gcs.storage.Client", return_value=mock_client):
            gcs_utils = GcsUtils()

        return gcs_utils, mock_client, mock_bucket

    @patch("gigl.common.utils.gcs.storage")
    def test_upload_from_filelike(self, mock_storage_client):
        # Mock the GCS client, bucket, and blob
        mock_client = MagicMock(spec=Client)
        mock_client.project = "test-project"
        mock_bucket = MagicMock(spec=Bucket)
        mock_blob = MagicMock(spec=Blob)

        mock_storage_client.Client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        # Create a file-like object
        filelike = BytesIO(b"test content")

        # Define GCS URI
        gcs_uri = GcsUri("gs://test-bucket/test-path/test-file.txt")

        # Call the function
        gcs_utils = GcsUtils()
        gcs_utils.upload_from_filelike(gcs_uri, filelike)

        # Assertions
        mock_client.bucket.assert_called_once_with("test-bucket")
        mock_bucket.blob.assert_called_once_with("test-path/test-file.txt")
        mock_blob.upload_from_file.assert_called_once_with(
            filelike, content_type="application/octet-stream"
        )

    def test_init_succeeds_when_client_has_no_project(self):
        # storage.Client(project=None) with an explicit None is a documented
        # "no project" mode where client.project is None. GcsUtils must still
        # be constructible in that mode (it is the default code path, e.g.
        # FileLoader() with no project).
        mock_client = MagicMock(spec=Client)
        mock_client.project = None

        with patch("gigl.common.utils.gcs.storage.Client", return_value=mock_client):
            gcs_utils = GcsUtils()

        self.assertIsNotNone(gcs_utils)

    def test_does_gcs_file_exist_uses_bucket_reference(self):
        gcs_utils, mock_client, mock_bucket = (
            self._create_gcs_utils_with_get_bucket_blocked()
        )
        mock_blob = MagicMock(spec=Blob)
        mock_blob.exists.return_value = True
        mock_bucket.blob.return_value = mock_blob

        gcs_path = GcsUri("gs://test-bucket/test-path/test-file.txt")

        self.assertTrue(gcs_utils.does_gcs_file_exist(gcs_path=gcs_path))
        mock_client.bucket.assert_called_once_with("test-bucket")
        mock_client.get_bucket.assert_not_called()
        mock_bucket.blob.assert_called_once_with("test-path/test-file.txt")
        mock_blob.exists.assert_called_once_with()

    def test_delete_gcs_file_if_exist_uses_bucket_reference(self):
        gcs_utils, mock_client, mock_bucket = (
            self._create_gcs_utils_with_get_bucket_blocked()
        )
        mock_blob = MagicMock(spec=Blob)
        mock_blob.exists.return_value = True
        mock_bucket.blob.return_value = mock_blob

        gcs_path = GcsUri("gs://test-bucket/test-path/test-file.txt")

        gcs_utils.delete_gcs_file_if_exist(gcs_path=gcs_path)

        mock_client.bucket.assert_called_once_with("test-bucket")
        mock_client.get_bucket.assert_not_called()
        mock_bucket.blob.assert_called_once_with("test-path/test-file.txt")
        mock_blob.exists.assert_called_once_with()
        mock_blob.delete.assert_called_once_with()

    def test_count_blobs_in_gcs_path_uses_bucket_reference(self):
        gcs_utils, mock_client, mock_bucket = (
            self._create_gcs_utils_with_get_bucket_blocked()
        )
        mock_text_blob = MagicMock(spec=Blob)
        mock_text_blob.name = "test-path/a.txt"
        mock_csv_blob = MagicMock(spec=Blob)
        mock_csv_blob.name = "test-path/b.csv"
        mock_bucket.list_blobs.return_value = [mock_text_blob, mock_csv_blob]

        gcs_path = GcsUri("gs://test-bucket/test-path/")

        self.assertEqual(
            gcs_utils.count_blobs_in_gcs_path(gcs_path=gcs_path, suffix=".txt"), 1
        )
        mock_client.bucket.assert_called_once_with("test-bucket")
        mock_client.get_bucket.assert_not_called()
        mock_bucket.list_blobs.assert_called_once_with(prefix="test-path/")

    def test_delete_files_uses_bucket_reference(self):
        gcs_utils, mock_client, mock_bucket = (
            self._create_gcs_utils_with_get_bucket_blocked()
        )
        mock_blob = MagicMock(spec=Blob)
        mock_bucket.blob.return_value = mock_blob

        gcs_path = GcsUri("gs://test-bucket/test-path/test-file.txt")

        gcs_utils.delete_files(gcs_files=[gcs_path])

        mock_client.bucket.assert_called_once_with("test-bucket")
        mock_client.get_bucket.assert_not_called()
        mock_bucket.blob.assert_called_once_with("test-path/test-file.txt")
        mock_client.batch.assert_called_once_with()
        mock_blob.delete.assert_called_once_with(timeout=DELETE_REQUEST_TIMEOUT_S)

    def test_delete_files_in_bucket_dir(self):
        # Mock the GCS client, bucket, and blob
        mock_client = MagicMock(spec=Client)
        mock_client.project = "test-project"
        mock_bucket = MagicMock(spec=Bucket)

        non_existent_bucket = "test-bucket"

        # Mock the blobs in the bucket
        mock_file1 = MagicMock(spec=Blob)
        mock_file2 = MagicMock(spec=Blob)
        mock_file3 = MagicMock(spec=Blob)
        mock_blobs = [
            mock_file1,
            mock_file2,
            mock_file3,
        ]
        # Since this will always return 3 blobs, delete_files_in_bucket_dir should throw assertion error
        # since it wont be able to validate deletion of all blobs
        mock_bucket.list_blobs.return_value = mock_blobs
        mock_client.bucket.return_value = mock_bucket
        mock_client.get_bucket.side_effect = AssertionError(
            "get_bucket should not be used for object-only operations"
        )

        with patch("gigl.common.utils.gcs.storage.Client", return_value=mock_client):
            # Define GCS URI
            gcs_uri = GcsUri(f"gs://{non_existent_bucket}/test-path/")

            # Call the function
            gcs_utils = GcsUtils()

            # This fails because we are simulating that there are still files in the bucket
            self.assertRaises(
                AssertionError, gcs_utils._delete_files_in_bucket_dir, gcs_uri
            )

            # Called once for checking what to delete, and once for checking if deletion was successful
            self.assertEqual(mock_bucket.list_blobs.call_count, 2)

            # We will simulate delete failing first time, and then succeeding
            mock_file1.reset_mock()  # Reset counts
            mock_file1.delete.side_effect = [Exception("Delete failed"), None]
            # We will also simulate list_blobs to return an empty list the second time its called to simulate successful deletion
            # the first time it will still return the 3 blobs
            mock_bucket.list_blobs.side_effect = [
                [mock_file1, mock_file2, mock_file3],  # First call returns 3 blobs
                [
                    mock_file1
                ],  # Seconds call returns 1 blob, which is the one that failed to delete first time
                [],  # Third call returns empty list, simulating successful deletion
            ]
            gcs_utils.delete_files_in_bucket_dir(gcs_uri)
            self.assertEqual(
                mock_file1.delete.call_count, 2
            )  # Fails first time, succeeds second time
            mock_client.get_bucket.assert_not_called()


if __name__ == "__main__":
    absltest.main()
