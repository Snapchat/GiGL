import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch

from google.cloud.storage.blob import Blob
from google.cloud.storage.bucket import Bucket
from google.cloud.storage.client import Client

from gigl.common import GcsUri
from gigl.common.utils.gcs import GcsUtils


class TestGcsUtils(unittest.TestCase):
    @patch("gigl.common.utils.gcs.storage")
    def test_upload_from_filelike(self, mock_storage_client):
        # Mock the GCS client, bucket, and blob
        mock_client = MagicMock(spec=Client)
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


if __name__ == "__main__":
    unittest.main()
