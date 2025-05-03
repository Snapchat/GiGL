import io
import tempfile
import unittest
from pathlib import Path
from typing import List, Optional
from unittest.mock import ANY, MagicMock, patch

import fastavro
import requests
import torch
from google.cloud.exceptions import GoogleCloudError
from parameterized import param, parameterized

from gigl.common import GcsUri, Uri
from gigl.common.data.export import EmbeddingExporter, load_embeddings_to_bigquery
from gigl.common.utils.retry import RetriesFailedException


class TestEmbeddingExporter(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self._temp_dir = tempfile.TemporaryDirectory()

        self.test_uris: List[GcsUri] = []

    def tearDown(self):
        super().tearDown()
        self._temp_dir.cleanup()

    def test_raises_with_nested_context(self):
        exporter = EmbeddingExporter(GcsUri("gs://test-bucket/test-folder"))
        with exporter:
            with self.assertRaises(RuntimeError):
                with exporter:
                    pass

        # Test can leave and re-enter.
        with exporter:
            pass

    def test_file_flush_threshold_must_be_nonnegative(self):
        with self.assertRaisesRegex(
            ValueError,
            "file_flush_threshold must be a non-negative integer, but got -1",
        ):
            EmbeddingExporter(
                GcsUri("gs://test-bucket/test-folder"),
                min_shard_size_threshold_bytes=-1,
            )

    @parameterized.expand(
        [
            param(
                "no_prefix", file_prefix=None, expected_file_name="shard_00000000.avro"
            ),
            param(
                "custom_prefix",
                file_prefix="my-prefix",
                expected_file_name="my-prefix_00000000.avro",
            ),
        ]
    )
    @patch("gigl.common.data.export.GcsUtils")
    def test_write_embeddings_to_gcs(
        self,
        _,
        mock_gcs_utils_class,
        file_prefix: Optional[str],
        expected_file_name: str,
    ):
        # Mock inputs
        gcs_base_uri = GcsUri("gs://test-bucket/test-folder")
        id_batches = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        embedding_batches = [
            torch.tensor([[1, 11], [2, 12], [3, 13]]),
            torch.tensor([[4, 14], [5, 15], [6, 16]]),
        ]
        embedding_type = "test_type"
        test_file = Path(self._temp_dir.name) / "test-file"

        # Mock GCS blob
        self.test_uri = None

        def mock_write(uri, buff: io.BytesIO, **kwargs):
            self.test_uri = uri
            with test_file.open("wb") as f:
                f.write(buff.getvalue())

        mock_gcs_utils = MagicMock()
        mock_gcs_utils.upload_from_filelike.side_effect = mock_write
        mock_gcs_utils_class.return_value = mock_gcs_utils

        with EmbeddingExporter(
            export_dir=gcs_base_uri, file_prefix=file_prefix
        ) as exporter:
            for id_batch, embedding_batch in zip(id_batches, embedding_batches):
                exporter.add_embedding(id_batch, embedding_batch, embedding_type)

        # Assertions
        self.assertEqual(self.test_uri, GcsUri.join(gcs_base_uri, expected_file_name))
        avro_reader = fastavro.reader(test_file.open("rb"))
        records = list(avro_reader)
        expected_records = [
            {"node_id": 1, "node_type": "test_type", "emb": [1.0, 11.0]},
            {"node_id": 2, "node_type": "test_type", "emb": [2.0, 12.0]},
            {"node_id": 3, "node_type": "test_type", "emb": [3.0, 13.0]},
            {"node_id": 4, "node_type": "test_type", "emb": [4.0, 14.0]},
            {"node_id": 5, "node_type": "test_type", "emb": [5.0, 15.0]},
            {"node_id": 6, "node_type": "test_type", "emb": [6.0, 16.0]},
        ]
        self.assertEqual(records, expected_records)

    @patch("gigl.common.data.export.GcsUtils")
    def test_write_embeddings_to_gcs_multiple_flushes(self, mock_gcs_utils_class):
        # Mock inputs
        gcs_base_uri = GcsUri("gs://test-bucket/test-folder")
        id_batches = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        embedding_batches = [
            torch.tensor([[1, 11], [2, 12], [3, 13]]),
            torch.tensor([[4, 14], [5, 15], [6, 16]]),
        ]
        embedding_type = "test_type"
        self.test_files = [
            Path(self._temp_dir.name) / f"test-file-{i}" for i in range(2)
        ]
        self.test_file_iter = iter(self.test_files)

        # Mock GCS blob
        def mock_write(uri, buff: io.BytesIO, **kwargs):
            self.test_uris.append(uri)
            next(self.test_file_iter).write_bytes(buff.getvalue())

        mock_gcs_utils = MagicMock()
        mock_gcs_utils.upload_from_filelike.side_effect = mock_write
        mock_gcs_utils_class.return_value = mock_gcs_utils

        # Write first batch using context manager
        id_embedding_batch_iter = zip(id_batches, embedding_batches)
        exporter = EmbeddingExporter(export_dir=gcs_base_uri)
        with exporter:
            id_batch, embedding_batch = next(id_embedding_batch_iter)
            exporter.add_embedding(id_batch, embedding_batch, embedding_type)

        # Write second batch with explict flush
        id_batch, embedding_batch = next(id_embedding_batch_iter)
        exporter.add_embedding(id_batch, embedding_batch, embedding_type)
        exporter.flush_embeddings()

        # Assertions
        self.assertEqual(
            self.test_uris,
            [
                GcsUri.join(gcs_base_uri, f"shard_{0:08}.avro"),
                GcsUri.join(gcs_base_uri, f"shard_{1:08}.avro"),
            ],
        )
        expected_records_by_batch = [
            [
                {"node_id": 1, "node_type": "test_type", "emb": [1.0, 11.0]},
                {"node_id": 2, "node_type": "test_type", "emb": [2.0, 12.0]},
                {"node_id": 3, "node_type": "test_type", "emb": [3.0, 13.0]},
            ],
            [
                {"node_id": 4, "node_type": "test_type", "emb": [4.0, 14.0]},
                {"node_id": 5, "node_type": "test_type", "emb": [5.0, 15.0]},
                {"node_id": 6, "node_type": "test_type", "emb": [6.0, 16.0]},
            ],
        ]
        for i, record_file in enumerate(self.test_files):
            with self.subTest(f"Records for batch {i}"):
                reader = fastavro.reader(record_file.open("rb"))
                records = list(reader)
                self.assertEqual(records, expected_records_by_batch[i])

    @patch("gigl.common.data.export.GcsUtils")
    def test_flushes_after_maximum_buffer_size(self, mock_gcs_utils_class):
        # Mock inputs
        gcs_base_uri = GcsUri("gs://test-bucket/test-folder")
        id_batches = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        embedding_batches = [
            torch.tensor([[1, 11], [2, 12], [3, 13]]),
            torch.tensor([[4, 14], [5, 15], [6, 16]]),
        ]
        embedding_type = "test_type"
        self.test_files = [
            Path(self._temp_dir.name) / f"test-file-{i}" for i in range(2)
        ]
        self.test_file_iter = iter(self.test_files)

        # Mock GCS blob
        def mock_write(uri, buff: io.BytesIO, **kwargs):
            self.test_uris.append(uri)
            next(self.test_file_iter).write_bytes(buff.getvalue())

        mock_gcs_utils = MagicMock()
        mock_gcs_utils.upload_from_filelike.side_effect = mock_write
        mock_gcs_utils_class.return_value = mock_gcs_utils

        with EmbeddingExporter(
            export_dir=gcs_base_uri, min_shard_size_threshold_bytes=1
        ) as exporter:
            for id_batch, embedding_batch in zip(id_batches, embedding_batches):
                exporter.add_embedding(id_batch, embedding_batch, embedding_type)

        self.assertEqual(
            self.test_uris,
            [
                GcsUri.join(gcs_base_uri, f"shard_{0:08}.avro"),
                GcsUri.join(gcs_base_uri, f"shard_{1:08}.avro"),
            ],
        )
        expected_records_by_batch = [
            [
                {"node_id": 1, "node_type": "test_type", "emb": [1.0, 11.0]},
                {"node_id": 2, "node_type": "test_type", "emb": [2.0, 12.0]},
                {"node_id": 3, "node_type": "test_type", "emb": [3.0, 13.0]},
            ],
            [
                {"node_id": 4, "node_type": "test_type", "emb": [4.0, 14.0]},
                {"node_id": 5, "node_type": "test_type", "emb": [5.0, 15.0]},
                {"node_id": 6, "node_type": "test_type", "emb": [6.0, 16.0]},
            ],
        ]
        for i, record_file in enumerate(self.test_files):
            with self.subTest(f"Records for batch {i}"):
                reader = fastavro.reader(record_file.open("rb"))
                records = list(reader)
                self.assertEqual(records, expected_records_by_batch[i])

    @patch("gigl.common.data.export.GcsUtils")
    def test_flush_resets_buffer(self, mock_gcs_utils_class):
        # Mock inputs
        gcs_base_uri = GcsUri("gs://test-bucket/test-folder")
        id_batch = torch.tensor([1, 2])
        embedding_batch = torch.tensor([[1, 11], [2, 12]])
        embedding_type = "test_type"
        self._mock_call_count = 0

        test_file = Path(self._temp_dir.name) / "test-file"

        def mock_upload(uri: Uri, buffer: io.BytesIO, content_type: str):
            if self._mock_call_count == 0:
                # Read the buffer, then fail.
                # We want to ensure that the buffer gets reset on retry.
                buffer.read()
                self._mock_call_count += 1
                google_cloud_error = GoogleCloudError("GCS upload failed")
                google_cloud_error.code = 503  # Service Unavailable
                raise google_cloud_error
            elif self._mock_call_count == 1:
                with test_file.open("wb") as f:
                    f.write(buffer.read())
                self._mock_call_count += 1
            else:
                self.fail(
                    f"Too many ({self._mock_call_count}) calls to upload, expected 2"
                )

        mock_gcs_utils = MagicMock()
        mock_gcs_utils.upload_from_filelike.side_effect = mock_upload
        mock_gcs_utils_class.return_value = mock_gcs_utils

        exporter = EmbeddingExporter(export_dir=gcs_base_uri)
        with EmbeddingExporter(export_dir=gcs_base_uri) as exporter:
            exporter.add_embedding(id_batch, embedding_batch, embedding_type)

        expected_records = [
            {"node_id": 1, "node_type": "test_type", "emb": [1.0, 11.0]},
            {"node_id": 2, "node_type": "test_type", "emb": [2.0, 12.0]},
        ]
        avro_reader = fastavro.reader(test_file.open("rb"))
        records = list(avro_reader)
        self.assertEqual(records, expected_records)

    @patch("time.sleep")
    @patch("gigl.common.data.export.GcsUtils")
    def test_write_embeddings_to_gcs_upload_retries_on_google_cloud_error_and_fails(
        self, mock_gcs_utils_class, mock_sleep
    ):
        # Mock inputs
        gcs_base_uri = GcsUri("gs://test-bucket/test-folder")
        id_batch = torch.tensor([1])
        embedding_batch = torch.tensor([[1, 11]])
        embedding_type = "test_type"

        mock_gcs_utils = MagicMock()
        google_cloud_error = GoogleCloudError("GCS upload failed")
        google_cloud_error.code = 503  # Service Unavailable
        mock_gcs_utils.upload_from_filelike.side_effect = google_cloud_error
        mock_gcs_utils_class.return_value = mock_gcs_utils
        exporter = EmbeddingExporter(export_dir=gcs_base_uri)
        exporter.add_embedding(id_batch, embedding_batch, embedding_type)

        # Assertions
        with self.assertRaisesRegex(RetriesFailedException, "GCS upload failed"):
            exporter.flush_embeddings()
        self.assertEqual(mock_gcs_utils.upload_from_filelike.call_count, 6)

    @patch("time.sleep")
    @patch("gigl.common.data.export.GcsUtils")
    def test_write_embeddings_to_gcs_upload_retries_on_request_exception_and_fails(
        self, mock_gcs_utils_class, mock_sleep
    ):
        # Constant message of request exception.
        CONNECTION_ABORTED_MESSAGE = "Connection aborted"

        # Mock inputs
        gcs_base_uri = GcsUri("gs://test-bucket/test-folder")
        id_batch = torch.tensor([1])
        embedding_batch = torch.tensor([[1, 11]])
        embedding_type = "test_type"

        mock_gcs_utils = MagicMock()
        mock_gcs_utils.upload_from_filelike.side_effect = (
            requests.exceptions.RequestException(CONNECTION_ABORTED_MESSAGE)
        )
        mock_gcs_utils_class.return_value = mock_gcs_utils
        exporter = EmbeddingExporter(export_dir=gcs_base_uri)
        exporter.add_embedding(id_batch, embedding_batch, embedding_type)

        # Assertions
        with self.assertRaisesRegex(RetriesFailedException, CONNECTION_ABORTED_MESSAGE):
            exporter.flush_embeddings()
        self.assertEqual(mock_gcs_utils.upload_from_filelike.call_count, 6)

    @patch("gigl.common.data.export.GcsUtils")
    def test_skips_flush_if_empty(self, mock_gcs_utils_class):
        gcs_base_uri = GcsUri("gs://test-bucket/test-folder")

        mock_gcs_utils_class.return_value.upload_from_filelike.side_effect = ValueError(
            "Should not be uploading if not data!"
        )
        exporter = EmbeddingExporter(export_dir=gcs_base_uri)
        exporter.flush_embeddings()

    @patch("gigl.common.data.export.bigquery.Client")
    def test_load_embedding_to_bigquery(self, mock_bigquery_client):
        # Mock inputs
        gcs_folder = GcsUri("gs://test-bucket/test-folder")
        project_id = "test-project"
        dataset_id = "test-dataset"
        table_id = "test-table"

        # Mock BigQuery client and load job
        mock_client = MagicMock()
        mock_client.load_table_from_uri.return_value.output_rows = 1000
        mock_bigquery_client.return_value = mock_client

        # Call the function
        load_embeddings_to_bigquery(gcs_folder, project_id, dataset_id, table_id)

        # Assertions
        mock_bigquery_client.assert_called_once_with(project=project_id)
        mock_client.load_table_from_uri.assert_called_once_with(
            source_uris=f"{gcs_folder.uri}/*.avro",
            destination=mock_client.dataset.return_value.table.return_value,
            job_config=ANY,
        )


if __name__ == "__main__":
    unittest.main()
