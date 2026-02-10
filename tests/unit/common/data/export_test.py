import io
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import ANY, MagicMock, patch
from uuid import uuid4

import fastavro
import requests
import torch
from absl.testing import absltest
from google.cloud.exceptions import GoogleCloudError
from parameterized import param, parameterized

from gigl.common import GcsUri, LocalUri, Uri, UriFactory
from gigl.common.data.export import (
    _EMBEDDING_KEY,
    _NODE_ID_KEY,
    _NODE_TYPE_KEY,
    _PREDICTION_KEY,
    EmbeddingExporter,
    PredictionExporter,
    load_embeddings_to_bigquery,
    load_predictions_to_bigquery,
)
from gigl.common.utils.retry import RetriesFailedException
from tests.test_assets.test_case import TestCase

TEST_NODE_TYPE = "test_type"


class TestEmbeddingExporter(TestCase):
    def setUp(self):
        super().setUp()
        self.maxDiff = 1_000  # Set a high maxDiff to see full error messages
        self._temp_dir = tempfile.TemporaryDirectory()
        self.local_export_dir = Path(self._temp_dir.name) / uuid4().hex / "local-export"

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
    def test_write_embeddings_to_local(
        self,
        _,
        file_prefix: Optional[str],
        expected_file_name: str,
    ):
        # Mock inputs
        id_batches = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        embedding_batches = [
            torch.tensor([[1, 11], [2, 12], [3, 13]]),
            torch.tensor([[4, 14], [5, 15], [6, 16]]),
        ]
        test_file = self.local_export_dir / expected_file_name
        with EmbeddingExporter(
            export_dir=LocalUri(self.local_export_dir), file_prefix=file_prefix
        ) as exporter:
            for id_batch, embedding_batch in zip(id_batches, embedding_batches):
                exporter.add_embedding(id_batch, embedding_batch, TEST_NODE_TYPE)

        avro_reader = fastavro.reader(test_file.open("rb"))
        records = list(avro_reader)
        expected_records = [
            {
                _NODE_ID_KEY: 1,
                _NODE_TYPE_KEY: TEST_NODE_TYPE,
                _EMBEDDING_KEY: [1.0, 11.0],
            },
            {
                _NODE_ID_KEY: 2,
                _NODE_TYPE_KEY: TEST_NODE_TYPE,
                _EMBEDDING_KEY: [2.0, 12.0],
            },
            {
                _NODE_ID_KEY: 3,
                _NODE_TYPE_KEY: TEST_NODE_TYPE,
                _EMBEDDING_KEY: [3.0, 13.0],
            },
            {
                _NODE_ID_KEY: 4,
                _NODE_TYPE_KEY: TEST_NODE_TYPE,
                _EMBEDDING_KEY: [4.0, 14.0],
            },
            {
                _NODE_ID_KEY: 5,
                _NODE_TYPE_KEY: TEST_NODE_TYPE,
                _EMBEDDING_KEY: [5.0, 15.0],
            },
            {
                _NODE_ID_KEY: 6,
                _NODE_TYPE_KEY: TEST_NODE_TYPE,
                _EMBEDDING_KEY: [6.0, 16.0],
            },
        ]
        self.assertEqual(records, expected_records)

    def test_write_embeddings_multiple_flushes(self):
        id_batches = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        embedding_batches = [
            torch.tensor([[1, 11], [2, 12], [3, 13]]),
            torch.tensor([[4, 14], [5, 15], [6, 16]]),
        ]

        # Write first batch using context manager
        id_embedding_batch_iter = zip(id_batches, embedding_batches)
        exporter = EmbeddingExporter(
            export_dir=UriFactory.create_uri(self.local_export_dir)
        )
        with exporter:
            id_batch, embedding_batch = next(id_embedding_batch_iter)
            exporter.add_embedding(id_batch, embedding_batch, TEST_NODE_TYPE)

        # Write second batch with explict flush
        id_batch, embedding_batch = next(id_embedding_batch_iter)
        exporter.add_embedding(id_batch, embedding_batch, TEST_NODE_TYPE)
        exporter.flush_records()

        # Assertions
        written_files = sorted(
            list(map(UriFactory.create_uri, self.local_export_dir.glob("*"))),
            key=lambda x: x.uri,
        )
        self.assertEqual(
            written_files,
            [
                LocalUri.join(self.local_export_dir, f"shard_{0:08}.avro"),
                LocalUri.join(self.local_export_dir, f"shard_{1:08}.avro"),
            ],
        )
        expected_records_by_batch = [
            [
                {
                    _NODE_ID_KEY: 1,
                    _NODE_TYPE_KEY: TEST_NODE_TYPE,
                    _EMBEDDING_KEY: [1.0, 11.0],
                },
                {
                    _NODE_ID_KEY: 2,
                    _NODE_TYPE_KEY: TEST_NODE_TYPE,
                    _EMBEDDING_KEY: [2.0, 12.0],
                },
                {
                    _NODE_ID_KEY: 3,
                    _NODE_TYPE_KEY: TEST_NODE_TYPE,
                    _EMBEDDING_KEY: [3.0, 13.0],
                },
            ],
            [
                {
                    _NODE_ID_KEY: 4,
                    _NODE_TYPE_KEY: TEST_NODE_TYPE,
                    _EMBEDDING_KEY: [4.0, 14.0],
                },
                {
                    _NODE_ID_KEY: 5,
                    _NODE_TYPE_KEY: TEST_NODE_TYPE,
                    _EMBEDDING_KEY: [5.0, 15.0],
                },
                {
                    _NODE_ID_KEY: 6,
                    _NODE_TYPE_KEY: TEST_NODE_TYPE,
                    _EMBEDDING_KEY: [6.0, 16.0],
                },
            ],
        ]
        for i, record_file in enumerate(written_files):
            with self.subTest(f"Records for batch {i}"):
                reader = fastavro.reader(Path(record_file.uri).open("rb"))
                records = list(reader)
                self.assertEqual(records, expected_records_by_batch[i])

    def test_flushes_after_maximum_buffer_size(self):
        id_batches = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        embedding_batches = [
            torch.tensor([[1, 11], [2, 12], [3, 13]]),
            torch.tensor([[4, 14], [5, 15], [6, 16]]),
        ]

        with EmbeddingExporter(
            export_dir=UriFactory.create_uri(self.local_export_dir),
            min_shard_size_threshold_bytes=1,
        ) as exporter:
            for id_batch, embedding_batch in zip(id_batches, embedding_batches):
                exporter.add_embedding(id_batch, embedding_batch, TEST_NODE_TYPE)

        # Assertions
        written_files = sorted(
            list(map(UriFactory.create_uri, self.local_export_dir.glob("*"))),
            key=lambda x: x.uri,
        )
        self.assertEqual(
            written_files,
            [
                LocalUri.join(self.local_export_dir, f"shard_{0:08}.avro"),
                LocalUri.join(self.local_export_dir, f"shard_{1:08}.avro"),
            ],
        )
        expected_records_by_batch = [
            [
                {
                    _NODE_ID_KEY: 1,
                    _NODE_TYPE_KEY: TEST_NODE_TYPE,
                    _EMBEDDING_KEY: [1.0, 11.0],
                },
                {
                    _NODE_ID_KEY: 2,
                    _NODE_TYPE_KEY: TEST_NODE_TYPE,
                    _EMBEDDING_KEY: [2.0, 12.0],
                },
                {
                    _NODE_ID_KEY: 3,
                    _NODE_TYPE_KEY: TEST_NODE_TYPE,
                    _EMBEDDING_KEY: [3.0, 13.0],
                },
            ],
            [
                {
                    _NODE_ID_KEY: 4,
                    _NODE_TYPE_KEY: TEST_NODE_TYPE,
                    _EMBEDDING_KEY: [4.0, 14.0],
                },
                {
                    _NODE_ID_KEY: 5,
                    _NODE_TYPE_KEY: TEST_NODE_TYPE,
                    _EMBEDDING_KEY: [5.0, 15.0],
                },
                {
                    _NODE_ID_KEY: 6,
                    _NODE_TYPE_KEY: TEST_NODE_TYPE,
                    _EMBEDDING_KEY: [6.0, 16.0],
                },
            ],
        ]
        for i, record_file in enumerate(written_files):
            with self.subTest(f"Records for batch {i}"):
                reader = fastavro.reader(Path(record_file.uri).open("rb"))
                records = list(reader)
                self.assertEqual(records, expected_records_by_batch[i])

    @patch("gigl.src.common.utils.file_loader.GcsUtils")
    def test_flush_resets_buffer(self, mock_gcs_utils_class):
        # Mock inputs
        gcs_base_uri = GcsUri("gs://test-bucket/test-folder")
        id_batch = torch.tensor([1, 2])
        embedding_batch = torch.tensor([[1, 11], [2, 12]])
        self._mock_call_count = 0

        test_file = Path(self._temp_dir.name) / "test-file"

        def mock_upload(gcs_path: Uri, filelike: io.BytesIO):
            if self._mock_call_count == 0:
                # Read the buffer, then fail.
                # We want to ensure that the buffer gets reset on retry.
                filelike.read()
                self._mock_call_count += 1
                google_cloud_error = GoogleCloudError("GCS upload failed")
                google_cloud_error.code = 503  # Service Unavailable
                raise google_cloud_error
            elif self._mock_call_count == 1:
                with test_file.open("wb") as f:
                    f.write(filelike.read())
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
            exporter.add_embedding(id_batch, embedding_batch, TEST_NODE_TYPE)

        expected_records = [
            {
                _NODE_ID_KEY: 1,
                _NODE_TYPE_KEY: TEST_NODE_TYPE,
                _EMBEDDING_KEY: [1.0, 11.0],
            },
            {
                _NODE_ID_KEY: 2,
                _NODE_TYPE_KEY: TEST_NODE_TYPE,
                _EMBEDDING_KEY: [2.0, 12.0],
            },
        ]
        avro_reader = fastavro.reader(test_file.open("rb"))
        records = list(avro_reader)
        self.assertEqual(records, expected_records)

    @patch("time.sleep")
    @patch("gigl.src.common.utils.file_loader.GcsUtils")
    def test_write_embeddings_to_gcs_upload_retries_on_google_cloud_error_and_fails(
        self, mock_gcs_utils_class, mock_sleep
    ):
        # Mock inputs
        gcs_base_uri = GcsUri("gs://test-bucket/test-folder")
        id_batch = torch.tensor([1])
        embedding_batch = torch.tensor([[1, 11]])

        mock_gcs_utils = MagicMock()
        google_cloud_error = GoogleCloudError("GCS upload failed")
        google_cloud_error.code = 503  # Service Unavailable
        mock_gcs_utils.upload_from_filelike.side_effect = google_cloud_error
        mock_gcs_utils_class.return_value = mock_gcs_utils
        exporter = EmbeddingExporter(export_dir=gcs_base_uri)
        exporter.add_embedding(id_batch, embedding_batch, TEST_NODE_TYPE)

        # Assertions
        with self.assertRaisesRegex(RetriesFailedException, "GCS upload failed"):
            exporter.flush_records()
        self.assertEqual(mock_gcs_utils.upload_from_filelike.call_count, 6)

    @patch("time.sleep")
    @patch("gigl.src.common.utils.file_loader.GcsUtils")
    def test_write_embeddings_to_gcs_upload_retries_on_request_exception_and_fails(
        self, mock_gcs_utils_class, mock_sleep
    ):
        # Constant message of request exception.
        CONNECTION_ABORTED_MESSAGE = "Connection aborted"

        # Mock inputs
        gcs_base_uri = GcsUri("gs://test-bucket/test-folder")
        id_batch = torch.tensor([1])
        embedding_batch = torch.tensor([[1, 11]])

        mock_gcs_utils = MagicMock()
        mock_gcs_utils.upload_from_filelike.side_effect = (
            requests.exceptions.RequestException(CONNECTION_ABORTED_MESSAGE)
        )
        mock_gcs_utils_class.return_value = mock_gcs_utils
        exporter = EmbeddingExporter(export_dir=gcs_base_uri)
        exporter.add_embedding(id_batch, embedding_batch, TEST_NODE_TYPE)

        # Assertions
        with self.assertRaisesRegex(RetriesFailedException, CONNECTION_ABORTED_MESSAGE):
            exporter.flush_records()
        self.assertEqual(mock_gcs_utils.upload_from_filelike.call_count, 6)

    @patch("gigl.src.common.utils.file_loader.GcsUtils")
    def test_skips_flush_if_empty(self, mock_gcs_utils_class):
        gcs_base_uri = GcsUri("gs://test-bucket/test-folder")

        mock_gcs_utils_class.return_value.upload_from_filelike.side_effect = ValueError(
            "Should not be uploading if not data!"
        )
        exporter = EmbeddingExporter(export_dir=gcs_base_uri)
        exporter.flush_records()

    @parameterized.expand(
        [
            param(
                "Test if we can load embeddings synchronously",
                should_run_async=False,
            ),
            param(
                "Test if we can load embeddings asynchronously",
                should_run_async=True,
            ),
        ]
    )
    @patch("gigl.common.data.export.bigquery.Client")
    def test_load_embedding_to_bigquery(
        self, _, mock_bigquery_client, should_run_async: bool
    ):
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
        load_job = load_embeddings_to_bigquery(
            gcs_folder,
            project_id,
            dataset_id,
            table_id,
            should_run_async=should_run_async,
        )

        # Assertions
        mock_bigquery_client.assert_called_once_with(project=project_id)
        mock_client.load_table_from_uri.assert_called_once_with(
            source_uris=f"{gcs_folder.uri}/*.avro",
            destination=mock_client.dataset.return_value.table.return_value,
            job_config=ANY,
        )
        self.assertEqual(load_job.output_rows, 1000)


class TestPredictionsExporter(TestCase):
    def setUp(self):
        super().setUp()
        self.maxDiff = 1_000  # Set a high maxDiff to see full error messages
        self._temp_dir = tempfile.TemporaryDirectory()
        self._local_export_dir = (
            Path(self._temp_dir.name) / uuid4().hex / "local-export"
        )

    def _create_prediction_record_dict(self, node_id: int, prediction: float) -> dict:
        return {
            _NODE_ID_KEY: node_id,
            _NODE_TYPE_KEY: TEST_NODE_TYPE,
            _PREDICTION_KEY: prediction,
        }

    def tearDown(self):
        super().tearDown()
        self._temp_dir.cleanup()

    def assertRecordsAlmostEqual(self, actual_records, expected_records, places=5):
        """Helper method to compare records with float tolerance."""
        self.assertEqual(len(actual_records), len(expected_records))
        for actual, expected in zip(actual_records, expected_records):
            self.assertEqual(actual[_NODE_ID_KEY], expected[_NODE_ID_KEY])
            self.assertEqual(actual[_NODE_TYPE_KEY], expected[_NODE_TYPE_KEY])
            self.assertAlmostEqual(
                actual[_PREDICTION_KEY], expected[_PREDICTION_KEY], places=places
            )

    def test_raises_with_nested_context(self):
        exporter = PredictionExporter(GcsUri("gs://test-bucket/test-folder"))
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
            PredictionExporter(
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
    def test_write_predictions_to_local(
        self,
        _,
        file_prefix: Optional[str],
        expected_file_name: str,
    ):
        # Mock inputs
        id_batches = [torch.tensor([1, 2]), torch.tensor([4])]
        prediction_batches = [
            torch.tensor([0.1, 0.2]),
            torch.tensor([0.4]),
        ]
        test_file = self._local_export_dir / expected_file_name
        with PredictionExporter(
            export_dir=LocalUri(self._local_export_dir), file_prefix=file_prefix
        ) as exporter:
            for id_batch, prediction_batch in zip(id_batches, prediction_batches):
                exporter.add_prediction(id_batch, prediction_batch, TEST_NODE_TYPE)

        avro_reader = fastavro.reader(test_file.open("rb"))
        records = list(avro_reader)
        expected_records = [
            {_NODE_ID_KEY: 1, _NODE_TYPE_KEY: TEST_NODE_TYPE, _PREDICTION_KEY: 0.1},
            {_NODE_ID_KEY: 2, _NODE_TYPE_KEY: TEST_NODE_TYPE, _PREDICTION_KEY: 0.2},
            {_NODE_ID_KEY: 4, _NODE_TYPE_KEY: TEST_NODE_TYPE, _PREDICTION_KEY: 0.4},
        ]
        self.assertRecordsAlmostEqual(records, expected_records)

    def test_write_predictions_multiple_flushes(self):
        id_batches = [torch.tensor([1, 2]), torch.tensor([4])]
        prediction_batches = [
            torch.tensor([0.1, 0.2]),
            torch.tensor([0.4]),
        ]

        # Write first batch using context manager
        id_prediction_batch_iter = zip(id_batches, prediction_batches)
        exporter = PredictionExporter(
            export_dir=UriFactory.create_uri(self._local_export_dir)
        )
        with exporter:
            id_batch, prediction_batch = next(id_prediction_batch_iter)
            exporter.add_prediction(id_batch, prediction_batch, TEST_NODE_TYPE)

        # Write second batch with explict flush
        id_batch, prediction_batch = next(id_prediction_batch_iter)
        exporter.add_prediction(id_batch, prediction_batch, TEST_NODE_TYPE)
        exporter.flush_records()

        # Assertions
        written_files = sorted(
            list(map(UriFactory.create_uri, self._local_export_dir.glob("*"))),
            key=lambda x: x.uri,
        )
        self.assertEqual(
            written_files,
            [
                LocalUri.join(self._local_export_dir, f"shard_{0:08}.avro"),
                LocalUri.join(self._local_export_dir, f"shard_{1:08}.avro"),
            ],
        )
        expected_records_by_batch = [
            [
                {
                    _NODE_ID_KEY: 1,
                    _NODE_TYPE_KEY: TEST_NODE_TYPE,
                    _PREDICTION_KEY: 0.1,
                },
                {
                    _NODE_ID_KEY: 2,
                    _NODE_TYPE_KEY: TEST_NODE_TYPE,
                    _PREDICTION_KEY: 0.2,
                },
            ],
            [
                {
                    _NODE_ID_KEY: 4,
                    _NODE_TYPE_KEY: TEST_NODE_TYPE,
                    _PREDICTION_KEY: 0.4,
                },
            ],
        ]
        for i, record_file in enumerate(written_files):
            with self.subTest(f"Records for batch {i}"):
                reader = fastavro.reader(Path(record_file.uri).open("rb"))
                records = list(reader)
                self.assertRecordsAlmostEqual(records, expected_records_by_batch[i])

    def test_flushes_after_maximum_buffer_size(self):
        id_batches = [torch.tensor([1, 2]), torch.tensor([4])]
        prediction_batches = [
            torch.tensor([0.1, 0.2]),
            torch.tensor([0.4]),
        ]

        with PredictionExporter(
            export_dir=UriFactory.create_uri(self._local_export_dir),
            min_shard_size_threshold_bytes=1,
        ) as exporter:
            for id_batch, prediction_batch in zip(id_batches, prediction_batches):
                exporter.add_prediction(id_batch, prediction_batch, TEST_NODE_TYPE)

        # Assertions
        written_files = sorted(
            list(map(UriFactory.create_uri, self._local_export_dir.glob("*"))),
            key=lambda x: x.uri,
        )
        self.assertEqual(
            written_files,
            [
                LocalUri.join(self._local_export_dir, f"shard_{0:08}.avro"),
                LocalUri.join(self._local_export_dir, f"shard_{1:08}.avro"),
            ],
        )
        expected_records_by_batch = [
            [
                {
                    _NODE_ID_KEY: 1,
                    _NODE_TYPE_KEY: TEST_NODE_TYPE,
                    _PREDICTION_KEY: 0.1,
                },
                {
                    _NODE_ID_KEY: 2,
                    _NODE_TYPE_KEY: TEST_NODE_TYPE,
                    _PREDICTION_KEY: 0.2,
                },
            ],
            [
                {
                    _NODE_ID_KEY: 4,
                    _NODE_TYPE_KEY: TEST_NODE_TYPE,
                    _PREDICTION_KEY: 0.4,
                },
            ],
        ]
        for i, record_file in enumerate(written_files):
            with self.subTest(f"Records for batch {i}"):
                reader = fastavro.reader(Path(record_file.uri).open("rb"))
                records = list(reader)
                self.assertRecordsAlmostEqual(records, expected_records_by_batch[i])

    @patch("gigl.src.common.utils.file_loader.GcsUtils")
    def test_flush_resets_buffer(self, mock_gcs_utils_class):
        # Mock inputs
        gcs_base_uri = GcsUri("gs://test-bucket/test-folder")
        id_batch = torch.tensor([1])
        prediction_batch = torch.tensor([0.1])
        self._mock_call_count = 0

        test_file = Path(self._temp_dir.name) / "test-file"

        def mock_upload(gcs_path: Uri, filelike: io.BytesIO):
            if self._mock_call_count == 0:
                # Read the buffer, then fail.
                # We want to ensure that the buffer gets reset on retry.
                filelike.read()
                self._mock_call_count += 1
                google_cloud_error = GoogleCloudError("GCS upload failed")
                google_cloud_error.code = 503  # Service Unavailable
                raise google_cloud_error
            elif self._mock_call_count == 1:
                with test_file.open("wb") as f:
                    f.write(filelike.read())
                self._mock_call_count += 1
            else:
                self.fail(
                    f"Too many ({self._mock_call_count}) calls to upload, expected 2"
                )

        mock_gcs_utils = MagicMock()
        mock_gcs_utils.upload_from_filelike.side_effect = mock_upload
        mock_gcs_utils_class.return_value = mock_gcs_utils

        exporter = PredictionExporter(export_dir=gcs_base_uri)
        with PredictionExporter(export_dir=gcs_base_uri) as exporter:
            exporter.add_prediction(id_batch, prediction_batch, TEST_NODE_TYPE)

        expected_records = [
            {_NODE_ID_KEY: 1, _NODE_TYPE_KEY: TEST_NODE_TYPE, _PREDICTION_KEY: 0.1},
        ]
        avro_reader = fastavro.reader(test_file.open("rb"))
        records = list(avro_reader)
        self.assertRecordsAlmostEqual(records, expected_records)

    @patch("time.sleep")
    @patch("gigl.src.common.utils.file_loader.GcsUtils")
    def test_write_predictions_to_gcs_upload_retries_on_google_cloud_error_and_fails(
        self, mock_gcs_utils_class, mock_sleep
    ):
        # Mock inputs
        gcs_base_uri = GcsUri("gs://test-bucket/test-folder")
        id_batch = torch.tensor([1])
        prediction_batch = torch.tensor([0.1])

        mock_gcs_utils = MagicMock()
        google_cloud_error = GoogleCloudError("GCS upload failed")
        google_cloud_error.code = 503  # Service Unavailable
        mock_gcs_utils.upload_from_filelike.side_effect = google_cloud_error
        mock_gcs_utils_class.return_value = mock_gcs_utils
        exporter = PredictionExporter(export_dir=gcs_base_uri)
        exporter.add_prediction(id_batch, prediction_batch, TEST_NODE_TYPE)

        # Assertions
        with self.assertRaisesRegex(RetriesFailedException, "GCS upload failed"):
            exporter.flush_records()
        self.assertEqual(mock_gcs_utils.upload_from_filelike.call_count, 6)

    @patch("time.sleep")
    @patch("gigl.src.common.utils.file_loader.GcsUtils")
    def test_write_predictions_to_gcs_upload_retries_on_request_exception_and_fails(
        self, mock_gcs_utils_class, mock_sleep
    ):
        # Constant message of request exception.
        CONNECTION_ABORTED_MESSAGE = "Connection aborted"

        # Mock inputs
        gcs_base_uri = GcsUri("gs://test-bucket/test-folder")
        id_batch = torch.tensor([1])
        prediction_batch = torch.tensor([0.1])

        mock_gcs_utils = MagicMock()
        mock_gcs_utils.upload_from_filelike.side_effect = (
            requests.exceptions.RequestException(CONNECTION_ABORTED_MESSAGE)
        )
        mock_gcs_utils_class.return_value = mock_gcs_utils
        exporter = PredictionExporter(export_dir=gcs_base_uri)
        exporter.add_prediction(id_batch, prediction_batch, TEST_NODE_TYPE)

        # Assertions
        with self.assertRaisesRegex(RetriesFailedException, CONNECTION_ABORTED_MESSAGE):
            exporter.flush_records()
        self.assertEqual(mock_gcs_utils.upload_from_filelike.call_count, 6)

    @patch("gigl.src.common.utils.file_loader.GcsUtils")
    def test_skips_flush_if_empty(self, mock_gcs_utils_class):
        gcs_base_uri = GcsUri("gs://test-bucket/test-folder")

        mock_gcs_utils_class.return_value.upload_from_filelike.side_effect = ValueError(
            "Should not be uploading if not data!"
        )
        exporter = PredictionExporter(export_dir=gcs_base_uri)
        exporter.flush_records()

    @parameterized.expand(
        [
            param(
                "Test if we can load predictions synchronously",
                should_run_async=False,
            ),
            param(
                "Test if we can load predictions asynchronously",
                should_run_async=True,
            ),
        ]
    )
    @patch("gigl.common.data.export.bigquery.Client")
    def test_load_prediction_to_bigquery(
        self, _, mock_bigquery_client, should_run_async: bool
    ):
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
        load_job = load_predictions_to_bigquery(
            gcs_folder,
            project_id,
            dataset_id,
            table_id,
            should_run_async=should_run_async,
        )

        # Assertions
        mock_bigquery_client.assert_called_once_with(project=project_id)
        mock_client.load_table_from_uri.assert_called_once_with(
            source_uris=f"{gcs_folder.uri}/*.avro",
            destination=mock_client.dataset.return_value.table.return_value,
            job_config=ANY,
        )
        self.assertEqual(load_job.output_rows, 1000)


if __name__ == "__main__":
    absltest.main()
