import io
import tempfile
import unittest
from pathlib import Path
from typing import Callable, Optional
from unittest.mock import ANY, MagicMock, patch
from uuid import uuid4

import fastavro
import requests
import torch
from google.cloud.bigquery.job import LoadJob
from google.cloud.exceptions import GoogleCloudError
from parameterized import param, parameterized

from gigl.common import GcsUri, LocalUri, Uri, UriFactory
from gigl.common.data.export import (
    _EMBEDDING_KEY,
    _NODE_ID_KEY,
    _NODE_TYPE_KEY,
    _PREDICTION_KEY,
    BaseGcsExporter,
    EmbeddingExporter,
    PredictionExporter,
    load_embeddings_to_bigquery,
    load_predictions_to_bigquery,
)
from gigl.common.utils.retry import RetriesFailedException

# Test constants
TEST_NODE_TYPE = "test_type"


class TestExporter(unittest.TestCase):
    """Test suite for both EmbeddingExporter and PredictionExporter."""

    def setUp(self):
        super().setUp()
        self.maxDiff = 1_000  # Set a high maxDiff to see full error messages
        self._temp_dir = tempfile.TemporaryDirectory()
        self.local_export_dir = Path(self._temp_dir.name) / uuid4().hex / "local-export"

    def tearDown(self):
        super().tearDown()
        self._temp_dir.cleanup()

    def assertPredictionRecordsAlmostEqual(
        self,
        actual_records,
        expected_records,
    ) -> None:
        """Helper method to compare prediction records with float tolerance. This is needed specifically
        for the prediction embeddings since they are stored as floats in the Avro file and may have small
        differences in precision as a result that may cause errors in the tests for exact matching.
        """
        self.assertEqual(len(actual_records), len(expected_records))
        for actual, expected in zip(actual_records, expected_records):
            self.assertEqual(actual[_NODE_ID_KEY], expected[_NODE_ID_KEY])
            self.assertEqual(actual[_NODE_TYPE_KEY], expected[_NODE_TYPE_KEY])
            self.assertAlmostEqual(
                actual[_PREDICTION_KEY], expected[_PREDICTION_KEY], places=5
            )

    @parameterized.expand(
        [
            param(
                "raises_with_nested_context_embedding", exporter_cls=EmbeddingExporter
            ),
            param(
                "raises_with_nested_context_prediction", exporter_cls=PredictionExporter
            ),
        ]
    )
    def test_raises_with_nested_context(self, _, exporter_cls: type[BaseGcsExporter]):
        exporter = exporter_cls(GcsUri("gs://test-bucket/test-folder"))
        with exporter:
            with self.assertRaises(RuntimeError):
                with exporter:
                    pass

        # Test can leave and re-enter.
        with exporter:
            pass

    @parameterized.expand(
        [
            param(
                "raises_with_negative_flush_threshold_embedding",
                exporter_cls=EmbeddingExporter,
            ),
            param(
                "raises_with_negative_flush_threshold_prediction",
                exporter_cls=PredictionExporter,
            ),
        ]
    )
    def test_file_flush_threshold_must_be_nonnegative(
        self, _, exporter_cls: type[BaseGcsExporter]
    ):
        with self.assertRaisesRegex(
            ValueError,
            "file_flush_threshold must be a non-negative integer, but got -1",
        ):
            exporter_cls(
                GcsUri("gs://test-bucket/test-folder"),
                min_shard_size_threshold_bytes=-1,
            )

    @parameterized.expand(
        [
            param(
                "no_prefix_embedding",
                exporter_cls=EmbeddingExporter,
                file_prefix=None,
                expected_file_name="shard_00000000.avro",
                id_batches=[torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])],
                data_batches=[
                    torch.tensor([[1, 11], [2, 12], [3, 13]]),
                    torch.tensor([[4, 14], [5, 15], [6, 16]]),
                ],
                expected_records=[
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
                ],
            ),
            param(
                "custom_prefix_embedding",
                exporter_cls=EmbeddingExporter,
                file_prefix="my-prefix",
                expected_file_name="my-prefix_00000000.avro",
                id_batches=[torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])],
                data_batches=[
                    torch.tensor([[1, 11], [2, 12], [3, 13]]),
                    torch.tensor([[4, 14], [5, 15], [6, 16]]),
                ],
                expected_records=[
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
                ],
            ),
            param(
                "no_prefix_prediction",
                exporter_cls=PredictionExporter,
                file_prefix=None,
                expected_file_name="shard_00000000.avro",
                id_batches=[torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])],
                data_batches=[
                    torch.tensor([0.1, 0.2, 0.3]),
                    torch.tensor([0.4, 0.5, 0.6]),
                ],
                expected_records=[
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
                    {
                        _NODE_ID_KEY: 3,
                        _NODE_TYPE_KEY: TEST_NODE_TYPE,
                        _PREDICTION_KEY: 0.3,
                    },
                    {
                        _NODE_ID_KEY: 4,
                        _NODE_TYPE_KEY: TEST_NODE_TYPE,
                        _PREDICTION_KEY: 0.4,
                    },
                    {
                        _NODE_ID_KEY: 5,
                        _NODE_TYPE_KEY: TEST_NODE_TYPE,
                        _PREDICTION_KEY: 0.5,
                    },
                    {
                        _NODE_ID_KEY: 6,
                        _NODE_TYPE_KEY: TEST_NODE_TYPE,
                        _PREDICTION_KEY: 0.6,
                    },
                ],
            ),
            param(
                "custom_prefix_prediction",
                exporter_cls=PredictionExporter,
                file_prefix="my-prefix",
                expected_file_name="my-prefix_00000000.avro",
                id_batches=[torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])],
                data_batches=[
                    torch.tensor([0.1, 0.2, 0.3]),
                    torch.tensor([0.4, 0.5, 0.6]),
                ],
                expected_records=[
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
                    {
                        _NODE_ID_KEY: 3,
                        _NODE_TYPE_KEY: TEST_NODE_TYPE,
                        _PREDICTION_KEY: 0.3,
                    },
                    {
                        _NODE_ID_KEY: 4,
                        _NODE_TYPE_KEY: TEST_NODE_TYPE,
                        _PREDICTION_KEY: 0.4,
                    },
                    {
                        _NODE_ID_KEY: 5,
                        _NODE_TYPE_KEY: TEST_NODE_TYPE,
                        _PREDICTION_KEY: 0.5,
                    },
                    {
                        _NODE_ID_KEY: 6,
                        _NODE_TYPE_KEY: TEST_NODE_TYPE,
                        _PREDICTION_KEY: 0.6,
                    },
                ],
            ),
        ]
    )
    def test_write_records_to_local(
        self,
        _,
        exporter_cls: type[BaseGcsExporter],
        file_prefix: Optional[str],
        expected_file_name: str,
        id_batches: list,
        data_batches: list,
        expected_records: list,
    ):
        test_file = self.local_export_dir / expected_file_name
        with exporter_cls(
            export_dir=LocalUri(self.local_export_dir), file_prefix=file_prefix
        ) as exporter:
            for id_batch, data_batch in zip(id_batches, data_batches):
                if isinstance(exporter, EmbeddingExporter):
                    exporter.add_embedding(id_batch, data_batch, TEST_NODE_TYPE)
                elif isinstance(exporter, PredictionExporter):
                    exporter.add_prediction(id_batch, data_batch, TEST_NODE_TYPE)
                else:
                    raise ValueError(f"Invalid exporter class: {type(exporter)}")

        avro_reader = fastavro.reader(test_file.open("rb"))
        records = list(avro_reader)

        # Use appropriate assertion method for predictions
        if isinstance(exporter, PredictionExporter):
            self.assertPredictionRecordsAlmostEqual(records, expected_records)
        else:
            self.assertEqual(records, expected_records)

    @parameterized.expand(
        [
            param(
                "multiple_flushes_embedding",
                exporter_cls=EmbeddingExporter,
                id_batches=[torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])],
                data_batches=[
                    torch.tensor([[1, 11], [2, 12], [3, 13]]),
                    torch.tensor([[4, 14], [5, 15], [6, 16]]),
                ],
                expected_records_by_batch=[
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
                ],
            ),
            param(
                "multiple_flushes_prediction",
                exporter_cls=PredictionExporter,
                id_batches=[torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])],
                data_batches=[
                    torch.tensor([0.1, 0.2, 0.3]),
                    torch.tensor([0.4, 0.5, 0.6]),
                ],
                expected_records_by_batch=[
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
                        {
                            _NODE_ID_KEY: 3,
                            _NODE_TYPE_KEY: TEST_NODE_TYPE,
                            _PREDICTION_KEY: 0.3,
                        },
                    ],
                    [
                        {
                            _NODE_ID_KEY: 4,
                            _NODE_TYPE_KEY: TEST_NODE_TYPE,
                            _PREDICTION_KEY: 0.4,
                        },
                        {
                            _NODE_ID_KEY: 5,
                            _NODE_TYPE_KEY: TEST_NODE_TYPE,
                            _PREDICTION_KEY: 0.5,
                        },
                        {
                            _NODE_ID_KEY: 6,
                            _NODE_TYPE_KEY: TEST_NODE_TYPE,
                            _PREDICTION_KEY: 0.6,
                        },
                    ],
                ],
            ),
        ]
    )
    def test_write_records_multiple_flushes(
        self,
        _,
        exporter_cls: type[BaseGcsExporter],
        id_batches: list,
        data_batches: list,
        expected_records_by_batch: list,
    ):
        # Write first batch using context manager
        id_data_batch_iter = zip(id_batches, data_batches)
        exporter = exporter_cls(export_dir=UriFactory.create_uri(self.local_export_dir))

        with exporter:
            id_batch, data_batch = next(id_data_batch_iter)
            if isinstance(exporter, EmbeddingExporter):
                exporter.add_embedding(id_batch, data_batch, TEST_NODE_TYPE)
            elif isinstance(exporter, PredictionExporter):
                exporter.add_prediction(id_batch, data_batch, TEST_NODE_TYPE)
            else:
                raise ValueError(f"Invalid exporter class: {type(exporter)}")

        # Write second batch with explicit flush
        id_batch, data_batch = next(id_data_batch_iter)
        if isinstance(exporter, EmbeddingExporter):
            exporter.add_embedding(id_batch, data_batch, TEST_NODE_TYPE)
            exporter.flush_embeddings()
        elif isinstance(exporter, PredictionExporter):
            exporter.add_prediction(id_batch, data_batch, TEST_NODE_TYPE)
            exporter.flush_predictions()
        else:
            raise ValueError(f"Invalid exporter class: {type(exporter)}")

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
        for i, record_file in enumerate(written_files):
            with self.subTest(f"Records for batch {i}"):
                reader = fastavro.reader(Path(record_file.uri).open("rb"))
                records = list(reader)
                # Use appropriate assertion method for predictions
                if isinstance(exporter, PredictionExporter):
                    self.assertPredictionRecordsAlmostEqual(
                        records, expected_records_by_batch[i]
                    )
                else:
                    self.assertEqual(records, expected_records_by_batch[i])

    @parameterized.expand(
        [
            param(
                "max_buffer_embedding",
                exporter_cls=EmbeddingExporter,
                id_batches=[torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])],
                data_batches=[
                    torch.tensor([[1, 11], [2, 12], [3, 13]]),
                    torch.tensor([[4, 14], [5, 15], [6, 16]]),
                ],
                expected_records_by_batch=[
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
                ],
            ),
            param(
                "max_buffer_prediction",
                exporter_cls=PredictionExporter,
                id_batches=[torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])],
                data_batches=[
                    torch.tensor([0.1, 0.2, 0.3]),
                    torch.tensor([0.4, 0.5, 0.6]),
                ],
                expected_records_by_batch=[
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
                        {
                            _NODE_ID_KEY: 3,
                            _NODE_TYPE_KEY: TEST_NODE_TYPE,
                            _PREDICTION_KEY: 0.3,
                        },
                    ],
                    [
                        {
                            _NODE_ID_KEY: 4,
                            _NODE_TYPE_KEY: TEST_NODE_TYPE,
                            _PREDICTION_KEY: 0.4,
                        },
                        {
                            _NODE_ID_KEY: 5,
                            _NODE_TYPE_KEY: TEST_NODE_TYPE,
                            _PREDICTION_KEY: 0.5,
                        },
                        {
                            _NODE_ID_KEY: 6,
                            _NODE_TYPE_KEY: TEST_NODE_TYPE,
                            _PREDICTION_KEY: 0.6,
                        },
                    ],
                ],
            ),
        ]
    )
    def test_flushes_after_maximum_buffer_size(
        self,
        _,
        exporter_cls: type[BaseGcsExporter],
        id_batches: list,
        data_batches: list,
        expected_records_by_batch: list,
    ):
        with exporter_cls(
            export_dir=UriFactory.create_uri(self.local_export_dir),
            min_shard_size_threshold_bytes=1,
        ) as exporter:
            for id_batch, data_batch in zip(id_batches, data_batches):
                if isinstance(exporter, EmbeddingExporter):
                    exporter.add_embedding(id_batch, data_batch, TEST_NODE_TYPE)
                elif isinstance(exporter, PredictionExporter):
                    exporter.add_prediction(id_batch, data_batch, TEST_NODE_TYPE)
                else:
                    raise ValueError(f"Invalid exporter class: {type(exporter)}")

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
        for i, record_file in enumerate(written_files):
            with self.subTest(f"Records for batch {i}"):
                reader = fastavro.reader(Path(record_file.uri).open("rb"))
                records = list(reader)
                # Use appropriate assertion method for predictions
                if isinstance(exporter, PredictionExporter):
                    self.assertPredictionRecordsAlmostEqual(
                        records, expected_records_by_batch[i]
                    )
                else:
                    self.assertEqual(records, expected_records_by_batch[i])

    @parameterized.expand(
        [
            param(
                "buffer_reset_embedding",
                EmbeddingExporter,
                torch.tensor([1, 2]),
                torch.tensor([[1, 11], [2, 12]]),
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
                ],
            ),
            param(
                "buffer_reset_prediction",
                PredictionExporter,
                torch.tensor([1, 2]),
                torch.tensor([0.1, 0.2]),
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
            ),
        ]
    )
    @patch("gigl.src.common.utils.file_loader.GcsUtils")
    def test_flush_resets_buffer(
        self,
        _,
        exporter_cls: type[BaseGcsExporter],
        id_batch: torch.Tensor,
        data_batch: torch.Tensor,
        expected_records: list,
        mock_gcs_utils_class,
    ):
        # Mock inputs
        gcs_base_uri = GcsUri("gs://test-bucket/test-folder")
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

        with exporter_cls(export_dir=gcs_base_uri) as exporter:
            if isinstance(exporter, EmbeddingExporter):
                exporter.add_embedding(id_batch, data_batch, TEST_NODE_TYPE)
            elif isinstance(exporter, PredictionExporter):
                exporter.add_prediction(id_batch, data_batch, TEST_NODE_TYPE)
            else:
                raise ValueError(f"Invalid exporter class: {type(exporter)}")

        avro_reader = fastavro.reader(test_file.open("rb"))
        records = list(avro_reader)
        # Use appropriate assertion method for predictions
        if isinstance(exporter, PredictionExporter):
            self.assertPredictionRecordsAlmostEqual(records, expected_records)
        else:
            self.assertEqual(records, expected_records)

    @parameterized.expand(
        [
            param(
                "retry_google_cloud_error_embedding",
                EmbeddingExporter,
                torch.tensor([1]),
                torch.tensor([[1, 11]]),
            ),
            param(
                "retry_google_cloud_error_prediction",
                PredictionExporter,
                torch.tensor([1]),
                torch.tensor([0.1]),
            ),
        ]
    )
    @patch("time.sleep")
    @patch("gigl.src.common.utils.file_loader.GcsUtils")
    def test_write_to_gcs_upload_retries_on_google_cloud_error_and_fails(
        self,
        _,
        exporter_cls: type[BaseGcsExporter],
        id_batch: torch.Tensor,
        data_batch: torch.Tensor,
        mock_gcs_utils_class,
        mock_sleep,
    ):
        # Mock inputs
        gcs_base_uri = GcsUri("gs://test-bucket/test-folder")

        mock_gcs_utils = MagicMock()
        google_cloud_error = GoogleCloudError("GCS upload failed")
        google_cloud_error.code = 503  # Service Unavailable
        mock_gcs_utils.upload_from_filelike.side_effect = google_cloud_error
        mock_gcs_utils_class.return_value = mock_gcs_utils

        exporter = exporter_cls(export_dir=gcs_base_uri)
        if isinstance(exporter, EmbeddingExporter):
            exporter.add_embedding(id_batch, data_batch, TEST_NODE_TYPE)
            flush_method = exporter.flush_embeddings
        elif isinstance(exporter, PredictionExporter):
            exporter.add_prediction(id_batch, data_batch, TEST_NODE_TYPE)
            flush_method = exporter.flush_predictions
        else:
            raise ValueError(f"Invalid exporter class: {type(exporter)}")

        # Assertions
        with self.assertRaisesRegex(RetriesFailedException, "GCS upload failed"):
            flush_method()
        self.assertEqual(mock_gcs_utils.upload_from_filelike.call_count, 6)

    @parameterized.expand(
        [
            param(
                "retry_request_exception_embedding",
                EmbeddingExporter,
                torch.tensor([1]),
                torch.tensor([[1, 11]]),
            ),
            param(
                "retry_request_exception_prediction",
                PredictionExporter,
                torch.tensor([1]),
                torch.tensor([0.1]),
            ),
        ]
    )
    @patch("time.sleep")
    @patch("gigl.src.common.utils.file_loader.GcsUtils")
    def test_write_to_gcs_upload_retries_on_request_exception_and_fails(
        self,
        _,
        exporter_cls: type[BaseGcsExporter],
        id_batch: torch.Tensor,
        data_batch: torch.Tensor,
        mock_gcs_utils_class,
        mock_sleep,
    ):
        # Constant message of request exception.
        CONNECTION_ABORTED_MESSAGE = "Connection aborted"

        # Mock inputs
        gcs_base_uri = GcsUri("gs://test-bucket/test-folder")

        mock_gcs_utils = MagicMock()
        mock_gcs_utils.upload_from_filelike.side_effect = (
            requests.exceptions.RequestException(CONNECTION_ABORTED_MESSAGE)
        )
        mock_gcs_utils_class.return_value = mock_gcs_utils

        exporter = exporter_cls(export_dir=gcs_base_uri)
        if isinstance(exporter, EmbeddingExporter):
            exporter.add_embedding(id_batch, data_batch, TEST_NODE_TYPE)
        elif isinstance(exporter, PredictionExporter):
            exporter.add_prediction(id_batch, data_batch, TEST_NODE_TYPE)
        else:
            raise ValueError(f"Invalid exporter class: {type(exporter)}")

        # Assertions
        with self.assertRaisesRegex(RetriesFailedException, CONNECTION_ABORTED_MESSAGE):
            if isinstance(exporter, EmbeddingExporter):
                exporter.flush_embeddings()
            elif isinstance(exporter, PredictionExporter):
                exporter.flush_predictions()
            else:
                raise ValueError(f"Invalid exporter class: {type(exporter)}")
        self.assertEqual(mock_gcs_utils.upload_from_filelike.call_count, 6)

    @parameterized.expand(
        [
            param("skip_flush_embedding", EmbeddingExporter),
            param("skip_flush_prediction", PredictionExporter),
        ]
    )
    @patch("gigl.src.common.utils.file_loader.GcsUtils")
    def test_skips_flush_if_empty(
        self, _, exporter_cls: type[BaseGcsExporter], mock_gcs_utils_class
    ):
        gcs_base_uri = GcsUri("gs://test-bucket/test-folder")

        mock_gcs_utils_class.return_value.upload_from_filelike.side_effect = ValueError(
            "Should not be uploading if not data!"
        )
        exporter = exporter_cls(export_dir=gcs_base_uri)

        if isinstance(exporter, EmbeddingExporter):
            exporter.flush_embeddings()
        elif isinstance(exporter, PredictionExporter):
            exporter.flush_predictions()
        else:
            raise ValueError(f"Invalid exporter class: {type(exporter)}")

    @parameterized.expand(
        [
            param(
                "load_embedding_sync",
                load_embeddings_to_bigquery,
                False,
            ),
            param(
                "load_embedding_async",
                load_embeddings_to_bigquery,
                True,
            ),
            param(
                "load_prediction_sync",
                load_predictions_to_bigquery,
                False,
            ),
            param(
                "load_prediction_async",
                load_predictions_to_bigquery,
                True,
            ),
        ]
    )
    @patch("gigl.common.data.export.bigquery.Client")
    def test_load_to_bigquery(
        self,
        _,
        load_function: Callable[[GcsUri, str, str, str, bool], LoadJob],
        should_run_async: bool,
        mock_bigquery_client,
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
        load_job = load_function(
            gcs_folder,
            project_id,
            dataset_id,
            table_id,
            should_run_async,
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
    unittest.main()
