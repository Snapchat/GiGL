"""Utility functions for exporting embeddings and predictions to Google Cloud Storage and BigQuery.

Note that we use avro files here since due to testing they are quicker to generate and upload
compared to parquet files.

However, if we switch to an on-line upload scheme, where we upload the embeddings as they are generated,
then we should look into if parquet or orc files are more performant in that modality.
"""
import io
import os
import time
from pathlib import Path
from typing import Final, Iterable, Optional, Sequence

import fastavro
import fastavro.types
import requests
import torch
from google.cloud import bigquery
from google.cloud.bigquery.job import LoadJob
from google.cloud.exceptions import GoogleCloudError
from typing_extensions import Self

from gigl.common import GcsUri, LocalUri, Uri
from gigl.common.logger import Logger
from gigl.common.utils.retry import retry
from gigl.src.common.utils.file_loader import FileLoader

logger = Logger()

# Shared key names between Avro and BigQuery schemas.
_NODE_ID_KEY: Final[str] = "node_id"
_NODE_TYPE_KEY: Final[str] = "node_type"
_EMBEDDING_KEY: Final[str] = "emb"
_PREDICTION_KEY: Final[str] = "pred"

# AVRO schema for embedding records.
EMBEDDING_AVRO_SCHEMA: Final[fastavro.types.Schema] = {
    "type": "record",
    "name": "Embedding",
    "fields": [
        {"name": _NODE_ID_KEY, "type": "long"},
        {"name": _NODE_TYPE_KEY, "type": "string"},
        {"name": _EMBEDDING_KEY, "type": {"type": "array", "items": "float"}},
    ],
}

# BigQuery schema for embedding records.
EMBEDDING_BIGQUERY_SCHEMA: Final[Sequence[bigquery.SchemaField]] = [
    bigquery.SchemaField(_NODE_ID_KEY, "INT64"),
    bigquery.SchemaField(_NODE_TYPE_KEY, "STRING"),
    bigquery.SchemaField(_EMBEDDING_KEY, "FLOAT64", mode="REPEATED"),
]

PREDICTION_AVRO_SCHEMA: Final[fastavro.types.Schema] = {
    "type": "record",
    "name": "Prediction",
    "fields": [
        {"name": _NODE_ID_KEY, "type": "long"},
        {"name": _NODE_TYPE_KEY, "type": "string"},
        {"name": _PREDICTION_KEY, "type": "float"},
    ],
}

# BigQuery schema for prediction records.
PREDICTION_BIGQUERY_SCHEMA: Final[Sequence[bigquery.SchemaField]] = [
    bigquery.SchemaField(_NODE_ID_KEY, "INT64"),
    bigquery.SchemaField(_NODE_TYPE_KEY, "STRING"),
    bigquery.SchemaField(_PREDICTION_KEY, "FLOAT64"),
]


class GcsExporter:
    def __init__(
        self,
        export_dir: Uri,
        avro_schema: fastavro.types.Schema,
        file_prefix: Optional[str] = None,
        min_shard_size_threshold_bytes: int = 0,
    ):
        """
        Initializes a BaseGcsExporter instance.

        Note that after every flush, either via exiting a context manager, by calling `flush_records()`,
        or when the buffer reaches the `file_flush_threshold`, a new avro file will be created, and
        subsequent calls to `add_record` will add to the new file. This means that after all
        records have been added the `export_dir` may look like the below:

        gs://my_bucket/records/
        ├── shard_00000000.avro
        ├── shard_00000001.avro
        └── shard_00000002.avro
        Args:
            export_dir (Uri): URI where the Avro files will be uploaded.
                                 If a GCS URI, this should be a fully qualified GCS path,
                                 e.g., 'gs://bucket_name/path/to/'.
                                 If a local URI (e.g. /tmp/gigl/records), then the directory
                                 will be created when GcsExporter is initialized.
            file_prefix (Optional[str]): An optional prefix to add to the file name. If provided then the
                                         the file names will be like $file_prefix_shard_00000000.avro.
            min_shard_size_threshold_bytes (int): The minimum size in bytes at which the buffer will be flushed to GCS.
                                        The buffer will contain the entire batch of records that caused it to
                                        reach the threshold, so the file sizes on GCS may be larger than this value.
                                        If set to zero, the default, then the buffer will be flushed only when
                                        `flush_records` is called or when the context manager is exited.
                                        An error will be thrown if this value is negative.
                                        Note that for the *last* shard, the buffer may be much smaller than this limit.
        """

        if min_shard_size_threshold_bytes < 0:
            raise ValueError(
                f"file_flush_threshold must be a non-negative integer, but got {min_shard_size_threshold_bytes}"
            )

        # TODO(kmonte): We may also want to support writing to disk instead of in-memory.
        self._buffer = io.BytesIO()
        self._num_records_written = 0
        self._num_files_written = 0
        self._in_context = False
        self._context_start_time = 0.0
        self._base_export_uri = export_dir
        self._write_time = 0.0
        self._file_utils = FileLoader()
        self._prefix = file_prefix
        self._min_shard_size_threshold_bytes = min_shard_size_threshold_bytes
        self._avro_schema = avro_schema

        if isinstance(
            self._base_export_uri, LocalUri
        ) and not self._file_utils.does_uri_exist(self._base_export_uri):
            logger.info(
                f"Creating local directory {self._base_export_uri.uri} for exporting records."
            )
            Path(self._base_export_uri.uri).mkdir(parents=True, exist_ok=True)

    def add_record(
        self,
        records: Iterable[dict],
    ):
        """
        Adds to the in-memory buffer the records.

        Args:
            records (Iterable[dict]): An iterable of dictionaries containing the records.
        """

        start = time.perf_counter()
        fastavro.writer(self._buffer, self._avro_schema, records)
        self._write_time += time.perf_counter() - start

        if (
            self._min_shard_size_threshold_bytes
            and self._buffer.tell() >= self._min_shard_size_threshold_bytes
        ):
            logger.info(
                f"Flushing buffer due to the buffer size ({self._buffer.tell():,} bytes) exceeding the threshold ({self._min_shard_size_threshold_bytes:,} bytes)."
            )
            self.flush_records()

    @retry(
        exception_to_check=(GoogleCloudError, requests.exceptions.RequestException),
        tries=5,
        max_delay_s=60,
    )
    def _flush(self):
        """Flushes the in-memory buffer, retrying on failure."""
        logger.info(
            f"Building and writing {self._num_records_written:,} records into in-memory buffer took {self._write_time:.2f} seconds"
        )
        buff_size = self._buffer.tell()

        start = time.perf_counter()
        # Reset the buffer to the beginning so we upload all records.
        # This is needed in both the happy path - writing we want to dump all the buffer,
        # and in the error case where we uploaded *some* of the buffer and then failed.
        self._buffer.seek(0)
        filename = (
            f"shard_{self._num_files_written:08}.avro"
            if not self._prefix
            else f"{self._prefix}_{self._num_files_written:08}.avro"
        )
        fname = self._base_export_uri.join(self._base_export_uri, filename)
        self._file_utils.load_from_filelike(
            fname,
            self._buffer,
        )
        self._num_files_written += 1
        # BytesIO.close() frees up memory used by the buffer.
        # https://docs.python.org/3/library/io.html#io.BytesIO
        # This also happens when the object is gc'd e.g. self._buffer = io.BytesIO()
        # but we prefer to be explicit here in case there are other references.
        self._buffer.close()
        self._buffer = io.BytesIO()

        logger.info(
            f"Upload the {buff_size:,} bytes Avro data took {time.perf_counter()- start:.2f} seconds"
        )
        self._num_records_written = 0
        self._write_time = 0.0

    def flush_records(self):
        """Flushes the in-memory buffer.

        After this method is called, the buffer is reset to an empty state.
        """
        if self._buffer.tell() == 0:
            logger.info("No records to flush, will skip upload.")
            return
        self._flush()

    def __enter__(self) -> Self:
        if self._in_context:
            raise RuntimeError(
                f"{type(self).__name__} is already in a context. Do not call `with {type(self).__name__}:` in a nested manner."
            )
        self._in_context = True

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.flush_records()
        self._in_context = False


class EmbeddingExporter(GcsExporter):
    def __init__(
        self,
        export_dir: Uri,
        file_prefix: Optional[str] = None,
        min_shard_size_threshold_bytes: int = 0,
    ):
        """
        Initializes an EmbeddingExporter instance, which will write embeddings to gcs with an embedding avro schema for
        writing an array of floats per record.

        Args:
            export_dir (Uri): URI where the Avro files will be uploaded.
            file_prefix (Optional[str]): An optional prefix to add to the file name. If provided then the
            min_shard_size_threshold_bytes (int): The minimum size in bytes at which the buffer will be flushed to GCS.
        """
        super().__init__(
            export_dir,
            EMBEDDING_AVRO_SCHEMA,
            file_prefix,
            min_shard_size_threshold_bytes,
        )

    def add_embedding(
        self,
        id_batch: torch.Tensor,
        embedding_batch: torch.Tensor,
        embedding_type: str,
    ):
        """
        Adds to the in-memory buffer the integer IDs and their corresponding embeddings.

        Args:
            id_batch (torch.Tensor): A torch.Tensor containing integer IDs.
            embedding_batch (torch.Tensor): A torch.Tensor containing embeddings corresponding to the integer IDs in `id_batch`.
            embedding_type (str): A tag for the type of the embeddings, e.g., 'user', 'content', etc.
        """
        # Convert torch tensors to NumPy arrays, and then to Python int(s)
        # and Python list(s). This is faster than converting torch tensors
        # directly to Python int(s) and Python list(s), as Numpy's implementation
        # is more efficient.
        ids = id_batch.numpy()
        embeddings = embedding_batch.numpy()

        self._num_records_written += len(ids)

        batched_records = (
            {
                _NODE_ID_KEY: int(node_id),
                _NODE_TYPE_KEY: embedding_type,
                _EMBEDDING_KEY: embedding.tolist(),
            }
            for node_id, embedding in zip(ids, embeddings)
        )

        self.add_record(batched_records)

    def flush_embeddings(self):
        """
        NOTE: This method is deprecated, and the `flush_records` method should be used instead.
        This method will be removed in a future version.
        """
        logger.warning(
            "flush_embeddings() is deprecated, and the `flush_records` method should be used instead. This method will be removed in a future version."
        )
        self.flush_records()


class PredictionExporter(GcsExporter):
    def __init__(
        self,
        export_dir: Uri,
        file_prefix: Optional[str] = None,
        min_shard_size_threshold_bytes: int = 0,
    ):
        """
        Initializes a PredictionExporter instance, which will write predictions to gcs with a prediction avro schema for
        writing a single float per record.

        Args:
            export_dir (Uri): URI where the Avro files will be uploaded.
            file_prefix (Optional[str]): An optional prefix to add to the file name. If provided then the
            min_shard_size_threshold_bytes (int): The minimum size in bytes at which the buffer will be flushed to GCS.
        """

        super().__init__(
            export_dir,
            PREDICTION_AVRO_SCHEMA,
            file_prefix,
            min_shard_size_threshold_bytes,
        )

    def add_prediction(
        self,
        id_batch: torch.Tensor,
        prediction_batch: torch.Tensor,
        prediction_type: str,
    ):
        """
        Adds to the in-memory buffer the integer IDs and their corresponding predictions.

        Args:
            id_batch (torch.Tensor): A torch.Tensor containing integer IDs.
            prediction_batch (torch.Tensor): A torch.Tensor containing predictions corresponding to the integer IDs in `id_batch`.
            prediction_type (str): A tag for the type of the predictions, e.g., 'user', 'content', etc.
        """
        # Convert torch tensors to NumPy arrays, and then to Python int(s)
        # and Python list(s). This is faster than converting torch tensors
        # directly to Python int(s) and Python list(s), as Numpy's implementation
        # is more efficient.
        ids = id_batch.numpy()
        predictions = prediction_batch.numpy()

        self._num_records_written += len(ids)

        batched_records = (
            {
                _NODE_ID_KEY: int(node_id),
                _NODE_TYPE_KEY: prediction_type,
                _PREDICTION_KEY: float(prediction),
            }
            for node_id, prediction in zip(ids, predictions)
        )

        self.add_record(batched_records)


# TODO(kmonte): We should migrate this over to `BqUtils.load_files_to_bq` once that is implemented.
def _load_records_to_bigquery(
    gcs_folder: GcsUri,
    project_id: str,
    dataset_id: str,
    table_id: str,
    schema: Sequence[bigquery.SchemaField],
    should_run_async: bool = False,
) -> LoadJob:
    """
    Loads multiple Avro files containing GNN records from GCS into BigQuery.

    Args:
        gcs_folder (GcsUri): The GCS folder containing the Avro files with records.
        project_id (str): The GCP project ID.
        dataset_id (str): The BigQuery dataset ID.
        table_id (str): The BigQuery table ID.
        schema (Sequence[bigquery.SchemaField]): The BigQuery schema for the records.
        should_run_async (bool): Whether loading to bigquery step should happen asynchronously. Defaults to False.

    Returns:
        LoadJob: A BigQuery LoadJob object representing the load operation, which allows
        user to monitor and retrieve details about the job status and result. The returned job will be done if
        `should_run_async=False` and will be returned immediately after creation (not necessarily complete) if
        `should_run_asnyc=True`.
    """
    start = time.perf_counter()
    logger.info(f"Loading records from {gcs_folder} to BigQuery.")
    # Initialize the BigQuery client
    bigquery_client = bigquery.Client(project=project_id)

    # Construct dataset and table references
    dataset_ref = bigquery_client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id)

    # Configure the load job
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.AVRO,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,  # Use WRITE_APPEND to append data
        schema=schema,
    )

    load_job = bigquery_client.load_table_from_uri(
        source_uris=os.path.join(gcs_folder.uri, "*.avro"),
        destination=table_ref,
        job_config=job_config,
    )

    if should_run_async:
        logger.info(
            f"Started loading process for {dataset_id}:{table_id} with job id {load_job.job_id}, running asynchronously"
        )
    else:
        load_job.result()  # Wait for the job to complete.
        logger.info(
            f"Loading {load_job.output_rows:,} rows into {dataset_id}:{table_id} in {time.perf_counter() - start:.2f} seconds."
        )

    return load_job


def load_embeddings_to_bigquery(
    gcs_folder: GcsUri,
    project_id: str,
    dataset_id: str,
    table_id: str,
    should_run_async: bool = False,
) -> LoadJob:
    """
    Loads multiple Avro files containing GNN embeddings from GCS into BigQuery.

    Note that this function will upload *all* Avro files in the GCS folder to BigQuery, recursively.
    So if you specify gcs_folder to be `gs://MY BUCKET/embeddings/` and if we have some nested directories, e.g.:

    gs://MY BUCKET/embeddings/shard_0000.avro
    gs://MY BUCKET/embeddings/nested/shard_0001.avro

    Both files will be uploaded to BigQuery.

    Args:
        gcs_folder (GcsUri): The GCS folder containing the Avro files with embeddings.
        project_id (str): The GCP project ID.
        dataset_id (str): The BigQuery dataset ID.
        table_id (str): The BigQuery table ID.
        should_run_async (bool): Whether loading to bigquery step should happen asynchronously. Defaults to False.

    Returns:
        LoadJob: A BigQuery LoadJob object representing the load operation, which allows
        user to monitor and retrieve details about the job status and result. The returned job will be done if
        `should_run_async=False` and will be returned immediately after creation (not necessarily complete) if
        `should_run_asnyc=True`.
    """
    return _load_records_to_bigquery(
        gcs_folder,
        project_id,
        dataset_id,
        table_id,
        EMBEDDING_BIGQUERY_SCHEMA,
        should_run_async,
    )


def load_predictions_to_bigquery(
    gcs_folder: GcsUri,
    project_id: str,
    dataset_id: str,
    table_id: str,
    should_run_async: bool = False,
) -> LoadJob:
    """
    Loads multiple Avro files containing GNN predictions from GCS into BigQuery.

    Note that this function will upload *all* Avro files in the GCS folder to BigQuery, recursively.
    So if you specify gcs_folder to be `gs://MY BUCKET/predictions/` and if we have some nested directories, e.g.:

    gs://MY BUCKET/predictions/shard_0000.avro
    gs://MY BUCKET/predictions/nested/shard_0001.avro

    Both files will be uploaded to BigQuery.

    Args:
        gcs_folder (GcsUri): The GCS folder containing the Avro files with predictions.
        project_id (str): The GCP project ID.
        dataset_id (str): The BigQuery dataset ID.
        table_id (str): The BigQuery table ID.
        should_run_async (bool): Whether loading to bigquery step should happen asynchronously. Defaults to False.

    Returns:
        LoadJob: A BigQuery LoadJob object representing the load operation, which allows
        user to monitor and retrieve details about the job status and result. The returned job will be done if
        `should_run_async=False` and will be returned immediately after creation (not necessarily complete) if
        `should_run_asnyc=True`.
    """
    return _load_records_to_bigquery(
        gcs_folder,
        project_id,
        dataset_id,
        table_id,
        PREDICTION_BIGQUERY_SCHEMA,
        should_run_async,
    )
