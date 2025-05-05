"""Utility functions for exporting embeddings to Google Cloud Storage and BigQuery.

Note that we use avro files here since due to testing they are quicker to generate and upload
compared to parquet files.

However, if we switch to an on-line upload scheme, where we upload the embeddings as they are generated,
then we should look into if parquet or orc files are more performant in that modality.
"""
import io
import os
import time
from typing import Final, Optional, Sequence

import fastavro
import fastavro.types
import requests
import torch
from google.cloud import bigquery
from google.cloud.bigquery.job import LoadJob
from google.cloud.exceptions import GoogleCloudError
from typing_extensions import Self

from gigl.common import GcsUri
from gigl.common.logger import Logger
from gigl.common.utils.gcs import GcsUtils
from gigl.common.utils.retry import retry

logger = Logger()

# Shared key names between Avro and BigQuery schemas.
_NODE_ID_KEY: Final[str] = "node_id"
_EMBEDDING_TYPE_KEY: Final[str] = "node_type"
_EMBEDDING_KEY: Final[str] = "emb"

# AVRO schema for embedding records.
AVRO_SCHEMA: Final[fastavro.types.Schema] = {
    "type": "record",
    "name": "Embedding",
    "fields": [
        {"name": _NODE_ID_KEY, "type": "long"},
        {"name": _EMBEDDING_TYPE_KEY, "type": "string"},
        {"name": _EMBEDDING_KEY, "type": {"type": "array", "items": "float"}},
    ],
}

# BigQuery schema for embedding records.
BIGQUERY_SCHEMA: Final[Sequence[bigquery.SchemaField]] = [
    bigquery.SchemaField(_NODE_ID_KEY, "INTEGER"),
    bigquery.SchemaField(_EMBEDDING_TYPE_KEY, "STRING"),
    bigquery.SchemaField(_EMBEDDING_KEY, "FLOAT64", mode="REPEATED"),
]


class EmbeddingExporter:
    def __init__(
        self,
        export_dir: GcsUri,
        file_prefix: Optional[str] = None,
        min_shard_size_threshold_bytes: int = 0,
    ):
        """
        Initializes an EmbeddingExporter instance.

        Note that after every flush, either via exiting a context manager, by calling `flush_embeddings()`,
        or when the buffer reaches the `file_flush_threshold`, a new avro file will be created, and
        subsequent calls to `add_embedding` will add to the new file. This means that after all
        embeddings have been added the `export_dir` may look like the below:

        gs://my_bucket/embeddings/
        ├── shard_00000000.avro
        ├── shard_00000001.avro
        └── shard_00000002.avro

        Args:
            export_dir (GcsUri): The Google Cloud Storage URI where the Avro files will be uploaded.
                                 This should be a fully qualified GCS path, e.g., 'gs://bucket_name/path/to/'.
            file_prefix (Optional[str]): An optional prefix to add to the file name. If provided then the
                                         the file names will be like $file_prefix_shard_00000000.avro.
            min_shard_size_threshold_bytes (int): The minimum size in bytes at which the buffer will be flushed to GCS.
                                        The buffer will contain the entire batch of embeddings that caused it to
                                        reach the threshold, so the file sizes on GCS may be larger than this value.
                                        If set to zero, the default, then the buffer will be flushed only when
                                        `flush_embeddings` is called or when the context manager is exited.
                                        An error will be thrown if this value is negative.
                                        Note that for the *last* shared, the buffer may be much smaller than this limit.
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
        self._base_gcs_uri = export_dir
        self._write_time = 0.0
        self._gcs_utils = GcsUtils()
        self._prefix = file_prefix
        self._min_shard_size_threshold_bytes = min_shard_size_threshold_bytes

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
        start = time.perf_counter()
        # Convert torch tensors to NumPy arrays, and then to Python int(s)
        # and Python list(s). This is faster than converting torch tensors
        # directly to Python int(s) and Python list(s), as Numpy's implementation
        # is more efficient.
        ids = id_batch.numpy()
        embeddings = embedding_batch.numpy()
        self._num_records_written += len(ids)

        batched_records = (
            {
                "node_id": int(node_id),
                "node_type": embedding_type,
                "emb": embedding.tolist(),
            }
            for node_id, embedding in zip(ids, embeddings)
        )

        # fastavro.writer can accept the generator directly.
        # Doing this appends to self._buffer, so in order to *read* all data from the buffer
        # we *must* call self._buffer.seek(0) before reading.
        fastavro.writer(self._buffer, AVRO_SCHEMA, batched_records)
        self._write_time += time.perf_counter() - start

        if (
            self._min_shard_size_threshold_bytes
            and self._buffer.tell() >= self._min_shard_size_threshold_bytes
        ):
            logger.info(
                f"Flushing buffer due to the buffer size ({self._buffer.tell():,} bytes) exceeding the threshold ({self._min_shard_size_threshold_bytes:,} bytes)."
            )
            self.flush_embeddings()

    @retry(
        exception_to_check=(GoogleCloudError, requests.exceptions.RequestException),
        tries=5,
        max_delay_s=60,
    )
    def _flush(self):
        """Flushes the in-memory buffer to GCS, retrying on failure."""
        logger.info(
            f"Building and writing {self._num_records_written:,} records into in-memory buffer took {self._write_time:.2f} seconds"
        )
        buff_size = self._buffer.tell()

        start = time.perf_counter()
        # Reset the buffer to the beginning so we upload all records.
        # This is needed in both the happy path - writing we want to dump all the buffer to GCS,
        # and in the error case where we uploaded *some* of the buffer and then failed.
        self._buffer.seek(0)
        filename = (
            f"shard_{self._num_files_written:08}.avro"
            if not self._prefix
            else f"{self._prefix}_{self._num_files_written:08}.avro"
        )
        self._gcs_utils.upload_from_filelike(
            GcsUri.join(self._base_gcs_uri, filename),
            self._buffer,
            content_type="application/octet-stream",
        )
        self._num_files_written += 1
        # BytesIO.close() frees up memory used by the buffer.
        # https://docs.python.org/3/library/io.html#io.BytesIO
        # This also happens when the object is gc'd e.g. self._buffer = io.BytesIO()
        # but we prefer to be explicit here in case there are other references.
        self._buffer.close()
        self._buffer = io.BytesIO()

        logger.info(
            f"Upload the {buff_size:,} bytes Avro data to GCS took {time.perf_counter()- start:.2f} seconds"
        )
        self._num_records_written = 0
        self._write_time = 0.0

    def flush_embeddings(self):
        """Flushes the in-memory buffer to GCS.

        After this method is called, the buffer is reset to an empty state.
        """
        if self._buffer.tell() == 0:
            logger.info("No embeddings to flush, will skip upload.")
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
        self.flush_embeddings()
        self._in_context = False


# TODO(kmonte): We should migrate this over to `BqUtils.load_files_to_bq` once that is implemented.
def load_embeddings_to_bigquery(
    gcs_folder: GcsUri, project_id: str, dataset_id: str, table_id: str
) -> LoadJob:
    """
    Loads multiple Avro files containing GNN embeddings from GCS into BigQuery.

    Note that this function will upload *all* Avro files in the GCS folder to BigQuery, recursively.
    So if we have some nested directories, e.g.:

    gs://MY BUCKET/embeddings/shard_0000.avro
    gs://MY BUCKET/embeddings/nested/shard_0001.avro

    Both files will be uploaded to BigQuery.

    Args:
        gcs_folder (GcsUri): The GCS folder containing the Avro files with embeddings.
        project_id (str): The GCP project ID.
        dataset_id (str): The BigQuery dataset ID.
        table_id (str): The BigQuery table ID.

    Returns:
        LoadJob: A BigQuery LoadJob object representing the load operation, which allows
        user to monitor and retrieve details about the job status and result.
    """
    start = time.perf_counter()
    logger.info(f"Loading embeddings from {gcs_folder} to BigQuery.")
    # Initialize the BigQuery client
    bigquery_client = bigquery.Client(project=project_id)

    # Construct dataset and table references
    dataset_ref = bigquery_client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id)

    # Configure the load job
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.AVRO,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,  # Use WRITE_APPEND to append data
        schema=BIGQUERY_SCHEMA,
    )

    load_job = bigquery_client.load_table_from_uri(
        source_uris=os.path.join(gcs_folder.uri, "*.avro"),
        destination=table_ref,
        job_config=job_config,
    )

    load_job.result()  # Wait for the job to complete.
    logger.info(
        f"Loading {load_job.output_rows:,} rows into {dataset_id}:{table_id} in {time.perf_counter() - start:.2f} seconds."
    )

    return load_job
