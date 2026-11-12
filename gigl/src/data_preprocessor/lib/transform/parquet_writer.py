"""Writes TFT output record batches as dense Parquet files.

Nodes are stored as ``node_id`` plus an optional ``features`` column, edges as ``src`` and ``dst`` plus an optional
``features`` column. ``features`` is a ``FixedSizeList<float32>`` whose slots follow ``feature_keys`` order, with every
feature cast to float32 and vectors flattened, matching the tensors the TFRecord loader builds. Each label key is its own
column (float32 for scalar labels, ``FixedSizeList<float32>`` for vector labels). Quantized features are stored as one
``FixedSizeBinary`` column of packed bytes per row, named after the packed feature key recorded in the quantization
metadata.

Files are named ``<parquet_uri_prefix>part-XXXXX-of-YYYYY.parquet``, for example::

    gs://my-bucket/user/part-00000-of-00004.parquet
    gs://my-bucket/user/part-00001-of-00004.parquet
    gs://my-bucket/user/part-00002-of-00004.parquet
    gs://my-bucket/user/part-00003-of-00004.parquet

A node entity with three features (one scalar, one 3-vector, one scalar), a packed column and a label has the schema::

    node_id: int64
    features: fixed_size_list<item: float>[5]
    node_packed_features: fixed_size_binary[16]
    label: float

An edge entity with features and no label has the schema::

    src: int64
    dst: int64
    features: fixed_size_list<item: float>[5]

A file can be read back into a ``[rows, width]`` numpy matrix with pyarrow:

    >>> table = pq.read_table("gs://my-bucket/user/part-00000-of-00004.parquet")
    >>> features = table.column("features").combine_chunks().flatten().to_numpy().reshape(table.num_rows, -1)
    >>> features.shape
    (100000, 5)
"""

import os
from typing import BinaryIO, Final, Iterator, Optional, Sequence

import apache_beam as beam
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from apache_beam.io import filebasedsink
from apache_beam.io.filesystem import CompressionTypes
from apache_beam.io.filesystems import FileSystems

from gigl.src.data_preprocessor.lib.transform.arrow_utils import list_column_to_numpy

NODE_ID_COLUMN: Final[str] = "node_id"
EDGE_SRC_COLUMN: Final[str] = "src"
EDGE_DST_COLUMN: Final[str] = "dst"
FEATURES_COLUMN: Final[str] = "features"

PARQUET_FILE_NAME_SUFFIX: Final[str] = ".parquet"
PARQUET_FILE_PREFIX: Final[str] = "part"
PARQUET_CODEC: Final[str] = "zstd"
# Ids and float features are close to unique, so dictionary encoding is wasted work that falls back to plain encoding.
PARQUET_USE_DICTIONARY: Final[bool] = False
NODE_ROW_GROUP_ROWS: Final[int] = 100_000
EDGE_ROW_GROUP_ROWS: Final[int] = 1_000_000
# The reader splits work by row group, not by file, so bigger files keep the same read parallelism while cutting the
# number of files to open and list. Files also end when a Dataflow bundle ends, so many are smaller than this.
TARGET_BYTES_PER_PARQUET_FILE: Final[int] = 1_000_000_000


def _fixed_size_list(values: np.ndarray) -> pa.Array:
    return pa.FixedSizeListArray.from_arrays(
        pa.array(np.ascontiguousarray(values).reshape(-1)), values.shape[1]
    )


def _fixed_size_binary(column: pa.Array, num_rows: int, width: int) -> pa.Array:
    """Returns a one-binary-value-per-row list column as a ``FixedSizeBinary(width)`` array.

    Arrow cannot cast variable-length binary to fixed-size binary, so the value bytes are reused directly.

    Args:
        column: A list column holding one binary value per row. May be a slice of a larger array.
        num_rows: Number of rows in the column.
        width: Number of bytes in every binary value.

    Returns:
        A ``FixedSizeBinary(width)`` array of ``num_rows`` rows sharing the column's value bytes.

    Raises:
        ValueError: If any binary value is not ``width`` bytes long.
    """
    values = column.flatten()
    offsets = np.frombuffer(values.buffers()[1], dtype=np.int32)
    offsets = offsets[values.offset : values.offset + num_rows + 1]
    if not np.array_equal(np.diff(offsets), np.full(num_rows, width)):
        raise ValueError(f"Packed feature values must all be {width} bytes long.")
    # A binary array's buffers are [validity bitmap, int32 offsets, value bytes]. The rows' bytes are contiguous and the
    # first row starts at offsets[0], which is non-zero when the column is a slice. Every row was checked to be
    # ``width`` long, so the next num_rows * width bytes are exactly the packed rows and are reused without copying.
    data = values.buffers()[2].slice(int(offsets[0]), num_rows * width)
    return pa.FixedSizeBinaryArray.from_buffers(
        pa.binary(width), num_rows, [None, data]
    )


class RecordBatchToDenseArrowTable(beam.DoFn):
    """Converts a TFT ``pa.RecordBatch`` into a ``pa.Table`` with the dense Parquet schema."""

    def __init__(
        self,
        id_keys: Sequence[str],
        id_columns: Sequence[str],
        feature_keys: Sequence[str],
        label_keys: Sequence[str],
        packed_feature_key: Optional[str] = None,
        packed_feature_dim: int = 0,
    ):
        """Initializes the converter.

        Args:
            id_keys: TFT output names of the id columns, in output order (``[node_id_key]`` or
                ``[src_key, dst_key]``).
            id_columns: Parquet column names for ``id_keys``.
            feature_keys: TFT output names concatenated, in order, into the ``features`` column.
            label_keys: TFT output names written as one float32 column each.
            packed_feature_key: TFT output name of the packed quantized bytes, written as a ``FixedSizeBinary`` column
                of the same name. ``None`` when features are not quantized.
            packed_feature_dim: Number of packed bytes per row. Required when ``packed_feature_key`` is set.
        """
        self._id_keys = list(id_keys)
        self._id_columns = list(id_columns)
        self._feature_keys = list(feature_keys)
        self._label_keys = list(label_keys)
        self._packed_feature_key = packed_feature_key
        self._packed_feature_dim = packed_feature_dim

    def process(self, batch: pa.RecordBatch) -> Iterator[pa.Table]:
        """Converts one record batch.

        Args:
            batch: A TFT output batch in which every column holds one list (or one scalar) per row.

        Yields:
            One table with the id, ``features``, packed and label columns. Nothing for an empty batch.

        Raises:
            ValueError: If a column holds nulls or a packed value is not ``packed_feature_dim`` bytes long.
        """
        num_rows = batch.num_rows
        if num_rows == 0:
            return
        names: list[str] = []
        arrays: list[pa.Array] = []
        for key, column_name in zip(self._id_keys, self._id_columns):
            ids = list_column_to_numpy(
                batch.column(batch.schema.get_field_index(key)), num_rows
            )
            names.append(column_name)
            arrays.append(pa.array(ids.reshape(-1).astype(np.int64, copy=False)))
        if self._feature_keys:
            feature_views: list[np.ndarray] = []
            total_width = 0
            for key in self._feature_keys:
                view = list_column_to_numpy(
                    batch.column(batch.schema.get_field_index(key)), num_rows
                )
                feature_views.append(view)
                total_width += view.shape[1]
            features = np.empty((num_rows, total_width), dtype=np.float32)
            start = 0
            for view in feature_views:
                end = start + view.shape[1]
                features[:, start:end] = view
                start = end
            names.append(FEATURES_COLUMN)
            arrays.append(
                pa.FixedSizeListArray.from_arrays(
                    pa.array(features.reshape(-1)), total_width
                )
            )
        if self._packed_feature_key is not None:
            packed = batch.column(
                batch.schema.get_field_index(self._packed_feature_key)
            )
            names.append(self._packed_feature_key)
            arrays.append(
                _fixed_size_binary(packed, num_rows, self._packed_feature_dim)
            )
        for key in self._label_keys:
            labels = list_column_to_numpy(
                batch.column(batch.schema.get_field_index(key)), num_rows
            ).astype(np.float32)
            names.append(key)
            if labels.shape[1] == 1:
                arrays.append(pa.array(labels.reshape(-1)))
            else:
                arrays.append(_fixed_size_list(labels))
        yield pa.Table.from_arrays(arrays, names=names)


class _BufferedParquetWriter:
    """Accumulates tables and writes row groups of ``row_group_rows`` rows.

    The Parquet schema is taken from the first table, so the feature width does not have to be known when the pipeline
    is built. A file that receives no rows still gets a valid, empty Parquet file containing only the id columns.
    """

    def __init__(
        self,
        file_handle: BinaryIO,
        codec: str,
        row_group_rows: int,
        empty_schema: pa.Schema,
    ):
        """Initializes the writer.

        Args:
            file_handle: Open binary handle the Parquet file is written to. Closed by ``close``.
            codec: Parquet compression codec.
            row_group_rows: Number of rows in every row group except the last.
            empty_schema: Schema of the file when no rows were written.
        """
        self._file_handle = file_handle
        self._codec = codec
        self._row_group_rows = row_group_rows
        self._empty_schema = empty_schema
        self._writer: Optional[pq.ParquetWriter] = None
        self._pending: list[pa.Table] = []
        self._pending_rows = 0
        self.uncompressed_bytes_written = 0

    def _open_writer(self, schema: pa.Schema) -> pq.ParquetWriter:
        return pq.ParquetWriter(
            self._file_handle,
            schema,
            compression=self._codec,
            use_dictionary=PARQUET_USE_DICTIONARY,
        )

    def write(self, table: pa.Table) -> None:
        """Buffers a table and writes any full row groups.

        Args:
            table: Rows to append. Every table must have the same schema.
        """
        self.uncompressed_bytes_written += table.nbytes
        self._pending.append(table)
        self._pending_rows += table.num_rows
        if self._pending_rows >= self._row_group_rows:
            self._flush(final=False)

    def _flush(self, final: bool) -> None:
        if not self._pending:
            return
        combined = pa.concat_tables(self._pending)
        if self._writer is None:
            self._writer = self._open_writer(combined.schema)
        offset = 0
        while combined.num_rows - offset >= self._row_group_rows:
            self._writer.write_table(
                combined.slice(offset, self._row_group_rows),
                row_group_size=self._row_group_rows,
            )
            offset += self._row_group_rows
        remainder = combined.slice(offset)
        if final and remainder.num_rows > 0:
            self._writer.write_table(remainder, row_group_size=self._row_group_rows)
            remainder = remainder.slice(remainder.num_rows)
        self._pending = [remainder] if remainder.num_rows > 0 else []
        self._pending_rows = remainder.num_rows

    def close(self) -> None:
        """Writes the remaining rows, finishes the Parquet footer and closes the file handle."""
        self._flush(final=True)
        if self._writer is None:
            self._writer = self._open_writer(self._empty_schema)
        self._writer.close()
        self._file_handle.close()


class _ParquetSinkWriter(filebasedsink.FileBasedSinkWriter):
    """Closes the current file once it holds TARGET_BYTES_PER_PARQUET_FILE of uncompressed data."""

    def at_capacity(self) -> bool:
        """Returns whether the current file has reached the target size."""
        return (
            self.temp_handle.uncompressed_bytes_written >= TARGET_BYTES_PER_PARQUET_FILE
        )


class _ParquetSink(filebasedsink.FileBasedSink):
    """Beam sink that writes ``pa.Table`` elements to Parquet files through ``_BufferedParquetWriter``."""

    def __init__(
        self,
        file_path_prefix: str,
        num_shards: int,
        row_group_rows: int,
        empty_schema: pa.Schema,
    ):
        """Initializes the sink.

        Args:
            file_path_prefix: Output path prefix, including the ``part`` file name prefix.
            num_shards: Number of output files. 0 starts a new file whenever the current one is at capacity.
            row_group_rows: Number of rows in every row group except the last of each file.
            empty_schema: Schema of files that receive no rows.
        """
        super().__init__(
            file_path_prefix,
            file_name_suffix=PARQUET_FILE_NAME_SUFFIX,
            num_shards=num_shards,
            coder=None,
            mime_type="application/x-parquet",
            # Compression is applied per column chunk by Parquet, not to the whole file.
            compression_type=CompressionTypes.UNCOMPRESSED,
        )
        self._row_group_rows = row_group_rows
        self._empty_schema = empty_schema

    def open_writer(self, init_result: str, uid: str) -> _ParquetSinkWriter:
        suffix = (
            "."
            + os.path.basename(self.file_path_prefix.get())
            + PARQUET_FILE_NAME_SUFFIX
        )
        return _ParquetSinkWriter(self, FileSystems.join(init_result, uid) + suffix)

    def open(self, temp_path: str) -> _BufferedParquetWriter:
        return _BufferedParquetWriter(
            super().open(temp_path),
            codec=PARQUET_CODEC,
            row_group_rows=self._row_group_rows,
            empty_schema=self._empty_schema,
        )

    def write_record(self, writer: _BufferedParquetWriter, table: pa.Table) -> None:
        writer.write(table)

    def close(self, writer: _BufferedParquetWriter) -> None:
        writer.close()


class WriteDenseParquet(beam.PTransform):
    """Writes TFT output record batches to ``<parquet_uri_prefix>part-XXXXX-of-YYYYY.parquet`` files."""

    def __init__(
        self,
        parquet_uri_prefix: str,
        id_keys: Sequence[str],
        id_columns: Sequence[str],
        feature_keys: Sequence[str],
        label_keys: Sequence[str],
        num_shards: int,
        packed_feature_key: Optional[str] = None,
        packed_feature_dim: int = 0,
    ):
        """Initializes the transform.

        Args:
            parquet_uri_prefix: Directory prefix (ending in ``/``) of the output files.
            id_keys: TFT output names of the id columns (one for nodes, two for edges).
            id_columns: Parquet column names for ``id_keys``.
            feature_keys: TFT output names concatenated into the ``features`` column. Empty for no features.
            label_keys: TFT output names written as one float32 column each.
            num_shards: Number of output files. 0 starts a new file whenever the current one holds
                ``TARGET_BYTES_PER_PARQUET_FILE`` of uncompressed data.
            packed_feature_key: TFT output name of the packed quantized bytes. ``None`` when features are not
                quantized.
            packed_feature_dim: Number of packed bytes per row. Required when ``packed_feature_key`` is set.
        """
        self._parquet_uri_prefix = parquet_uri_prefix
        self._id_keys = id_keys
        self._id_columns = id_columns
        self._feature_keys = feature_keys
        self._label_keys = label_keys
        self._num_shards = num_shards
        self._packed_feature_key = packed_feature_key
        self._packed_feature_dim = packed_feature_dim

    def expand(
        self, record_batches: beam.PCollection[pa.RecordBatch]
    ) -> beam.pvalue.PDone:
        """Converts the record batches and writes them to Parquet files.

        Args:
            record_batches: TFT output record batches.

        Returns:
            The result of the Parquet write.
        """
        if len(self._id_columns) == 1:
            row_group_rows = NODE_ROW_GROUP_ROWS
        else:
            row_group_rows = EDGE_ROW_GROUP_ROWS
        empty_schema = pa.schema(
            [pa.field(column, pa.int64()) for column in self._id_columns]
        )
        return (
            record_batches
            | "Convert RecordBatches to dense Arrow tables"
            >> beam.ParDo(
                RecordBatchToDenseArrowTable(
                    id_keys=self._id_keys,
                    id_columns=self._id_columns,
                    feature_keys=self._feature_keys,
                    label_keys=self._label_keys,
                    packed_feature_key=self._packed_feature_key,
                    packed_feature_dim=self._packed_feature_dim,
                )
            )
            | "Write Parquet files"
            >> beam.io.Write(
                _ParquetSink(
                    file_path_prefix=self._parquet_uri_prefix + PARQUET_FILE_PREFIX,
                    num_shards=self._num_shards,
                    row_group_rows=row_group_rows,
                    empty_schema=empty_schema,
                )
            )
        )
