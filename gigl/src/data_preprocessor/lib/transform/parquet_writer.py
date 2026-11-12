"""Writes TFT output record batches as dense Parquet files.

Nodes are stored as ``node_id`` plus an optional ``features`` column, edges as ``src`` and ``dst`` plus an optional
``features`` column. ``features`` is a ``FixedSizeList<float32>`` whose slots follow ``feature_keys`` order, with every
feature cast to float32 and vectors flattened, matching the tensors the TFRecord loader builds. Each label key is its own
column (float32 for scalar labels, ``FixedSizeList<float32>`` for vector labels). Quantized features are stored as one
``FixedSizeBinary`` column of packed bytes per row, named after the packed feature key recorded in the quantization
metadata.
"""

import math
from typing import Optional, Sequence

import apache_beam as beam
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from apache_beam.io import filebasedsink
from apache_beam.io.filesystem import CompressionTypes

NODE_ID_COLUMN = "node_id"
EDGE_SRC_COLUMN = "src"
EDGE_DST_COLUMN = "dst"
FEATURES_COLUMN = "features"

PARQUET_FILE_NAME_SUFFIX = ".parquet"
PARQUET_FILE_PREFIX = "part"
PARQUET_CODEC = "zstd"
NODE_ROW_GROUP_ROWS = 100_000
EDGE_ROW_GROUP_ROWS = 1_000_000
TARGET_BYTES_PER_PARQUET_FILE = 200_000_000


def estimate_num_parquet_shards(num_rows: int, bytes_per_row: int) -> int:
    """Number of output files so that each holds about TARGET_BYTES_PER_PARQUET_FILE of uncompressed data."""
    return max(1, math.ceil(num_rows * bytes_per_row / TARGET_BYTES_PER_PARQUET_FILE))


def _flat_values(column: pa.Array, num_rows: int) -> np.ndarray:
    """Returns the column as a ``[num_rows, width]`` numpy array, flattening list-typed columns row-major."""
    if pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
        column = column.flatten()
    values = column.to_numpy(zero_copy_only=False)
    return values.reshape(num_rows, -1)


def _fixed_size_list(values: np.ndarray) -> pa.Array:
    return pa.FixedSizeListArray.from_arrays(
        pa.array(np.ascontiguousarray(values).reshape(-1)), values.shape[1]
    )


def _fixed_size_binary(column: pa.Array, num_rows: int, width: int) -> pa.Array:
    """Returns a one-binary-value-per-row list column as a ``FixedSizeBinary(width)`` array.

    Arrow cannot cast variable-length binary to fixed-size binary, so the value bytes are reused directly.
    """
    values = column.flatten()
    offsets = np.frombuffer(values.buffers()[1], dtype=np.int32)
    offsets = offsets[values.offset : values.offset + num_rows + 1]
    if not np.array_equal(np.diff(offsets), np.full(num_rows, width)):
        raise ValueError(f"Packed feature values must all be {width} bytes long.")
    data = values.buffers()[2].slice(int(offsets[0]), num_rows * width)
    return pa.FixedSizeBinaryArray.from_buffers(
        pa.binary(width), num_rows, [None, data]
    )


class RecordBatchToDenseArrowTable(beam.DoFn):
    """Converts a TFT ``pa.RecordBatch`` into a ``pa.Table`` with the dense Parquet schema.

    Args:
        id_keys: TFT output names of the id columns, in output order (``[node_id_key]`` or ``[src_key, dst_key]``).
        id_columns: Parquet column names for ``id_keys``.
        feature_keys: TFT output names concatenated, in order, into the ``features`` column.
        label_keys: TFT output names written as one float32 column each.
        packed_feature_key: TFT output name of the packed quantized bytes, written as a ``FixedSizeBinary`` column of the
            same name. ``None`` when features are not quantized.
        packed_feature_dim: Number of packed bytes per row. Required when ``packed_feature_key`` is set.
    """

    def __init__(
        self,
        id_keys: Sequence[str],
        id_columns: Sequence[str],
        feature_keys: Sequence[str],
        label_keys: Sequence[str],
        packed_feature_key: Optional[str] = None,
        packed_feature_dim: int = 0,
    ):
        self._id_keys = list(id_keys)
        self._id_columns = list(id_columns)
        self._feature_keys = list(feature_keys)
        self._label_keys = list(label_keys)
        self._packed_feature_key = packed_feature_key
        self._packed_feature_dim = packed_feature_dim

    def process(self, batch: pa.RecordBatch):
        num_rows = batch.num_rows
        if num_rows == 0:
            return
        names = []
        arrays = []
        for key, column_name in zip(self._id_keys, self._id_columns):
            ids = _flat_values(
                batch.column(batch.schema.get_field_index(key)), num_rows
            )
            names.append(column_name)
            arrays.append(pa.array(ids.reshape(-1).astype(np.int64, copy=False)))
        if self._feature_keys:
            feature_blocks = [
                _flat_values(
                    batch.column(batch.schema.get_field_index(key)), num_rows
                ).astype(np.float32)
                for key in self._feature_keys
            ]
            names.append(FEATURES_COLUMN)
            arrays.append(_fixed_size_list(np.concatenate(feature_blocks, axis=1)))
        if self._packed_feature_key is not None:
            packed = batch.column(
                batch.schema.get_field_index(self._packed_feature_key)
            )
            names.append(self._packed_feature_key)
            arrays.append(
                _fixed_size_binary(packed, num_rows, self._packed_feature_dim)
            )
        for key in self._label_keys:
            labels = _flat_values(
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
        file_handle,
        codec: str,
        row_group_rows: int,
        empty_schema: pa.Schema,
    ):
        self._file_handle = file_handle
        self._codec = codec
        self._row_group_rows = row_group_rows
        self._empty_schema = empty_schema
        self._writer: Optional[pq.ParquetWriter] = None
        self._pending: list[pa.Table] = []
        self._pending_rows = 0

    def _get_writer(self, schema: pa.Schema) -> pq.ParquetWriter:
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self._file_handle, schema, compression=self._codec
            )
        return self._writer

    def write(self, table: pa.Table) -> None:
        self._pending.append(table)
        self._pending_rows += table.num_rows
        if self._pending_rows >= self._row_group_rows:
            self._flush(final=False)

    def _flush(self, final: bool) -> None:
        if not self._pending:
            return
        combined = pa.concat_tables(self._pending)
        writer = self._get_writer(combined.schema)
        offset = 0
        while combined.num_rows - offset >= self._row_group_rows:
            writer.write_table(
                combined.slice(offset, self._row_group_rows),
                row_group_size=self._row_group_rows,
            )
            offset += self._row_group_rows
        remainder = combined.slice(offset)
        if final and remainder.num_rows > 0:
            writer.write_table(remainder, row_group_size=self._row_group_rows)
            remainder = remainder.slice(remainder.num_rows)
        self._pending = [remainder] if remainder.num_rows > 0 else []
        self._pending_rows = remainder.num_rows

    def close(self) -> None:
        self._flush(final=True)
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self._file_handle, self._empty_schema, compression=self._codec
            )
        self._writer.close()
        self._file_handle.close()


class _ParquetSink(filebasedsink.FileBasedSink):
    def __init__(
        self,
        file_path_prefix: str,
        num_shards: int,
        row_group_rows: int,
        empty_schema: pa.Schema,
    ):
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
    """Writes TFT output record batches to ``<parquet_uri_prefix>part-XXXXX-of-YYYYY.parquet`` files.

    Args:
        parquet_uri_prefix: Directory prefix (ending in ``/``) of the output files.
        id_keys: TFT output names of the id columns (one for nodes, two for edges).
        id_columns: Parquet column names for ``id_keys``.
        feature_keys: TFT output names concatenated into the ``features`` column. Empty for no features.
        label_keys: TFT output names written as one float32 column each.
        num_shards: Number of output files. 0 lets the runner choose.
        packed_feature_key: TFT output name of the packed quantized bytes. ``None`` when features are not quantized.
        packed_feature_dim: Number of packed bytes per row. Required when ``packed_feature_key`` is set.
    """

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
        self._parquet_uri_prefix = parquet_uri_prefix
        self._id_keys = id_keys
        self._id_columns = id_columns
        self._feature_keys = feature_keys
        self._label_keys = label_keys
        self._num_shards = num_shards
        self._packed_feature_key = packed_feature_key
        self._packed_feature_dim = packed_feature_dim

    def expand(self, record_batches):
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
