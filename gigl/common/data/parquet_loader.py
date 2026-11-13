"""Loads node and edge tensors from Parquet files.

``ParquetDataLoader`` reads the files of one entity (one node type or one edge type) under
``SerializedTFRecordInfo.parquet_uri_prefix``. Column names and types are defined in
``gigl.common.data.parquet_format``.

An entity directory::

    <parquet_uri_prefix>/
        _row_counts.json                  the manifest
        part-00000-of-00003.parquet       row groups of 100000, 100000, 523 rows
        part-00001-of-00003.parquet       row groups of 100000, 4100 rows
        part-00002-of-00003.parquet       one row group of 61 rows

    _row_counts.json:
        {"version": 1, "files": {"part-00000-of-00003.parquet": [100000, 100000, 523],
                                 "part-00001-of-00003.parquet": [100000, 4100],
                                 "part-00002-of-00003.parquet": [61]}}

One node file::

    part-00000-of-00003.parquet
    +-- row group 0 (100000 rows)
    |     column chunk  node_id               int64
    |     column chunk  features              fixed_size_list<float32>[feature_dim]  optional
    |     column chunk  <packed_feature_key>  fixed_size_binary[packed_feature_dim]  optional
    |     column chunk  <label key>           float32 or fixed_size_list<float32>    optional, one per label
    +-- row group 1 (100000 rows)             same column chunks
    +-- row group 2 (523 rows)                same column chunks
    +-- footer
          schema
          per row group: row count, byte range of each column chunk

Edge files have the same layout with ``src`` and ``dst`` (int64) in place of ``node_id``.

The writer (``gigl.src.data_preprocessor.lib.transform.parquet_writer``) targets:

- row groups of 100,000 rows for nodes and 1,000,000 rows for edges (the last row group of a file is smaller),
- files of 1 GB of uncompressed data. Files are zstd-compressed and also end when a Beam bundle ends, so many files
  are much smaller than this.

Terms:

- **Row group**: a block of consecutive rows, stored column by column.
  pyarrow can read one row group without the others but not part of one, so it is the unit of work here.
- **Column chunk**: one column's values within one row group. Only the requested columns' chunks are read.
  pyarrow also reads a gap of up to 8 KiB between two requested chunks instead of making a second request.
- **Footer**: the schema and, per row group, the row count and the byte range of each column chunk.
  Reading it costs storage requests, so only the footers of files this rank reads from are fetched.
- **Manifest**: ``_row_counts.json``, the row count of every row group of every file
  (see ``gigl.common.data.parquet_manifest``).
  Every rank computes its row range from it without reading other ranks' footers.
  Loading fails if it is missing or does not list exactly the files found.
- **Row range**: the rank's contiguous slice of all rows, counted in sorted file order.
  Rank ``r`` of ``world_size`` keeps rows ``[r * total // world_size, (r + 1) * total // world_size)``.
- **Read task / plan**: the plan is a list of ``_ReadTask``, one per row group that overlaps the row range:

  - ``file_index``: index into the sorted file list.
  - ``row_group``: row group of that file.
  - ``row_offset``: first row of the row group to keep (0 unless the range starts inside it).
  - ``num_rows``: rows of the row group to keep.
  - ``out_offset``: output row that the first kept row is written to.

  Tasks write disjoint output rows, so the copies need no lock.
- **Shared handle**: one open file handle per file, used by all of that file's tasks (see ``_SharedHandles``).

``ParquetDataLoader.load_as_torch_tensors``::

    list .parquet files (sorted) -> read manifest -> row range -> plan tasks
        -> read footers of this rank's files
        -> allocate each output tensor once, at its final size
        -> thread pool: read each task's row group, copy its columns into the outputs at out_offset
        -> LoadedEntityTensors

Example: files ``a: [4, 4]``, ``b: [2]``, ``c: [4]`` (row counts per row group) hold 14 rows; 2 ranks keep rows
``[0, 7)`` and ``[7, 14)``. Row group ``a[1]`` straddles the boundary, so both ranks read it::

    global rows    0 1 2 3 | 4 5 6 7 | 8 9 | 10 11 12 13
    row group      a[0]    | a[1]    | b[0]| c[0]
    rank 0         x x x x   x x x
    rank 1                         x   x x   x  x  x  x

    (file, row_group, row_offset, num_rows, out_offset)
    rank 0: (a, 0, 0, 4, 0)  (a, 1, 0, 3, 4)
    rank 1: (a, 1, 3, 1, 0)  (b, 0, 0, 2, 1)  (c, 0, 0, 4, 3)

Rank 0 reads only the footer of ``a``.

Outputs are allocated with ``gigl.utils.share_memory.allocate_preshared``, which picks the tensor's final home:

- a file-backed tensor in ``GIGL_TENSOR_SPILL_DIR`` if that is set and the tensor is at least
  ``GIGL_TENSOR_SPILL_MIN_BYTES`` (2 GiB by default),
- an ordinary in-memory tensor if it is smaller than that threshold,
- otherwise a tensor in POSIX shared memory. Allocation fails if the shared-memory mount lacks the room.

So large outputs are never copied into shared memory later, and no per-row-group batches are kept to concatenate.
Columns are copied as numpy arrays into numpy views of the outputs, which write straight into the tensors' memory.

The loader runs in a process started with ``spawn`` (see ``gigl.common.data.load_torch_tensors``).
The libcurl inside the pyarrow 10 wheels waits on sockets with ``select()``, which cannot take a file-descriptor number
of 1024 or higher.
A training process can hold more than 1024 open descriptors, so its new sockets get numbers above the limit and GCS
requests fail with a ``pa.ArrowException`` that mentions ``select/poll``.
A freshly spawned process starts with a few dozen descriptors.
The pyarrow wheels use ``poll()`` from version 13.
The log line at the end of a load reports the descriptor count at its start.
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Final, Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import torch

from gigl.common import Uri
from gigl.common.data import parquet_manifest
from gigl.common.data.loader_types import (
    LoadedEntityTensors,
    SerializedTFRecordInfo,
    empty_entity_tensors,
)
from gigl.common.data.parquet_format import (
    EDGE_DST_COLUMN,
    EDGE_SRC_COLUMN,
    FEATURES_COLUMN,
    NODE_ID_COLUMN,
)
from gigl.common.logger import Logger
from gigl.common.utils.retry import retry
from gigl.utils.share_memory import allocate_preshared

logger = Logger()

_MAX_THREADS: Final[int] = 64
_RETRY_TRIES: Final[int] = 2
_RETRY_DELAY_SECONDS: Final[int] = 2
_RETRY_BACKOFF: Final[int] = 4
_RETRY_MAX_DELAY_SECONDS: Final[int] = 30
# Retries a call that raises ``OSError``, which is what pyarrow raises for I/O failures.
# The call is made ``_RETRY_TRIES + 1`` times in total, with sleeps of 2s and then 8s in between.
# ``FileNotFoundError`` and ``PermissionError`` are ``OSError`` too, so they are also retried before they are raised.
# Other pyarrow errors are not retried.
# The GCS client retries connection failures, timeouts and HTTP 408, 429 and 5xx responses itself; when it gives up,
# pyarrow raises them as ``OSError``.
_retry_on_os_error = retry(
    exception_to_check=OSError,
    tries=_RETRY_TRIES,
    delay_s=_RETRY_DELAY_SECONDS,
    backoff=_RETRY_BACKOFF,
    max_delay_s=_RETRY_MAX_DELAY_SECONDS,
    should_throw_original_exception=True,
)


@dataclass(frozen=True)
class _ReadTask:
    """Reads ``num_rows`` rows starting at ``row_offset`` of one row group into ``out_offset`` of the output."""

    file_index: int
    row_group: int
    row_offset: int
    num_rows: int
    out_offset: int


class _SharedHandles:
    """One handle per file, shared by all of the file's read tasks and closed after the last one.

    A file has many row groups, each read as its own task, and opening a file on GCS costs an object metadata request
    (pyarrow 10 also starts a read of the object).
    An open takes about 130 ms (median) on pyarrow 10 and 29 ms on pyarrow 25.
    An open followed by a footer read takes about 150 ms on pyarrow 10 and 134 ms on pyarrow 25.
    Opening a fresh handle per row-group read instead of sharing one per file is slower on both versions (GCS, one node
    and one edge directory, median of 3 runs):

    - pyarrow 10: nodes within noise, edges 1.2-1.3x slower.
    - pyarrow 25: nodes 1.2-1.4x slower, edges 3.5-4.8x slower (2.98 s vs 0.62 s at 48 threads).

    Each open handle also holds a socket or file descriptor.
    Tasks start in file order and a file's handle closes after its last task, so the number of open files stays near
    the number of reader threads instead of growing with the number of row groups.
    A slow task can keep an earlier file open while later files are opened.

    This is not what avoids pyarrow 10's limit on descriptor numbers; running in a spawned process does (see the module
    docstring).

    ``get`` opens a file's handle on first use, under a per-file lock.
    ``finish_task`` counts down the file's remaining tasks and closes the handle after the last one.
    ``close_all`` closes the handles left open when tasks are cancelled; call it only once no task is running.
    """

    def __init__(
        self, fs: pafs.FileSystem, files: Sequence[str], tasks: Sequence[_ReadTask]
    ) -> None:
        self._fs = fs
        self._files = files
        self._remaining: list[int] = [0] * len(files)
        for task in tasks:
            self._remaining[task.file_index] += 1
        self._locks: list[threading.Lock] = [threading.Lock() for _ in files]
        self._handles: list[Optional[pa.NativeFile]] = [None] * len(files)

    def get(self, file_index: int) -> pa.NativeFile:
        """Returns the shared handle of the file, opening it on first use."""
        with self._locks[file_index]:
            handle = self._handles[file_index]
            if handle is None:
                handle = self._fs.open_input_file(self._files[file_index])
                self._handles[file_index] = handle
            return handle

    def finish_task(self, file_index: int) -> None:
        """Marks one task of the file done and closes the file's handle after its last task."""
        with self._locks[file_index]:
            self._remaining[file_index] -= 1
            if self._remaining[file_index] == 0:
                self._close_locked(file_index)

    def close_all(self) -> None:
        """Closes every handle that is still open."""
        for file_index in range(len(self._files)):
            with self._locks[file_index]:
                self._close_locked(file_index)

    def _close_locked(self, file_index: int) -> None:
        handle = self._handles[file_index]
        if handle is not None:
            self._handles[file_index] = None
            handle.close()


def _open_fd_count() -> int:
    """Number of file descriptors open in this process, or -1 where ``/proc`` is unavailable."""
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:
        return -1


def _column_width(field: pa.Field) -> int:
    """Number of values per row of a column: the list size of a fixed-size list, 1 for scalars."""
    if pa.types.is_fixed_size_list(field.type):
        return field.type.list_size
    return 1


def _copy_into(column: pa.ChunkedArray, destination: np.ndarray) -> None:
    """Copies ``column`` into the leading rows of ``destination``, casting to its dtype.

    Fixed-size-list columns fill ``[rows, width]`` destinations and scalar columns fill ``[rows]`` or
    ``[rows, 1]`` ones.

    Raises:
        ValueError: If the column has null values, or its width is not the width of ``destination``.
            numpy would otherwise repeat a width-1 column across a wider destination.
    """
    if destination.ndim == 2:
        destination_width = destination.shape[1]
    else:
        destination_width = 1
    offset = 0
    for chunk in column.chunks:
        if chunk.null_count:
            raise ValueError(f"Parquet column has {chunk.null_count} null values")
        if pa.types.is_fixed_size_list(chunk.type):
            width = chunk.type.list_size
            flat_values = chunk.flatten()
            # The null count of a list array does not include null values inside the lists.
            if flat_values.null_count:
                raise ValueError(
                    f"Parquet column has {flat_values.null_count} null list values"
                )
            values = flat_values.to_numpy(zero_copy_only=False)
        else:
            width = 1
            values = chunk.to_numpy(zero_copy_only=False)
        if width != destination_width:
            raise ValueError(
                f"Parquet column has {width} values per row but {destination_width} were expected"
            )
        if destination.ndim == 2:
            values = values.reshape(len(chunk), width)
        destination[offset : offset + len(chunk)] = values
        offset += len(chunk)


def _copy_packed_into(column: pa.ChunkedArray, destination: np.ndarray) -> None:
    """Copies a ``FixedSizeBinary`` column into the leading rows of the ``[rows, width]`` uint8 ``destination``."""
    offset = 0
    for chunk in column.chunks:
        if chunk.null_count:
            raise ValueError(f"Parquet column has {chunk.null_count} null values")
        width = chunk.type.byte_width
        if width != destination.shape[1]:
            raise ValueError(
                f"Packed feature column has {width} bytes per row but {destination.shape[1]} were expected"
            )
        values = np.frombuffer(
            chunk.buffers()[1],
            dtype=np.uint8,
            count=len(chunk) * width,
            offset=chunk.offset * width,
        )
        destination[offset : offset + len(chunk)] = values.reshape(len(chunk), width)
        offset += len(chunk)


class ParquetDataLoader:
    """Loads one entity (node, edge, positive label or negative label) of one type from Parquet files.

    Produces the same ``LoadedEntityTensors`` as ``TFRecordDataLoader.load_as_torch_tensors`` for the same data.

    ``TFRecordDataLoader`` gives each rank whole files.
    Dataflow ends files early at bundle boundaries, so file sizes vary and whole-file assignment gives uneven ranks.
    This loader instead gives each rank a contiguous range of rows, so ranks differ by at most one row.
    The module docstring describes how a rank's range is planned, read and copied into its outputs.

    Args:
        rank (int): Rank of the current machine.
        world_size (int): Number of machines.
        num_threads (Optional[int]): Number of reader threads. Defaults to the CPU count, at most 64.
    """

    def __init__(
        self, rank: int, world_size: int, num_threads: Optional[int] = None
    ) -> None:
        self._rank = rank
        self._world_size = world_size
        if num_threads is None:
            num_threads = min(os.cpu_count() or 1, _MAX_THREADS)
        self._num_threads = num_threads

    @staticmethod
    def _filesystem_and_path(uri: Uri) -> tuple[pafs.FileSystem, str]:
        return parquet_manifest.filesystem_and_path(uri)

    def _plan_tasks(
        self, row_groups_per_file: Sequence[Sequence[int]]
    ) -> list[_ReadTask]:
        """Returns the read tasks that make up this rank's row range, in file and row-group order.

        See the module docstring for the row range and a worked example.

        Args:
            row_groups_per_file (Sequence[Sequence[int]]): Row counts of each row group of each file, in file order.

        Returns:
            list[_ReadTask]: One task per row group that overlaps the range.
        """
        total_rows = sum(sum(file_rows) for file_rows in row_groups_per_file)
        range_start = self._rank * total_rows // self._world_size
        range_end = (self._rank + 1) * total_rows // self._world_size

        tasks: list[_ReadTask] = []
        group_start = 0
        for file_index, file_rows in enumerate(row_groups_per_file):
            for row_group, group_rows in enumerate(file_rows):
                group_end = group_start + group_rows
                keep_start = max(group_start, range_start)
                keep_end = min(group_end, range_end)
                if keep_start < keep_end:
                    tasks.append(
                        _ReadTask(
                            file_index=file_index,
                            row_group=row_group,
                            row_offset=keep_start - group_start,
                            num_rows=keep_end - keep_start,
                            out_offset=keep_start - range_start,
                        )
                    )
                group_start = group_end
        return tasks

    def load_as_torch_tensors(
        self, serialized_tf_record_info: SerializedTFRecordInfo
    ) -> LoadedEntityTensors:
        """Loads this rank's share of the Parquet files at ``parquet_uri_prefix`` as torch tensors.

        Args:
            serialized_tf_record_info (SerializedTFRecordInfo): How the entity is stored.
                Must have ``parquet_uri_prefix`` set.

        Returns:
            LoadedEntityTensors: The (ids, features, packed features, labels) of this rank's rows.
                Packed features are the uint8 ``[rows, packed_feature_dim]`` packed bytes and are None
                unless ``packed_feature_key`` is set.

        Raises:
            ValueError: If ``parquet_uri_prefix`` is not set, or the manifest is malformed, has an unknown version or
                does not list exactly the files found.
            FileNotFoundError: If the manifest is missing.
            OSError: If listing the files or reading a footer or row group still fails after the retries.
        """
        info = serialized_tf_record_info
        if info.parquet_uri_prefix is None:
            raise ValueError("ParquetDataLoader requires parquet_uri_prefix to be set")
        fd_count = _open_fd_count()
        wall_start = time.perf_counter()

        fs, path = self._filesystem_and_path(info.parquet_uri_prefix)
        files: list[str] = _retry_on_os_error(parquet_manifest.list_parquet_files)(
            fs, path
        )
        list_s = time.perf_counter() - wall_start
        if not files:
            logger.warning(
                f"Found no Parquet files under {info.parquet_uri_prefix.uri}"
            )
            return empty_entity_tensors(info)

        # Reads with pre_buffer=True run on pyarrow's I/O pool, which caps the number of in-flight requests
        # (8 threads by default).
        pa.set_io_thread_count(max(pa.io_thread_count(), self._num_threads))

        with ThreadPoolExecutor(self._num_threads) as pool:
            footer_start = time.perf_counter()

            @_retry_on_os_error
            def read_footer(file: str) -> pq.FileMetaData:
                """Opens ``file``, reads its footer and closes it again."""
                handle = fs.open_input_file(file)
                try:
                    return pq.read_metadata(handle)
                finally:
                    handle.close()

            row_groups_per_file = parquet_manifest.read_manifest_row_groups(
                fs, path, files
            )
            tasks = self._plan_tasks(row_groups_per_file)
            # Reads and the schema need the footers of the files this rank reads from.
            footer_file_indices = sorted({task.file_index for task in tasks})
            footer_list = pool.map(
                read_footer, [files[index] for index in footer_file_indices]
            )
            metadatas: dict[int, pq.FileMetaData] = dict(
                zip(footer_file_indices, footer_list)
            )
            footers_s = time.perf_counter() - footer_start
            num_rows = sum(task.num_rows for task in tasks)
            if num_rows == 0:
                logger.info(f"No rows to load for rank: {self._rank}.")
                return empty_entity_tensors(info)

            if info.is_node_entity:
                id_columns = [NODE_ID_COLUMN]
            else:
                id_columns = [EDGE_SRC_COLUMN, EDGE_DST_COLUMN]
            columns: list[str] = list(id_columns)
            has_features = bool(info.feature_keys)
            if has_features:
                columns.append(FEATURES_COLUMN)
            if info.packed_feature_key is not None:
                columns.append(info.packed_feature_key)
            columns.extend(info.label_keys)

            # Shards without row groups carry only the id columns, so the schema comes from a file with rows.
            schema = metadatas[tasks[0].file_index].schema.to_arrow_schema()
            label_widths: list[int] = []
            for key in info.label_keys:
                label_widths.append(_column_width(schema.field(key)))
            num_label_columns = sum(label_widths)

            if info.is_node_entity:
                ids = allocate_preshared((num_rows,), torch.int64)
            else:
                ids = allocate_preshared((2, num_rows), torch.int64)
            features: Optional[torch.Tensor] = None
            if has_features:
                features = allocate_preshared(
                    (num_rows, info.feature_dim), torch.float32
                )
            packed_features: Optional[torch.Tensor] = None
            if info.packed_feature_key is not None:
                packed_features = allocate_preshared(
                    (num_rows, info.packed_feature_dim), torch.uint8
                )
            labels: Optional[torch.Tensor] = None
            if info.label_keys:
                labels = allocate_preshared(
                    (num_rows, num_label_columns), torch.float32
                )
            ids_np: np.ndarray = ids.numpy()
            features_np: Optional[np.ndarray] = None
            if features is not None:
                features_np = features.numpy()
            packed_features_np: Optional[np.ndarray] = None
            if packed_features is not None:
                packed_features_np = packed_features.numpy()
            labels_np: Optional[np.ndarray] = None
            if labels is not None:
                labels_np = labels.numpy()

            shared_handles = _SharedHandles(fs, files, tasks)

            # A retry reads through the same shared handle.
            # Every positional read opens its own request, so a failed read leaves nothing behind in the handle.
            @_retry_on_os_error
            def read_row_group(task: _ReadTask) -> pa.Table:
                parquet_file = pq.ParquetFile(
                    shared_handles.get(task.file_index),
                    metadata=metadatas[task.file_index],
                    pre_buffer=True,
                )
                return parquet_file.read_row_group(
                    task.row_group, columns=columns, use_threads=False
                )

            def read_task(task: _ReadTask) -> None:
                try:
                    table = read_row_group(task)
                finally:
                    shared_handles.finish_task(task.file_index)
                table = table.slice(task.row_offset, task.num_rows)
                out = slice(task.out_offset, task.out_offset + task.num_rows)
                if info.is_node_entity:
                    _copy_into(table.column(NODE_ID_COLUMN), ids_np[out])
                else:
                    _copy_into(table.column(EDGE_SRC_COLUMN), ids_np[0, out])
                    _copy_into(table.column(EDGE_DST_COLUMN), ids_np[1, out])
                if features_np is not None:
                    _copy_into(table.column(FEATURES_COLUMN), features_np[out])
                if packed_features_np is not None:
                    _copy_packed_into(
                        table.column(info.packed_feature_key),
                        packed_features_np[out],
                    )
                if labels_np is not None:
                    label_offset = 0
                    for key, width in zip(info.label_keys, label_widths):
                        _copy_into(
                            table.column(key),
                            labels_np[out, label_offset : label_offset + width],
                        )
                        label_offset += width

            read_start = time.perf_counter()
            # Iterating drains the map: it waits for the tasks in order and re-raises the first exception here,
            # cancelling the tasks that have not started.
            # Each file's handle closes after its last task; close_all closes the handles of files whose last task was
            # cancelled, once shutdown has waited for the tasks still running.
            try:
                for _ in pool.map(read_task, tasks):
                    pass
            finally:
                pool.shutdown(wait=True)
                shared_handles.close_all()
            read_s = time.perf_counter() - read_start

        compressed_bytes = 0
        for task in tasks:
            row_group_metadata = metadatas[task.file_index].row_group(task.row_group)
            for column_index in range(row_group_metadata.num_columns):
                compressed_bytes += row_group_metadata.column(
                    column_index
                ).total_compressed_size
        wall_s = time.perf_counter() - wall_start
        logger.info(
            f"Loaded Parquet entity: rank={self._rank} files={len(files)} "
            f"row_groups={len(tasks)} list_s={list_s:.2f} footers_s={footers_s:.2f} "
            f"read_s={read_s:.2f} bytes={compressed_bytes} rows={num_rows} "
            f"threads={self._num_threads} fd_count={fd_count} "
            f"wall_s={wall_s:.2f} "
            f"rows_per_s={num_rows / max(wall_s, 1e-9):.0f}"
        )
        return LoadedEntityTensors(
            ids=ids,
            features=features,
            packed_features=packed_features,
            labels=labels,
        )
