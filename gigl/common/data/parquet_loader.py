"""Loads node and edge tensors from Parquet files.

``ParquetDataLoader`` reads the files of one entity under ``SerializedTFRecordInfo.parquet_uri_prefix``.
The file layout is described in ``gigl.common.data.parquet_format``.

How ``ParquetDataLoader.load_as_torch_tensors`` works
=====================================================

Every rank reads a contiguous slice of the entity's rows, and all ranks together read every row exactly once::

    list the .parquet files under parquet_uri_prefix (sorted)
                         |
                         v
    read the manifest: row counts of each row group of each file
                         |
                         v
    compute this rank's row range [range_start, range_end)
                         |
                         v
    plan read tasks: one per row group that overlaps the range,
    sliced where the range starts or ends inside a row group
                         |
                         v
    read the footers of only the files this rank's tasks touch
                         |
                         v
    allocate each output tensor once, at its final size (allocate_preshared)
                         |
                         v
    take numpy views of the output tensors (same memory, no copy)
                         |
                         v
    thread pool: read each task's row group and copy its columns
    into the numpy views at the task's out_offset
                         |
                         v
    return the output tensors as LoadedEntityTensors

Terms:

- **Row group**: the unit a Parquet file is divided into.
  A row group is a horizontal chunk of rows, stored column by column, whose byte ranges are recorded in the footer.
  It is the smallest unit pyarrow reads on its own, so it is the unit of work here.
  The writer targets row groups of about 100k rows for nodes and 1M rows for edges, in files of about 1 GB, so one
  file holds many row groups.
- **Footer (file metadata)**: the block at the end of a Parquet file that holds the schema and, for each row group,
  its row count and the byte ranges of its columns.
  Reading it costs a storage request, so only the footers of files this rank reads from are fetched.
  ``pq.ParquetFile`` gets the footer passed in and does not fetch it again for each row group.
- **Manifest**: ``_row_counts.json`` written once next to the data files, listing the row count of
  every row group of every file (see ``gigl.common.data.parquet_manifest``).
  It lets every rank compute every rank's range without reading every footer.
  Loading fails if the manifest is missing or does not list exactly the files found.
- **Row range per rank**: the rows of all files, concatenated in sorted file order, split into ``world_size``
  contiguous ranges that differ in size by at most one row.
  Rank ``r`` keeps rows ``[r * total // world_size, (r + 1) * total // world_size)``.
- **Read tasks (the plan)**: a list of ``_ReadTask``, one per row group that overlaps the rank's range:

  - ``file_index``: which file, as an index into the sorted file list.
  - ``row_group``: which row group of that file.
  - ``row_offset``: the first row of the row group to keep. 0 unless the range starts inside the row group.
  - ``num_rows``: how many rows of the row group to keep.
  - ``out_offset``: the output row the first kept row is written to.

  Tasks do not overlap in the output, so threads write to disjoint rows and need no lock around the copies.
- **Shared handles** (``_SharedHandles``): one open file handle per file, shared by all tasks of that file and closed
  after its last task, so a file is opened once rather than once per row group.

Worked example: three files with row groups ``a.parquet: [4, 4]``, ``b.parquet: [2]``, ``c.parquet: [4]`` hold 14
rows, and 2 ranks split them into rows ``[0, 7)`` and ``[7, 14)``.
The second row group of ``a.parquet`` covers rows ``[4, 8)`` and so lies on the boundary; both ranks read it::

    global rows    0 1 2 3 | 4 5 6 7 | 8 9 | 10 11 12 13
    row group      a[0]    | a[1]    | b[0]| c[0]
    rank 0         x x x x   x x x
    rank 1                         x   x x   x  x  x  x

    rank 0 tasks: (a, 0, row_offset=0, num_rows=4, out_offset=0)
                  (a, 1, row_offset=0, num_rows=3, out_offset=4)
    rank 1 tasks: (a, 1, row_offset=3, num_rows=1, out_offset=0)
                  (b, 0, row_offset=0, num_rows=2, out_offset=1)
                  (c, 0, row_offset=0, num_rows=4, out_offset=3)

Rank 0 reads only the footer of ``a.parquet``.

Outputs are allocated once with ``gigl.utils.share_memory.allocate_preshared``, which picks the tensor's final home:

- a file-backed tensor in ``GIGL_TENSOR_SPILL_DIR`` if that is set and the tensor is at least
  ``GIGL_TENSOR_SPILL_MIN_BYTES`` (2 GiB by default),
- an ordinary in-memory tensor if it is smaller than that threshold,
- otherwise a tensor in POSIX shared memory.

The large outputs never need a later copy into shared memory, and no per-row-group batches are kept to concatenate.

The copies go through numpy views of the output tensors.
pyarrow converts a column to a numpy array, and assigning it into a numpy view of an output writes straight into the
tensor's memory, with no intermediate torch tensor per row group.

The loader runs in a process started with ``spawn`` (see ``gigl.common.data.load_torch_tensors``).
The bundled curl in pyarrow 10 waits on sockets with ``select()``, which fails for file descriptors numbered 1024 or
above, and a freshly spawned process starts with few open descriptors.
The log line at the end of a load reports the descriptor count at its start.
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Final, Optional, Sequence, TypeVar

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
# Each retried read makes ``_RETRY_TRIES + 1`` attempts in total, sleeping 2s and then 8s between them.
_RETRY_TRIES: Final[int] = 2
_RETRY_DELAY_SECONDS: Final[int] = 2
_RETRY_BACKOFF: Final[int] = 4
_RETRY_MAX_DELAY_SECONDS: Final[int] = 30

_T = TypeVar("_T")


@dataclass(frozen=True)
class _ReadTask:
    """Reads ``num_rows`` rows starting at ``row_offset`` of one row group into ``out_offset`` of the output."""

    file_index: int
    row_group: int
    row_offset: int
    num_rows: int
    out_offset: int


class _TransientReadError(OSError):
    """A read failed with an error that a retry with a fresh file handle may not hit again."""


def _is_retryable(error: Exception) -> bool:
    """Whether ``error`` is a transient I/O failure worth retrying with a fresh handle.

    Missing files and permission errors are permanent.
    Generic ``pa.ArrowException`` is what pyarrow raises for storage-client failures without a more specific status
    (for example an unknown error from the GCS client).
    Its subclasses (invalid data, type errors, ...) are not transient.

    Args:
        error (Exception): The error raised by a read.

    Returns:
        bool: True if the read should be retried.
    """
    if isinstance(error, (FileNotFoundError, PermissionError)):
        return False
    if isinstance(error, OSError):
        return True
    return type(error) is pa.ArrowException


@retry(
    exception_to_check=_TransientReadError,
    tries=_RETRY_TRIES,
    delay_s=_RETRY_DELAY_SECONDS,
    backoff=_RETRY_BACKOFF,
    max_delay_s=_RETRY_MAX_DELAY_SECONDS,
    should_throw_original_exception=True,
)
def _retry_transient(description: str, operation: Callable[[], _T]) -> _T:
    """Runs ``operation``, retrying it when it fails with a transient I/O error.

    Transient failures are raised as ``_TransientReadError`` so that ``retry`` retries them, for three attempts in total.
    Other errors propagate unchanged and are not retried.
    ``operation`` must open any file handle it uses so that every retry starts with a fresh one.

    Args:
        description (str): What the operation does, for the error message.
        operation (Callable[[], _T]): The read to run.

    Returns:
        _T: The result of ``operation``.

    Raises:
        _TransientReadError: If every attempt failed with a transient error.
    """
    try:
        return operation()
    except Exception as e:
        if _is_retryable(e):
            raise _TransientReadError(f"{description} failed: {e}") from e
        raise


class _SharedHandles:
    """One lazily opened handle per file, shared by concurrent reads and closed after the file's last task.

    Opening a file is expensive on GCS with pyarrow 10: it costs an extra metadata request and a new TLS connection,
    about 70 ms each.
    Files hold many row groups (the writer targets files of about 1 GB with 100k-row node and 1M-row edge row groups),
    so each file is opened once and that handle serves every row-group read of the file.
    The manifest only saves reading the footers of other ranks' files; this rank still opens its own files.

    Tasks run in file order on a pool, so at most as many files as pool threads are open at once.

    TODO: Re-measure and remove once pyarrow is upgraded (the open cost may drop).
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
    """
    offset = 0
    for chunk in column.chunks:
        if chunk.null_count:
            raise ValueError(f"Parquet column has {chunk.null_count} null values")
        if pa.types.is_fixed_size_list(chunk.type):
            values = chunk.flatten().to_numpy(zero_copy_only=False)
            values = values.reshape(len(chunk), chunk.type.list_size)
        else:
            values = chunk.to_numpy(zero_copy_only=False)
            if destination.ndim == 2:
                values = values.reshape(len(chunk), 1)
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
        """
        info = serialized_tf_record_info
        if info.parquet_uri_prefix is None:
            raise ValueError("ParquetDataLoader requires parquet_uri_prefix to be set")
        fd_count = _open_fd_count()
        wall_start = time.perf_counter()

        fs, path = self._filesystem_and_path(info.parquet_uri_prefix)
        files: list[str] = _retry_transient(
            "Listing Parquet files",
            lambda: parquet_manifest.list_parquet_files(fs, path),
        )
        list_s = time.perf_counter() - wall_start
        if not files:
            logger.warning(
                f"Found no Parquet files under {info.parquet_uri_prefix.uri}"
            )
            return empty_entity_tensors(info)

        # Reads with pre_buffer=True run on pyarrow's I/O pool, which caps the number of in-flight requests.
        pa.set_io_thread_count(max(pa.io_thread_count(), self._num_threads))

        with ThreadPoolExecutor(self._num_threads) as pool:
            footer_start = time.perf_counter()

            def read_footer(file: str) -> pq.FileMetaData:
                """Opens ``file``, reads its footer and closes it again."""

                def open_and_read_footer() -> pq.FileMetaData:
                    handle = fs.open_input_file(file)
                    try:
                        return pq.read_metadata(handle)
                    finally:
                        handle.close()

                return _retry_transient(
                    f"Reading footer of {file}", open_and_read_footer
                )

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

            def read_with_shared_handle(task: _ReadTask) -> pa.Table:
                parquet_file = pq.ParquetFile(
                    shared_handles.get(task.file_index),
                    metadata=metadatas[task.file_index],
                    pre_buffer=True,
                )
                return parquet_file.read_row_group(
                    task.row_group, columns=columns, use_threads=False
                )

            def read_with_fresh_handle(task: _ReadTask) -> pa.Table:
                handle = fs.open_input_file(files[task.file_index])
                try:
                    parquet_file = pq.ParquetFile(
                        handle,
                        metadata=metadatas[task.file_index],
                        pre_buffer=True,
                    )
                    return parquet_file.read_row_group(
                        task.row_group, columns=columns, use_threads=False
                    )
                finally:
                    handle.close()

            def read_row_group(task: _ReadTask) -> pa.Table:
                description = (
                    f"Reading row group {task.row_group} of {files[task.file_index]}"
                )
                # Concurrent reads share one handle (positional reads). Retries each open a fresh handle.
                try:
                    return read_with_shared_handle(task)
                except Exception as e:
                    if not _is_retryable(e):
                        raise
                    logger.warning(f"{description} failed, retrying: {e}")
                return _retry_transient(
                    description, lambda: read_with_fresh_handle(task)
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
            # cancelled.
            try:
                for _ in pool.map(read_task, tasks):
                    pass
            finally:
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
