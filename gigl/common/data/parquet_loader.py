"""Loads node and edge tensors from Parquet files (see ``ParquetDataLoader``).

Expected layout under ``SerializedTFRecordInfo.parquet_uri_prefix``: ``part-XXXXX-of-YYYYY.parquet`` files with

* nodes: ``node_id`` (int64) and, if the node type has features, ``features`` as
  ``FixedSizeList<float32>[feature_dim]`` with slots in ``feature_keys`` order;
* edges: ``src`` and ``dst`` (int64) and, if the edge type has features, ``features`` as above;
* if the entity has packed features, one ``FixedSizeBinary[packed_feature_dim]`` column named ``packed_feature_key``
  with the packed bytes of each row; ``features`` then holds only the features that are not packed;
* optional label columns named by ``label_keys``, each a numeric scalar or a fixed-size list.
"""

import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, TypeVar

import numpy as np
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import torch

from gigl.common import Uri
from gigl.common.data import parquet_manifest
from gigl.common.data.loader_types import LoadedEntityTensors, SerializedTFRecordInfo
from gigl.common.logger import Logger
from gigl.utils.share_memory import allocate_preshared

logger = Logger()

_NODE_ID_COLUMN = "node_id"
_SRC_COLUMN = "src"
_DST_COLUMN = "dst"
_FEATURES_COLUMN = "features"
_MAX_THREADS = 64
_MAX_ATTEMPTS = 3
# (min, max) seconds of the uniformly jittered sleep before a retry.
_RETRY_BACKOFF_SECONDS = (2.0, 30.0)

_T = TypeVar("_T")


@dataclass(frozen=True)
class _ReadTask:
    """Reads ``num_rows`` rows starting at ``row_offset`` of one row group into ``out_offset`` of the output."""

    file_index: int
    row_group: int
    row_offset: int
    num_rows: int
    out_offset: int


def _is_retryable(error: Exception) -> bool:
    """Whether ``error`` is a transient I/O failure worth retrying with a fresh handle.

    Missing files and permission errors are permanent. Generic ``pa.ArrowException`` is what pyarrow raises for
    storage-client failures without a more specific status (for example an unknown error from the GCS client);
    its subclasses (invalid data, type errors, ...) are not transient.
    """
    if isinstance(error, (FileNotFoundError, PermissionError)):
        return False
    if isinstance(error, OSError):
        return True
    return type(error) is pa.ArrowException


class _Retrier:
    """Retries operations that fail with a transient I/O error and counts file opens, open handles and retries.

    Operations take the zero-based attempt number so they can re-open a fresh file handle on retries. Handles
    must be opened and closed through ``open`` and ``close`` so that ``max_open_handles`` is accurate.
    """

    def __init__(self, fs: pafs.FileSystem) -> None:
        self._fs = fs
        self._lock = threading.Lock()
        self._open_handles = 0
        self.opens = 0
        self.max_open_handles = 0
        self.retries = 0

    def open(self, file: str) -> pa.NativeFile:
        handle = self._fs.open_input_file(file)
        with self._lock:
            self.opens += 1
            self._open_handles += 1
            self.max_open_handles = max(self.max_open_handles, self._open_handles)
        return handle

    def close(self, handle: pa.NativeFile) -> None:
        handle.close()
        with self._lock:
            self._open_handles -= 1

    def call(self, description: str, operation: Callable[[int], _T]) -> _T:
        for attempt in range(_MAX_ATTEMPTS):
            try:
                return operation(attempt)
            except Exception as e:
                if attempt == _MAX_ATTEMPTS - 1 or not _is_retryable(e):
                    raise
                backoff = random.uniform(*_RETRY_BACKOFF_SECONDS)
                logger.warning(
                    f"{description} failed on attempt {attempt + 1} of {_MAX_ATTEMPTS}, "
                    f"retrying in {backoff:.1f}s: {e}"
                )
                with self._lock:
                    self.retries += 1
                time.sleep(backoff)
        raise AssertionError("unreachable")


class _SharedHandles:
    """One lazily opened handle per file, shared by concurrent reads and closed after the file's last task.

    Tasks run in file order on a pool, so at most as many files as pool threads are open at once.
    """

    def __init__(
        self, retrier: _Retrier, files: Sequence[str], tasks: Sequence[_ReadTask]
    ) -> None:
        self._retrier = retrier
        self._files = files
        self._remaining = [0] * len(files)
        for task in tasks:
            self._remaining[task.file_index] += 1
        self._locks = [threading.Lock() for _ in files]
        self._handles: list[Optional[pa.NativeFile]] = [None] * len(files)

    def get(self, file_index: int) -> pa.NativeFile:
        with self._locks[file_index]:
            handle = self._handles[file_index]
            if handle is None:
                handle = self._retrier.open(self._files[file_index])
                self._handles[file_index] = handle
            return handle

    def finish_task(self, file_index: int) -> None:
        """Marks one task of the file done and closes the file's handle after its last task."""
        with self._locks[file_index]:
            self._remaining[file_index] -= 1
            if self._remaining[file_index] == 0:
                self._close_locked(file_index)

    def close_all(self) -> None:
        for file_index in range(len(self._files)):
            with self._locks[file_index]:
                self._close_locked(file_index)

    def _close_locked(self, file_index: int) -> None:
        handle = self._handles[file_index]
        if handle is not None:
            self._handles[file_index] = None
            self._retrier.close(handle)


def _open_fd_count() -> int:
    """Number of file descriptors open in this process, or -1 where ``/proc`` is unavailable."""
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:
        return -1


def _column_width(field: pa.Field) -> int:
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

    Produces the same ``LoadedEntityTensors`` as ``TFRecordDataLoader.load_as_torch_tensors`` for the same
    data. The difference is how data is split across ranks: ``TFRecordDataLoader`` gives each rank whole
    files, here the rows of all files, in sorted file order, are split into ``world_size`` contiguous
    ranges of equal size (they differ by at most one row) and each rank reads the row groups that overlap
    its range. A row group on a range boundary is read by both neighboring ranks, which each keep
    their own rows.

    Row counts come from the manifest written next to the files (see ``gigl.common.data.parquet_manifest``) when it
    matches the files found, and from the file footers otherwise.

    Outputs are allocated once from those counts, in shared memory (or spilled to disk, see
    ``gigl.utils.share_memory.allocate_preshared``), and filled by a thread pool. No intermediate batches are kept and
    the outputs need no later copy into shared memory.

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

    @staticmethod
    def _list_files(fs: pafs.FileSystem, path: str) -> list[str]:
        return parquet_manifest.list_parquet_files(fs, path)

    @staticmethod
    def _manifest_row_groups(
        fs: pafs.FileSystem, path: str, files: Sequence[str]
    ) -> Optional[list[list[int]]]:
        """Row-group row counts of ``files`` from the manifest, or None (with a warning) if it cannot be used."""
        try:
            return parquet_manifest.read_manifest_row_groups(fs, path, list(files))
        except (FileNotFoundError, ValueError, KeyError, TypeError) as e:
            logger.warning(
                f"Not using the row-count manifest under {path}, reading all footers instead: "
                f"{type(e).__name__}: {e}"
            )
            return None

    def _plan_tasks(
        self, row_groups_per_file: Sequence[Sequence[int]]
    ) -> list[_ReadTask]:
        """Row groups (and slices of boundary row groups) making up this rank's row range."""
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

    @staticmethod
    def _empty_tensors(info: SerializedTFRecordInfo) -> LoadedEntityTensors:
        if info.is_node_entity:
            ids = torch.empty(0, dtype=torch.int64)
        else:
            ids = torch.empty(2, 0, dtype=torch.int64)
        features = None
        if info.feature_keys:
            features = torch.empty(0, info.feature_dim)
        labels = None
        if info.label_keys:
            labels = torch.empty(0, len(info.label_keys))
        packed_features = None
        if info.packed_feature_key is not None:
            packed_features = torch.empty(0, info.packed_feature_dim, dtype=torch.uint8)
        return LoadedEntityTensors(
            ids=ids,
            features=features,
            packed_features=packed_features,
            labels=labels,
        )

    def load_as_torch_tensors(
        self, serialized_tf_record_info: SerializedTFRecordInfo
    ) -> LoadedEntityTensors:
        """Loads this rank's share of the Parquet files at ``parquet_uri_prefix`` as torch tensors.

        Args:
            serialized_tf_record_info (SerializedTFRecordInfo): How the entity is stored. Must have
                ``parquet_uri_prefix`` set.
        Returns:
            LoadedEntityTensors: The (ids, features, packed features, labels) of this rank's rows.
                Packed features are the uint8 ``[rows, packed_feature_dim]`` packed bytes and are None
                unless ``packed_feature_key`` is set.
        """
        info = serialized_tf_record_info
        if info.parquet_uri_prefix is None:
            raise ValueError("ParquetDataLoader requires parquet_uri_prefix to be set")
        fd_count = _open_fd_count()
        wall_start = time.perf_counter()

        fs, path = self._filesystem_and_path(info.parquet_uri_prefix)
        retrier = _Retrier(fs)
        files = retrier.call(
            "Listing Parquet files", lambda attempt: self._list_files(fs, path)
        )
        list_s = time.perf_counter() - wall_start
        if not files:
            logger.warning(
                f"Found no Parquet files under {info.parquet_uri_prefix.uri}"
            )
            return self._empty_tensors(info)

        # Reads with pre_buffer=True run on pyarrow's I/O pool, which caps the number of in-flight requests.
        pa.set_io_thread_count(max(pa.io_thread_count(), self._num_threads))

        with ThreadPoolExecutor(self._num_threads) as pool:
            footer_start = time.perf_counter()

            def read_footer(file: str) -> pq.FileMetaData:
                """Opens ``file``, reads its footer and closes it again."""

                def attempt_read(attempt: int) -> pq.FileMetaData:
                    handle = retrier.open(file)
                    try:
                        return pq.read_metadata(handle)
                    finally:
                        retrier.close(handle)

                return retrier.call(f"Reading footer of {file}", attempt_read)

            def read_footers(file_indices: Sequence[int]) -> dict[int, pq.FileMetaData]:
                metadata_list = pool.map(
                    read_footer, [files[index] for index in file_indices]
                )
                return dict(zip(file_indices, metadata_list))

            row_groups_per_file = self._manifest_row_groups(fs, path, files)
            manifest_used = row_groups_per_file is not None
            metadatas: dict[int, pq.FileMetaData] = {}
            if row_groups_per_file is None:
                metadatas = read_footers(range(len(files)))
                row_groups_per_file = []
                for index in range(len(files)):
                    metadata = metadatas[index]
                    row_groups_per_file.append(
                        [
                            metadata.row_group(group).num_rows
                            for group in range(metadata.num_row_groups)
                        ]
                    )
            tasks = self._plan_tasks(row_groups_per_file)
            if manifest_used:
                # Reads and the schema still need the footers of the files this rank reads from.
                metadatas = read_footers(sorted({task.file_index for task in tasks}))
            footers_s = time.perf_counter() - footer_start
            num_rows = sum(task.num_rows for task in tasks)
            if num_rows == 0:
                logger.info(f"No rows to load for rank: {self._rank}.")
                return self._empty_tensors(info)

            if info.is_node_entity:
                id_columns = [_NODE_ID_COLUMN]
            else:
                id_columns = [_SRC_COLUMN, _DST_COLUMN]
            columns = list(id_columns)
            has_features = bool(info.feature_keys)
            if has_features:
                columns.append(_FEATURES_COLUMN)
            if info.packed_feature_key is not None:
                columns.append(info.packed_feature_key)
            columns.extend(info.label_keys)

            # Shards without row groups carry only the id columns, so the schema comes from a file with rows.
            schema = metadatas[tasks[0].file_index].schema.to_arrow_schema()
            label_widths = [_column_width(schema.field(key)) for key in info.label_keys]
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
            ids_np = ids.numpy()
            features_np = None
            if features is not None:
                features_np = features.numpy()
            packed_features_np = None
            if packed_features is not None:
                packed_features_np = packed_features.numpy()
            labels_np = None
            if labels is not None:
                labels_np = labels.numpy()

            shared_handles = _SharedHandles(retrier, files, tasks)

            def read_row_group(task: _ReadTask, attempt: int) -> pa.Table:
                # Concurrent reads share one handle (positional reads). A retry uses its own fresh handle.
                if attempt == 0:
                    parquet_file = pq.ParquetFile(
                        shared_handles.get(task.file_index),
                        metadata=metadatas[task.file_index],
                        pre_buffer=True,
                    )
                    return parquet_file.read_row_group(
                        task.row_group, columns=columns, use_threads=False
                    )
                fresh_handle = retrier.open(files[task.file_index])
                try:
                    parquet_file = pq.ParquetFile(
                        fresh_handle,
                        metadata=metadatas[task.file_index],
                        pre_buffer=True,
                    )
                    return parquet_file.read_row_group(
                        task.row_group, columns=columns, use_threads=False
                    )
                finally:
                    retrier.close(fresh_handle)

            def read_task(task: _ReadTask) -> None:
                try:
                    table = retrier.call(
                        f"Reading row group {task.row_group} of {files[task.file_index]}",
                        lambda attempt: read_row_group(task, attempt),
                    )
                finally:
                    shared_handles.finish_task(task.file_index)
                table = table.slice(task.row_offset, task.num_rows)
                out = slice(task.out_offset, task.out_offset + task.num_rows)
                if info.is_node_entity:
                    _copy_into(table.column(_NODE_ID_COLUMN), ids_np[out])
                else:
                    _copy_into(table.column(_SRC_COLUMN), ids_np[0, out])
                    _copy_into(table.column(_DST_COLUMN), ids_np[1, out])
                if features_np is not None:
                    _copy_into(table.column(_FEATURES_COLUMN), features_np[out])
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
            try:
                for _ in pool.map(read_task, tasks):
                    pass
            finally:
                shared_handles.close_all()
            read_s = time.perf_counter() - read_start

        compressed_bytes = 0
        for task in tasks:
            row_group_metadata = metadatas[task.file_index].row_group(task.row_group)
            compressed_bytes += sum(
                row_group_metadata.column(i).total_compressed_size
                for i in range(row_group_metadata.num_columns)
            )
        wall_s = time.perf_counter() - wall_start
        logger.info(
            f"Loaded Parquet entity: rank={self._rank} files={len(files)} "
            f"row_groups={len(tasks)} manifest_used={manifest_used} list_s={list_s:.2f} footers_s={footers_s:.2f} "
            f"read_s={read_s:.2f} bytes={compressed_bytes} rows={num_rows} "
            f"opens={retrier.opens} max_open_handles={retrier.max_open_handles} "
            f"threads={self._num_threads} fd_count={fd_count} retries={retrier.retries} "
            f"wall_s={wall_s:.2f} "
            f"rows_per_s={num_rows / max(wall_s, 1e-9):.0f}"
        )
        return LoadedEntityTensors(
            ids=ids,
            features=features,
            packed_features=packed_features,
            labels=labels,
        )
