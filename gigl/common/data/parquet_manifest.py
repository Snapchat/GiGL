"""Row-count manifest of a directory of Parquet files.

Reading the footer of every Parquet file of an entity to learn its row-group sizes costs one storage request per file on
every rank. The manifest ``_row_counts.json`` is written once, next to the data files, and holds the same information::

    {"version": 1, "files": {"part-00000-of-00002.parquet": [100000, 100000, 523], ...}}

Keys are file names relative to the directory and values are the row counts of the file's row groups, in order. The name
does not end in ``.parquet``, so file listings of the data never include it.

The data preprocessor writes one directory per node type and per edge type. The manifest sits next to the data files:

- Node type ``user``::

    gs://<temp bucket>/<job name>/data_preprocess/staging/transformed_node_features_dir/user/parquet/_row_counts.json
    gs://<temp bucket>/<job name>/data_preprocess/staging/transformed_node_features_dir/user/parquet/part-00000-of-00042.parquet
    gs://<temp bucket>/<job name>/data_preprocess/staging/transformed_node_features_dir/user/parquet/part-00001-of-00042.parquet

- Edge type ``user-follows-user``::

    gs://<temp bucket>/<job name>/data_preprocess/staging/transformed_edge_features_dir/user-follows-user/parquet/_row_counts.json
    gs://<temp bucket>/<job name>/data_preprocess/staging/transformed_edge_features_dir/user-follows-user/parquet/part-00000-of-00042.parquet
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Any, Final

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from gigl.common import GcsUri, Uri
from gigl.common.logger import Logger

logger = Logger()

MANIFEST_FILE_NAME: Final[str] = "_row_counts.json"
MANIFEST_VERSION: Final[int] = 1
PARQUET_SUFFIX: Final[str] = ".parquet"
GCS_RETRY_TIME_LIMIT: Final[timedelta] = timedelta(seconds=300)
_DEFAULT_THREADS: Final[int] = 32
_FOOTER_ATTEMPTS: Final[int] = 3


def filesystem_and_path(uri: Uri) -> tuple[pafs.FileSystem, str]:
    """Returns the filesystem for ``uri`` and the path of ``uri`` within it.

    GCS URIs get a GCS filesystem with a longer retry limit, and the path is ``<bucket>/<object path>``. Any other URI
    is resolved by pyarrow.

    Args:
        uri: The location of the directory or file.

    Returns:
        The filesystem and the scheme-less path within it, such as ``my-bucket/my_job/parquet/``.
    """
    if isinstance(uri, GcsUri):
        fs = pafs.GcsFileSystem(retry_time_limit=GCS_RETRY_TIME_LIMIT)
        return fs, f"{uri.bucket}/{uri.path}"
    return pafs.FileSystem.from_uri(uri.uri)


def list_parquet_files(fs: pafs.FileSystem, path: str) -> list[str]:
    """Returns the sorted paths of the Parquet data files directly under ``path``.

    Args:
        fs: The filesystem that holds ``path``.
        path: The scheme-less directory path within ``fs``.

    Returns:
        The sorted full paths of the files that end in ``.parquet``. The manifest itself is not included.
    """
    selector = pafs.FileSelector(path.rstrip("/"), recursive=False)
    infos = fs.get_file_info(selector)
    parquet_files: list[str] = []
    for info in infos:
        if info.type == pafs.FileType.File and info.path.endswith(PARQUET_SUFFIX):
            parquet_files.append(info.path)
    return sorted(parquet_files)


def manifest_path(path: str) -> str:
    """Returns the path of the manifest in the directory ``path``.

    Args:
        path: The scheme-less directory path, with or without a trailing slash.

    Returns:
        ``path`` joined with the manifest file name.
    """
    return path.rstrip("/") + "/" + MANIFEST_FILE_NAME


def read_manifest_row_groups(
    fs: pafs.FileSystem, path: str, files: list[str]
) -> list[list[int]]:
    """Returns the row-group row counts of ``files`` (as listed under ``path``) from the manifest under ``path``.

    Args:
        fs: The filesystem that holds ``path``.
        path: The scheme-less directory path within ``fs``.
        files: The data file paths listed under ``path``, as returned by ``list_parquet_files``.

    Returns:
        One list of row-group row counts per entry of ``files``, in the same order.

    Raises:
        FileNotFoundError: If there is no manifest.
        ValueError: If the manifest is malformed, has an unknown version, or does not list exactly ``files``.
    """
    if fs.get_file_info(manifest_path(path)).type == pafs.FileType.NotFound:
        raise FileNotFoundError(manifest_path(path))
    with fs.open_input_file(manifest_path(path)) as file:
        manifest = json.loads(file.read())
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError(f"unsupported manifest version {manifest.get('version')!r}")
    manifest_files = manifest["files"]
    names = [os.path.basename(file) for file in files]
    if set(names) != set(manifest_files):
        raise ValueError(
            f"manifest lists {len(manifest_files)} files but {len(names)} were found"
        )
    return [manifest_files[name] for name in names]


def _read_row_groups(fs: pafs.FileSystem, file: str) -> list[int]:
    """Returns the row counts of the row groups of ``file``, retrying transient read failures."""
    for attempt in range(_FOOTER_ATTEMPTS):
        try:
            metadata = pq.read_metadata(file, filesystem=fs)
            return [
                metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)
            ]
        except (OSError, pa.ArrowException) as e:
            if attempt == _FOOTER_ATTEMPTS - 1:
                raise
            logger.warning(f"Reading footer of {file} failed, retrying: {e}")
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def build_manifest(
    uri_prefix: Uri, num_threads: int = _DEFAULT_THREADS
) -> dict[str, Any]:
    """Builds the manifest of the Parquet files under ``uri_prefix`` by reading each footer once.

    Args:
        uri_prefix: The directory that holds the Parquet files.
        num_threads: The number of threads that read footers in parallel.

    Returns:
        The manifest, with the keys ``version`` and ``files``.
    """
    fs, path = filesystem_and_path(uri_prefix)
    files = list_parquet_files(fs, path)
    with ThreadPoolExecutor(num_threads) as pool:
        row_groups = list(pool.map(lambda file: _read_row_groups(fs, file), files))
    names = [os.path.basename(file) for file in files]
    return {"version": MANIFEST_VERSION, "files": dict(zip(names, row_groups))}


def write_manifest(
    uri_prefix: Uri, num_threads: int = _DEFAULT_THREADS
) -> dict[str, Any]:
    """Builds the manifest of the Parquet files under ``uri_prefix`` and writes it there.

    Args:
        uri_prefix: The directory that holds the Parquet files.
        num_threads: The number of threads that read footers in parallel.

    Returns:
        The manifest that was written.
    """
    manifest = build_manifest(uri_prefix, num_threads)
    fs, path = filesystem_and_path(uri_prefix)
    with fs.open_output_stream(manifest_path(path)) as stream:
        stream.write(json.dumps(manifest).encode())
    logger.info(
        f"Wrote row-count manifest of {len(manifest['files'])} files to {manifest_path(uri_prefix.uri)}"
    )
    return manifest
