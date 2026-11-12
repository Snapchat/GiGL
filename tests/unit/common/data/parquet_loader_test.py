import io
import json
import os
import re
import resource
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Union
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import tensorflow as tf
import torch
from absl.testing import absltest
from parameterized import parameterized
from torch.testing import assert_close

from gigl.common import Uri, UriFactory
from gigl.common.data import parquet_format, parquet_manifest
from gigl.common.data.load_torch_tensors import (
    SerializedGraphMetadata,
    load_torch_tensors_from_tf_record,
)
from gigl.common.data.loader_types import (
    LoadedEntityTensors,
    SerializedTFRecordInfo,
    TFDatasetOptions,
)
from gigl.common.data.parquet_loader import ParquetDataLoader
from gigl.common.data.tf_record_loader import TFRecordDataLoader
from gigl.src.data_preprocessor.lib.types import FeatureSpecDict
from tests.test_assets.test_case import TestCase

_FD_SETSIZE = 1024
# Retries sleep between attempts; tests skip the wait.
_RETRY_SLEEP = "gigl.common.utils.retry.time.sleep"
_NUM_NODES = 53
_FEATURE_DIM = 3
# Files with different numbers of row groups of different sizes, 53 rows in total.
_ROWS_PER_FILE = [20, 1, 17, 15]
_ROW_GROUP_SIZE = 7

_NODE_FEATURE_SPEC: FeatureSpecDict = {
    "node_id": tf.io.FixedLenFeature([], tf.int64),
    "feature_a": tf.io.FixedLenFeature([], tf.float32),
    "feature_b": tf.io.FixedLenFeature([2], tf.int64),
    "label_a": tf.io.FixedLenFeature([], tf.int64),
    "label_b": tf.io.FixedLenFeature([2], tf.float32),
}
_EDGE_FEATURE_SPEC: FeatureSpecDict = {
    "src": tf.io.FixedLenFeature([], tf.int64),
    "dst": tf.io.FixedLenFeature([], tf.int64),
    "weight": tf.io.FixedLenFeature([], tf.float32),
}


def _node_data() -> dict[str, np.ndarray]:
    ids = np.arange(_NUM_NODES, dtype=np.int64) * 3 + 11
    return {
        "node_id": ids,
        "feature_a": (np.arange(_NUM_NODES) * 0.37).astype(np.float32),
        # Large integers lose precision when cast to float32; both loaders must round alike.
        "feature_b": np.stack(
            [ids * 1_000_003 + 1, np.arange(_NUM_NODES, dtype=np.int64) % 5], axis=1
        ),
        "label_a": np.arange(_NUM_NODES, dtype=np.int64) % 4,
        "label_b": np.stack(
            [np.arange(_NUM_NODES) * 0.5, np.arange(_NUM_NODES) * -1.5], axis=1
        ).astype(np.float32),
    }


def _write_parquet_nodes(
    directory: Path, data: dict[str, np.ndarray], with_features: bool, with_labels: bool
) -> None:
    """Writes node data in the Parquet layout read by ParquetDataLoader, split over several files."""
    features = np.concatenate(
        [data["feature_a"][:, None], data["feature_b"].astype(np.float32)], axis=1
    )
    columns: dict[str, pa.Array] = {"node_id": pa.array(data["node_id"])}
    if with_features:
        columns["features"] = pa.FixedSizeListArray.from_arrays(
            pa.array(features.reshape(-1)), _FEATURE_DIM
        )
    if with_labels:
        columns["label_a"] = pa.array(data["label_a"])
        columns["label_b"] = pa.FixedSizeListArray.from_arrays(
            pa.array(data["label_b"].reshape(-1)), 2
        )
    table = pa.table(columns)
    start = 0
    for file_index, num_rows in enumerate(_ROWS_PER_FILE):
        pq.write_table(
            table.slice(start, num_rows),
            directory / f"part-{file_index:05d}-of-{len(_ROWS_PER_FILE):05d}.parquet",
            row_group_size=_ROW_GROUP_SIZE,
            compression="zstd",
        )
        start += num_rows
    parquet_manifest.write_manifest(UriFactory.create_uri(directory))


def _write_tfrecord_nodes(directory: Path, data: dict[str, np.ndarray]) -> None:
    start = 0
    for file_index, num_rows in enumerate(_ROWS_PER_FILE):
        path = (
            directory / f"part-{file_index:05d}-of-{len(_ROWS_PER_FILE):05d}.tfrecord"
        )
        with tf.io.TFRecordWriter(str(path)) as writer:
            for row in range(start, start + num_rows):
                feature = {
                    "node_id": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[data["node_id"][row]])
                    ),
                    "feature_a": tf.train.Feature(
                        float_list=tf.train.FloatList(value=[data["feature_a"][row]])
                    ),
                    "feature_b": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=data["feature_b"][row])
                    ),
                    "label_a": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[data["label_a"][row]])
                    ),
                    "label_b": tf.train.Feature(
                        float_list=tf.train.FloatList(value=data["label_b"][row])
                    ),
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
        start += num_rows


def _node_info(
    directory: Path,
    parquet: bool,
    with_features: bool = True,
    with_labels: bool = True,
) -> SerializedTFRecordInfo:
    uri = UriFactory.create_uri(directory)
    feature_keys = []
    if with_features:
        feature_keys = ["feature_a", "feature_b"]
    label_keys = []
    if with_labels:
        label_keys = ["label_a", "label_b"]
    return SerializedTFRecordInfo(
        tfrecord_uri_prefix=uri,
        parquet_uri_prefix=uri if parquet else None,
        feature_keys=feature_keys,
        feature_spec=_NODE_FEATURE_SPEC,
        feature_dim=_FEATURE_DIM if with_features else 0,
        entity_key="node_id",
        label_keys=label_keys,
    )


def _present(tensor: Optional[torch.Tensor]) -> torch.Tensor:
    assert tensor is not None
    return tensor


def _sorted_by_id(loaded_ids, *others: Optional[torch.Tensor]) -> list:
    order = torch.argsort(loaded_ids)
    return [loaded_ids[order]] + [
        None if other is None else other[order] for other in others
    ]


class ParquetDataLoaderTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self._temp_dir.name)
        self.data = _node_data()

    def tearDown(self) -> None:
        self._temp_dir.cleanup()
        super().tearDown()

    @parameterized.expand([(1,), (2,), (3,), (4,), (7,)])
    def test_nodes_with_features_and_labels_split_exactly_across_ranks(
        self, world_size: int
    ) -> None:
        directory = self.root / "nodes"
        directory.mkdir()
        _write_parquet_nodes(directory, self.data, True, True)
        info = _node_info(directory, parquet=True)

        features = np.concatenate(
            [
                self.data["feature_a"][:, None],
                self.data["feature_b"].astype(np.float32),
            ],
            axis=1,
        )
        labels = np.concatenate(
            [self.data["label_a"][:, None].astype(np.float32), self.data["label_b"]],
            axis=1,
        )
        rank_sizes = []
        all_ids, all_features, all_labels = [], [], []
        for rank in range(world_size):
            loaded = ParquetDataLoader(rank, world_size).load_as_torch_tensors(info)
            self.assertEqual(loaded.ids.dtype, torch.int64)
            self.assertEqual(_present(loaded.features).dtype, torch.float32)
            self.assertEqual(_present(loaded.labels).dtype, torch.float32)
            self.assertIsNone(loaded.packed_features)
            rank_sizes.append(loaded.ids.shape[0])
            all_ids.append(loaded.ids)
            all_features.append(loaded.features)
            all_labels.append(loaded.labels)

        self.assertLessEqual(max(rank_sizes) - min(rank_sizes), 1)
        # Ranks hold contiguous row ranges, so concatenating them restores the stored order.
        assert_close(torch.cat(all_ids), torch.from_numpy(self.data["node_id"]))
        assert_close(torch.cat(all_features), torch.from_numpy(features))
        assert_close(torch.cat(all_labels), torch.from_numpy(labels))

    def test_nodes_without_features_or_labels(self) -> None:
        directory = self.root / "nodes"
        directory.mkdir()
        _write_parquet_nodes(directory, self.data, False, False)
        info = _node_info(
            directory, parquet=True, with_features=False, with_labels=False
        )

        loaded = ParquetDataLoader(1, 2).load_as_torch_tensors(info)

        assert_close(loaded.ids, torch.from_numpy(self.data["node_id"][26:]))
        self.assertIsNone(loaded.features)
        self.assertIsNone(loaded.labels)

    def test_nodes_with_labels_only(self) -> None:
        directory = self.root / "nodes"
        directory.mkdir()
        _write_parquet_nodes(directory, self.data, False, True)
        info = _node_info(
            directory, parquet=True, with_features=False, with_labels=True
        )

        loaded = ParquetDataLoader(0, 1).load_as_torch_tensors(info)

        self.assertIsNone(loaded.features)
        self.assertEqual(_present(loaded.labels).shape, (_NUM_NODES, 3))

    @parameterized.expand([(1,), (2,), (4,)])
    def test_edges_with_and_without_features(self, world_size: int) -> None:
        src = np.arange(40, dtype=np.int64)
        dst = (src * 7) % 40
        weight = (src * 0.25).astype(np.float32)
        for with_features in [True, False]:
            directory = self.root / f"edges_{with_features}"
            directory.mkdir()
            columns: dict[str, pa.Array] = {"src": pa.array(src), "dst": pa.array(dst)}
            if with_features:
                columns["features"] = pa.FixedSizeListArray.from_arrays(
                    pa.array(weight), 1
                )
            table = pa.table(columns)
            for file_index in range(3):
                pq.write_table(
                    table.slice(file_index * 14, 14 if file_index < 2 else 12),
                    directory / f"part-{file_index:05d}-of-00003.parquet",
                    row_group_size=5,
                )
            uri = UriFactory.create_uri(directory)
            parquet_manifest.write_manifest(uri)
            feature_keys = []
            if with_features:
                feature_keys = ["weight"]
            info = SerializedTFRecordInfo(
                tfrecord_uri_prefix=uri,
                parquet_uri_prefix=uri,
                feature_keys=feature_keys,
                feature_spec=_EDGE_FEATURE_SPEC,
                feature_dim=len(feature_keys),
                entity_key=("src", "dst"),
            )

            loaded = [
                ParquetDataLoader(rank, world_size).load_as_torch_tensors(info)
                for rank in range(world_size)
            ]

            assert_close(
                torch.cat([part.ids for part in loaded], dim=1),
                torch.from_numpy(np.stack([src, dst])),
            )
            if with_features:
                assert_close(
                    torch.cat([_present(part.features) for part in loaded]),
                    torch.from_numpy(weight).reshape(-1, 1),
                )
            else:
                self.assertTrue(all(part.features is None for part in loaded))

    def test_more_ranks_than_rows_gives_empty_ranks(self) -> None:
        directory = self.root / "nodes"
        directory.mkdir()
        _write_parquet_nodes(directory, self.data, True, True)
        info = _node_info(directory, parquet=True)

        sizes = []
        for rank in range(60):
            loaded = ParquetDataLoader(rank, 60).load_as_torch_tensors(info)
            sizes.append(loaded.ids.shape[0])
            if loaded.ids.shape[0] == 0:
                self.assertEqual(_present(loaded.features).shape, (0, _FEATURE_DIM))
                self.assertEqual(_present(loaded.labels).shape, (0, 2))
        self.assertEqual(sum(sizes), _NUM_NODES)
        self.assertIn(0, sizes)

    def test_empty_directory_gives_empty_tensors(self) -> None:
        directory = self.root / "empty"
        directory.mkdir()
        loaded = ParquetDataLoader(0, 2).load_as_torch_tensors(
            _node_info(directory, parquet=True)
        )
        self.assertEqual(loaded.ids.shape, (0,))
        self.assertEqual(_present(loaded.features).shape, (0, _FEATURE_DIM))

    @parameterized.expand([(1,), (3,)])
    def test_matches_tfrecord_loader(self, world_size: int) -> None:
        parquet_dir = self.root / "parquet"
        tfrecord_dir = self.root / "tfrecord"
        parquet_dir.mkdir()
        tfrecord_dir.mkdir()
        _write_parquet_nodes(parquet_dir, self.data, True, True)
        _write_tfrecord_nodes(tfrecord_dir, self.data)

        parquet_parts = []
        tfrecord_parts = []
        for rank in range(world_size):
            parquet_parts.append(
                ParquetDataLoader(rank, world_size).load_as_torch_tensors(
                    _node_info(parquet_dir, parquet=True)
                )
            )
            tfrecord_parts.append(
                TFRecordDataLoader(rank, world_size).load_as_torch_tensors(
                    _node_info(tfrecord_dir, parquet=False),
                    TFDatasetOptions(deterministic=True),
                )
            )

        # TFRecord assigns whole files to ranks and Parquet assigns row ranges, so only the union over
        # ranks (sorted by id) is comparable, and for world size 1 that is also the per-rank result.
        parquet_ids, parquet_features, parquet_labels = _sorted_by_id(
            torch.cat([p.ids for p in parquet_parts]),
            torch.cat([_present(p.features) for p in parquet_parts]),
            torch.cat([_present(p.labels) for p in parquet_parts]),
        )
        tfrecord_ids, tfrecord_features, tfrecord_labels = _sorted_by_id(
            torch.cat([p.ids for p in tfrecord_parts]),
            torch.cat([_present(p.features) for p in tfrecord_parts]),
            torch.cat([_present(p.labels) for p in tfrecord_parts]),
        )
        assert_close(parquet_ids, tfrecord_ids, rtol=0, atol=0)
        assert_close(parquet_features, tfrecord_features, rtol=0, atol=0)
        assert_close(parquet_labels, tfrecord_labels, rtol=0, atol=0)
        self.assertEqual(parquet_features.dtype, tfrecord_features.dtype)

    def _write_local_graph(self) -> tuple[SerializedGraphMetadata, np.ndarray]:
        node_dir = self.root / "nodes"
        edge_dir = self.root / "edges"
        node_dir.mkdir()
        edge_dir.mkdir()
        _write_parquet_nodes(node_dir, self.data, True, False)
        src = np.arange(30, dtype=np.int64)
        dst = (src + 1) % 30
        pq.write_table(
            pa.table({"src": pa.array(src), "dst": pa.array(dst)}),
            edge_dir / "part-00000-of-00001.parquet",
            row_group_size=8,
        )
        edge_uri = UriFactory.create_uri(edge_dir)
        parquet_manifest.write_manifest(edge_uri)
        edge_info = SerializedTFRecordInfo(
            tfrecord_uri_prefix=edge_uri,
            parquet_uri_prefix=edge_uri,
            feature_keys=[],
            feature_spec=_EDGE_FEATURE_SPEC,
            feature_dim=0,
            entity_key=("src", "dst"),
        )
        metadata = SerializedGraphMetadata(
            node_entity_info=_node_info(node_dir, parquet=True, with_labels=False),
            edge_entity_info=edge_info,
        )
        return metadata, np.stack([src, dst])

    def test_load_torch_tensors_from_tf_record_reads_parquet(self) -> None:
        metadata, edge_index = self._write_local_graph()

        loaded = load_torch_tensors_from_tf_record(
            world_size=2,
            serialized_graph_metadata=metadata,
            should_load_tensors_in_parallel=True,
            rank=1,
        )

        assert_close(loaded.node_ids, torch.from_numpy(self.data["node_id"][26:]))
        assert_close(loaded.edge_index, torch.from_numpy(edge_index[:, 15:]))
        assert isinstance(loaded.node_features, torch.Tensor)
        assert isinstance(loaded.edge_index, torch.Tensor)
        self.assertEqual(loaded.node_features.shape, (_NUM_NODES - 26, _FEATURE_DIM))
        self.assertTrue(loaded.edge_index.is_shared())

    def test_load_torch_tensors_from_tf_record_reads_parquet_with_high_fd_numbers(
        self,
    ) -> None:
        metadata, edge_index = self._write_local_graph()
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard_limit, 4096), hard_limit))
        held_fds: list[int] = []
        log_path = self.root / "child.log"
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)
        log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT)
        try:
            for _ in range(_FD_SETSIZE + 100):
                held_fds.append(os.open(os.devnull, os.O_RDONLY))
            # Spawned children inherit the redirected stdout and stderr, which carry their log lines.
            os.dup2(log_fd, 1)
            os.dup2(log_fd, 2)
            loaded = load_torch_tensors_from_tf_record(
                world_size=1,
                serialized_graph_metadata=metadata,
                should_load_tensors_in_parallel=True,
                rank=0,
            )
        finally:
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            for fd in [saved_stdout, saved_stderr, log_fd, *held_fds]:
                os.close(fd)
            resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

        assert_close(loaded.node_ids, torch.from_numpy(self.data["node_id"]))
        assert_close(loaded.edge_index, torch.from_numpy(edge_index))
        fd_counts = re.findall(r"fd_count=(\d+)", log_path.read_text())
        self.assertNotEqual(fd_counts, [])
        for fd_count in fd_counts:
            self.assertLess(int(fd_count), _FD_SETSIZE)


_PACKED_DIM = 3
_PACKED_ROWS_PER_FILE = [5, 1, 4]


def _packed_bytes(num_rows: int) -> np.ndarray:
    """Distinct ``[num_rows, _PACKED_DIM]`` uint8 rows including 0 and 255."""
    return (
        ((np.arange(num_rows * _PACKED_DIM) * 37 + 250) % 256)
        .astype(np.uint8)
        .reshape(num_rows, _PACKED_DIM)
    )


def _write_packed_entity_files(
    root: Path, is_node: bool, ids: np.ndarray, floats: np.ndarray, packed: np.ndarray
) -> tuple[Path, Path]:
    """Writes the same rows as Parquet (the preprocessor's column layout) and as TFRecords.

    Packed features are one ``FixedSizeBinary`` column named after the packed feature key, the remaining
    features are in the float32 ``features`` column.
    """
    parquet_dir = root / "parquet"
    tfrecord_dir = root / "tfrecord"
    parquet_dir.mkdir()
    tfrecord_dir.mkdir()
    if is_node:
        id_columns = {"node_id": ids}
    else:
        id_columns = {"src": ids, "dst": ids + 100}
    columns: dict[str, pa.Array] = {
        name: pa.array(values) for name, values in id_columns.items()
    }
    columns["features"] = pa.FixedSizeListArray.from_arrays(
        pa.array(floats.reshape(-1)), floats.shape[1]
    )
    columns["packed_features"] = pa.FixedSizeBinaryArray.from_buffers(
        pa.binary(_PACKED_DIM), len(ids), [None, pa.py_buffer(packed.tobytes())]
    )
    table = pa.table(columns)
    start = 0
    for file_index, num_rows in enumerate(_PACKED_ROWS_PER_FILE):
        name = f"part-{file_index:05d}-of-{len(_PACKED_ROWS_PER_FILE):05d}"
        pq.write_table(
            table.slice(start, num_rows),
            parquet_dir / f"{name}.parquet",
            row_group_size=2,
        )
        with tf.io.TFRecordWriter(str(tfrecord_dir / f"{name}.tfrecord")) as writer:
            for row in range(start, start + num_rows):
                feature = {
                    name: tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[values[row]])
                    )
                    for name, values in id_columns.items()
                }
                feature["feature_a"] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=floats[row])
                )
                feature["packed_features"] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[packed[row].tobytes()])
                )
                writer.write(
                    tf.train.Example(
                        features=tf.train.Features(feature=feature)
                    ).SerializeToString()
                )
        start += num_rows
    parquet_manifest.write_manifest(UriFactory.create_uri(parquet_dir))
    return parquet_dir, tfrecord_dir


class ParquetPackedFeaturesTest(TestCase):
    """Packed features load from Parquet exactly as they do from TFRecords."""

    def setUp(self) -> None:
        super().setUp()
        self._temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self._temp_dir.name)

    def tearDown(self) -> None:
        self._temp_dir.cleanup()
        super().tearDown()

    @parameterized.expand([("nodes", True, 1), ("nodes", True, 2), ("edges", False, 1)])
    def test_matches_tfrecord_loader(
        self, _name: str, is_node: bool, world_size: int
    ) -> None:
        num_rows = sum(_PACKED_ROWS_PER_FILE)
        ids = np.arange(num_rows, dtype=np.int64) + 7
        floats = (np.arange(num_rows * 2) * 0.25).astype(np.float32).reshape(-1, 2)
        packed = _packed_bytes(num_rows)
        parquet_dir, tfrecord_dir = _write_packed_entity_files(
            self.root, is_node, ids, floats, packed
        )
        if is_node:
            entity_key: Union[str, tuple[str, str]] = "node_id"
            feature_spec: FeatureSpecDict = {
                "node_id": tf.io.FixedLenFeature([], tf.int64)
            }
        else:
            entity_key = ("src", "dst")
            feature_spec = {
                "src": tf.io.FixedLenFeature([], tf.int64),
                "dst": tf.io.FixedLenFeature([], tf.int64),
            }
        feature_spec = {
            **feature_spec,
            "feature_a": tf.io.FixedLenFeature([2], tf.float32),
            "packed_features": tf.io.FixedLenFeature([], tf.string),
        }

        def info(directory: Path, parquet: bool) -> SerializedTFRecordInfo:
            uri = UriFactory.create_uri(directory)
            return SerializedTFRecordInfo(
                tfrecord_uri_prefix=uri,
                parquet_uri_prefix=uri if parquet else None,
                feature_keys=["feature_a"],
                feature_spec=feature_spec,
                feature_dim=2,
                entity_key=entity_key,
                packed_feature_key="packed_features",
                packed_feature_dim=_PACKED_DIM,
            )

        parquet_parts = [
            ParquetDataLoader(rank, world_size).load_as_torch_tensors(
                info(parquet_dir, parquet=True)
            )
            for rank in range(world_size)
        ]
        tfrecord_parts = [
            TFRecordDataLoader(rank, world_size).load_as_torch_tensors(
                info(tfrecord_dir, parquet=False),
                TFDatasetOptions(deterministic=True),
            )
            for rank in range(world_size)
        ]

        # TFRecord gives ranks whole files and Parquet gives them row ranges, so compare the union over
        # ranks sorted by the (source) node id.
        def sorted_union(parts: list[LoadedEntityTensors]) -> list[torch.Tensor]:
            ids_tensor = torch.cat([part.ids for part in parts], dim=-1)
            order = torch.argsort(ids_tensor if is_node else ids_tensor[0])
            result = [ids_tensor[..., order]]
            result.append(torch.cat([_present(part.features) for part in parts])[order])
            result.append(
                torch.cat([_present(part.packed_features) for part in parts])[order]
            )
            return result

        parquet_result = sorted_union(parquet_parts)
        tfrecord_result = sorted_union(tfrecord_parts)
        for actual, expected in zip(parquet_result, tfrecord_result):
            assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(parquet_result[2].dtype, torch.uint8)
        assert_close(parquet_result[2], torch.from_numpy(packed), rtol=0, atol=0)

    def test_no_files_gives_empty_packed_features(self) -> None:
        (self.root / "empty").mkdir()
        loaded = ParquetDataLoader(0, 1).load_as_torch_tensors(
            SerializedTFRecordInfo(
                tfrecord_uri_prefix=UriFactory.create_uri(self.root / "empty"),
                parquet_uri_prefix=UriFactory.create_uri(self.root / "empty"),
                feature_keys=[],
                feature_spec={},
                feature_dim=0,
                entity_key="node_id",
                packed_feature_key="packed_features",
                packed_feature_dim=_PACKED_DIM,
            )
        )
        packed = _present(loaded.packed_features)
        self.assertEqual(packed.shape, (0, _PACKED_DIM))
        self.assertEqual(packed.dtype, torch.uint8)

    def test_packed_column_of_the_wrong_width_is_rejected(self) -> None:
        pq.write_table(
            pa.table(
                {
                    "node_id": pa.array([1, 2], type=pa.int64()),
                    "packed_features": pa.FixedSizeBinaryArray.from_buffers(
                        pa.binary(2), 2, [None, pa.py_buffer(b"\x01\x02\x03\x04")]
                    ),
                }
            ),
            self.root / "part-00000-of-00001.parquet",
        )
        parquet_manifest.write_manifest(UriFactory.create_uri(self.root))
        info = SerializedTFRecordInfo(
            tfrecord_uri_prefix=UriFactory.create_uri(self.root),
            parquet_uri_prefix=UriFactory.create_uri(self.root),
            feature_keys=[],
            feature_spec={},
            feature_dim=0,
            entity_key="node_id",
            packed_feature_key="packed_features",
            packed_feature_dim=_PACKED_DIM,
        )
        with self.assertRaisesRegex(ValueError, "bytes per row"):
            ParquetDataLoader(0, 1).load_as_torch_tensors(info)


class _FailingReadsFile(io.RawIOBase):
    """A local file whose first ``failures.remaining`` reads (across all instances) raise ``OSError``."""

    def __init__(self, path: str, failures: "_Failures") -> None:
        super().__init__()
        self._file = open(path, "rb")
        self._failures = failures

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._file.seek(offset, whence)

    def tell(self) -> int:
        return self._file.tell()

    def readinto(self, buffer) -> int:
        self._failures.maybe_fail()
        return self._file.readinto(buffer)

    def close(self) -> None:
        self._file.close()
        super().close()


class _Failures:
    """Fails ``remaining`` reads, counting only reads after ``start_after_opens`` file opens."""

    def __init__(self, remaining: int, start_after_opens: int = 0) -> None:
        self.remaining = remaining
        self.start_after_opens = start_after_opens
        self.opens = 0
        self._lock = threading.Lock()

    def maybe_fail(self) -> None:
        with self._lock:
            if self.remaining > 0 and self.opens >= self.start_after_opens:
                self.remaining -= 1
                raise OSError("injected read failure")


class _FailingReadsHandler(pafs.FileSystemHandler):
    """Local file system whose input files fail their first reads, see ``_FailingReadsFile``."""

    def __init__(self, failures: _Failures) -> None:
        self._local = pafs.LocalFileSystem()
        self._failures = failures

    def get_type_name(self) -> str:
        return "failing_reads"

    def normalize_path(self, path: str) -> str:
        return path

    def get_file_info(self, paths):
        return self._local.get_file_info(paths)

    def get_file_info_selector(self, selector):
        return self._local.get_file_info(selector)

    def open_input_file(self, path: str):
        if not path.endswith(parquet_format.PARQUET_FILE_SUFFIX):
            return self._local.open_input_file(path)
        self._failures.opens += 1
        return pa.PythonFile(_FailingReadsFile(path, self._failures), mode="r")

    def _unsupported(self, *args, **kwargs):
        raise NotImplementedError("Read-only test file system")

    copy_file = create_dir = delete_dir = delete_dir_contents = _unsupported
    delete_file = delete_root_dir_contents = move = open_append_stream = _unsupported
    open_input_stream = open_output_stream = _unsupported


class _FailingReadsLoader(ParquetDataLoader):
    """Must run with one reader thread: ``pa.PythonFile`` handles do not support concurrent positional reads."""

    def __init__(self, failures: _Failures, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._failures = failures

    def _filesystem_and_path(self, uri: Uri):
        fs = pafs.PyFileSystem(_FailingReadsHandler(self._failures))
        return fs, uri.uri


class _CountingFile(io.RawIOBase):
    """A local file that reports opens and closes to ``_OpenHandleCounter``."""

    def __init__(self, path: str, counter: "_OpenHandleCounter") -> None:
        super().__init__()
        self._file = open(path, "rb")
        self._counter = counter
        counter.opened()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._file.seek(offset, whence)

    def tell(self) -> int:
        return self._file.tell()

    def readinto(self, buffer) -> int:
        return self._file.readinto(buffer)

    def close(self) -> None:
        if not self.closed:
            self._file.close()
            self._counter.closed()
        super().close()


class _OpenHandleCounter:
    def __init__(self) -> None:
        self.currently_open = 0
        self.max_open = 0
        self.total_opens = 0
        self.attempts_by_path: dict[str, int] = {}
        self._lock = threading.Lock()

    def opened(self) -> None:
        with self._lock:
            self.currently_open += 1
            self.total_opens += 1
            self.max_open = max(self.max_open, self.currently_open)

    def closed(self) -> None:
        with self._lock:
            self.currently_open -= 1


class _CountingHandler(_FailingReadsHandler):
    """Local file system that counts open input files, see ``_CountingFile``."""

    def __init__(self, counter: _OpenHandleCounter, error: Optional[Exception] = None):
        super().__init__(_Failures(remaining=0))
        self._counter = counter
        self._error = error

    def open_input_file(self, path: str):
        if not path.endswith(parquet_format.PARQUET_FILE_SUFFIX):
            return self._local.open_input_file(path)
        if self._error is not None:
            counts = self._counter.attempts_by_path
            counts[path] = counts.get(path, 0) + 1
            raise self._error
        return pa.PythonFile(_CountingFile(path, self._counter), mode="r")


class _CountingLoader(ParquetDataLoader):
    def __init__(self, handler: _CountingHandler, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._handler = handler

    def _filesystem_and_path(self, uri: Uri):
        return pafs.PyFileSystem(self._handler), uri.uri


class _FlakyOpenHandler(_CountingHandler):
    """Fails the first ``failures_remaining`` opens with a generic ``pa.ArrowException``."""

    def __init__(self, failures_remaining: int) -> None:
        super().__init__(_OpenHandleCounter())
        self.failures_remaining = failures_remaining

    def open_input_file(self, path: str):
        if (
            path.endswith(parquet_format.PARQUET_FILE_SUFFIX)
            and self.failures_remaining > 0
        ):
            self.failures_remaining -= 1
            raise pa.ArrowException("Unknown error: transient storage failure")
        return super().open_input_file(path)


class ParquetDataLoaderSharedHandleAndRetryTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self._temp_dir.name)
        self.data = _node_data()
        _write_parquet_nodes(self.root, self.data, True, True)

    def tearDown(self) -> None:
        self._temp_dir.cleanup()
        super().tearDown()

    def test_concurrent_row_group_reads_on_one_handle_are_correct(self) -> None:
        path = str(self.root / "part-00000-of-00004.parquet")
        expected = pq.read_table(path)
        with pafs.LocalFileSystem().open_input_file(path) as handle:
            metadata = pq.read_metadata(handle)
            self.assertGreater(metadata.num_row_groups, 1)

            def read(row_group: int) -> pa.Table:
                parquet_file = pq.ParquetFile(
                    handle, metadata=metadata, pre_buffer=True
                )
                return parquet_file.read_row_group(row_group, use_threads=False)

            row_groups = list(range(metadata.num_row_groups)) * 20
            with ThreadPoolExecutor(16) as pool:
                tables = list(pool.map(read, row_groups))
        for row_group, table in zip(row_groups, tables):
            start = row_group * _ROW_GROUP_SIZE
            self.assertTrue(table.equals(expected.slice(start, table.num_rows)))

    def test_each_file_is_opened_once(self) -> None:
        failures = _Failures(remaining=0)
        info = _node_info(self.root, parquet=True)
        loaded = _FailingReadsLoader(
            failures, 0, 1, num_threads=1
        ).load_as_torch_tensors(info)
        assert_close(loaded.ids, torch.from_numpy(self.data["node_id"]))
        # Once to read each footer and once to read each file's row groups.
        self.assertEqual(failures.opens, 2 * len(_ROWS_PER_FILE))

    @parameterized.expand(
        [
            ("footer_reads", 0),
            ("row_group_reads", len(_ROWS_PER_FILE)),
        ]
    )
    def test_transient_read_failures_are_retried_and_data_is_correct(
        self, _name: str, start_after_opens: int
    ) -> None:
        failures = _Failures(remaining=2, start_after_opens=start_after_opens)
        info = _node_info(self.root, parquet=True)
        with mock.patch(_RETRY_SLEEP):
            loaded = _FailingReadsLoader(
                failures, 0, 1, num_threads=1
            ).load_as_torch_tensors(info)
        self.assertEqual(failures.remaining, 0)
        assert_close(loaded.ids, torch.from_numpy(self.data["node_id"]))
        features = np.concatenate(
            [
                self.data["feature_a"][:, None],
                self.data["feature_b"].astype(np.float32),
            ],
            axis=1,
        )
        assert_close(loaded.features, torch.from_numpy(features))

    def test_open_handles_are_bounded_by_the_thread_count(self) -> None:
        counter = _OpenHandleCounter()
        info = _node_info(self.root, parquet=True)
        loaded = _CountingLoader(
            _CountingHandler(counter), 0, 1, num_threads=2
        ).load_as_torch_tensors(info)
        assert_close(loaded.ids, torch.from_numpy(self.data["node_id"]))
        self.assertGreater(counter.total_opens, 0)
        self.assertEqual(counter.currently_open, 0)
        self.assertLessEqual(counter.max_open, 2)

    @parameterized.expand(
        [
            ("unknown_arrow_error", pa.ArrowException("unknown error"), OSError, 3),
            ("os_error", OSError("connection reset"), OSError, 3),
            ("file_not_found", FileNotFoundError("gone"), FileNotFoundError, 1),
            ("permission_denied", PermissionError("denied"), PermissionError, 1),
            ("invalid_data", pa.ArrowInvalid("bad"), pa.ArrowInvalid, 1),
        ]
    )
    def test_open_failures_are_retried_only_when_transient(
        self,
        _name: str,
        error: Exception,
        expected_error: type[Exception],
        expected_attempts: int,
    ) -> None:
        counter = _OpenHandleCounter()
        info = _node_info(self.root, parquet=True)
        with mock.patch(_RETRY_SLEEP):
            with self.assertRaises(expected_error):
                _CountingLoader(
                    _CountingHandler(counter, error), 0, 1, num_threads=1
                ).load_as_torch_tensors(info)
        self.assertGreater(len(counter.attempts_by_path), 0)
        for attempts in counter.attempts_by_path.values():
            self.assertEqual(attempts, expected_attempts)

    def test_arrow_exception_during_reads_is_retried_and_data_is_correct(
        self,
    ) -> None:
        handler = _FlakyOpenHandler(failures_remaining=2)
        info = _node_info(self.root, parquet=True)
        with mock.patch(_RETRY_SLEEP):
            loaded = _CountingLoader(
                handler, 0, 1, num_threads=1
            ).load_as_torch_tensors(info)
        self.assertEqual(handler.failures_remaining, 0)
        assert_close(loaded.ids, torch.from_numpy(self.data["node_id"]))

    def test_persistent_read_failures_raise(self) -> None:
        failures = _Failures(remaining=10_000)
        info = _node_info(self.root, parquet=True)
        with mock.patch(_RETRY_SLEEP):
            with self.assertRaises(OSError):
                _FailingReadsLoader(
                    failures, 0, 1, num_threads=1
                ).load_as_torch_tensors(info)


class ParquetRequiredManifestTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self._temp_dir.name)
        _write_parquet_nodes(self.root, _node_data(), True, True)
        self.info = _node_info(self.root, parquet=True)
        self.manifest_path = self.root / parquet_manifest.MANIFEST_FILE_NAME

    def tearDown(self) -> None:
        self._temp_dir.cleanup()
        super().tearDown()

    def test_missing_manifest_raises(self) -> None:
        self.manifest_path.unlink()
        with self.assertRaises(FileNotFoundError):
            ParquetDataLoader(0, 1).load_as_torch_tensors(self.info)

    def test_manifest_that_does_not_match_the_files_raises(self) -> None:
        pq.write_table(
            pq.read_table(self.root / "part-00001-of-00004.parquet"),
            self.root / "part-00004-of-00004.parquet",
        )
        with self.assertRaises(ValueError):
            ParquetDataLoader(0, 1).load_as_torch_tensors(self.info)

    def test_manifest_with_unknown_version_raises(self) -> None:
        manifest = json.loads(self.manifest_path.read_text())
        manifest["version"] = parquet_manifest.MANIFEST_VERSION + 1
        self.manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            ParquetDataLoader(0, 1).load_as_torch_tensors(self.info)


if __name__ == "__main__":
    absltest.main()
