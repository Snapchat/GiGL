import glob
import json
import os
import tempfile

import apache_beam as beam
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tensorflow as tf
import torch
from apache_beam.testing.test_pipeline import TestPipeline
from tensorflow_transform.tf_metadata.dataset_metadata import DatasetMetadata

from gigl.common import LocalUri
from gigl.common.beam.better_tfrecordio import BetterWriteToTFRecord
from gigl.common.data.dataloaders import (
    LoadedEntityTensors,
    SerializedTFRecordInfo,
    TFDatasetOptions,
    TFRecordDataLoader,
)
from gigl.common.data.parquet_manifest import MANIFEST_FILE_NAME, write_manifest
from gigl.common.utils.feature_quantization.numpy_ops import quantize_ndarray
from gigl.src.data_preprocessor.lib.transform.parquet_writer import (
    WriteDenseParquet,
)
from tests.test_assets.test_case import TestCase

_FEATURE_KEYS = ["int_feature", "vector_feature", "float_feature", "int_vector"]
_LABEL_KEYS = ["label"]
_FEATURE_SPEC = {
    "node_id": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
    "src": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
    "dst": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
    "float_feature": tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
    "int_feature": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
    "vector_feature": tf.io.FixedLenFeature(shape=[3], dtype=tf.float32),
    "int_vector": tf.io.FixedLenFeature(shape=[2], dtype=tf.int64),
    "label": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
}


def _make_batches(
    id_names: list[str], num_rows: int, batch_sizes: list[int]
) -> list[pa.RecordBatch]:
    """Builds TFT-style record batches (every column is a list array) with shuffled ids."""
    rng = np.random.default_rng(0)
    ids = [rng.permutation(num_rows * 10)[:num_rows].astype(np.int64) for _ in id_names]
    int_feature = rng.integers(0, 1 << 24, size=num_rows)
    float_feature = rng.normal(size=num_rows).astype(np.float32)
    vector_feature = rng.normal(size=(num_rows, 3)).astype(np.float32)
    int_vector = rng.integers(0, 1 << 24, size=(num_rows, 2))
    label = rng.integers(0, 5, size=num_rows)

    def as_list_array(values: np.ndarray, value_type: pa.DataType) -> pa.Array:
        values = values.reshape(num_rows, -1)
        return pa.ListArray.from_arrays(
            pa.array(np.arange(num_rows + 1, dtype=np.int32) * values.shape[1]),
            pa.array(values.reshape(-1), type=value_type),
        )

    columns = {
        name: as_list_array(column, pa.int64()) for name, column in zip(id_names, ids)
    }
    columns["float_feature"] = as_list_array(float_feature, pa.float32())
    columns["int_feature"] = as_list_array(int_feature, pa.int64())
    columns["vector_feature"] = as_list_array(vector_feature, pa.float32())
    columns["int_vector"] = as_list_array(int_vector, pa.int64())
    columns["label"] = as_list_array(label, pa.int64())
    full = pa.RecordBatch.from_pydict(columns)

    batches = []
    start = 0
    for size in batch_sizes:
        batches.append(full.slice(start, size))
        start += size
    return batches


def _load_parquet(prefix: str, id_columns: list[str]) -> LoadedEntityTensors:
    tables = [pq.read_table(path) for path in sorted(glob.glob(f"{prefix}*.parquet"))]
    table = pa.concat_tables([t for t in tables if t.num_rows > 0])
    ids = torch.from_numpy(
        np.stack([table[c].to_numpy() for c in id_columns]).squeeze()
    )
    features = torch.from_numpy(
        table["features"]
        .combine_chunks()
        .flatten()
        .to_numpy()
        .reshape(table.num_rows, -1)
    )
    labels = torch.from_numpy(table["label"].to_numpy()).unsqueeze(-1)
    return LoadedEntityTensors(
        ids=ids, features=features, quantized_features=None, labels=labels
    )


def _sort_by_id(tensors: LoadedEntityTensors) -> LoadedEntityTensors:
    ids = tensors.ids
    if ids.dim() == 1:
        order = torch.argsort(ids)
        ids = ids[order]
    else:
        order = torch.argsort(ids[0] * (ids.max() + 1) + ids[1])
        ids = ids[:, order]
    assert tensors.features is not None
    assert tensors.labels is not None
    return LoadedEntityTensors(
        ids=ids,
        features=tensors.features[order],
        quantized_features=None,
        labels=tensors.labels[order],
    )


class ParquetWriterRoundTripTest(TestCase):
    def _write_both_formats_and_load(
        self,
        temp_dir: str,
        id_columns: list[str],
        feature_keys: list[str],
        num_shards: int,
        batches: list[pa.RecordBatch],
    ) -> tuple[LoadedEntityTensors, LoadedEntityTensors]:
        tfrecord_dir = os.path.join(temp_dir, "tfrecord") + "/"
        parquet_dir = os.path.join(temp_dir, "parquet") + "/"
        feature_spec = {name: _FEATURE_SPEC[name] for name in batches[0].schema.names}
        metadata = DatasetMetadata.from_feature_spec(feature_spec)
        with TestPipeline() as pipeline:
            record_batches = pipeline | beam.Create(batches)
            record_batches | "Write TFRecord" >> BetterWriteToTFRecord(
                file_path_prefix=tfrecord_dir + "part",
                transformed_metadata=metadata,
                num_shards=num_shards,
            )
            record_batches | "Write Parquet" >> WriteDenseParquet(
                parquet_uri_prefix=parquet_dir,
                id_keys=id_columns,
                id_columns=id_columns,
                feature_keys=feature_keys,
                label_keys=_LABEL_KEYS,
                num_shards=num_shards,
            )

        entity_key: str | tuple[str, str]
        if len(id_columns) == 1:
            entity_key = id_columns[0]
        else:
            entity_key = (id_columns[0], id_columns[1])
        tfrecord_loaded = TFRecordDataLoader(
            rank=0, world_size=1
        ).load_as_torch_tensors(
            serialized_tf_record_info=SerializedTFRecordInfo(
                tfrecord_uri_prefix=LocalUri(tfrecord_dir),
                feature_keys=feature_keys,
                feature_spec=feature_spec,
                feature_dim=7,
                entity_key=entity_key,
                label_keys=_LABEL_KEYS,
            ),
            tf_dataset_options=TFDatasetOptions(deterministic=True),
        )
        parquet_loaded = _load_parquet(parquet_dir + "part", id_columns)
        return _sort_by_id(tfrecord_loaded), _sort_by_id(parquet_loaded)

    def _assert_bitwise_equal(
        self, expected: LoadedEntityTensors, actual: LoadedEntityTensors
    ) -> None:
        assert expected.features is not None and actual.features is not None
        assert expected.labels is not None and actual.labels is not None
        self.assert_tensor_equality(expected.ids, actual.ids)
        self.assertEqual(actual.features.dtype, torch.float32)
        self.assertTrue(
            torch.equal(
                expected.features.view(torch.int32), actual.features.view(torch.int32)
            )
        )
        self.assert_tensor_equality(expected.labels, actual.labels)

    def test_nodes_match_tfrecord_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tfrecord_loaded, parquet_loaded = self._write_both_formats_and_load(
                temp_dir=temp_dir,
                id_columns=["node_id"],
                feature_keys=_FEATURE_KEYS,
                num_shards=2,
                batches=_make_batches(["node_id"], 25, [10, 1, 14]),
            )
            self._assert_bitwise_equal(tfrecord_loaded, parquet_loaded)

    def test_edges_match_tfrecord_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tfrecord_loaded, parquet_loaded = self._write_both_formats_and_load(
                temp_dir=temp_dir,
                id_columns=["src", "dst"],
                feature_keys=_FEATURE_KEYS,
                num_shards=3,
                batches=_make_batches(["src", "dst"], 25, [7, 7, 11]),
            )
            self._assert_bitwise_equal(tfrecord_loaded, parquet_loaded)

    def test_parquet_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_dir = os.path.join(temp_dir, "parquet") + "/"
            with TestPipeline() as pipeline:
                _ = (
                    pipeline
                    | beam.Create(_make_batches(["node_id"], 25, [10, 15]))
                    | WriteDenseParquet(
                        parquet_uri_prefix=parquet_dir,
                        id_keys=["node_id"],
                        id_columns=["node_id"],
                        feature_keys=_FEATURE_KEYS,
                        label_keys=_LABEL_KEYS,
                        num_shards=2,
                    )
                )
            files = sorted(os.listdir(parquet_dir))
            self.assertEqual(
                files, ["part-00000-of-00002.parquet", "part-00001-of-00002.parquet"]
            )
            parquet_file = pq.ParquetFile(os.path.join(parquet_dir, files[0]))
            self.assertEqual(
                parquet_file.schema_arrow.field("features").type,
                pa.list_(pa.float32(), 7),
            )
            self.assertEqual(
                parquet_file.schema_arrow.field("node_id").type, pa.int64()
            )
            self.assertEqual(
                parquet_file.metadata.row_group(0).column(0).compression, "ZSTD"
            )

    def test_empty_shard_is_valid_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_dir = os.path.join(temp_dir, "parquet") + "/"
            with TestPipeline() as pipeline:
                _ = (
                    pipeline
                    | beam.Create(_make_batches(["node_id"], 5, [5]))
                    | WriteDenseParquet(
                        parquet_uri_prefix=parquet_dir,
                        id_keys=["node_id"],
                        id_columns=["node_id"],
                        feature_keys=_FEATURE_KEYS,
                        label_keys=_LABEL_KEYS,
                        num_shards=4,
                    )
                )
            total_rows = 0
            for path in glob.glob(f"{parquet_dir}*.parquet"):
                total_rows += pq.ParquetFile(path).metadata.num_rows
            self.assertEqual(total_rows, 5)

    def test_manifest_of_written_files_matches_footers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_dir = os.path.join(temp_dir, "parquet") + "/"
            with TestPipeline() as pipeline:
                _ = (
                    pipeline
                    | beam.Create(_make_batches(["node_id"], 25, [10, 15]))
                    | WriteDenseParquet(
                        parquet_uri_prefix=parquet_dir,
                        id_keys=["node_id"],
                        id_columns=["node_id"],
                        feature_keys=_FEATURE_KEYS,
                        label_keys=_LABEL_KEYS,
                        num_shards=4,
                    )
                )
            write_manifest(parquet_dir)
            with open(os.path.join(parquet_dir, MANIFEST_FILE_NAME)) as manifest_file:
                manifest = json.load(manifest_file)
            expected = {}
            for path in sorted(glob.glob(f"{parquet_dir}*.parquet")):
                metadata = pq.ParquetFile(path).metadata
                expected[os.path.basename(path)] = [
                    metadata.row_group(i).num_rows
                    for i in range(metadata.num_row_groups)
                ]
            self.assertEqual(manifest["files"], expected)
            self.assertEqual(sum(sum(rows) for rows in expected.values()), 25)
            # Empty shards are listed with no row groups.
            self.assertIn([], expected.values())

    def test_packed_features_are_fixed_size_binary(self) -> None:
        num_rows = 25
        rng = np.random.default_rng(1)
        packed = quantize_ndarray(
            rng.normal(size=(num_rows, 10)).astype(np.float32),
            bits=2,
            clip_min=-2.0,
            clip_max=2.0,
        )
        self.assertEqual(packed.shape, (num_rows, 3))
        batch = pa.RecordBatch.from_pydict(
            {
                "node_id": pa.array(
                    [[i] for i in range(num_rows)], type=pa.list_(pa.int64())
                ),
                "float_feature": pa.array(
                    [[float(i)] for i in range(num_rows)], type=pa.list_(pa.float32())
                ),
                "label": pa.array(
                    [[i % 5] for i in range(num_rows)], type=pa.list_(pa.int64())
                ),
                "node_packed_features": pa.array(
                    [[row.tobytes()] for row in packed], type=pa.list_(pa.binary())
                ),
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_dir = os.path.join(temp_dir, "parquet") + "/"
            with TestPipeline() as pipeline:
                _ = (
                    pipeline
                    | beam.Create([batch.slice(0, 10), batch.slice(10, 15)])
                    | WriteDenseParquet(
                        parquet_uri_prefix=parquet_dir,
                        id_keys=["node_id"],
                        id_columns=["node_id"],
                        feature_keys=["float_feature"],
                        label_keys=_LABEL_KEYS,
                        num_shards=1,
                        packed_feature_key="node_packed_features",
                        packed_feature_dim=3,
                    )
                )
            path = glob.glob(f"{parquet_dir}*.parquet")[0]
            table = pq.read_table(path)
            self.assertEqual(
                table.schema.field("node_packed_features").type, pa.binary(3)
            )
            self.assertEqual(
                table.schema.field("features").type, pa.list_(pa.float32(), 1)
            )
            written = np.stack(
                [
                    np.frombuffer(value, dtype=np.uint8)
                    for value in table["node_packed_features"].to_pylist()
                ]
            )
            order = np.argsort(table["node_id"].to_numpy())
            np.testing.assert_array_equal(written[order], packed)
