import json
import os
import tempfile

import apache_beam as beam
import pyarrow as pa
import tensorflow as tf
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_transform.tf_metadata.dataset_metadata import DatasetMetadata

from gigl.src.data_preprocessor.lib.transform.feature_quantization import (
    apply_feature_quantization_transform,
)
from gigl.src.data_preprocessor.lib.types import FeatureQuantizationSpec
from tests.test_assets.test_case import TestCase


def _column_pylist(batch: pa.RecordBatch, name: str) -> list:
    return batch.column(batch.schema.names.index(name)).to_pylist()


def _record_batch_summary(batch: pa.RecordBatch) -> dict[str, object]:
    return {
        "names": batch.schema.names,
        "node_id": _column_pylist(batch, "node_id"),
        "label": _column_pylist(batch, "label"),
        "node_packed_features": _column_pylist(batch, "node_packed_features"),
    }


class FeatureQuantizationTransformTest(TestCase):
    def test_apply_feature_quantization_transform_quantizes_multi_bit_features(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = os.path.join(temp_dir, "feature_quantization_metadata.json")
            batch = pa.RecordBatch.from_arrays(
                [
                    pa.array([10, 11], type=pa.int64()),
                    pa.array([-2.0, 8.0], type=pa.float32()),
                    pa.array([-2.0, 8.0], type=pa.float32()),
                    pa.array([0, 1], type=pa.int64()),
                ],
                names=["node_id", "f0", "f1", "label"],
            )
            transform_output_metadata = DatasetMetadata.from_feature_spec(
                {
                    "node_id": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
                    "f0": tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
                    "f1": tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
                    "label": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
                }
            )

            with TestPipeline() as pipeline:
                transformed_batches, physical_metadata = (
                    apply_feature_quantization_transform(
                        logical_features=pipeline
                        | "Create RecordBatch" >> beam.Create([batch]),
                        transform_output_metadata=transform_output_metadata,
                        analyzed_logical_metadata=None,
                        quantization_spec=FeatureQuantizationSpec(
                            feature_keys=["f0", "f1"], bits=2
                        ),
                        logical_feature_keys=["f0", "f1"],
                        metadata_path=metadata_path,
                    )
                )
                physical_features = {
                    feature.name: feature
                    for feature in physical_metadata.schema.feature
                }
                self.assertEqual(
                    set(physical_features),
                    {"node_id", "label", "node_packed_features"},
                )
                self.assertEqual(
                    physical_features["node_packed_features"].type,
                    schema_pb2.BYTES,
                )
                self.assertEqual(
                    physical_features["node_packed_features"].value_count.min,
                    1,
                )
                self.assertEqual(
                    physical_features["node_packed_features"].value_count.max,
                    1,
                )

                # These values sit exactly at the learned clip bounds, so this
                # does not depend on mid-bucket rounding: min/min maps to
                # 00000000 and max/max maps to 11110000 with two padded codes.
                assert_that(
                    transformed_batches
                    | "Summarize RecordBatch" >> beam.Map(_record_batch_summary),
                    equal_to(
                        [
                            {
                                "names": [
                                    "node_id",
                                    "label",
                                    "node_packed_features",
                                ],
                                "node_id": [10, 11],
                                "label": [0, 1],
                                "node_packed_features": [
                                    [bytes([0])],
                                    [bytes([240])],
                                ],
                            }
                        ]
                    ),
                )

            with open(metadata_path) as metadata_file:
                metadata = json.load(metadata_file)
            self.assertEqual(metadata["packed_feature_key"], "node_packed_features")
            self.assertEqual(metadata["quantized_feature_indices"], [0, 1])
            self.assertEqual(metadata["bits"], 2)
            self.assertEqual(metadata["clip_min"], -2.0)
            self.assertEqual(metadata["clip_max"], 8.0)
