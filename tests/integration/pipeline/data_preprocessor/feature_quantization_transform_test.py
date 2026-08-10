import json
import os
import tempfile

import apache_beam as beam
import pyarrow as pa
import tensorflow as tf
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to
from parameterized import parameterized
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_transform.tf_metadata.dataset_metadata import DatasetMetadata

from gigl.src.data_preprocessor.lib.transform.feature_quantization import (
    apply_feature_quantization_transform,
)
from gigl.src.data_preprocessor.lib.types import FeatureQuantizationSpec
from tests.test_assets.test_case import TestCase


class FeatureQuantizationTransformTest(TestCase):
    @parameterized.expand(
        [
            (
                "multibit",
                2,
                [(-2.0, -2.0), (8.0, 8.0)],
                {"clip_min": -2.0, "clip_max": 8.0},
            ),
            (
                "single_bit",
                1,
                [(-4.0, -2.0), (4.0, 8.0)],
                {"neg_mean": -3.0, "pos_mean": 6.0},
            ),
        ]
    )
    def test_apply_feature_quantization_transform_writes_metadata(
        self,
        _: str,
        bits: int,
        feature_values: list[tuple[float, float]],
        expected_stats: dict[str, float],
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = os.path.join(temp_dir, "feature_quantization_metadata.json")
            logical_feature_keys = ["f0", "f1"]
            batches = [
                pa.RecordBatch.from_arrays(
                    [
                        pa.array([node_id], type=pa.int64()),
                        pa.array([f0], type=pa.float32()),
                        pa.array([f1], type=pa.float32()),
                        pa.array([node_id], type=pa.int64()),
                    ],
                    names=["node_id", "f0", "f1", "label"],
                )
                for node_id, (f0, f1) in enumerate(feature_values)
            ]
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
                        | "Create RecordBatches" >> beam.Create(batches),
                        logical_metadata=transform_output_metadata,
                        logical_feature_keys=logical_feature_keys,
                        quantization_spec=FeatureQuantizationSpec(
                            feature_keys=logical_feature_keys, bits=bits
                        ),
                        quantization_metadata_path=metadata_path,
                    )
                )
                physical_features = {
                    feature.name: feature
                    for feature in physical_metadata.schema.feature
                }
                self.assertEqual(
                    set(physical_features), {"node_id", "label", "node_packed_features"}
                )
                packed_feature = physical_features["node_packed_features"]
                self.assertEqual(
                    (
                        packed_feature.type,
                        packed_feature.value_count.min,
                        packed_feature.value_count.max,
                    ),
                    (schema_pb2.BYTES, 1, 1),
                )

                assert_that(
                    transformed_batches
                    | "Extract quantized feature names"
                    >> beam.Map(lambda batch: batch.schema.names),
                    equal_to(
                        [["node_id", "label", "node_packed_features"]] * len(batches)
                    ),
                )

            with open(metadata_path) as metadata_file:
                metadata = json.load(metadata_file)
            self.assertEqual(
                metadata,
                {
                    "packed_feature_key": "node_packed_features",
                    "quantized_feature_indices": [0, 1],
                    "bits": bits,
                    **expected_stats,
                },
            )
