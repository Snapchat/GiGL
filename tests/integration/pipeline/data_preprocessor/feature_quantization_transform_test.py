import json
import os
import tempfile

import apache_beam as beam
import pyarrow as pa
import tensorflow as tf
import tensorflow_data_validation as tfdv
import torch
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to
from parameterized import parameterized
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_transform.tf_metadata.dataset_metadata import DatasetMetadata
from torch_geometric.data import Data

from gigl.common.beam.better_tfrecordio import BetterWriteToTFRecord
from gigl.common.data.dataloaders import TFDatasetOptions, TFRecordDataLoader
from gigl.distributed.utils.neighborloader import (
    EDGE_PACKED_FEATURES_METADATA_KEY,
    materialize_quantized_edge_features,
)
from gigl.distributed.utils.serialized_graph_metadata_translator import (
    convert_pb_to_serialized_graph_metadata,
)
from gigl.src.common.types.pb_wrappers.graph_metadata import GraphMetadataPbWrapper
from gigl.src.common.types.pb_wrappers.preprocessed_metadata import (
    PreprocessedMetadataPbWrapper,
)
from gigl.src.data_preprocessor.lib.transform.feature_quantization import (
    EDGE_PACKED_FEATURE_KEY,
    NODE_PACKED_FEATURE_KEY,
    apply_feature_quantization_transform,
)
from gigl.src.data_preprocessor.lib.types import FeatureQuantizationSpec
from snapchat.research.gbml import graph_schema_pb2, preprocessed_metadata_pb2
from tests.test_assets.test_case import TestCase


class FeatureQuantizationTransformTest(TestCase):
    def test_edge_quantization_round_trips_through_storage_and_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = os.path.join(temp_dir, "feature_quantization_metadata.json")
            tfrecord_prefix = os.path.join(temp_dir, "edges")
            schema_path = os.path.join(temp_dir, "schema.pbtxt")
            logical_metadata = DatasetMetadata.from_feature_spec(
                {
                    "src": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
                    "dst": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
                    "quantized": tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
                    "raw": tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
                }
            )
            logical_batches = [
                pa.RecordBatch.from_arrays(
                    [
                        pa.array([[0], [1]], type=pa.list_(pa.int64())),
                        pa.array([[1], [0]], type=pa.list_(pa.int64())),
                        pa.array([[-2.0], [8.0]], type=pa.list_(pa.float32())),
                        pa.array([[10.0], [20.0]], type=pa.list_(pa.float32())),
                    ],
                    names=["src", "dst", "quantized", "raw"],
                )
            ]

            with TestPipeline() as pipeline:
                transformed_batches, physical_metadata = (
                    apply_feature_quantization_transform(
                        logical_features=pipeline
                        | "Create edge RecordBatches" >> beam.Create(logical_batches),
                        logical_metadata=logical_metadata,
                        logical_feature_keys=["quantized", "raw"],
                        quantization_spec=FeatureQuantizationSpec(
                            feature_keys=["quantized"], bits=2
                        ),
                        quantization_metadata_path=metadata_path,
                        packed_feature_key=EDGE_PACKED_FEATURE_KEY,
                    )
                )
                transformed_batches | "Write edge TFRecords" >> BetterWriteToTFRecord(
                    file_path_prefix=tfrecord_prefix,
                    transformed_metadata=physical_metadata,
                    num_shards=1,
                )

            tfdv.write_schema_text(logical_metadata.schema, schema_path)
            with tf.io.gfile.GFile(metadata_path) as metadata_file:
                quantization_metadata = json.loads(metadata_file.read())
            quantization_metadata_pb = preprocessed_metadata_pb2.PreprocessedMetadata.FeatureQuantizationMetadata(
                packed_feature_key=quantization_metadata["packed_feature_key"],
                quantized_feature_indices=quantization_metadata[
                    "quantized_feature_indices"
                ],
            )
            quantization_metadata_pb.multi_bit_state.bits = quantization_metadata[
                "bits"
            ]
            quantization_metadata_pb.multi_bit_state.clip_min = quantization_metadata[
                "clip_min"
            ]
            quantization_metadata_pb.multi_bit_state.clip_max = quantization_metadata[
                "clip_max"
            ]
            self.assertEqual(
                quantization_metadata_pb.packed_feature_key, "edge_packed_features"
            )
            self.assertEqual(
                list(quantization_metadata_pb.quantized_feature_indices), [0]
            )

            preprocessed_metadata_pb = preprocessed_metadata_pb2.PreprocessedMetadata()
            preprocessed_metadata_pb.condensed_node_type_to_preprocessed_metadata[
                0
            ].node_id_key = "node_id"
            edge_metadata = (
                preprocessed_metadata_pb.condensed_edge_type_to_preprocessed_metadata[0]
            )
            edge_metadata.src_node_id_key = "src"
            edge_metadata.dst_node_id_key = "dst"
            edge_metadata.main_edge_info.CopyFrom(
                preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataInfo(
                    feature_keys=["quantized", "raw"],
                    feature_dim=2,
                    tfrecord_uri_prefix=temp_dir,
                    schema_uri=schema_path,
                    quantized_feature_metadata=quantization_metadata_pb,
                )
            )
            graph_metadata_pb = graph_schema_pb2.GraphMetadata(
                node_types=["node"],
                edge_types=[
                    graph_schema_pb2.EdgeType(
                        src_node_type="node", relation="connects", dst_node_type="node"
                    )
                ],
                condensed_node_type_map={0: "node"},
                condensed_edge_type_map={
                    0: graph_schema_pb2.EdgeType(
                        src_node_type="node", relation="connects", dst_node_type="node"
                    )
                },
            )
            serialized_metadata = convert_pb_to_serialized_graph_metadata(
                preprocessed_metadata_pb_wrapper=PreprocessedMetadataPbWrapper(
                    preprocessed_metadata_pb
                ),
                graph_metadata_pb_wrapper=GraphMetadataPbWrapper(graph_metadata_pb),
                tfrecord_uri_pattern="edges.*\\.tfrecord",
            )
            loaded = TFRecordDataLoader(rank=0, world_size=1).load_as_torch_tensors(
                serialized_tf_record_info=serialized_metadata.edge_entity_info,
                tf_dataset_options=TFDatasetOptions(deterministic=True),
            )

            assert loaded.features is not None
            assert loaded.quantized_features is not None
            self.assert_tensor_equality(loaded.ids, torch.tensor([[0, 1], [1, 0]]))
            self.assert_tensor_equality(loaded.features, torch.tensor([[10.0], [20.0]]))
            self.assert_tensor_equality(
                loaded.quantized_features, torch.tensor([[0], [192]], dtype=torch.uint8)
            )
            materialized, remaining_metadata = materialize_quantized_edge_features(
                data=Data(edge_index=loaded.ids, edge_attr=loaded.features),
                metadata={EDGE_PACKED_FEATURES_METADATA_KEY: loaded.quantized_features},
                edge_quantization_metadata=serialized_metadata.edge_quantization_metadata,
            )
            self.assert_tensor_equality(
                materialized.edge_attr, torch.tensor([[-2.0, 10.0], [8.0, 20.0]])
            )
            self.assertEqual(remaining_metadata, {})

    def test_apply_feature_quantization_transform_rejects_reserved_schema_key(
        self,
    ) -> None:
        logical_metadata = DatasetMetadata.from_feature_spec(
            {
                "f0": tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
                "edge_packed_features": tf.io.FixedLenFeature(
                    shape=[], dtype=tf.string
                ),
            }
        )

        with (
            self.assertRaisesRegex(ValueError, "Reserved packed feature key"),
            TestPipeline() as pipeline,
        ):
            apply_feature_quantization_transform(
                logical_features=pipeline
                | "Create collision input"
                >> beam.Create(
                    [
                        pa.RecordBatch.from_arrays(
                            [pa.array([1.0]), pa.array([b"existing"])],
                            names=["f0", "edge_packed_features"],
                        )
                    ]
                ),
                logical_metadata=logical_metadata,
                logical_feature_keys=["f0"],
                quantization_spec=FeatureQuantizationSpec(feature_keys=["f0"], bits=2),
                quantization_metadata_path="unused",
                packed_feature_key="edge_packed_features",
            )

    @parameterized.expand(
        [
            (
                "multibit",
                2,
                [(-2.0, -2.0), (8.0, 8.0)],
                {"clip_min": -2.0, "clip_max": 8.0},
                False,
            ),
            (
                "single_bit",
                1,
                [(-4.0, -2.0), (4.0, 8.0)],
                {"neg_mean": -3.0, "pos_mean": 6.0},
                True,
            ),
        ]
    )
    def test_apply_feature_quantization_transform_writes_metadata(
        self,
        _: str,
        bits: int,
        feature_values: list[tuple[float, float]],
        expected_stats: dict[str, float],
        use_deferred_metadata: bool,
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
                logical_metadata = transform_output_metadata
                if use_deferred_metadata:
                    logical_metadata = (
                        pipeline
                        | "Create deferred metadata"
                        >> beam.Create([transform_output_metadata])
                    )
                transformed_batches, physical_metadata = (
                    apply_feature_quantization_transform(
                        logical_features=pipeline
                        | "Create RecordBatches" >> beam.Create(batches),
                        logical_metadata=logical_metadata,
                        logical_feature_keys=logical_feature_keys,
                        quantization_spec=FeatureQuantizationSpec(
                            feature_keys=logical_feature_keys, bits=bits
                        ),
                        quantization_metadata_path=metadata_path,
                        packed_feature_key=NODE_PACKED_FEATURE_KEY,
                    )
                )
                if use_deferred_metadata:
                    assert isinstance(physical_metadata, beam.pvalue.AsSingleton)
                    assert_that(
                        pipeline
                        | "Create metadata validation input" >> beam.Create([None])
                        | "Extract deferred physical feature names"
                        >> beam.Map(
                            lambda _, metadata: sorted(
                                feature.name for feature in metadata.schema.feature
                            ),
                            metadata=physical_metadata,
                        ),
                        equal_to([["label", "node_id", "node_packed_features"]]),
                        label="assert_deferred_physical_feature_names",
                    )
                else:
                    physical_features = {
                        feature.name: feature
                        for feature in physical_metadata.schema.feature
                    }
                    self.assertEqual(
                        set(physical_features),
                        {"node_id", "label", "node_packed_features"},
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
                    label="assert_quantized_feature_names",
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
