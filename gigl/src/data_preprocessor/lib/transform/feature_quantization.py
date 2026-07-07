import json
from typing import Iterable

import apache_beam as beam
import numpy as np
import pyarrow as pa
from apache_beam.transforms.stats import ApproximateQuantiles
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_transform.tf_metadata import schema_utils
from tensorflow_transform.tf_metadata.dataset_metadata import DatasetMetadata

from gigl.common.logger import Logger
from gigl.common.utils.feature_quantization.numpy_ops import quantize_ndarray
from gigl.common.utils.tensorflow_schema import feature_spec_to_feature_index_map
from gigl.src.data_preprocessor.lib.types import FeatureQuantizationSpec

logger = Logger()
_NODE_PACKED_FEATURE_KEY = "node_packed_features"
_CentroidAcc = tuple[float, int, float, int]


def apply_feature_quantization_transform(
    transformed_features: beam.PCollection[pa.RecordBatch],
    transformed_metadata: DatasetMetadata,
    analyzed_metadata: beam.PCollection[DatasetMetadata] | None,
    spec: FeatureQuantizationSpec,
    feature_keys: list[str],
    metadata_path: str,
):
    logger.info(f"Applying Beam feature quantization with spec: {spec}")
    stats = _build_feature_quantization_stats(transformed_features, spec)
    logical_metadata = (
        transformed_metadata
        if analyzed_metadata is None
        else beam.pvalue.AsSingleton(analyzed_metadata)
    )
    _ = (
        stats
        | "Build feature quantization metadata JSON"
        >> beam.Map(
            _feature_quantization_metadata_json,
            spec=spec,
            feature_keys=feature_keys,
            dataset_metadata=logical_metadata,
        )
        | "Write feature quantization metadata"
        >> beam.io.WriteToText(
            metadata_path,
            num_shards=1,
            shard_name_template="",
        )
    )
    transformed_features = transformed_features | (
        "Quantize transformed feature RecordBatches"
        >> beam.Map(
            _quantize_record_batch,
            spec=spec,
            stats=beam.pvalue.AsSingleton(stats),
        )
    )
    # Encode TFRecords with the compact physical schema. The persisted schema
    # remains the original logical TFT schema because dequantization scatters
    # features back.
    if analyzed_metadata is None:
        physical_metadata = DatasetMetadata(
            _apply_feature_quantization_schema(transformed_metadata.schema, spec)
        )
    else:
        physical_metadata = analyzed_metadata | (
            "Apply feature quantization schema"
            >> beam.Map(
                lambda metadata, spec: DatasetMetadata(
                    _apply_feature_quantization_schema(metadata.schema, spec)
                ),
                spec=spec,
            )
        )
        physical_metadata = beam.pvalue.AsSingleton(physical_metadata)
    return transformed_features, physical_metadata


def _build_feature_quantization_stats(
    record_batches: beam.PCollection[pa.RecordBatch],
    spec: FeatureQuantizationSpec,
) -> beam.PCollection[dict[str, float]]:
    if spec.bits not in (1, 2, 4, 8):
        raise ValueError(f"bits must be one of 1, 2, 4, or 8, got {spec.bits}.")
    if not spec.feature_keys:
        raise ValueError("Feature quantization expects at least one feature key.")
    logger.info(
        f"Building Beam feature quantization stats for {len(spec.feature_keys)} "
        f"features with bits={spec.bits}: {spec.feature_keys}"
    )
    if spec.bits == 1:
        return (
            record_batches
            | "Compute centroid quantization stats"
            >> beam.CombineGlobally(_CentroidStatsFn(spec.feature_keys))
        )
    return (
        record_batches
        | "Build linear quantization value batches"
        >> beam.Map(_build_abs_feature_values, feature_keys=spec.feature_keys)
        | "Compute linear quantization quantiles"
        >> ApproximateQuantiles.Globally(num_quantiles=1000, input_batched=True)
        | "Build linear quantization stats" >> beam.Map(_linear_stats_from_quantiles)
    )


def _quantize_record_batch(
    batch: pa.RecordBatch,
    spec: FeatureQuantizationSpec,
    stats: dict[str, float],
) -> pa.RecordBatch:
    features = _build_feature_matrix(batch, spec.feature_keys)
    packed = quantize_ndarray(features, bits=spec.bits, stats=stats)
    drop_keys = set(spec.feature_keys) | {_NODE_PACKED_FEATURE_KEY}
    schema_names = batch.schema.names
    keep_indices = [i for i, name in enumerate(schema_names) if name not in drop_keys]
    arrays = [batch.column(i) for i in keep_indices]
    names = [schema_names[i] for i in keep_indices]
    arrays.append(
        pa.array([[row.tobytes()] for row in packed], type=pa.list_(pa.binary()))
    )
    names.append(_NODE_PACKED_FEATURE_KEY)
    return pa.RecordBatch.from_arrays(arrays, names=names)


def _feature_quantization_metadata_json(
    stats: dict[str, float],
    spec: FeatureQuantizationSpec,
    feature_keys: list[str],
    dataset_metadata: DatasetMetadata,
) -> str:
    raw_feature_spec = schema_utils.schema_as_feature_spec(
        dataset_metadata.schema
    ).feature_spec
    feature_key_set = set(feature_keys)
    missing = [
        key
        for key in spec.feature_keys
        if key not in raw_feature_spec or key not in feature_key_set
    ]
    if missing:
        raise ValueError(
            f"Quantized feature keys missing from feature outputs: {missing}"
        )
    feature_spec = {key: raw_feature_spec[key] for key in feature_keys}
    feature_index = feature_spec_to_feature_index_map(feature_spec)
    quantized_feature_indices = []
    for key in spec.feature_keys:
        start, end = feature_index[key]
        if end - start != 1:
            raise ValueError(
                f"Feature quantization expects scalar features, got {key}."
            )
        quantized_feature_indices.append(start)
    metadata = {
        "packed_feature_key": _NODE_PACKED_FEATURE_KEY,
        "quantized_feature_indices": quantized_feature_indices,
        "bits": spec.bits,
        **stats,
    }
    logger.info(f"Writing feature quantization metadata: {metadata}")
    return json.dumps(metadata)


def _apply_feature_quantization_schema(
    schema: schema_pb2.Schema, spec: FeatureQuantizationSpec
) -> schema_pb2.Schema:
    drop_keys = set(spec.feature_keys) | {_NODE_PACKED_FEATURE_KEY}
    quantized_schema = schema_pb2.Schema()
    quantized_schema.CopyFrom(schema)
    kept_features = [
        feature for feature in quantized_schema.feature if feature.name not in drop_keys
    ]
    del quantized_schema.feature[:]
    quantized_schema.feature.extend(kept_features)
    packed_feature = quantized_schema.feature.add()
    packed_feature.name = _NODE_PACKED_FEATURE_KEY
    packed_feature.type = schema_pb2.BYTES
    packed_feature.value_count.min = 1
    packed_feature.value_count.max = 1
    logger.info(
        f"Updated transformed schema for feature quantization: dropped "
        f"{len(spec.feature_keys)} features and added bytes feature "
        f"{_NODE_PACKED_FEATURE_KEY}."
    )
    return quantized_schema


def _build_abs_feature_values(
    batch: pa.RecordBatch, feature_keys: list[str]
) -> list[float]:
    values = np.abs(_build_feature_matrix(batch, feature_keys).reshape(-1))
    return values[np.isfinite(values)].astype(float).tolist()


def _build_feature_matrix(batch: pa.RecordBatch, feature_keys: list[str]) -> np.ndarray:
    key_to_idx = {name: i for i, name in enumerate(batch.schema.names)}
    cols: list[np.ndarray] = []
    for key in feature_keys:
        if key not in key_to_idx:
            raise ValueError(f"Feature key {key} not found in RecordBatch.")
        col = batch.column(key_to_idx[key])
        values = np.asarray(col.to_numpy(zero_copy_only=False), dtype=np.float32)
        if values.ndim != 1:
            raise ValueError(
                f"Feature quantization currently expects scalar features; "
                f"got {key} with shape {values.shape}."
            )
        cols.append(values)
    return np.stack(cols, axis=1)


def _linear_stats_from_quantiles(quantiles: list[float]) -> dict[str, float]:
    if not quantiles:
        raise ValueError("Cannot compute quantization stats from no values.")
    # Store symmetric clip bounds from the approximate 99.5th abs-value percentile.
    clip_max = max(float(quantiles[round(0.995 * (len(quantiles) - 1))]), 1e-5)
    stats = {"clip_min": -clip_max, "clip_max": clip_max}
    logger.info(f"Computed Beam feature quantization stats: {stats}")
    return stats


class _CentroidStatsFn(beam.CombineFn):
    def __init__(self, feature_keys: list[str]):
        self._feature_keys = feature_keys

    def create_accumulator(self) -> _CentroidAcc:
        return 0.0, 0, 0.0, 0

    def add_input(
        self, accumulator: _CentroidAcc, batch: pa.RecordBatch
    ) -> _CentroidAcc:
        neg_sum, neg_count, pos_sum, pos_count = accumulator
        values = _build_feature_matrix(batch, self._feature_keys).reshape(-1)
        values = values[np.isfinite(values)]
        neg = values <= 0
        pos = values > 0
        return (
            neg_sum + float(values[neg].sum()),
            neg_count + int(neg.sum()),
            pos_sum + float(values[pos].sum()),
            pos_count + int(pos.sum()),
        )

    def merge_accumulators(self, accumulators: Iterable[_CentroidAcc]) -> _CentroidAcc:
        neg_sum = neg_count = pos_sum = pos_count = 0
        for n_sum, n_count, p_sum, p_count in accumulators:
            neg_sum += n_sum
            neg_count += n_count
            pos_sum += p_sum
            pos_count += p_count
        return neg_sum, neg_count, pos_sum, pos_count

    def extract_output(self, accumulator: _CentroidAcc) -> dict[str, float]:
        neg_sum, neg_count, pos_sum, pos_count = accumulator
        stats = {
            "neg_mean": neg_sum / neg_count if neg_count else 0.0,
            "pos_mean": pos_sum / pos_count if pos_count else 0.0,
        }
        logger.info(f"Computed Beam feature quantization stats: {stats}")
        return stats
