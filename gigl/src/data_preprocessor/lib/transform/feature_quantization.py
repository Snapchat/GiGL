import json
from typing import Iterable

import apache_beam as beam
import numpy as np
import pyarrow as pa
import tensorflow_transform.tf_metadata.dataset_metadata as dataset_metadata
from apache_beam.transforms.stats import ApproximateQuantiles
from tensorflow_metadata.proto.v0 import schema_pb2

from gigl.src.data_preprocessor.lib.types import FeatureQuantizationSpec


def build_feature_quantization_stats(
    transformed_features: beam.PCollection[pa.RecordBatch],
    spec: FeatureQuantizationSpec,
) -> beam.PCollection[dict[str, float]]:
    if spec.bits not in (1, 2, 4, 8):
        raise ValueError(f"bits must be one of 1, 2, 4, or 8, got {spec.bits}.")
    if not spec.feature_keys:
        raise ValueError("Feature quantization expects at least one feature key.")
    if spec.bits == 1:
        return (
            transformed_features
            | "Extract centroid quantization values"
            >> beam.FlatMap(_iter_feature_values, feature_keys=spec.feature_keys)
            | "Compute centroid quantization stats"
            >> beam.CombineGlobally(_CentroidStatsFn())
        )
    return (
        transformed_features
        | "Extract linear quantization abs values"
        >> beam.FlatMap(
            _iter_feature_values, feature_keys=spec.feature_keys, use_abs=True
        )
        | "Compute linear quantization quantiles"
        >> ApproximateQuantiles.Globally(num_quantiles=1000)
        | "Build linear quantization stats" >> beam.Map(_linear_stats_from_quantiles)
    )


def quantize_record_batch(
    record_batch: pa.RecordBatch,
    spec: FeatureQuantizationSpec,
    stats: dict[str, float],
) -> pa.RecordBatch:
    features = _build_feature_matrix(record_batch, spec.feature_keys)
    if spec.bits == 1:
        # 1-bit quantization keeps only sign; values restore from neg/pos means.
        codes = (features > 0).astype(np.uint8)
    else:
        # Linearly map clipped values into integer buckets.
        levels = (1 << spec.bits) - 1
        clip_min = stats["clip_min"]
        clip_max = stats["clip_max"]
        clipped = np.clip(features, clip_min, clip_max)
        scaled = (clipped - clip_min) / (clip_max - clip_min)
        codes = np.rint(scaled * levels).astype(np.uint8)

    packed = _pack_codes(codes, spec.bits)
    keep_indices = [
        i
        for i, name in enumerate(record_batch.schema.names)
        if name not in spec.feature_keys and name != spec.quantized_feature_key
    ]
    arrays = [record_batch.column(i) for i in keep_indices]
    names = [record_batch.schema.names[i] for i in keep_indices]
    arrays.append(pa.array([row.tobytes() for row in packed], type=pa.binary()))
    names.append(spec.quantized_feature_key)
    return pa.RecordBatch.from_arrays(arrays, names=names)


def feature_quantization_metadata_json(
    stats: dict[str, float],
    spec: FeatureQuantizationSpec,
) -> str:
    per_byte = 8 // spec.bits
    dim = len(spec.feature_keys)
    metadata = {
        "quantized_feature_key": spec.quantized_feature_key,
        "dequantized_feature_keys": list(spec.feature_keys),
        "packed_feature_dim": (dim + per_byte - 1) // per_byte,
        "dequantized_feature_dim": dim,
        "bits": spec.bits,
    }
    metadata.update(stats)
    return json.dumps(metadata)


def apply_feature_quantization_schema(
    metadata: dataset_metadata.DatasetMetadata,
    spec: FeatureQuantizationSpec,
) -> dataset_metadata.DatasetMetadata:
    features_to_drop = set(spec.feature_keys)
    schema = schema_pb2.Schema()
    schema.CopyFrom(metadata.schema)
    kept_features = [
        feature
        for feature in schema.feature
        if feature.name not in features_to_drop
        and feature.name != spec.quantized_feature_key
    ]
    del schema.feature[:]
    schema.feature.extend(kept_features)
    quantized_feature = schema.feature.add()
    quantized_feature.name = spec.quantized_feature_key
    quantized_feature.type = schema_pb2.BYTES
    quantized_feature.value_count.min = 1
    quantized_feature.value_count.max = 1
    return dataset_metadata.DatasetMetadata(schema)


def _iter_feature_values(
    record_batch: pa.RecordBatch,
    feature_keys: list[str],
    use_abs: bool = False,
) -> Iterable[float]:
    values = _build_feature_matrix(record_batch, feature_keys).reshape(-1)
    if use_abs:
        values = np.abs(values)
    for value in values:
        if np.isfinite(value):
            yield float(value)


def _build_feature_matrix(
    record_batch: pa.RecordBatch, feature_keys: list[str]
) -> np.ndarray:
    key_to_index = {name: i for i, name in enumerate(record_batch.schema.names)}
    columns: list[np.ndarray] = []
    for feature_key in feature_keys:
        if feature_key not in key_to_index:
            raise ValueError(f"Feature key {feature_key} not found in RecordBatch.")
        values = np.asarray(
            record_batch.column(key_to_index[feature_key]).to_numpy(
                zero_copy_only=False
            ),
            dtype=np.float32,
        )
        if values.ndim != 1:
            raise ValueError(
                f"Feature quantization currently expects scalar features; "
                f"got {feature_key} with shape {values.shape}."
            )
        columns.append(values)
    if not columns:
        return np.empty((record_batch.num_rows, 0), dtype=np.float32)
    return np.stack(columns, axis=1)


def _pack_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    # Cast to uint16 for integer shift/multiply/sum packing math.
    per_byte = 8 // bits
    pad = (-codes.shape[-1]) % per_byte
    if pad:
        # Pad only the feature dimension of this 2D [row, feature] array.
        codes = np.pad(codes, ((0, 0), (0, pad)), constant_values=0)
    # Group the padded feature dimension into chunks that each form one byte.
    codes = codes.reshape(codes.shape[0], -1, per_byte).astype(np.uint16)
    # Place the first code in each chunk into the highest bits of the byte.
    shifts = bits * np.arange(per_byte - 1, -1, -1, dtype=np.uint16)
    weights = (1 << shifts).astype(np.uint16)
    return np.sum(codes * weights, axis=-1).astype(np.uint8)


def _linear_stats_from_quantiles(quantiles: list[float]) -> dict[str, float]:
    if not quantiles:
        raise ValueError("Cannot compute quantization stats from no values.")
    # Store symmetric clip bounds from the approximate 99.5th abs-value percentile.
    index = round(0.995 * (len(quantiles) - 1))
    clip_max = max(float(quantiles[index]), 1e-5)
    return {"clip_min": -clip_max, "clip_max": clip_max}


class _CentroidStatsFn(beam.CombineFn):
    def create_accumulator(self) -> tuple[float, int, float, int]:
        return 0.0, 0, 0.0, 0

    def add_input(
        self, accumulator: tuple[float, int, float, int], value: float
    ) -> tuple[float, int, float, int]:
        neg_sum, neg_count, pos_sum, pos_count = accumulator
        if value <= 0:
            return neg_sum + value, neg_count + 1, pos_sum, pos_count
        return neg_sum, neg_count, pos_sum + value, pos_count + 1

    def merge_accumulators(
        self, accumulators: Iterable[tuple[float, int, float, int]]
    ) -> tuple[float, int, float, int]:
        neg_sum = neg_count = pos_sum = pos_count = 0
        for acc_neg_sum, acc_neg_count, acc_pos_sum, acc_pos_count in accumulators:
            neg_sum += acc_neg_sum
            neg_count += acc_neg_count
            pos_sum += acc_pos_sum
            pos_count += acc_pos_count
        return neg_sum, neg_count, pos_sum, pos_count

    def extract_output(
        self, accumulator: tuple[float, int, float, int]
    ) -> dict[str, float]:
        neg_sum, neg_count, pos_sum, pos_count = accumulator
        # Store mean values for negative and positive buckets.
        return {
            "neg_mean": neg_sum / neg_count if neg_count else 0.0,
            "pos_mean": pos_sum / pos_count if pos_count else 0.0,
        }
