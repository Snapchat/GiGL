import json
from typing import Iterable

import apache_beam as beam
import numpy as np
import pyarrow as pa
from apache_beam.transforms.stats import ApproximateQuantiles
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_transform.tf_metadata.dataset_metadata import DatasetMetadata

from gigl.common.logger import Logger
from gigl.src.data_preprocessor.lib.types import FeatureQuantizationSpec

logger = Logger()
_NODE_QUANTIZED_FEATURE_KEY = "node_quantized_features"
_CentroidAcc = tuple[float, int, float, int]


def build_feature_quantization_stats(
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


def quantize_record_batch(
    batch: pa.RecordBatch,
    spec: FeatureQuantizationSpec,
    stats: dict[str, float],
) -> pa.RecordBatch:
    features = _build_feature_matrix(batch, spec.feature_keys)
    if spec.bits == 1:
        # 1-bit quantization keeps only sign; values restore from neg/pos means.
        codes = (features > 0).astype(np.uint8)
    else:
        # Linearly map clipped values into integer buckets.
        levels = (1 << spec.bits) - 1
        lo, hi = stats["clip_min"], stats["clip_max"]
        clipped = np.clip(features, lo, hi)
        scaled = (clipped - lo) / (hi - lo)
        codes = np.rint(scaled * levels).astype(np.uint8)

    packed = _pack_codes(codes, spec.bits)
    drop_keys = set(spec.feature_keys) | {_NODE_QUANTIZED_FEATURE_KEY}
    keep_indices = [
        i for i, name in enumerate(batch.schema.names) if name not in drop_keys
    ]
    arrays = [batch.column(i) for i in keep_indices]
    names = [batch.schema.names[i] for i in keep_indices]
    arrays.append(
        pa.array([[row.tobytes()] for row in packed], type=pa.list_(pa.binary()))
    )
    names.append(_NODE_QUANTIZED_FEATURE_KEY)
    return pa.RecordBatch.from_arrays(arrays, names=names)


def feature_quantization_metadata_json(
    stats: dict[str, float],
    spec: FeatureQuantizationSpec,
) -> str:
    per_byte = 8 // spec.bits
    dim = len(spec.feature_keys)
    metadata = {
        "quantized_feature_key": _NODE_QUANTIZED_FEATURE_KEY,
        "dequantized_feature_keys": list(spec.feature_keys),
        "packed_feature_dim": (dim + per_byte - 1) // per_byte,
        "dequantized_feature_dim": dim,
        "bits": spec.bits,
    }
    metadata.update(stats)
    logger.info(f"Writing feature quantization metadata: {metadata}")
    return json.dumps(metadata)


def apply_feature_quantization_schema(
    metadata: DatasetMetadata,
    spec: FeatureQuantizationSpec,
) -> DatasetMetadata:
    drop_keys = set(spec.feature_keys) | {_NODE_QUANTIZED_FEATURE_KEY}
    schema = schema_pb2.Schema()
    schema.CopyFrom(metadata.schema)
    kept_features = [
        feature for feature in schema.feature if feature.name not in drop_keys
    ]
    del schema.feature[:]
    schema.feature.extend(kept_features)
    quantized_feature = schema.feature.add()
    quantized_feature.name = _NODE_QUANTIZED_FEATURE_KEY
    quantized_feature.type = schema_pb2.BYTES
    quantized_feature.value_count.min = 1
    quantized_feature.value_count.max = 1
    logger.info(
        f"Updated transformed schema for feature quantization: dropped "
        f"{len(spec.feature_keys)} features and added bytes feature "
        f"{_NODE_QUANTIZED_FEATURE_KEY}."
    )
    return DatasetMetadata(schema)


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
    if not cols:
        return np.empty((batch.num_rows, 0), dtype=np.float32)
    return np.stack(cols, axis=1)


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
        # Store mean values for negative and positive buckets.
        stats = {
            "neg_mean": neg_sum / neg_count if neg_count else 0.0,
            "pos_mean": pos_sum / pos_count if pos_count else 0.0,
        }
        logger.info(f"Computed Beam feature quantization stats: {stats}")
        return stats
