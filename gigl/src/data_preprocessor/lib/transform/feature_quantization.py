import json
from typing import Final, Iterable, TypeAlias

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
_NODE_PACKED_FEATURE_KEY: Final[str] = "node_packed_features"
_SignStats: TypeAlias = tuple[float, int, float, int]


def apply_feature_quantization_transform(
    logical_features: beam.PCollection[pa.RecordBatch],
    logical_metadata: DatasetMetadata | beam.PCollection[DatasetMetadata],
    logical_feature_keys: list[str],
    quantization_spec: FeatureQuantizationSpec,
    quantization_metadata_path: str,
) -> tuple[beam.PCollection[pa.RecordBatch], DatasetMetadata | beam.pvalue.AsSingleton]:
    missing = set(quantization_spec.feature_keys) - set(logical_feature_keys)
    if missing:
        raise ValueError(f"Quantized features missing: {missing}")

    logical_metadata_is_eager = isinstance(logical_metadata, DatasetMetadata)
    if logical_metadata_is_eager:
        metadata_for_json = logical_metadata
    else:
        metadata_for_json = beam.pvalue.AsSingleton(logical_metadata)

    logger.info(f"Applying feature quantization with spec: {quantization_spec}")
    quantization_stats = _build_quantization_stats(logical_features, quantization_spec)
    (
        quantization_stats
        | "Build quantization stats JSON"
        >> beam.Map(
            _quantization_stats_to_json,
            quantization_spec=quantization_spec,
            logical_feature_keys=logical_feature_keys,
            logical_metadata=metadata_for_json,
        )
        | "Write quantization stats"
        >> beam.io.WriteToText(
            quantization_metadata_path, num_shards=1, shard_name_template=""
        )
    )

    quantized_features = logical_features | (
        "Quantize feature RecordBatches"
        >> beam.Map(
            _quantize_record_batch,
            quantization_spec=quantization_spec,
            quantization_stats=beam.pvalue.AsSingleton(quantization_stats),
        )
    )

    if logical_metadata_is_eager:
        physical_feature_metadata = DatasetMetadata(
            _apply_quantization_schema(logical_metadata.schema, quantization_spec)
        )
    else:
        physical_feature_metadata = logical_metadata | (
            "Apply feature quantization schema"
            >> beam.Map(
                lambda metadata, quantization_spec: DatasetMetadata(
                    _apply_quantization_schema(metadata.schema, quantization_spec)
                ),
                quantization_spec=quantization_spec,
            )
        )
        physical_feature_metadata = beam.pvalue.AsSingleton(physical_feature_metadata)
    return quantized_features, physical_feature_metadata


def _build_quantization_stats(
    logical_features: beam.PCollection[pa.RecordBatch],
    quantization_spec: FeatureQuantizationSpec,
) -> beam.PCollection[dict[str, float]]:
    logger.info(
        f"Building Beam feature quantization stats for {len(quantization_spec.feature_keys)} "
        f"features with bits={quantization_spec.bits}: {quantization_spec.feature_keys}"
    )
    if quantization_spec.bits == 1:
        return (
            logical_features
            | "Compute single bit quantization stats"
            >> beam.CombineGlobally(_PosNegMeanFn(quantization_spec.feature_keys))
        )
    return (
        logical_features
        | "Build multi-bit quantization value batches"
        >> beam.Map(
            _flatten_feature_values,
            quantized_feature_keys=quantization_spec.feature_keys,
        )
        | "Compute multi-bit quantization quantiles"
        >> ApproximateQuantiles.Globally(num_quantiles=1000, input_batched=True)
        | "Build multi-bit quantization stats"
        >> beam.Map(_multi_bit_stats_from_quantiles)
    )


def _quantize_record_batch(
    batch: pa.RecordBatch,
    quantization_spec: FeatureQuantizationSpec,
    quantization_stats: dict[str, float],
) -> pa.RecordBatch:
    feature_matrix = _build_feature_matrix(batch, quantization_spec.feature_keys)
    if quantization_spec.bits == 1:
        packed = quantize_ndarray(feature_matrix, bits=quantization_spec.bits)
    else:
        packed = quantize_ndarray(
            feature_matrix,
            bits=quantization_spec.bits,
            clip_min=quantization_stats["clip_min"],
            clip_max=quantization_stats["clip_max"],
        )

    quantized_feature_keys = set(quantization_spec.feature_keys)
    arrays = [
        batch.column(i)
        for i, name in enumerate(batch.schema.names)
        if name not in quantized_feature_keys
    ]
    names = [name for name in batch.schema.names if name not in quantized_feature_keys]
    arrays.append(
        pa.array([[row.tobytes()] for row in packed], type=pa.list_(pa.binary()))
    )
    names.append(_NODE_PACKED_FEATURE_KEY)
    return pa.RecordBatch.from_arrays(arrays, names=names)


def _quantization_stats_to_json(
    quantization_stats: dict[str, float],
    quantization_spec: FeatureQuantizationSpec,
    logical_feature_keys: list[str],
    logical_metadata: DatasetMetadata,
) -> str:
    metadata = {
        "packed_feature_key": _NODE_PACKED_FEATURE_KEY,
        "quantized_feature_indices": _quantized_feature_indices(
            logical_metadata, logical_feature_keys, quantization_spec.feature_keys
        ),
        "bits": quantization_spec.bits,
        **quantization_stats,
    }
    logger.info(f"Writing feature quantization metadata: {metadata}")
    return json.dumps(metadata)


def _quantized_feature_indices(
    logical_metadata: DatasetMetadata,
    logical_feature_keys: list[str],
    quantized_feature_keys: list[str],
) -> list[int]:
    raw_feature_spec = schema_utils.schema_as_feature_spec(
        logical_metadata.schema
    ).feature_spec
    logical_feature_spec = {key: raw_feature_spec[key] for key in logical_feature_keys}
    logical_feature_index = feature_spec_to_feature_index_map(logical_feature_spec)

    feature_indices: list[int] = []
    for key in quantized_feature_keys:
        start, end = logical_feature_index[key]
        if end - start != 1:
            raise ValueError(f"Quantization expects scalar features, got {key}")
        feature_indices.append(start)
    return feature_indices


def _apply_quantization_schema(
    schema: schema_pb2.Schema, quantization_spec: FeatureQuantizationSpec
) -> schema_pb2.Schema:
    drop_keys = set(quantization_spec.feature_keys) | {_NODE_PACKED_FEATURE_KEY}
    quantized_schema = schema_pb2.Schema()
    quantized_schema.CopyFrom(schema)
    del quantized_schema.feature[:]
    quantized_schema.feature.extend(
        feature for feature in schema.feature if feature.name not in drop_keys
    )
    packed_feature = quantized_schema.feature.add()
    packed_feature.name = _NODE_PACKED_FEATURE_KEY
    packed_feature.type = schema_pb2.BYTES
    packed_feature.value_count.min = 1
    packed_feature.value_count.max = 1
    logger.info(
        f"Updated transformed schema for feature quantization: dropped "
        f"{len(quantization_spec.feature_keys)} features and added bytes feature "
        f"{_NODE_PACKED_FEATURE_KEY}."
    )
    return quantized_schema


def _flatten_feature_values(
    batch: pa.RecordBatch, quantized_feature_keys: list[str]
) -> list[float]:
    return _build_feature_matrix(batch, quantized_feature_keys).ravel().tolist()


def _build_feature_matrix(
    batch: pa.RecordBatch, quantized_feature_keys: list[str]
) -> np.ndarray:
    key_to_idx: dict[str, int] = {name: i for i, name in enumerate(batch.schema.names)}
    cols: list[np.ndarray] = []
    for key in quantized_feature_keys:
        if key not in key_to_idx:
            raise ValueError(f"Feature key {key} not found in RecordBatch.")

        col = batch.column(key_to_idx[key])
        values = np.asarray(col.to_numpy(zero_copy_only=False), dtype=np.float32)
        if values.ndim != 1:
            raise ValueError(
                f"Quantization expects scalar features, got {key} with shape {values.shape}."
            )
        cols.append(values)

    feature_matrix = np.stack(cols, axis=1)
    if not np.isfinite(feature_matrix).all():
        raise ValueError("Feature quantization expects finite feature values.")
    return feature_matrix


def _multi_bit_stats_from_quantiles(quantiles: list[float]) -> dict[str, float]:
    if not quantiles:
        raise ValueError("Cannot compute quantization stats from no values.")
    quantile_count = len(quantiles) - 1
    clip_min = float(quantiles[round(0.005 * quantile_count)])
    clip_max = float(quantiles[round(0.995 * quantile_count)])
    if clip_max <= clip_min:
        clip_max = clip_min + 1e-5
    stats = {"clip_min": clip_min, "clip_max": clip_max}
    logger.info(f"Computed feature quantization stats: {stats}")
    return stats


class _PosNegMeanFn(beam.CombineFn):
    """Accumulates mean positive and negative feature values for 1-bit quantization."""

    def __init__(self, feature_keys: list[str]) -> None:
        self._feature_keys = feature_keys

    def create_accumulator(self) -> _SignStats:
        return 0.0, 0, 0.0, 0

    def add_input(self, accumulator: _SignStats, batch: pa.RecordBatch) -> _SignStats:
        neg_sum, neg_count, pos_sum, pos_count = accumulator
        values = _build_feature_matrix(batch, self._feature_keys).ravel()
        neg = values <= 0
        pos = values > 0
        return (
            neg_sum + float(values[neg].sum()),
            neg_count + int(neg.sum()),
            pos_sum + float(values[pos].sum()),
            pos_count + int(pos.sum()),
        )

    def merge_accumulators(self, accumulators: Iterable[_SignStats]) -> _SignStats:
        neg_sum = neg_count = pos_sum = pos_count = 0
        for n_sum, n_count, p_sum, p_count in accumulators:
            neg_sum += n_sum
            neg_count += n_count
            pos_sum += p_sum
            pos_count += p_count
        return neg_sum, neg_count, pos_sum, pos_count

    def extract_output(self, accumulator: _SignStats) -> dict[str, float]:
        neg_sum, neg_count, pos_sum, pos_count = accumulator
        stats = {
            "neg_mean": neg_sum / neg_count if neg_count else 0.0,
            "pos_mean": pos_sum / pos_count if pos_count else 0.0,
        }
        logger.info(f"Computed Beam feature quantization stats: {stats}")
        return stats
