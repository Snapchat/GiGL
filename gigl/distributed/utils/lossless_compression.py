"""Build-time lossless compression of feature columns into the packed feature sidecar.

Feature quantization already stores selected columns of a node or edge type as packed ``uint8`` rows next to the raw
float matrix, with a ``FeatureQuantizationMetadata`` describing them, and every later stage (partitioning, GLT
storage, sampling, graph-store metadata fetching, loader decode) is generic over that pair. This module adds columns
to the same sidecar: categorical id columns stored as narrow integers (``FeatureCodec``), chosen when the dataset is
built instead of in the preprocessor, so existing preprocessed data needs no rerun.

``compress_lossless_columns`` is the pure tensor operation; ``compress_loaded_features`` applies it to the tensors of
all node or edge types and makes the codec identical on every rank by merging per-rank column statistics over RPC.

Integer columns come back equal to the values that were stored; a stored ``-0.0`` is read back as ``+0.0``.
"""

from dataclasses import replace
from typing import Hashable, Mapping, Optional, TypeVar, Union

import torch
from graphlearn_torch.distributed.rpc import all_gather
from jaxtyping import Float, UInt8

from gigl.common.logger import Logger
from gigl.common.utils.feature_quantization.lossless import ColumnStats, FeatureCodec
from gigl.types.graph import FeatureQuantizationMetadata

logger = Logger()

_Key = TypeVar("_Key", bound=Hashable)

# Declared kind of a column that is not compressed.
_UNCOMPRESSED_KIND = "float32"
_ENCODE_CHUNK_ROWS = 1 << 16


def _candidate_columns(
    declared: FeatureCodec,
    metadata: Optional[FeatureQuantizationMetadata],
) -> tuple[list[int], dict[int, int]]:
    """Validates ``declared`` against the entity's metadata.

    Returns:
        The logical columns declared as compressed, and the position of every logical raw column in the raw matrix.
    """
    if metadata is None:
        logical_to_raw = {column: column for column in range(declared.num_columns)}
    else:
        if metadata.lossless_feature_indices:
            raise ValueError("The features already have a lossless region")
        if declared.num_columns != metadata.feature_dim:
            raise ValueError(
                f"The compression spec has {declared.num_columns} columns but the features have {metadata.feature_dim}"
            )
        logical_to_raw = {
            column: position
            for position, column in enumerate(metadata.raw_feature_indices)
        }
    candidates = [
        column
        for column, kind in enumerate(declared.column_dtypes)
        if kind != _UNCOMPRESSED_KIND
    ]
    lossy = [
        column for column in candidates if declared.column_dtypes[column] == "bfloat16"
    ]
    if lossy:
        raise ValueError(
            f"Columns {lossy} are declared bfloat16, which is lossy; lossless compression supports only integer kinds"
        )
    already_packed = [column for column in candidates if column not in logical_to_raw]
    if already_packed:
        raise ValueError(
            f"Columns {already_packed} are already stored quantized and cannot also be compressed losslessly"
        )
    return candidates, logical_to_raw


def _check_width(
    features: Optional[Float[torch.Tensor, "rows raw_dim"]],
    logical_to_raw: dict[int, int],
) -> None:
    if features is not None and features.shape[1] != len(logical_to_raw):
        raise ValueError(
            f"The compression spec expects {len(logical_to_raw)} raw feature columns but the features have "
            f"{features.shape[1]}"
        )


def lossless_column_stats(
    features: Optional[Float[torch.Tensor, "rows raw_dim"]],
    metadata: Optional[FeatureQuantizationMetadata],
    declared: FeatureCodec,
) -> Optional[ColumnStats]:
    """Statistics of the declared columns of one partition's raw float features.

    Args:
        features: Raw float matrix of this partition; ``None`` means the partition has no rows.
        metadata: Existing quantization metadata of the entity, if any.
        declared: Storage kind of every logical column.

    Returns:
        ``None`` if no column is declared; otherwise statistics over the declared columns (those of zero rows for an
        empty partition, which do not change the merged result).
    """
    candidates, logical_to_raw = _candidate_columns(declared, metadata)
    _check_width(features, logical_to_raw)
    if not candidates:
        return None
    if features is None:
        return ColumnStats.empty(len(candidates))
    return ColumnStats.from_features(
        features, [logical_to_raw[column] for column in candidates]
    )


def compress_lossless_columns(
    features: Optional[Float[torch.Tensor, "rows raw_dim"]],
    quantized_features: Optional[UInt8[torch.Tensor, "rows packed_dim"]],
    metadata: Optional[FeatureQuantizationMetadata],
    declared: FeatureCodec,
    stats: Optional[ColumnStats],
    strict: bool = False,
) -> tuple[
    Optional[Float[torch.Tensor, "rows _remaining_dim"]],
    Optional[UInt8[torch.Tensor, "rows _new_packed_dim"]],
    Optional[FeatureQuantizationMetadata],
]:
    """Moves the declared columns of one node or edge type from the raw float matrix into the packed sidecar.

    A declared column is only moved if ``stats`` (the statistics of the data, merged over all partitions) show it fits
    its storage kind; the others stay in the raw matrix and are logged, or raise if ``strict``. Nothing is changed if
    no column is moved or moving them would not make rows smaller.

    Peak host memory is the input raw matrix, plus the remaining raw columns, plus the packed sidecar; the encode works in
    row chunks, so it needs no other full-size temporary. The input matrix can be released once the caller drops it.

    Args:
        features: Raw float matrix ``[N, raw_dim]`` holding every column that is not already quantized; ``None`` if
            this partition has no rows.
        quantized_features: Existing packed sidecar ``[N, packed_dim]`` from preprocessing, if any.
        metadata: Quantization metadata describing ``quantized_features``, if any.
        declared: Storage kind of every logical column; ``float32`` leaves a column alone.
        stats: ``lossless_column_stats`` of the data, merged over partitions.
        strict: Raise instead of leaving a column that does not fit in the raw matrix.

    Returns:
        The remaining raw matrix (``None`` if empty), the new sidecar and the metadata describing it.
    """
    candidates, logical_to_raw = _candidate_columns(declared, metadata)
    _check_width(features, logical_to_raw)
    if not candidates or stats is None:
        return features, quantized_features, metadata
    fitted = FeatureCodec(
        tuple(declared.column_dtypes[column] for column in candidates)
    ).fitted_to(stats, strict=strict)
    moved = [
        (column, kind)
        for column, kind in zip(candidates, fitted.column_dtypes)
        if kind != _UNCOMPRESSED_KIND
    ]
    if not moved:
        return features, quantized_features, metadata
    region = FeatureCodec(tuple(kind for _, kind in moved))
    if region.row_bytes >= 4 * len(moved):
        logger.info(
            "Lossless compression would not shrink the rows; leaving them as float32"
        )
        return features, quantized_features, metadata

    moved_columns = [column for column, _ in moved]
    if metadata is None:
        new_metadata = FeatureQuantizationMetadata(
            bits=None,
            feature_dim=declared.num_columns,
            lossless_feature_indices=tuple(moved_columns),
            lossless_column_kinds=region.column_dtypes,
        )
    else:
        new_metadata = replace(
            metadata,
            lossless_feature_indices=tuple(moved_columns),
            lossless_column_kinds=region.column_dtypes,
        )

    num_rows = 0 if features is None else features.shape[0]
    packed = torch.zeros(num_rows, new_metadata.packed_feature_dim, dtype=torch.uint8)
    if quantized_features is not None:
        packed[:, : quantized_features.shape[1]] = quantized_features
    if features is not None:
        moved_positions = [logical_to_raw[column] for column in moved_columns]
        offset = new_metadata.lossless_byte_offset
        for start in range(0, num_rows, _ENCODE_CHUNK_ROWS):
            chunk = features[start : start + _ENCODE_CHUNK_ROWS][:, moved_positions]
            packed[start : start + chunk.shape[0], offset:] = region.encode(chunk)
    moved_set = set(moved_columns)
    remaining_positions = [
        position
        for column, position in logical_to_raw.items()
        if column not in moved_set
    ]
    remaining = (
        features[:, remaining_positions]
        if features is not None and remaining_positions
        else None
    )
    return remaining, packed, new_metadata


def compress_loaded_features(
    features: Optional[Union[torch.Tensor, Mapping[_Key, torch.Tensor]]],
    quantized_features: Optional[Union[torch.Tensor, Mapping[_Key, torch.Tensor]]],
    metadata: Optional[
        Union[FeatureQuantizationMetadata, Mapping[_Key, FeatureQuantizationMetadata]]
    ],
    declared: Union[FeatureCodec, Mapping[_Key, FeatureCodec]],
    is_homogeneous: bool,
    default_key: _Key,
    strict: bool = False,
) -> tuple[
    Optional[Union[torch.Tensor, dict[_Key, torch.Tensor]]],
    Optional[Union[torch.Tensor, dict[_Key, torch.Tensor]]],
    Optional[
        Union[FeatureQuantizationMetadata, dict[_Key, FeatureQuantizationMetadata]]
    ],
]:
    """Applies ``compress_lossless_columns`` to every node or edge type of the loaded tensors of this rank.

    Every rank must call this with the same ``declared`` (it joins one RPC ``all_gather``), so that all ranks end up with
    identical metadata even if a column only fails to fit on one rank's partition or a rank has no rows.

    Peak host memory is the input tensors of all compressed types, plus their remaining raw columns, plus the packed
    sidecars, until the caller releases the inputs.

    Args:
        features: Raw float matrix, or one per type for heterogeneous graphs. May already be keyed by ``default_key``
            for a homogeneous graph. A type missing here has no rows on this rank.
        quantized_features: Existing packed sidecar(s) from preprocessing.
        metadata: Existing quantization metadata, a single value for homogeneous graphs.
        declared: A codec for a homogeneous graph; a mapping with a codec for each type to compress otherwise.
        is_homogeneous: Whether the graph is homogeneous, which decides the form of ``declared`` and ``metadata``.
        default_key: Key the homogeneous graph's single type uses when tensors are keyed.
        strict: Raise instead of leaving a column that does not fit in the raw matrix.

    Returns:
        ``features``, ``quantized_features`` and ``metadata`` after compression, in the same form as the inputs
        (``metadata`` is a single value for homogeneous graphs).
    """
    if isinstance(declared, FeatureCodec):
        if not is_homogeneous:
            raise ValueError("A heterogeneous graph needs a FeatureCodec per type")
        declared_by_type: dict[_Key, FeatureCodec] = {default_key: declared}
    else:
        if is_homogeneous:
            raise ValueError("A homogeneous graph needs a single FeatureCodec")
        declared_by_type = dict(declared)

    metadata_by_type: dict[_Key, FeatureQuantizationMetadata] = {}
    if isinstance(metadata, FeatureQuantizationMetadata):
        metadata_by_type[default_key] = metadata
    elif metadata is not None:
        metadata_by_type.update(metadata)

    def by_type(
        tensors: Optional[Union[torch.Tensor, Mapping[_Key, torch.Tensor]]],
    ) -> dict[_Key, torch.Tensor]:
        if tensors is None:
            return {}
        if isinstance(tensors, torch.Tensor):
            return {default_key: tensors}
        return dict(tensors)

    features_by_type = by_type(features)
    quantized_by_type = by_type(quantized_features)

    # Every rank joins the gather before any data-dependent check, so a rank whose shard holds no rows for a type
    # (no feature tensor at all) cannot leave the other ranks waiting.
    local_report: dict[_Key, tuple[bool, Optional[ColumnStats]]] = {}
    for key, codec in declared_by_type.items():
        local_report[key] = (
            key in features_by_type or key in metadata_by_type,
            lossless_column_stats(
                features_by_type.get(key), metadata_by_type.get(key), codec
            ),
        )
    gathered = list(all_gather(local_report).values())
    missing = [
        key
        for key in declared_by_type
        if not any(rank_report[key][0] for rank_report in gathered)
    ]
    if missing:
        raise ValueError(
            f"No features to compress for {sorted(map(str, missing))} on any rank"
        )

    new_features = dict(features_by_type)
    new_quantized = dict(quantized_by_type)
    new_metadata = dict(metadata_by_type)
    for key, codec in declared_by_type.items():
        merged_stats = [rank_report[key][1] for rank_report in gathered]
        merged = (
            ColumnStats.merge(merged_stats)
            if any(stats is not None for stats in merged_stats)
            else None
        )
        remaining, packed, updated = compress_lossless_columns(
            features_by_type.get(key),
            quantized_by_type.get(key),
            metadata_by_type.get(key),
            codec,
            merged,
            strict=strict,
        )
        for tensors, value in ((new_features, remaining), (new_quantized, packed)):
            if value is None:
                tensors.pop(key, None)
            else:
                tensors[key] = value
        if updated is None:
            new_metadata.pop(key, None)
        else:
            new_metadata[key] = updated

    keyed = not is_homogeneous or any(
        not isinstance(original, torch.Tensor) and original is not None
        for original in (features, quantized_features)
    )

    def restore(
        tensors: dict[_Key, torch.Tensor],
    ) -> Optional[Union[torch.Tensor, dict[_Key, torch.Tensor]]]:
        if not tensors:
            return None
        return tensors if keyed else tensors[default_key]

    restored_metadata: Optional[
        Union[FeatureQuantizationMetadata, dict[_Key, FeatureQuantizationMetadata]]
    ]
    if not new_metadata:
        restored_metadata = None
    elif is_homogeneous:
        restored_metadata = new_metadata[default_key]
    else:
        restored_metadata = new_metadata
    return restore(new_features), restore(new_quantized), restored_metadata
