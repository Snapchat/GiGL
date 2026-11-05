"""Build-time lossless compression of feature columns into the packed feature sidecar.

Feature quantization already stores selected columns of a node or edge type as packed ``uint8`` rows next to the raw
float matrix, with a ``FeatureQuantizationMetadata`` describing them. Every later stage is generic over that pair:

- partitioning
- GLT storage
- sampling
- graph-store metadata fetching
- loader decode

This module adds columns to the same sidecar: every column that is integral on all ranks (one-hot columns, categorical
ids) is stored as the narrowest integer kind that holds its values (``FeatureCodec``). The kind is inferred from the
data when the dataset is built instead of declared in the preprocessor, so existing preprocessed data needs no rerun.
Columns that preprocessing already quantized are not in the raw matrix and are left alone.

``compress_lossless_columns`` is the pure tensor operation; ``compress_loaded_features`` applies it to the tensors of
all node or edge types and makes the codec identical on every rank by merging per-rank column statistics over RPC.

Compression is lossless and bit-exact: a column holding a non-integral value, a value no integer kind holds, or a
``-0.0`` stays float32.
"""

import time
from collections import Counter
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

_UNCOMPRESSED_KIND = "float32"
_ENCODE_CHUNK_ROWS = 1 << 16


def compress_lossless_columns(
    features: Optional[Float[torch.Tensor, "rows raw_dim"]],
    quantized_features: Optional[UInt8[torch.Tensor, "rows packed_dim"]],
    metadata: Optional[FeatureQuantizationMetadata],
    stats: Optional[ColumnStats],
    name: str,
) -> tuple[
    Optional[Float[torch.Tensor, "rows _remaining_dim"]],
    Optional[UInt8[torch.Tensor, "rows _new_packed_dim"]],
    Optional[FeatureQuantizationMetadata],
]:
    """Moves the integral columns of one node or edge type from the raw float matrix into the packed sidecar.

    The storage kind of each raw column comes from ``stats`` (``FeatureCodec.from_stats``), the statistics of the data
    merged over all partitions. Nothing is changed if no column is moved or moving them would not make rows smaller.

    Peak host memory is the input raw matrix, plus the remaining raw columns, plus the packed sidecar; the encode works in
    row chunks, so it needs no other full-size temporary. The input matrix can be released once the caller drops it.

    Args:
        features: Raw float matrix ``[N, raw_dim]`` holding every column that is not already quantized; ``None`` if
            this partition has no rows.
        quantized_features: Existing packed sidecar ``[N, packed_dim]`` from preprocessing, if any.
        metadata: Quantization metadata describing ``quantized_features``, if any.
        stats: ``ColumnStats`` of the raw matrix, merged over partitions; ``None`` if no partition has a raw matrix.
        name: Name of the node or edge type, for log messages.

    Returns:
        The remaining raw matrix (``None`` if empty), the new sidecar and the metadata describing it.

    Raises:
        ValueError: If the metadata already has a lossless region, or ``stats`` does not match the raw column count.
    """
    if metadata is not None and metadata.lossless_feature_indices:
        raise ValueError("The features already have a lossless region")
    if stats is None:
        return features, quantized_features, metadata
    num_raw_columns = stats.minimum.numel()
    if metadata is None:
        raw_columns: tuple[int, ...] = tuple(range(num_raw_columns))
    else:
        raw_columns = metadata.raw_feature_indices
    if num_raw_columns != len(raw_columns):
        raise ValueError(
            f"{name} has {len(raw_columns)} raw feature columns but the statistics cover {num_raw_columns}"
        )
    codec = FeatureCodec.from_stats(stats)
    logger.info(f"{name}: columns by chosen kind {dict(Counter(codec.column_dtypes))}")
    moved_positions: list[int] = []
    for position, kind in enumerate(codec.column_dtypes):
        if kind != _UNCOMPRESSED_KIND:
            moved_positions.append(position)
    if not moved_positions:
        return features, quantized_features, metadata
    region = FeatureCodec(tuple(codec.column_dtypes[p] for p in moved_positions))
    if region.row_bytes >= 4 * len(moved_positions):
        logger.info(
            f"Lossless compression would not shrink the rows of {name}; leaving them as float32"
        )
        return features, quantized_features, metadata

    moved_columns = [raw_columns[position] for position in moved_positions]
    if metadata is None:
        new_metadata = FeatureQuantizationMetadata(
            bits=None,
            feature_dim=num_raw_columns,
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
        offset = new_metadata.lossless_byte_offset
        for start in range(0, num_rows, _ENCODE_CHUNK_ROWS):
            chunk = features[start : start + _ENCODE_CHUNK_ROWS][:, moved_positions]
            packed[start : start + chunk.shape[0], offset:] = region.encode(chunk)
    remaining_positions: list[int] = []
    for position in range(num_raw_columns):
        if position not in moved_positions:
            remaining_positions.append(position)
    remaining: Optional[torch.Tensor] = None
    if features is not None and remaining_positions:
        remaining = features[:, remaining_positions]
    bytes_before = 4 * num_raw_columns
    if metadata is not None:
        bytes_before += metadata.packed_feature_dim
    bytes_after = 4 * len(remaining_positions) + new_metadata.packed_feature_dim
    logger.info(
        f"{name}: bytes per row {bytes_before} -> {bytes_after} "
        f"({len(moved_positions)} of {num_raw_columns} raw columns compressed)"
    )
    return remaining, packed, new_metadata


def compress_loaded_features(
    features: Optional[Union[torch.Tensor, Mapping[_Key, torch.Tensor]]],
    quantized_features: Optional[Union[torch.Tensor, Mapping[_Key, torch.Tensor]]],
    metadata: Optional[
        Union[FeatureQuantizationMetadata, Mapping[_Key, FeatureQuantizationMetadata]]
    ],
    is_homogeneous: bool,
    default_key: _Key,
) -> tuple[
    Optional[Union[torch.Tensor, dict[_Key, torch.Tensor]]],
    Optional[Union[torch.Tensor, dict[_Key, torch.Tensor]]],
    Optional[
        Union[FeatureQuantizationMetadata, dict[_Key, FeatureQuantizationMetadata]]
    ],
]:
    """Applies ``compress_lossless_columns`` to every node or edge type of the loaded tensors of this rank.

    Every rank must call this (it joins one RPC ``all_gather`` of the column statistics), so that all ranks end up with
    identical metadata even if a column only fails to fit on one rank's partition or a rank has no rows.

    Every type with a raw feature matrix on any rank is a candidate. Columns that preprocessing already quantized are
    not in the raw matrix and stay as they are.

    Logs on each rank the seconds spent scanning the statistics, merging them across ranks, encoding, and in total.

    Peak host memory is the input tensors of all compressed types, plus their remaining raw columns, plus the packed
    sidecars, until the caller releases the inputs.

    Args:
        features: Raw float matrix, or one per type for heterogeneous graphs. May already be keyed by ``default_key``
            for a homogeneous graph. A type missing here has no rows on this rank.
        quantized_features: Existing packed sidecar(s) from preprocessing.
        metadata: Existing quantization metadata, a single value for homogeneous graphs.
        is_homogeneous: Whether the graph is homogeneous, which decides the form of ``metadata``.
        default_key: Key the homogeneous graph's single type uses when tensors are keyed.

    Returns:
        ``features``, ``quantized_features`` and ``metadata`` after compression, in the same form as the inputs
        (``metadata`` is a single value for homogeneous graphs).
    """
    start_time = time.time()
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

    local_stats: dict[_Key, ColumnStats] = {}
    for key, raw in features_by_type.items():
        local_stats[key] = ColumnStats.from_features(raw)
    scan_done_time = time.time()
    # Every rank joins the gather, even one that holds no rows of a type, so the others never wait.
    gathered = list(all_gather(local_stats).values())
    types_to_compress: list[_Key] = []
    for rank_stats in gathered:
        for key in rank_stats:
            if key not in types_to_compress:
                types_to_compress.append(key)
    merged_by_type: dict[_Key, ColumnStats] = {}
    for key in types_to_compress:
        merged_by_type[key] = ColumnStats.merge(
            [rank_stats.get(key) for rank_stats in gathered]
        )
    merge_done_time = time.time()

    new_features = dict(features_by_type)
    new_quantized = dict(quantized_by_type)
    new_metadata = dict(metadata_by_type)
    for key in types_to_compress:
        remaining, packed, updated = compress_lossless_columns(
            features_by_type.get(key),
            quantized_by_type.get(key),
            metadata_by_type.get(key),
            merged_by_type[key],
            str(key),
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
    end_time = time.time()
    logger.info(
        f"Lossless feature compression took {end_time - start_time:.2f}s on this rank: "
        f"stats scan {scan_done_time - start_time:.2f}s, "
        f"cross-rank merge {merge_done_time - scan_done_time:.2f}s, "
        f"encode {end_time - merge_done_time:.2f}s"
    )

    keyed = not is_homogeneous
    for original in (features, quantized_features):
        if original is not None and not isinstance(original, torch.Tensor):
            keyed = True

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
