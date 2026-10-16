"""Derives lossless feature compression specs from GiGL's preprocessing metadata.

``build_dataset`` takes explicit ``FeatureCodec`` declarations (one storage kind per logical feature column) and knows
nothing about schemas or task configs. This module is the bridge for callers that do have them: it declares each
column from the feature schema's ``int_domain``.
"""

from typing import Optional, Sequence, Union

from gigl.common.data.load_torch_tensors import SerializedGraphMetadata
from gigl.common.utils.feature_quantization.lossless import (
    FeatureCodec,
    kind_holding_domain,
)
from gigl.common.utils.tensorflow_schema import feature_spec_to_feature_index_map
from gigl.src.common.types.graph_data import NodeType
from gigl.src.common.types.pb_wrappers.graph_metadata import GraphMetadataPbWrapper
from gigl.src.common.types.pb_wrappers.preprocessed_metadata import (
    PreprocessedMetadataPbWrapper,
)
from gigl.src.data_preprocessor.lib.types import FeatureSchema
from gigl.types.graph import FeatureQuantizationMetadata


def codec_from_feature_schema(
    schema: FeatureSchema,
    feature_keys: Sequence[str],
) -> FeatureCodec:
    """Declares the storage kind of every column of a matrix described by a feature schema.

    The matrix columns are the ``feature_keys`` fields concatenated in that order. A schema feature with an
    ``int_domain`` is declared as the narrowest integer kind that holds the domain; every other column (a feature
    without an ``int_domain``, or a domain no integer kind holds) is declared ``float32``, i.e. left uncompressed.

    The result is a declaration, not a guarantee about any data: ``build_dataset`` checks it against the data.

    Args:
        schema: Feature schema holding the spec and ``int_domain`` of every feature key.
        feature_keys: Features that make up the matrix, in column order.
    """
    feature_index = feature_spec_to_feature_index_map(
        {key: schema.feature_spec[key] for key in feature_keys}
    )
    kinds = ["float32"] * max((end for _, end in feature_index.values()), default=0)
    for name, (start, end) in feature_index.items():
        feature = schema.schema[name]
        if feature.HasField("int_domain"):
            kind = kind_holding_domain(
                int(feature.int_domain.min), int(feature.int_domain.max)
            )
            kinds[start:end] = [kind or "float32"] * (end - start)
    return FeatureCodec(tuple(kinds))


def exclude_quantized_columns(
    codec: FeatureCodec, metadata: Optional[FeatureQuantizationMetadata]
) -> FeatureCodec:
    """Leaves columns that preprocessing already quantized uncompressed (they cannot also be packed losslessly)."""
    if metadata is None:
        return codec
    kinds = list(codec.column_dtypes)
    for column in metadata.quantized_feature_indices:
        kinds[column] = "float32"
    return FeatureCodec(tuple(kinds))


def derive_node_lossless_codecs(
    preprocessed_metadata_pb_wrapper: PreprocessedMetadataPbWrapper,
    graph_metadata_pb_wrapper: GraphMetadataPbWrapper,
    serialized_graph_metadata: SerializedGraphMetadata,
) -> Optional[Union[FeatureCodec, dict[NodeType, FeatureCodec]]]:
    """Declares a ``FeatureCodec`` over the logical node feature columns of each node type.

    Args:
        preprocessed_metadata_pb_wrapper: Preprocessed metadata holding the feature schemas and feature keys.
        graph_metadata_pb_wrapper: Graph metadata mapping node types to their condensed types.
        serialized_graph_metadata: The graph's serialized metadata; node columns that preprocessing quantized are left
            uncompressed.

    Returns:
        A codec for a homogeneous graph, or a dict with a codec for every node type that has features otherwise;
        ``None`` if no node type has features.
    """
    schemas = preprocessed_metadata_pb_wrapper.condensed_node_type_to_feature_schema_map
    keys = preprocessed_metadata_pb_wrapper.condensed_node_type_to_feature_keys_map
    node_quantization_metadata = serialized_graph_metadata.node_quantization_metadata
    quantization_by_type = (
        {}
        if node_quantization_metadata is None
        else (
            {
                graph_metadata_pb_wrapper.homogeneous_node_type: node_quantization_metadata
            }
            if isinstance(node_quantization_metadata, FeatureQuantizationMetadata)
            else node_quantization_metadata
        )
    )
    codecs = {
        node_type: exclude_quantized_columns(
            codec_from_feature_schema(schemas[condensed], keys[condensed]),
            quantization_by_type.get(node_type),
        )
        for node_type, condensed in (
            graph_metadata_pb_wrapper.node_type_to_condensed_node_type_map.items()
        )
        if keys[condensed]
    }
    if not codecs:
        return None
    if graph_metadata_pb_wrapper.is_heterogeneous:
        return codecs
    return codecs[graph_metadata_pb_wrapper.homogeneous_node_type]
