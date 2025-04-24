from typing import Dict, Optional, Tuple, Union

from gigl.common import UriFactory
from gigl.common.data.dataloaders import SerializedTFRecordInfo
from gigl.common.data.load_torch_tensors import SerializedGraphMetadata
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.src.common.types.pb_wrappers.graph_metadata import GraphMetadataPbWrapper
from gigl.src.common.types.pb_wrappers.preprocessed_metadata import (
    PreprocessedMetadataPbWrapper,
)
from gigl.src.data_preprocessor.lib.types import FeatureSpecDict
from gigl.types.graph import to_homogeneous
from snapchat.research.gbml.preprocessed_metadata_pb2 import PreprocessedMetadata


def _build_serialized_tfrecord_entity_info(
    preprocessed_metadata: Union[
        PreprocessedMetadata.NodeMetadataOutput, PreprocessedMetadata.EdgeMetadataInfo
    ],
    feature_spec_dict: FeatureSpecDict,
    entity_key: Union[str, Tuple[str, str]],
    tfrecord_uri_pattern: str,
) -> SerializedTFRecordInfo:
    """
    Populates a SerializedTFRecordInfo field from provided arguments for either a node or edge entity of a single node/edge type.
    Args:
        preprocessed_metadata(Union[
            PreprocessedMetadata.NodeMetadataOutput, PreprocessedMetadata.EdgeMetadataInfo
        ]): Preprocessed metadata pb for either NodeMetadataOutput or EdgeMetadataInfo
        feature_spec_dict (FeatureSpecDict): Feature spec to register to SerializedTFRecordInfo
        entity_key (Union[str, Tuple[str, str]]): Entity key to register to SerializedTFRecordInfo, is a str if Node entity or Tuple[str, str] if Edge entity
        tfrecord_uri_pattern (str): Regex pattern for loading serialized tf records
    Returns:
        SerializedTFRecordInfo: Stored metadata for current entity
    """
    return SerializedTFRecordInfo(
        tfrecord_uri_prefix=UriFactory.create_uri(
            preprocessed_metadata.tfrecord_uri_prefix
        ),
        feature_keys=list(preprocessed_metadata.feature_keys),
        feature_spec=feature_spec_dict,
        feature_dim=preprocessed_metadata.feature_dim,
        entity_key=entity_key,
        tfrecord_uri_pattern=tfrecord_uri_pattern,
    )


def convert_pb_to_serialized_graph_metadata(
    preprocessed_metadata_pb_wrapper: PreprocessedMetadataPbWrapper,
    graph_metadata_pb_wrapper: GraphMetadataPbWrapper,
    tfrecord_uri_pattern: str = ".*-of-.*\.tfrecord(\.gz)?$",
) -> SerializedGraphMetadata:
    """
    Populates a SerializedGraphMetadata field from PreprocessedMetadataPbWrapper and GraphMetadataPbWrapper, containing information for loading tensors for all entities and node/edge types.
    Args:
        preprocessed_metadata_pb_wrapper (PreprocessedMetadataPbWrapper): Preprocessed Metadata Pb Wrapper to translate into SerializedGraphMetadata
        graph_metadata_pb_wrapper (GraphMetadataPbWrapper): Graph Metadata Pb Wrapper to translate into Dataset Metadata
        tfrecord_uri_pattern (str): Regex pattern for loading serialized tf records
    Returns:
        SerializedGraphMetadata: Dataset Metadata for all entity and node/edge types.
    """

    node_entity_info: Dict[NodeType, SerializedTFRecordInfo] = {}
    edge_entity_info: Dict[EdgeType, SerializedTFRecordInfo] = {}
    positive_label_entity_info: Dict[EdgeType, Optional[SerializedTFRecordInfo]] = {}
    negative_label_entity_info: Dict[EdgeType, Optional[SerializedTFRecordInfo]] = {}

    preprocessed_metadata_pb = preprocessed_metadata_pb_wrapper.preprocessed_metadata_pb

    for node_type in graph_metadata_pb_wrapper.node_types:
        condensed_node_type = (
            graph_metadata_pb_wrapper.node_type_to_condensed_node_type_map[node_type]
        )
        node_metadata = (
            preprocessed_metadata_pb.condensed_node_type_to_preprocessed_metadata[
                condensed_node_type
            ]
        )

        node_feature_spec_dict = (
            preprocessed_metadata_pb_wrapper.condensed_node_type_to_feature_schema_map[
                condensed_node_type
            ].feature_spec
        )

        node_key = node_metadata.node_id_key

        node_entity_info[node_type] = _build_serialized_tfrecord_entity_info(
            preprocessed_metadata=node_metadata,
            feature_spec_dict=node_feature_spec_dict,
            entity_key=node_key,
            tfrecord_uri_pattern=tfrecord_uri_pattern,
        )

    for edge_type in graph_metadata_pb_wrapper.edge_types:
        condensed_edge_type = (
            graph_metadata_pb_wrapper.edge_type_to_condensed_edge_type_map[edge_type]
        )

        edge_metadata = (
            preprocessed_metadata_pb.condensed_edge_type_to_preprocessed_metadata[
                condensed_edge_type
            ]
        )

        edge_feature_spec_dict = (
            preprocessed_metadata_pb_wrapper.condensed_edge_type_to_feature_schema_map[
                condensed_edge_type
            ].feature_spec
        )

        edge_key = (
            edge_metadata.src_node_id_key,
            edge_metadata.dst_node_id_key,
        )

        edge_entity_info[edge_type] = _build_serialized_tfrecord_entity_info(
            preprocessed_metadata=edge_metadata.main_edge_info,
            feature_spec_dict=edge_feature_spec_dict,
            entity_key=edge_key,
            tfrecord_uri_pattern=tfrecord_uri_pattern,
        )

        if preprocessed_metadata_pb_wrapper.has_pos_edge_features(
            condensed_edge_type=condensed_edge_type
        ):
            pos_edge_feature_spec_dict = preprocessed_metadata_pb_wrapper.condensed_edge_type_to_pos_edge_feature_schema_map[
                condensed_edge_type
            ].feature_spec

            positive_label_entity_info[
                edge_type
            ] = _build_serialized_tfrecord_entity_info(
                preprocessed_metadata=edge_metadata.positive_edge_info,
                feature_spec_dict=pos_edge_feature_spec_dict,
                entity_key=edge_key,
                tfrecord_uri_pattern=tfrecord_uri_pattern,
            )
        else:
            positive_label_entity_info[edge_type] = None

        if preprocessed_metadata_pb_wrapper.has_hard_neg_edge_features(
            condensed_edge_type=condensed_edge_type
        ):
            hard_neg_edge_feature_spec_dict = preprocessed_metadata_pb_wrapper.condensed_edge_type_to_hard_neg_edge_feature_schema_map[
                condensed_edge_type
            ].feature_spec

            negative_label_entity_info[
                edge_type
            ] = _build_serialized_tfrecord_entity_info(
                preprocessed_metadata=edge_metadata.negative_edge_info,
                feature_spec_dict=hard_neg_edge_feature_spec_dict,
                entity_key=edge_key,
                tfrecord_uri_pattern=tfrecord_uri_pattern,
            )
        else:
            negative_label_entity_info[edge_type] = None

    if not graph_metadata_pb_wrapper.is_heterogeneous:
        # If our input is homogeneous, we remove the node/edge type component of the metadata fields.
        return SerializedGraphMetadata(
            node_entity_info=to_homogeneous(node_entity_info),
            edge_entity_info=to_homogeneous(edge_entity_info),
            positive_label_entity_info=to_homogeneous(positive_label_entity_info),
            negative_label_entity_info=to_homogeneous(negative_label_entity_info),
        )
    else:
        return SerializedGraphMetadata(
            node_entity_info=node_entity_info,
            edge_entity_info=edge_entity_info,
            positive_label_entity_info=positive_label_entity_info
            if not all(
                entity_info is None
                for entity_info in positive_label_entity_info.values()
            )
            else None,
            negative_label_entity_info=negative_label_entity_info
            if not all(
                entity_info is None
                for entity_info in negative_label_entity_info.values()
            )
            else None,
        )
