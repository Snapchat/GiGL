import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional

import tensorflow as tf
import torch
from tensorflow_transform.tf_metadata import schema_utils

from gigl.common import GcsUri, LocalUri, UriFactory
from gigl.common.logger import Logger
from gigl.common.utils.proto_utils import ProtoUtils
from gigl.src.common.types.graph_data import (
    CondensedEdgeType,
    CondensedNodeType,
    EdgeType,
    EdgeUsageType,
    NodeType,
)
from gigl.src.common.utils.file_loader import FileLoader
from gigl.src.data_preprocessor.lib.transform.tf_value_encoder import TFValueEncoder
from gigl.src.data_preprocessor.lib.types import FeatureSpecDict, InstanceDict
from gigl.src.mocking.lib.constants import (
    get_example_task_edge_features_gcs_dir,
    get_example_task_edge_features_schema_gcs_path,
    get_example_task_node_features_gcs_dir,
    get_example_task_node_features_schema_gcs_path,
)
from gigl.src.mocking.lib.feature_handling import get_feature_field_name
from gigl.src.mocking.lib.mocked_dataset_resources import MockedDatasetInfo
from snapchat.research.gbml import gbml_config_pb2, preprocessed_metadata_pb2

logger = Logger()


@dataclass
class _PreprocessMetadata:
    features_uri: GcsUri
    schema_uri: GcsUri
    feature_cols: List[str]


@dataclass
class _NodePreprocessMetadata(_PreprocessMetadata):
    id_col: str
    label_col: Optional[str] = None


@dataclass
class _EdgePreprocessMetadata(_PreprocessMetadata):
    src_id_col: str
    dst_id_col: str


class _InstanceDictToTFExample:
    """
    Uses a feature spec to process a raw instance dict (read from some tabular data) as a TFExample.
    """

    def __init__(self, feature_spec: FeatureSpecDict):
        self.feature_spec = feature_spec

    def process(self, element: InstanceDict) -> bytes:
        # Each row is a single instance dict from the original tabular input (BQ, GCS, etc.)
        example = dict()
        for key in self.feature_spec.keys():
            # prepare each value associated with a key that appears in the feature_spec.
            # only the instance dict keys the user specifies wanting in the feature_spec will pass through here
            value = element[key]
            if value is None:
                logger.debug(f"Found key {key} with missing value in sample {element}")
            example[key] = TFValueEncoder.encode_value_as_feature(
                value=value, dtype=self.feature_spec[key].dtype
            )
        example_proto = tf.train.Example(features=tf.train.Features(feature=example))
        serialized_proto = example_proto.SerializeToString()
        return serialized_proto


def _generate_preprocessed_node_tfrecord_data(
    data: MockedDatasetInfo,
    version: str,
    node_type: NodeType,
    num_node_features: int,
    node_features: torch.Tensor,
    node_labels: Optional[torch.Tensor],
) -> _NodePreprocessMetadata:
    feature_names: List[str] = [
        get_feature_field_name(n=i) for i in range(num_node_features)
    ]
    feature_spec_dict = {
        data.node_id_column_name: tf.io.FixedLenFeature(shape=[], dtype=tf.int64)
    }
    feature_spec_dict.update(
        {
            col: tf.io.FixedLenFeature(shape=[], dtype=tf.float32)
            for col in feature_names
        }
    )
    if node_labels is not None:
        feature_spec_dict.update(
            {
                data.node_label_column_name: tf.io.FixedLenFeature(
                    shape=[], dtype=tf.int64
                )
            }
        )

    id2tfe_encoder = _InstanceDictToTFExample(feature_spec=feature_spec_dict)

    tfrecords = []
    instance_dict_feats: Dict[str, torch.Tensor]
    for node_id, node_feature_values in enumerate(node_features):
        instance_dict_feats = {data.node_id_column_name: torch.LongTensor([node_id])}
        instance_dict_feats.update(
            {
                feat_name: feat_value
                for feat_name, feat_value in zip(feature_names, node_feature_values)
            }
        )
        if node_labels is not None:
            instance_dict_feats.update(
                {data.node_label_column_name: node_labels[node_id]}
            )
        tfrecord_bytes = id2tfe_encoder.process(element=instance_dict_feats)
        tfrecords.append(tfrecord_bytes)

    # Write features to GCS.
    features_path = get_example_task_node_features_gcs_dir(
        task_name=data.name, version=version, node_type=node_type
    )
    with tf.io.TFRecordWriter(
        GcsUri.join(features_path, "data.tfrecord").uri
    ) as writer:
        for tfrecord in tfrecords:
            writer.write(tfrecord)
    logger.info(
        f"Wrote preprocessed node TFRecords for type {node_type} to prefix {features_path.uri}"
    )

    # Write schema to GCS.
    node_schema_uri = get_example_task_node_features_schema_gcs_path(
        task_name=data.name, version=version, node_type=node_type
    )
    node_schema = schema_utils.schema_from_feature_spec(feature_spec=feature_spec_dict)
    file_loader = FileLoader()
    temp_file_handle = tempfile.NamedTemporaryFile()
    with open(temp_file_handle.name, "w") as f:
        f.write(repr(node_schema))
    file_loader.load_file(
        file_uri_src=LocalUri(temp_file_handle.name),
        file_uri_dst=node_schema_uri,
    )
    logger.info(
        f"Wrote preprocessed node TFRecords schema for type {node_type} to prefix {node_schema_uri.uri}"
    )
    return _NodePreprocessMetadata(
        features_uri=features_path,
        schema_uri=node_schema_uri,
        feature_cols=feature_names,
        id_col=data.node_id_column_name,
        label_col=data.node_label_column_name if node_labels is not None else None,
    )


def _generate_preprocessed_edge_tfrecord_data(
    data: MockedDatasetInfo,
    version: str,
    edge_type: EdgeType,
    edge_index: torch.Tensor,
    num_edge_features: int,
    edge_features: Optional[torch.Tensor],
    edge_usage_type: EdgeUsageType,
) -> _EdgePreprocessMetadata:
    feature_names: List[str] = [
        get_feature_field_name(n=i) for i in range(num_edge_features)
    ]
    feature_spec_dict = {
        data.edge_src_column_name: tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        data.edge_dst_column_name: tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
    }
    feature_spec_dict.update(
        {
            col: tf.io.FixedLenFeature(shape=[], dtype=tf.float32)
            for col in feature_names
        }
    )

    id2tfe_encoder = _InstanceDictToTFExample(feature_spec=feature_spec_dict)

    tfrecords = []
    for edge_id, (src_id, dst_id) in enumerate(zip(edge_index[0, :], edge_index[1, :])):
        instance_dict_feats = {
            data.edge_src_column_name: src_id,
            data.edge_dst_column_name: dst_id,
        }
        if edge_features is not None:
            edge_feature_values = edge_features[edge_id, :]
            instance_dict_feats.update(
                {
                    feat_name: feat_value
                    for feat_name, feat_value in zip(feature_names, edge_feature_values)
                }
            )
        tfrecord_bytes = id2tfe_encoder.process(element=instance_dict_feats)
        tfrecords.append(tfrecord_bytes)

    # Write features to GCS.
    features_path = get_example_task_edge_features_gcs_dir(
        task_name=data.name,
        version=version,
        edge_type=edge_type,
        edge_usage_type=edge_usage_type,
    )
    with tf.io.TFRecordWriter(
        GcsUri.join(features_path, "data.tfrecord").uri
    ) as writer:
        for tfrecord in tfrecords:
            writer.write(tfrecord)
    logger.info(
        f"Wrote preprocessed edge TFRecords for type {edge_type} to prefix {features_path.uri}"
    )

    # Write schema to GCS.
    edge_schema_uri = get_example_task_edge_features_schema_gcs_path(
        task_name=data.name,
        version=version,
        edge_type=edge_type,
        edge_usage_type=edge_usage_type,
    )
    edge_schema = schema_utils.schema_from_feature_spec(feature_spec=feature_spec_dict)
    file_loader = FileLoader()
    temp_file_handle = tempfile.NamedTemporaryFile()
    with open(temp_file_handle.name, "w") as f:
        f.write(repr(edge_schema))
    file_loader.load_file(
        file_uri_src=LocalUri(temp_file_handle.name),
        file_uri_dst=edge_schema_uri,
    )
    logger.info(
        f"Wrote preprocessed edge TFRecords schema for type {edge_type} to {edge_schema_uri.uri}"
    )

    return _EdgePreprocessMetadata(
        features_uri=features_path,
        schema_uri=edge_schema_uri,
        feature_cols=feature_names,
        src_id_col=data.edge_src_column_name,
        dst_id_col=data.edge_dst_column_name,
    )


def generate_preprocessed_tfrecord_data(
    mocked_dataset_info: MockedDatasetInfo,
    version: str,
    gbml_config_pb: gbml_config_pb2.GbmlConfig,
):
    graph_metadata_pb_wrapper = mocked_dataset_info.graph_metadata_pb_wrapper
    num_features_by_node_type = mocked_dataset_info.num_node_features
    node_features_by_node_type = mocked_dataset_info.node_feats
    node_labels_by_node_type = mocked_dataset_info.node_labels
    node_types = mocked_dataset_info.node_types

    condensed_node_type_to_preprocessed_metadata: Dict[
        CondensedNodeType,
        preprocessed_metadata_pb2.PreprocessedMetadata.NodeMetadataOutput,
    ] = dict()
    for node_type in node_types:
        condensed_node_type = (
            graph_metadata_pb_wrapper.node_type_to_condensed_node_type_map[node_type]
        )
        num_features = num_features_by_node_type[node_type]
        node_features = node_features_by_node_type[node_type]

        if not num_features:
            continue

        node_labels = (
            node_labels_by_node_type[node_type]
            if node_labels_by_node_type is not None
            else None
        )

        node_preprocess_metadata = _generate_preprocessed_node_tfrecord_data(
            data=mocked_dataset_info,
            version=version,
            node_type=node_type,
            num_node_features=num_features,
            node_features=node_features,
            node_labels=node_labels,
        )

        condensed_node_type_to_preprocessed_metadata[
            condensed_node_type
        ] = preprocessed_metadata_pb2.PreprocessedMetadata.NodeMetadataOutput(
            node_id_key=node_preprocess_metadata.id_col,
            feature_keys=node_preprocess_metadata.feature_cols,
            label_keys=[node_preprocess_metadata.label_col] if node_preprocess_metadata.label_col is not None else None,  # type: ignore
            tfrecord_uri_prefix=node_preprocess_metadata.features_uri.uri,
            schema_uri=node_preprocess_metadata.schema_uri.uri,
            feature_dim=num_features,
        )

    num_features_by_edge_type = mocked_dataset_info.num_edge_features
    edge_features_by_edge_type = mocked_dataset_info.edge_feats
    edge_index_by_edge_type = mocked_dataset_info.edge_index
    edge_types = mocked_dataset_info.edge_types
    condensed_edge_type_to_preprocessed_metadata: Dict[
        CondensedEdgeType,
        preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataOutput,
    ] = dict()
    for edge_type in edge_types:
        condensed_edge_type = (
            graph_metadata_pb_wrapper.edge_type_to_condensed_edge_type_map[edge_type]
        )
        num_features = num_features_by_edge_type[edge_type]
        edge_features = (
            edge_features_by_edge_type[edge_type]
            if edge_features_by_edge_type is not None
            else None
        )
        edge_index = edge_index_by_edge_type[edge_type]

        edge_preprocess_metadata = _generate_preprocessed_edge_tfrecord_data(
            data=mocked_dataset_info,
            version=version,
            edge_type=edge_type,
            edge_index=edge_index,
            num_edge_features=num_features,
            edge_features=edge_features,
            edge_usage_type=EdgeUsageType.MAIN,
        )
        main_edge_metadata_info_pb = (
            preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataInfo(
                feature_keys=edge_preprocess_metadata.feature_cols,
                tfrecord_uri_prefix=edge_preprocess_metadata.features_uri.uri,
                schema_uri=edge_preprocess_metadata.schema_uri.uri,
                feature_dim=num_features,
            )
        )
        if (
            mocked_dataset_info.user_defined_edge_index
            and edge_type in mocked_dataset_info.user_defined_edge_index
        ):
            edge_preprocess_metadata_pb_dict = {}
            for (
                user_def_label,
                user_def_edge_index,
            ) in mocked_dataset_info.user_defined_edge_index[edge_type].items():
                num_edge_feats = mocked_dataset_info.num_user_def_edge_features[
                    edge_type
                ][user_def_label]
                user_defined_edge_feats = (
                    mocked_dataset_info.user_defined_edge_feats[edge_type][
                        user_def_label
                    ]
                    if mocked_dataset_info.user_defined_edge_feats
                    and edge_type in mocked_dataset_info.user_defined_edge_feats
                    else None
                )
                user_def_edge_preprocess_metadata = (
                    _generate_preprocessed_edge_tfrecord_data(
                        data=mocked_dataset_info,
                        version=version,
                        edge_type=edge_type,
                        edge_index=user_def_edge_index,
                        num_edge_features=num_edge_feats,
                        edge_features=user_defined_edge_feats,
                        edge_usage_type=user_def_label,
                    )
                )

                user_def_edge_metadata_info_pb = preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataInfo(
                    feature_keys=user_def_edge_preprocess_metadata.feature_cols,
                    tfrecord_uri_prefix=user_def_edge_preprocess_metadata.features_uri.uri,
                    schema_uri=user_def_edge_preprocess_metadata.schema_uri.uri,
                    feature_dim=num_edge_feats,
                )

                edge_preprocess_metadata_pb_dict[
                    user_def_label
                ] = user_def_edge_metadata_info_pb

            condensed_edge_type_to_preprocessed_metadata[
                condensed_edge_type
            ] = preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataOutput(
                src_node_id_key=edge_preprocess_metadata.src_id_col,
                dst_node_id_key=edge_preprocess_metadata.dst_id_col,
                main_edge_info=main_edge_metadata_info_pb,
                positive_edge_info=edge_preprocess_metadata_pb_dict.get(
                    EdgeUsageType.POSITIVE, None
                ),
                negative_edge_info=edge_preprocess_metadata_pb_dict.get(
                    EdgeUsageType.NEGATIVE, None
                ),
            )
        else:
            condensed_edge_type_to_preprocessed_metadata[
                condensed_edge_type
            ] = preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataOutput(
                src_node_id_key=edge_preprocess_metadata.src_id_col,
                dst_node_id_key=edge_preprocess_metadata.dst_id_col,
                main_edge_info=main_edge_metadata_info_pb,
            )

    # Assemble Preprocessed Metadata pb and write out.
    preprocessed_metadata_pb = preprocessed_metadata_pb2.PreprocessedMetadata()
    for (
        condensed_node_type,
        node_metadata_output,
    ) in condensed_node_type_to_preprocessed_metadata.items():
        preprocessed_metadata_pb.condensed_node_type_to_preprocessed_metadata[
            condensed_node_type
        ].CopyFrom(node_metadata_output)
    for (
        condensed_edge_type,
        edge_metadata_output,
    ) in condensed_edge_type_to_preprocessed_metadata.items():
        preprocessed_metadata_pb.condensed_edge_type_to_preprocessed_metadata[
            condensed_edge_type
        ].CopyFrom(edge_metadata_output)

    preprocessed_metadata_uri = UriFactory.create_uri(
        gbml_config_pb.shared_config.preprocessed_metadata_uri
    )
    proto_utils = ProtoUtils()
    proto_utils.write_proto_to_yaml(
        proto=preprocessed_metadata_pb, uri=preprocessed_metadata_uri
    )
    logger.info(
        f"Wrote preprocessed metadata proto to {preprocessed_metadata_uri.uri}."
    )
