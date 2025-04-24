from dataclasses import dataclass
from typing import List, Optional, Union

import gigl.src.common.constants.gcs as gcs_constants
from gigl.common import GcsUri, HttpUri
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.features import FeatureTypes
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.src.data_preprocessor.lib.types import (
    EdgeOutputIdentifier,
    NodeOutputIdentifier,
)


@dataclass
class TransformedFeaturesInfo:
    # TODO: (svij-sc): refactor to have a EdgeTransformedFeaturesInfo and a NodeTransformedFeaturesInfo
    feature_type: FeatureTypes
    entity_type: Union[NodeType, EdgeType]
    visualized_facets_file_path: GcsUri
    stats_file_path: GcsUri
    raw_data_schema_file_path: GcsUri
    tft_temp_directory_path: GcsUri
    transformed_features_file_prefix: GcsUri
    transformed_features_schema_path: GcsUri
    transform_directory_path: GcsUri
    dataflow_console_uri: Optional[HttpUri] = None
    identifier_output: Optional[
        Union[NodeOutputIdentifier, EdgeOutputIdentifier]
    ] = None
    features_outputs: Optional[List[str]] = None
    label_outputs: Optional[List[str]] = None
    feature_dim_output: Optional[int] = None
    custom_identifier: Optional[str] = None

    def __init__(
        self,
        applied_task_identifier: AppliedTaskIdentifier,
        feature_type: FeatureTypes,
        entity_type: Union[NodeType, EdgeType],
        custom_identifier: str = "",
    ) -> None:
        self.feature_type = feature_type
        self.entity_type = entity_type

        self.transform_directory_path = gcs_constants.get_tf_transform_directory_path(
            applied_task_identifier=applied_task_identifier,
            feature_type=feature_type,
            entity_type=entity_type,
            custom_identifier=custom_identifier,
        )

        self.visualized_facets_file_path = (
            gcs_constants.get_tf_transform_visualized_facets_file_path(
                applied_task_identifier=applied_task_identifier,
                feature_type=feature_type,
                entity_type=entity_type,
                custom_identifier=custom_identifier,
            )
        )

        self.stats_file_path = gcs_constants.get_tf_transform_stats_file_path(
            applied_task_identifier=applied_task_identifier,
            feature_type=feature_type,
            entity_type=entity_type,
            custom_identifier=custom_identifier,
        )

        self.raw_data_schema_file_path = (
            gcs_constants.get_tf_transform_raw_data_schema_file_path(
                applied_task_identifier=applied_task_identifier,
                feature_type=feature_type,
                entity_type=entity_type,
                custom_identifier=custom_identifier,
            )
        )

        self.tft_temp_directory_path = (
            gcs_constants.get_tf_transform_temp_directory_path(
                applied_task_identifier=applied_task_identifier,
                feature_type=feature_type,
                entity_type=entity_type,
                custom_identifier=custom_identifier,
            )
        )

        self.transformed_features_file_prefix = (
            gcs_constants.get_transformed_features_file_prefix(
                applied_task_identifier=applied_task_identifier,
                feature_type=feature_type,
                entity_type=entity_type,
                custom_identifier=custom_identifier,
            )
        )

        self.transformed_features_schema_path = (
            gcs_constants.get_tf_transformed_features_schema_path(
                applied_task_identifier=applied_task_identifier,
                feature_type=feature_type,
                entity_type=entity_type,
                custom_identifier=custom_identifier,
            )
        )
        self.transformed_features_transform_fn_assets_path = gcs_constants.get_tf_transformed_features_transform_fn_assets_directory_path(
            applied_task_identifier=applied_task_identifier,
            feature_type=feature_type,
            entity_type=entity_type,
            custom_identifier=custom_identifier,
        )
