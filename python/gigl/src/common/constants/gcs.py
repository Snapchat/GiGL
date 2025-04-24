from typing import Optional, Union

from gigl.common import GcsUri
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.dataset_split import DatasetSplit
from gigl.src.common.types.features import FeatureTypes
from gigl.src.common.types.graph_data import EdgeType, NodeType

_CONFIG_POPULATOR = "config_populator"
_DATA_PREPROCESSOR_PREFIX = "data_preprocess"
_SPLIT_GENERATOR_PREFIX = "split_generator"
_SUBGRAPH_SAMPLER_PREFIX = "subgraph_sampler"
_TRAINER_PREFIX = "trainer"
_INFERENCER_PREFIX = "inferencer"
_POST_PROCESSOR_PREFIX = "post_processor"


def get_applied_task_temp_gcs_path(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the GCS path for the temp_assets bucket for a given gigl job.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.

    Returns:
        GcsUri: The GCS path for the temp assets bucket.
    """
    return GcsUri.join(
        get_resource_config().temp_assets_bucket_path, applied_task_identifier
    )


def get_applied_task_temp_regional_gcs_path(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the GCS path for the temp regional assets for a given gigl job.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.

    Returns:
        GcsUri: The GCS path for the temp regional assets.
    """
    return GcsUri.join(
        get_resource_config().temp_assets_regional_bucket_path, applied_task_identifier
    )


def get_applied_task_perm_gcs_path(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the GCS path for the perm assets bucket for a given gigl job.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.

    Returns:
        GcsUri: The GCS path for the perm assets bucket.
    """
    return GcsUri.join(
        get_resource_config().perm_assets_bucket_path, applied_task_identifier
    )


def get_data_preprocessor_assets_temp_gcs_path(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the GCS path for temporary data preprocessor assets for a given gigl job.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.

    Returns:
        GcsUri: The GCS path for temporary data preprocessor assets.
    """
    return GcsUri.join(
        get_applied_task_temp_regional_gcs_path(
            applied_task_identifier=applied_task_identifier
        ),
        _DATA_PREPROCESSOR_PREFIX,
    )


def get_data_preprocessor_assets_perm_gcs_path(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the GCS path for the data preprocessor perm assets for a given gigl job.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.

    Returns:
        GcsUri: The GCS path for the data preprocessor perm assets.
    """
    return GcsUri.join(
        get_applied_task_perm_gcs_path(applied_task_identifier=applied_task_identifier),
        _DATA_PREPROCESSOR_PREFIX,
    )


def get_data_preprocessor_staging_gcs_path(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the GCS path for the staging directory of the data preprocessor assets for a given gigl job.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.

    Returns:
        GcsUri: The GCS path for the staging directory of the data preprocessor assets.
    """
    return GcsUri.join(
        get_data_preprocessor_assets_temp_gcs_path(
            applied_task_identifier=applied_task_identifier
        ),
        "staging",
    )


def get_tf_transform_directory_path(
    applied_task_identifier: AppliedTaskIdentifier,
    feature_type: FeatureTypes,
    entity_type: Union[NodeType, EdgeType],
    custom_identifier: Optional[str] = "",
) -> GcsUri:
    """
    Returns the GCS path used by Data Preprocessor for TensorFlow Transform (TFT) assets.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.
        feature_type (FeatureTypes): The type of feature.
        entity_type (Union[NodeType, EdgeType]): The type of entity (node or edge).
        custom_identifier (Optional[str]): Custom identifier for the directory path. Defaults to "".

    Returns:
        GcsUri: The GCS path for the directory used by Data Preprocessor for TensorFlow Transform (TFT) assets.
    """
    gcs_preprocess_staging_path: GcsUri = get_data_preprocessor_assets_perm_gcs_path(
        applied_task_identifier=applied_task_identifier,
    )
    return GcsUri.join(
        gcs_preprocess_staging_path,
        feature_type.value,
        str(entity_type),
        str(custom_identifier),
        "tft_transform_dir",
    )


def get_tf_transformed_features_schema_path(
    applied_task_identifier: AppliedTaskIdentifier,
    feature_type: FeatureTypes,
    entity_type: Union[NodeType, EdgeType],
    custom_identifier: str = "",
) -> GcsUri:
    """
    Returns the GCS path for the schema.pbtxt file of the transformed features (Data Preprocessor)

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.
        feature_type (FeatureTypes): The type of the features.
        entity_type (Union[NodeType, EdgeType]): The type of the entity (node or edge).
        custom_identifier (Optional[str]): Custom identifier for the GCS path. Defaults to "".

    Returns:
        GcsUri: The GCS path for the schema.pbtxt file of the transformed features.
    """
    gcs_tf_transform_directory_path = get_tf_transform_directory_path(
        applied_task_identifier=applied_task_identifier,
        feature_type=feature_type,
        entity_type=entity_type,
        custom_identifier=custom_identifier,
    )
    return GcsUri.join(
        gcs_tf_transform_directory_path, "transformed_metadata", "schema.pbtxt"
    )


def get_tf_transformed_features_transform_fn_assets_directory_path(
    applied_task_identifier: AppliedTaskIdentifier,
    feature_type: FeatureTypes,
    entity_type: Union[NodeType, EdgeType],
    custom_identifier: Optional[str] = "",
) -> GcsUri:
    """
    Returns the directory path for the assets of the transformed features' transform_fn (Data Preprocessor).

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.
        feature_type (FeatureTypes): The type of the feature.
        entity_type (Union[NodeType, EdgeType]): The type of the entity.
        custom_identifier (Optional[str]): Custom identifier for the directory path. Defaults to "".

    Returns:
        GcsUri: The GCS path for the directory of the assets of the transformed features' transform_fn.
    """
    gcs_tf_transform_directory_path = get_tf_transform_directory_path(
        applied_task_identifier=applied_task_identifier,
        feature_type=feature_type,
        entity_type=entity_type,
        custom_identifier=custom_identifier,
    )
    return GcsUri.join(gcs_tf_transform_directory_path, "transform_fn", "assets")


def get_tf_transform_temp_directory_path(
    applied_task_identifier: AppliedTaskIdentifier,
    feature_type: FeatureTypes,
    entity_type: Union[NodeType, EdgeType],
    custom_identifier: str = "",
) -> GcsUri:
    """
    Returns the GCS path for the "tft_temp_dir" used by Data Preprocessor for TensorFlow Transform (TFT) temp assets.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.
        feature_type (FeatureTypes): The type of feature.
        entity_type (Union[NodeType, EdgeType]): The type of entity.
        custom_identifier (Optional[str]): Custom identifier for the directory path. Defaults to "".

    Returns:
        GcsUri: The GCS path for the "tft_temp_dir" used by Data Preprocessor for TensorFlow Transform (TFT) temp assets.
    """
    gcs_preprocess_staging_path: GcsUri = get_data_preprocessor_staging_gcs_path(
        applied_task_identifier=applied_task_identifier,
    )
    return GcsUri.join(
        gcs_preprocess_staging_path,
        feature_type.value,
        str(entity_type),
        custom_identifier,
        "tft_temp_dir",
    )


def get_tf_transform_stats_directory_path(
    applied_task_identifier: AppliedTaskIdentifier,
    feature_type: FeatureTypes,
    entity_type: Union[NodeType, EdgeType],
    custom_identifier: str = "",
) -> GcsUri:
    """
    Returns the GCS path for the "stats" directory used by Data Preprocessor for TensorFlow Transform (TFT) assets.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.
        feature_type (FeatureTypes): The type of feature.
        entity_type (Union[NodeType, EdgeType]): The type of entity (node or edge).
        custom_identifier (Optional[str]): Custom identifier for the directory path. Defaults to "".

    Returns:
        GcsUri: The GCS path for the "stats" directory used by Data Preprocessor for TensorFlow Transform (TFT) assets.
    """
    gcs_tf_transform_directory_path = get_tf_transform_directory_path(
        applied_task_identifier=applied_task_identifier,
        feature_type=feature_type,
        entity_type=entity_type,
        custom_identifier=custom_identifier,
    )
    return GcsUri.join(gcs_tf_transform_directory_path, "stats")


def get_tf_transform_visualized_facets_file_path(
    applied_task_identifier: AppliedTaskIdentifier,
    feature_type: FeatureTypes,
    entity_type: Union[NodeType, EdgeType],
    custom_identifier: str = "",
) -> GcsUri:
    """
    Returns the GCS path for the visualized facets overview HTML file.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.
        feature_type (FeatureTypes): The type of feature.
        entity_type (Union[NodeType, EdgeType]): The type of entity (node or edge).
        custom_identifier (Optional[str]): Custom identifier for the file path. Defaults to "".

    Returns:
        GcsUri: The GCS path for the visualized facets overview HTML file.
    """
    stats_dir_path = get_tf_transform_stats_directory_path(
        applied_task_identifier=applied_task_identifier,
        feature_type=feature_type,
        entity_type=entity_type,
        custom_identifier=custom_identifier,
    )
    return GcsUri.join(stats_dir_path, "facets_overview.html")


def get_tf_transform_stats_file_path(
    applied_task_identifier: AppliedTaskIdentifier,
    feature_type: FeatureTypes,
    entity_type: Union[NodeType, EdgeType],
    custom_identifier: str = "",
) -> GcsUri:
    """
    Returns the GCS path for the TensorFlow transform stats file.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.
        feature_type (FeatureTypes): The feature type.
        entity_type (Union[NodeType, EdgeType]): The entity type.
        custom_identifier (Optional[str]): The custom identifier. Defaults to "".

    Returns:
        GcsUri: The GCS path for the TensorFlow transform stats file.
    """
    stats_dir_path = get_tf_transform_stats_directory_path(
        applied_task_identifier=applied_task_identifier,
        feature_type=feature_type,
        entity_type=entity_type,
        custom_identifier=custom_identifier,
    )
    return GcsUri.join(stats_dir_path, "stats.tfrecord")


def get_tf_transform_raw_data_schema_file_path(
    applied_task_identifier: AppliedTaskIdentifier,
    feature_type: FeatureTypes,
    entity_type: Union[NodeType, EdgeType],
    custom_identifier: str = "",
) -> GcsUri:
    """
    Returns the GCS path for the raw data schema file used in TensorFlow Transform.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.
        feature_type (FeatureTypes): The type of feature.
        entity_type (Union[NodeType, EdgeType]): The type of entity (node or edge).
        custom_identifier (Optional[str]): Custom identifier for the file path. Defaults to "".

    Returns:
        GcsUri: The GCS path for the raw data schema file used in TensorFlow Transform.
    """
    tf_transform_dir = get_tf_transform_directory_path(
        applied_task_identifier=applied_task_identifier,
        feature_type=feature_type,
        entity_type=entity_type,
        custom_identifier=custom_identifier,
    )
    return GcsUri.join(tf_transform_dir, "raw_features_schema.pbtxt")


def get_transformed_features_directory_path(
    applied_task_identifier: AppliedTaskIdentifier,
    feature_type: FeatureTypes,
    entity_type: Union[NodeType, EdgeType],
    custom_identifier: str = "",
) -> GcsUri:
    """
    Returns the GCS path for the directory where transformed features are written by Data Preprocessor.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.
        feature_type (FeatureTypes): The type of feature.
        entity_type (Union[NodeType, EdgeType]): The type of entity (node or edge).
        custom_identifier (Optional[str]): Custom identifier for the directory path. Defaults to "".

    Returns:
        GcsUri: The GCS path for the directory of the transformed features.
    """
    gcs_preprocess_staging_path: GcsUri = get_data_preprocessor_staging_gcs_path(
        applied_task_identifier=applied_task_identifier,
    )
    return GcsUri.join(
        gcs_preprocess_staging_path,
        f"transformed_{feature_type.value}_features_dir",
        str(entity_type),
        custom_identifier,
    )


def get_transformed_features_file_prefix(
    applied_task_identifier: AppliedTaskIdentifier,
    feature_type: FeatureTypes,
    entity_type: Union[NodeType, EdgeType],
    custom_identifier: str = "",
) -> GcsUri:
    """
    Returns the GCS file prefix for transformed features.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.
        feature_type (FeatureTypes): The type of the feature.
        entity_type (Union[NodeType, EdgeType]): The type of the entity.
        custom_identifier (Optional[str]): Custom identifier for the file prefix. Defaults to "".

    Returns:
        GcsUri: The GCS file prefix for transformed features.
    """
    transformed_features_path = get_transformed_features_directory_path(
        applied_task_identifier=applied_task_identifier,
        feature_type=feature_type,
        entity_type=entity_type,
        custom_identifier=custom_identifier,
    )
    return GcsUri.join(transformed_features_path, "features/")


def get_preprocessed_metadata_proto_gcs_path(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the GCS path for the generated PreprocessedMetadata yaml file.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.

    Returns:
        GcsUri: The GCS path for the preprocessed metadata proto file.
    """
    return GcsUri.join(
        get_data_preprocessor_assets_perm_gcs_path(
            applied_task_identifier=applied_task_identifier
        ),
        "preprocessed_metadata.yaml",
    )


def get_split_generator_assets_temp_gcs_path(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the temporary GCS path for Split Generator assets.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.

    Returns:
        GcsUri: The temporary GCS path for Split Generator assets.
    """
    return GcsUri.join(
        get_applied_task_temp_gcs_path(applied_task_identifier=applied_task_identifier),
        _SPLIT_GENERATOR_PREFIX,
    )


def get_dataflow_staging_gcs_path(
    applied_task_identifier: AppliedTaskIdentifier,
    job_name: str,
) -> GcsUri:
    """
    Returns the GCS path for the staging directory used for Dataflow Jobs.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.
        job_name (str): The name of the Dataflow job.

    Returns:
        GcsUri: The GCS path for the staging directory used for Dataflow Jobs.
    """
    return GcsUri.join(
        get_applied_task_temp_gcs_path(
            applied_task_identifier=applied_task_identifier,
        ),
        job_name,
        "staging",
    )


def get_dataflow_temp_gcs_path(
    applied_task_identifier: AppliedTaskIdentifier,
    job_name: str,
) -> GcsUri:
    """
    Returns the GCS path for the "tmp" directory used for Dataflow Jobs.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.
        job_name (str): The name of the Dataflow job.

    Returns:
        GcsUri: The GCS path for the "tmp" directory used for Dataflow Jobs.
    """
    return GcsUri.join(
        get_applied_task_temp_gcs_path(
            applied_task_identifier=applied_task_identifier,
        ),
        job_name,
        "tmp",
    )


def get_split_dataset_output_gcs_file_prefix(
    applied_task_identifier: AppliedTaskIdentifier, dataset_split: DatasetSplit
) -> GcsUri:
    """
    Returns the GCS file prefix for the samples output by Split Generator.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.
        dataset_split (DatasetSplit): The dataset split.

    Returns:
        GcsUri: The GCS file prefix for the samples output by Split Generator.
    """
    return GcsUri.join(
        get_split_generator_assets_temp_gcs_path(
            applied_task_identifier=applied_task_identifier
        ),
        dataset_split.value,
        "samples/",
    )


def get_subgraph_sampler_root_dir(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the GCS path which Subgraph Sampler uses to store temp assets.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.

    Returns:
        GcsUri: The GCS path which Subgraph Sampler uses to store temp assets.
    """
    return GcsUri.join(
        get_applied_task_temp_gcs_path(applied_task_identifier=applied_task_identifier),
        "subgraph_sampler",
    )


def get_subgraph_sampler_supervised_node_classification_task_dir(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the GCS path which Subgraph Sampler uses to store temp assets for supervised node classification.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.

    Returns:
        GcsUri: The GCS path which Subgraph Sampler uses to store temp assets for supervised node classification.
    """
    return GcsUri.join(
        get_subgraph_sampler_root_dir(applied_task_identifier=applied_task_identifier),
        "supervised_node_classification",
    )


def get_subgraph_sampler_node_anchor_based_link_prediction_task_dir(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the GCS path which Subgraph Sampler uses to store temp assets for node anchor based link prediction.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.

    Returns:
        GcsUri: The GCS path which Subgraph Sampler uses to store temp assets for node anchor based link prediction.
    """
    return GcsUri.join(
        get_subgraph_sampler_root_dir(applied_task_identifier=applied_task_identifier),
        "node_anchor_based_link_prediction",
    )


def get_subgraph_sampler_supervised_link_based_task_dir(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the GCS path which Subgraph Sampler uses to store temp assets for supervised link based tasks.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.

    Returns:
        GcsUri: The GCS path which Subgraph Sampler uses to store temp assets for supervised link based tasks.
    """
    return GcsUri.join(
        get_subgraph_sampler_root_dir(applied_task_identifier=applied_task_identifier),
        "supervised_link_based",
    )


def get_subgraph_sampler_node_neighborhood_samples_dir(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the GCS path which Subgraph Sampler uses to store node neighborhood samples.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.

    Returns:
        GcsUri: The GCS path which Subgraph Sampler uses to store node neighborhood samples.
    """
    return GcsUri.join(
        get_subgraph_sampler_root_dir(applied_task_identifier=applied_task_identifier),
        "node_neighborhood_samples",
    )


def get_subgraph_sampler_supervised_node_classification_labeled_samples_prefix(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the GCS file prefix for labeled samples output by Subgraph Sampler for supervised node classification.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.

    Returns:
        GcsUri: The GCS file prefix for labeled samples output by Subgraph Sampler for supervised node classification.
    """
    return GcsUri.join(
        get_subgraph_sampler_supervised_node_classification_task_dir(
            applied_task_identifier=applied_task_identifier
        ),
        "labeled",
        "samples/",
    )


def get_subgraph_sampler_supervised_node_classification_unlabeled_samples_prefix(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the GCS file prefix for unlabeled samples output by Subgraph Sampler for supervised node classification.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.

    Returns:
        GcsUri: The GCS file prefix for unlabeled samples output by Subgraph Sampler for supervised node classification.
    """
    return GcsUri.join(
        get_subgraph_sampler_supervised_node_classification_task_dir(
            applied_task_identifier=applied_task_identifier
        ),
        "unlabeled",
        "samples/",
    )


def get_subgraph_sampler_supervised_link_based_task_labeled_samples_prefix(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the GCS file prefix for labeled samples output by Subgraph Sampler for supervised link based tasks.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.

    Returns:
        GcsUri: The GCS file prefix for labeled samples output by Subgraph Sampler for supervised link based tasks.
    """
    return GcsUri.join(
        get_subgraph_sampler_supervised_link_based_task_dir(
            applied_task_identifier=applied_task_identifier
        ),
        "labeled",
        "samples/",
    )


def get_subgraph_sampler_supervised_link_based_task_unlabeled_samples_prefix(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the GCS file prefix for unlabeled samples output by Subgraph Sampler for supervised link based tasks.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.

    Returns:
        GcsUri: The GCS file prefix for unlabeled samples output by Subgraph Sampler for supervised link based tasks.
    """
    return GcsUri.join(
        get_subgraph_sampler_supervised_link_based_task_dir(
            applied_task_identifier=applied_task_identifier
        ),
        "unlabeled",
        "samples/",
    )


def get_subgraph_sampler_node_neighborhood_samples_path_prefix(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the GCS file prefix for node neighborhood samples output by Subgraph Sampler.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.

    Returns:
        GcsUri: The GCS file prefix for node neighborhood samples output by Subgraph Sampler.
    """
    return GcsUri.join(
        get_subgraph_sampler_node_neighborhood_samples_dir(
            applied_task_identifier=applied_task_identifier
        ),
        "samples/",
    )


def get_subgraph_sampler_node_anchor_based_link_prediction_samples_prefix(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the GCS file prefix for samples output by Subgraph Sampler for node anchor based link prediction.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.

    Returns:
        GcsUri: The GCS file prefix for samples output by Subgraph Sampler for node anchor based link prediction.
    """
    return GcsUri.join(
        get_subgraph_sampler_node_anchor_based_link_prediction_task_dir(
            applied_task_identifier=applied_task_identifier
        ),
        "node_anchor_based_link_prediction_samples",
        "samples/",
    )


def get_subgraph_sampler_node_anchor_based_link_prediction_random_negatives_samples_prefix(
    applied_task_identifier: AppliedTaskIdentifier,
    node_type: NodeType,
) -> GcsUri:
    """
    Returns the GCS file prefix for random negative samples output by Subgraph Sampler for node anchor based link prediction.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.

    Returns:
        GcsUri: The GCS file prefix for random negative samples output by Subgraph Sampler for node anchor based link prediction.
    """
    return GcsUri.join(
        get_subgraph_sampler_node_anchor_based_link_prediction_task_dir(
            applied_task_identifier=applied_task_identifier
        ),
        "random_negative_rooted_neighborhood_samples",
        node_type,
        "samples/",
    )


def get_subgraph_sampler_flattened_graph_metadata_output_gcs_path(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the GCS path for the flattened graph metadata yaml output by Subgraph Sampler.
    See: proto/snapchat/research/gbml/flattened_graph_metadata.proto for more details.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.

    Returns:
        GcsUri: The GCS path for the flattened graph metadata yaml output by Subgraph Sampler.
    """
    return GcsUri.join(
        get_subgraph_sampler_root_dir(applied_task_identifier=applied_task_identifier),
        "flattened_graph_metadata.yaml",
    )


def get_split_dataset_main_samples_gcs_file_prefix(
    applied_task_identifier: AppliedTaskIdentifier, dataset_split: DatasetSplit
) -> GcsUri:
    """
    Returns the GCS file prefix for the main samples output by Split Generator.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.
        dataset_split (DatasetSplit): The dataset split.

    Returns:
        GcsUri: The GCS file prefix for the main samples output by Split Generator.
    """
    return GcsUri.join(
        get_split_generator_assets_temp_gcs_path(
            applied_task_identifier=applied_task_identifier
        ),
        dataset_split.value,
        "main_samples",
        "samples/",
    )


def get_split_dataset_random_negatives_gcs_file_prefix(
    applied_task_identifier: AppliedTaskIdentifier,
    node_type: NodeType,
    dataset_split: DatasetSplit,
) -> GcsUri:
    """
    Returns the GCS file prefix for the random negative samples output by Split Generator.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.
        dataset_split (DatasetSplit): The dataset split.

    Returns:
        GcsUri: The GCS file prefix for the random negative samples output by Split Generator.
    """
    return GcsUri.join(
        get_split_generator_assets_temp_gcs_path(
            applied_task_identifier=applied_task_identifier
        ),
        dataset_split.value,
        "random_negatives",
        node_type,
        "neighborhoods/",
    )


def get_config_populator_assets_perm_gcs_path(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the GCS path for the config populator perm assets for a given gigl job (Used to write Frozen GBML Config).

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.

    Returns:
        GcsUri: The GCS path for the config populator perm assets.
    """
    return GcsUri.join(
        get_applied_task_perm_gcs_path(applied_task_identifier=applied_task_identifier),
        _CONFIG_POPULATOR,
    )


def get_frozen_gbml_config_proto_gcs_path(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the GCS path for the frozen GBML config proto file.
    See: proto/snapchat/research/gbml/gbml_config.proto for more details.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.

    Returns:
        GcsUri: The GCS path for the frozen GBML config proto file.
    """
    return GcsUri.join(
        get_config_populator_assets_perm_gcs_path(
            applied_task_identifier=applied_task_identifier
        ),
        "frozen_gbml_config.yaml",
    )


def get_trainer_asset_dir_gcs_path(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the GCS path for perm assets written by the Trainer (e.g. trained models, eval metrics, etc.)

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.

    Returns:
        GcsUri: The GCS path for perm assets written by the Trainer.
    """
    return GcsUri.join(
        get_applied_task_perm_gcs_path(applied_task_identifier=applied_task_identifier),
        _TRAINER_PREFIX,
    )


def get_trained_models_dir_gcs_path(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the GCS path for the trained models directory.
    """
    return GcsUri.join(
        get_trainer_asset_dir_gcs_path(applied_task_identifier=applied_task_identifier),
        "models",
    )


def get_trained_model_gcs_path(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the GCS path for the trained model output by the Trainer (model.pt)

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.

    Returns:
        GcsUri: The GCS path for the trained model output by the Trainer.
    """
    return GcsUri.join(
        get_trained_models_dir_gcs_path(
            applied_task_identifier=applied_task_identifier
        ),
        "model.pt",
    )


def get_trained_scripted_model_gcs_path(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the GCS path for the scripted model output by the Trainer (scripted_model.pt)

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.

    Returns:
        GcsUri: The GCS path for the scripted model output by the Trainer.
    """
    return GcsUri.join(
        get_trained_models_dir_gcs_path(
            applied_task_identifier=applied_task_identifier
        ),
        "scripted_model.pt",
    )


def get_trained_model_eval_metrics_gcs_path(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the GCS path for the eval metrics output by the Trainer (eval_metrics.json)

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.

    Returns:
        GcsUri: The GCS path for the eval metrics output by the Trainer.
    """
    return GcsUri.join(
        get_trained_models_dir_gcs_path(
            applied_task_identifier=applied_task_identifier
        ),
        "trainer_eval_metrics.json",
    )


def get_trained_model_metadata_proto_gcs_path(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the trained model metadata yaml file outputted by the Trainer.
    See: proto/snapchat/research/gbml/trained_model_metadata.proto for more details.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.

    Returns:
        GcsUri: The GCS path for the trained model metadata yaml file outputted by the Trainer.
    """
    return GcsUri.join(
        get_trainer_asset_dir_gcs_path(applied_task_identifier=applied_task_identifier),
        "trained_model_metadata.yaml",
    )


def get_tensorboard_logs_gcs_path(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the GCS path that is used to store tensorboard logs.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.

    Returns:
        GcsUri: The GCS path that is used to store tensorboard logs.
    """
    return GcsUri.join(
        get_trainer_asset_dir_gcs_path(applied_task_identifier=applied_task_identifier),
        "tensorboard_logs/",
    )


def get_inferencer_asset_dir_gcs_path(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the GCS path for perm assets written by the Inferencer (e.g. embeddings, predictions, etc.)

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.

    Returns:
        GcsUri: The GCS path for perm assets written by the Inferencer.
    """
    return GcsUri.join(
        get_applied_task_perm_gcs_path(applied_task_identifier=applied_task_identifier),
        _INFERENCER_PREFIX,
    )


def get_inferencer_embeddings_gcs_prefix(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the GCS directory for embeddings output by the Inferencer.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.

    Returns:
        GcsUri: The GCS directory for embeddings output by the Inferencer.
    """
    return GcsUri.join(
        get_inferencer_asset_dir_gcs_path(
            applied_task_identifier=applied_task_identifier
        ),
        "embeddings/",
    )


def get_inferencer_predictions_gcs_prefix(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the GCS directory for predictions output by the Inferencer.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.

    Returns:
        GcsUri: The GCS directory for predictions output by the Inferencer.
    """
    return GcsUri.join(
        get_inferencer_asset_dir_gcs_path(
            applied_task_identifier=applied_task_identifier
        ),
        "predictions/",
    )


def get_post_processor_asset_dir_gcs_path(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the GCS path for perm assets written by the Post Processor (e.g. eval metrics, etc.)
    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.
    Returns:
        GcsUri: The GCS path for perm assets written by the Post Processor.
    """
    return GcsUri.join(
        get_applied_task_perm_gcs_path(applied_task_identifier=applied_task_identifier),
        _POST_PROCESSOR_PREFIX,
    )


def get_post_processor_metrics_gcs_path(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the GCS path for the eval metrics output by the Post Processor (post_processor_metrics.json)
    Args:
        applied_task_identifier (AppliedTaskIdentifier): The job name.
    Returns:
        GcsUri: The GCS path for the eval metrics output by the Post Processor.
    """
    return GcsUri.join(
        get_post_processor_asset_dir_gcs_path(
            applied_task_identifier=applied_task_identifier
        ),
        "post_processor_metrics.json",
    )
