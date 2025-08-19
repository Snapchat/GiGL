import gigl.src.common.constants.gcs as gcs_constants
from gigl.common import GcsUri
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.graph_data import EdgeType


def get_applied_task_staging_path(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the GCS URI for the staging/temp path of the applied task.

    Args:
        applied_task_identifeir: The identifier of the applied task.

    Returns:
        GcsUri: The GCS URI for the staging path of the applied task.
    """
    return gcs_constants.get_applied_task_temp_regional_gcs_path(
        applied_task_identifier=applied_task_identifier
    )


def get_edge_dataset_output_path(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the GCS URI for edge data to be written to (to read during training).

    Args:
        applied_task_identifier: The identifier of the applied task.

    Returns:
        GcsUri: The GCS URI for the staging path of the applied task.
    """
    return GcsUri.join(
        get_applied_task_staging_path(applied_task_identifier=applied_task_identifier),
        "edge_dataset",
    )


def get_embedding_output_path(applied_task_identifier: AppliedTaskIdentifier) -> GcsUri:
    """
    Returns the GCS URI for the embeddings to be written to.

    Args:
        applied_task_identifier: The identifier of the applied task.

    Returns:
        GcsUri: The GCS URI for the staging path of the applied task.
    """
    return GcsUri.join(
        get_applied_task_staging_path(applied_task_identifier=applied_task_identifier),
        "embeddings",
    )


def get_embedding_output_path_for_edge_type(
    applied_task_identifier: AppliedTaskIdentifier,
    edge_type: EdgeType,
) -> GcsUri:
    """
    Returns the GCS URI for the embedding output path for a specific edge type.

    Args:
        applied_task_identifeir: The identifier of the applied task.
        edge_type: The edge type for which to get the embedding output path.

    Returns:
        GcsUri: The GCS URI for the embedding output path for the specified edge type.
    """
    return GcsUri.join(
        get_embedding_output_path(applied_task_identifier=applied_task_identifier),
        f"{edge_type.src_node_type}_{edge_type.relation}_{edge_type.dst_node_type}",
    )


def get_embedding_output_path_for_src_node(
    applied_task_identifier: AppliedTaskIdentifier, edge_type: EdgeType
):
    """
    Returns the GCS URI for the embedding output path for a specific source node type.

    Args:
        applied_task_identifeir: The identifier of the applied task.
        edge_type: The edge type for which to get the embedding output path.

    Returns:
        GcsUri: The GCS URI for the embedding output path for the specified source node type.
    """
    return GcsUri.join(
        get_embedding_output_path(applied_task_identifier=applied_task_identifier),
        f"{edge_type.src_node_type}_{edge_type.relation}_{edge_type.dst_node_type}",
        f"src_embeddings",
    )


def get_embedding_output_path_for_dst_node(
    applied_task_identifier: AppliedTaskIdentifier, edge_type: EdgeType
):
    """
    Returns the GCS URI for the embedding output path for a specific source node type.

    Args:
        applied_task_identifeir: The identifier of the applied task.
        edge_type: The edge type for which to get the embedding output path.

    Returns:
        GcsUri: The GCS URI for the embedding output path for the specified source node type.
    """
    return GcsUri.join(
        get_embedding_output_path(applied_task_identifier=applied_task_identifier),
        f"{edge_type.src_node_type}_{edge_type.relation}_{edge_type.dst_node_type}",
        f"dst_embeddings",
    )


def get_enumerated_config_output_path(
    applied_task_identifier: AppliedTaskIdentifier,
) -> GcsUri:
    """
    Returns the GCS URI for the path to the config to be run post enumeration of data.

    Args:
        applied_task_identifeir: The identifier of the applied task.

    Returns:
        GcsUri: The GCS URI for the staging path of the applied task.
    """
    return GcsUri.join(
        get_applied_task_staging_path(applied_task_identifier=applied_task_identifier),
        "post_enumeration_config.yaml",
    )
