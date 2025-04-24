from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.graph_data import NodeType
from gigl.src.common.utils.bq import BqUtils


def get_embeddings_dataset_bq_path() -> str:
    """
    Returns the path to where the table for embeddings will be stored in BQ, specified in GiGLResourceConfig
    """
    project = get_resource_config().project
    dataset = get_resource_config().embedding_bq_dataset_name
    bq_dataset_path = BqUtils.join_path(project, dataset)
    return bq_dataset_path


def get_embeddings_table(
    applied_task_identifier: AppliedTaskIdentifier, node_type: NodeType
) -> str:
    """
    Returns the full BQ table path where embeddings will be stored

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The name provided for the gigl job

    """
    embeddings_table_name = f"embeddings_{node_type}_{applied_task_identifier}"
    bq_table_path = BqUtils.join_path(
        get_embeddings_dataset_bq_path(), embeddings_table_name
    )
    return bq_table_path


def get_predictions_table(
    applied_task_identifier: AppliedTaskIdentifier, node_type: NodeType
) -> str:
    """
    This function return the BQ table path where predictions will be stored

    Args:
        applied_task_identifier (AppliedTaskIdentifier): The name provided for the gigl job
    """
    predictions_table_name = f"predictions_{node_type}_{applied_task_identifier}"
    bq_table_path = BqUtils.join_path(
        get_embeddings_dataset_bq_path(), predictions_table_name
    )
    return bq_table_path
