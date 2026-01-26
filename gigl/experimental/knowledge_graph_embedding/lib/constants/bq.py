import gigl.src.common.constants.bq as bq_constants
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.graph_data import EdgeType
from gigl.src.common.utils.bq import BqUtils


def get_src_node_embedding_table_for_edge_type(
    applied_task_identifier: AppliedTaskIdentifier,
    edge_type: EdgeType,
    is_enumerated: bool = False,
):
    """
    Returns the BigQuery table for the embeddings of a specific source node type.

    Args:
        applied_task_identifier: The identifier of the applied task.
        edge_type: The edge type for which to get the embedding table.
        is_enumerated: Whether the embeddings are enumerated (default is False).

    Returns:
        str: The BigQuery table name for the embeddings of the specified source node type.
    """
    bq_table_path = BqUtils.join_path(
        bq_constants.get_embeddings_dataset_bq_path(),
        f"{applied_task_identifier}_{edge_type.src_node_type}_{edge_type.relation}_{edge_type.dst_node_type}_src_embeddings{'_enumerated' if is_enumerated else ''}",
    )
    return bq_table_path


def get_dst_node_embedding_table_for_edge_type(
    applied_task_identifier: AppliedTaskIdentifier,
    edge_type: EdgeType,
    is_enumerated: bool = False,
):
    """
    Returns the BigQuery table for the embeddings of a specific destination node type.

    Args:
        applied_task_identifier: The identifier of the applied task.
        edge_type: The edge type for which to get the embedding table.
        is_enumerated: Whether the embeddings are enumerated (default is False).

    Returns:
        str: The BigQuery table name for the embeddings of the specified source node type.
    """
    bq_table_path = BqUtils.join_path(
        bq_constants.get_embeddings_dataset_bq_path(),
        f"{applied_task_identifier}_{edge_type.src_node_type}_{edge_type.relation}_{edge_type.dst_node_type}_dst_embeddings{'_enumerated' if is_enumerated else ''}",
    )
    return bq_table_path
