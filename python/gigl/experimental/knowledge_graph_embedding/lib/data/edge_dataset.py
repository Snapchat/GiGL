from enum import Enum
from typing import Dict, List

import gigl.experimental.knowledge_graph_embedding.lib.constants.gcs as gcs_constants
import torch.distributed as dist
from gigl.experimental.knowledge_graph_embedding.common.graph_dataset import (
    CONDENSED_EDGE_TYPE_FIELD,
    DST_FIELD,
    SRC_FIELD,
    BigQueryHeterogeneousGraphIterableDataset,
    GcsJSONLHeterogeneousGraphIterableDataset,
    GcsParquetHeterogeneousGraphIterableDataset,
)
from torch.utils.data import IterableDataset

from gigl.common.logger import Logger
from gigl.common.types.uri.gcs_uri import GcsUri
from gigl.common.utils.gcs import GcsUtils
from gigl.common.utils.torch_training import is_distributed_available_and_initialized
from gigl.distributed.dist_context import DistributedContext
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.dataset_split import DatasetSplit
from gigl.src.common.types.graph_data import EdgeType
from gigl.src.common.types.pb_wrappers.graph_metadata import GraphMetadataPbWrapper
from gigl.src.common.utils.bq import BqUtils
from gigl.src.data_preprocessor.lib.enumerate.utils import EnumeratorEdgeTypeMetadata
from gigl.src.data_preprocessor.lib.ingest.bigquery import BigqueryEdgeDataReference

logger = Logger()


def _build_intermediate_edges_table(
    enumerated_edge_metadata: List[EnumeratorEdgeTypeMetadata],
    applied_task_identifier: AppliedTaskIdentifier,
    output_bq_dataset: str,
    graph_metadata: GraphMetadataPbWrapper,
    bq_utils: BqUtils,
    split_columns: List[str] = list(),
    train_split_clause: str = "rand_split BETWEEN 0 AND 0.8",
    val_split_clause: str = "rand_split BETWEEN 0.8 AND 0.9",
    test_split_clause: str = "rand_split BETWEEN 0.9 AND 1",
) -> str:
    """
    Build an intermediate edges table by unioning multiple edge tables with split metadata.

    This function creates a BigQuery table that combines all edge tables from the enumerated
    edge metadata into a single intermediate table. Each edge is mapped to a condensed edge
    type and includes split information (either from provided split columns or random splits).

    Args:
        enumerated_edge_metadata: List of metadata objects containing edge table references
            and identifiers for source and destination nodes.
        applied_task_identifier: Unique identifier for the current applied task, used in
            the intermediate table name.
        output_bq_dataset: BigQuery dataset where the intermediate table will be created.
        graph_metadata: Wrapper containing graph metadata including edge type mappings.
        bq_utils: BigQuery utilities instance for executing queries.
        split_columns: Optional list of column names to use for data splitting. If empty,
            a random split column will be generated.
        train_split_clause: SQL WHERE clause defining the training split condition.
        val_split_clause: SQL WHERE clause defining the validation split condition.
        test_split_clause: SQL WHERE clause defining the test split condition.

    Returns:
        str: The fully qualified BigQuery table path of the created intermediate edges table.
    """
    # Create an intermediate edges table with some split-related metadata.
    has_split_columns = len(split_columns) > 0
    split_column_selector = (
        ", ".join(split_columns) if has_split_columns else "RAND() AS rand_split"
    )
    if has_split_columns:
        logger.info(f"Using split columns: {split_columns}")
    else:
        logger.info("No split columns provided. Using random transductive split.")

    logger.info(
        f"Using train/val/test clauses: '{train_split_clause}', '{val_split_clause}', '{test_split_clause}'"
    )

    edge_table_queries: List[str] = list()
    for edge_metadata in enumerated_edge_metadata:
        enumerated_reference = edge_metadata.enumerated_edge_data_reference
        edge_table = BqUtils.format_bq_path(bq_path=enumerated_reference.reference_uri)
        condensed_edge_type = graph_metadata.edge_type_to_condensed_edge_type_map[
            enumerated_reference.edge_type
        ]
        edge_table_query = f"""
            SELECT
                {enumerated_reference.src_identifier} AS {SRC_FIELD},
                {enumerated_reference.dst_identifier} AS {DST_FIELD},
                {condensed_edge_type} AS {CONDENSED_EDGE_TYPE_FIELD},
                {split_column_selector}
            FROM
                `{edge_table}`
        """
        edge_table_queries.append(edge_table_query)

    union_edges_query = " UNION ALL ".join(edge_table_queries)
    logger.info(f"Will write train/val/test datasets to BQ dataset {output_bq_dataset}")
    intermediate_edges_table = BqUtils.join_path(
        BqUtils.format_bq_path(output_bq_dataset),
        f"intermediate_{applied_task_identifier}",
    )
    bq_utils.run_query(
        query=union_edges_query,
        destination=intermediate_edges_table,
        write_disposition="WRITE_TRUNCATE",
        labels={},
    )

    return intermediate_edges_table


class EdgeDatasetFormat(str, Enum):
    """
    Enumeration of supported edge dataset output formats.

    This enum defines the different formats in which edge datasets can be stored
    and accessed. Each format has different performance characteristics and use cases:

    - GCS_JSONL: Stores data as JSONL (JSON Lines) files in Google Cloud Storage.
      Good for debugging and human-readable data inspection.
    - GCS_PARQUET: Stores data as Parquet files in Google Cloud Storage.
      Optimized for analytical workloads with efficient compression and columnar storage.
    - BIGQUERY: Keeps data in BigQuery tables for direct querying.
      Best for large-scale datasets that benefit from BigQuery's distributed processing.
    """
    GCS_JSONL = "JSONL"
    GCS_PARQUET = "PARQUET"
    BIGQUERY = "BIGQUERY"


def build_edge_datasets(
    distributed_context: DistributedContext,
    enumerated_edge_metadata: List[EnumeratorEdgeTypeMetadata],
    applied_task_identifier: AppliedTaskIdentifier,
    output_bq_dataset: str,
    graph_metadata: GraphMetadataPbWrapper,
    split_columns: List[str] = list(),
    train_split_clause: str = "rand_split BETWEEN 0 AND 0.8",
    val_split_clause: str = "rand_split BETWEEN 0.8 AND 0.9",
    test_split_clause: str = "rand_split BETWEEN 0.9 AND 1",
    format: EdgeDatasetFormat = EdgeDatasetFormat.GCS_PARQUET,
) -> Dict[DatasetSplit, IterableDataset]:
    """
    Build edge datasets for training, validation, and testing.  This function
    reads edge data from BigQuery, filters it based on the provided split clauses,
    and writes the filtered data to either BigQuery or GCS in the specified format.

    This function is designed to work in a distributed environment, where
    multiple processes may be running in parallel. It ensures that the resources
    are created only once and that all processes wait for each other to finish
    before proceeding.  It uses PyTorch's distributed package to manage the
    distributed context.  It also handles the initialization and destruction of
    the distributed process group if necessary.

    Args:
        distributed_context: The distributed context for the current process.
        enumerated_edge_metadata: Metadata for the edges to be processed.
        applied_task_identifier: Identifier for the applied task.
        output_bq_dataset: BigQuery dataset to write the output to.
        graph_metadata: Metadata for the graph.
        split_columns: List of columns to use for splitting the data.
        train_split_clause: SQL clause for training data split.
        val_split_clause: SQL clause for validation data split.
        test_split_clause: SQL clause for testing data split.
        format: Format of the output datasets (GCS or BigQuery).
        project: GCP project ID.
    """

    # Only init torch distributed if not already initialized
    we_initialized = False
    if not is_distributed_available_and_initialized():
        logger.info(
            f"Building edge datasets -- Initializing torch distributed for {distributed_context.global_rank}..."
        )
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            world_size=distributed_context.global_world_size,
            rank=distributed_context.global_rank,
            init_method=f"tcp://{distributed_context.main_worker_ip_address}:23456",
        )
        logger.info(
            f"Using backend: {dist.get_backend()} for distributed dataset building."
        )
        we_initialized = True

    bq_utils = BqUtils(project=get_resource_config().project)
    gcs_utils = GcsUtils(project=get_resource_config().project)

    MIXED_EDGE_TYPE = EdgeType("mixed", "mixed", "mixed")
    heterogeneous_kwargs = {
        "src_field": SRC_FIELD,
        "dst_field": DST_FIELD,
        "condensed_edge_type_field": CONDENSED_EDGE_TYPE_FIELD,
    }

    split_info = [
        (DatasetSplit.TRAIN, train_split_clause),
        (DatasetSplit.VAL, val_split_clause),
        (DatasetSplit.TEST, test_split_clause),
    ]

    def create_resources() -> None:
        """
        Create the required resources for edge datasets.

        This nested function handles the creation of all necessary resources for the edge
        datasets. It first builds an intermediate edges table by combining all edge tables,
        then creates separate train/validation/test tables by filtering the intermediate
        table according to the provided split clauses. If the output format is GCS-based
        (JSONL or PARQUET), it also exports the BigQuery tables to GCS.

        The function creates:
        1. An intermediate edges table containing all edges with split metadata
        2. Separate BigQuery tables for train/validation/test splits
        3. GCS exports of the split tables (if format is GCS_JSONL or GCS_PARQUET)
        """

        intermediate_edges_table = _build_intermediate_edges_table(
            enumerated_edge_metadata=enumerated_edge_metadata,
            applied_task_identifier=applied_task_identifier,
            output_bq_dataset=output_bq_dataset,
            graph_metadata=graph_metadata,
            bq_utils=bq_utils,
            split_columns=split_columns,
            train_split_clause=train_split_clause,
            val_split_clause=val_split_clause,
            test_split_clause=test_split_clause,
        )
        for split, split_clause in split_info:
            table_reference = BigqueryEdgeDataReference(
                reference_uri=BqUtils.join_path(
                    BqUtils.format_bq_path(output_bq_dataset),
                    f"{split.value}_edges_{applied_task_identifier}",
                ),
                src_identifier=SRC_FIELD,
                dst_identifier=DST_FIELD,
                edge_type=MIXED_EDGE_TYPE,
            )
            random_column_field = "row_id"
            maybe_extra_field_selector = (
                f", RAND() as {random_column_field}"
                if format == EdgeDatasetFormat.BIGQUERY
                else ""
            )
            query = f"SELECT * {maybe_extra_field_selector} FROM `{intermediate_edges_table}` WHERE {split_clause} ORDER BY RAND()"

            bq_utils.run_query(
                query=query,
                destination=table_reference.reference_uri,
                write_disposition="WRITE_TRUNCATE",
                labels=dict(),
            )
            if format in (EdgeDatasetFormat.GCS_JSONL, EdgeDatasetFormat.GCS_PARQUET):
                gcs_target_path = GcsUri.join(
                    gcs_constants.get_edge_dataset_output_path(
                        applied_task_identifier=applied_task_identifier,
                    ),
                    f"{split.value}_edges",
                )
                destination_glob_path = GcsUri.join(gcs_target_path, "shard-*")
                bq_utils.export_to_gcs(
                    bq_table_path=table_reference.reference_uri,
                    destination_gcs_uri=destination_glob_path,
                    destination_format="NEWLINE_DELIMITED_JSON"
                    if format == EdgeDatasetFormat.GCS_JSONL
                    else "PARQUET",
                )

    def instantiate_datasets() -> Dict[DatasetSplit, IterableDataset]:
        """
        Instantiate and return the edge datasets for each data split.

        This nested function creates IterableDataset instances for train, validation,
        and test splits. The type of dataset created depends on the specified format:
        - BIGQUERY: Creates BigQueryHeterogeneousGraphIterableDataset instances that
          read directly from BigQuery tables
        - GCS_JSONL/GCS_PARQUET: Creates GcsJSONLHeterogeneousGraphIterableDataset or
          GcsParquetHeterogeneousGraphIterableDataset instances that read from GCS files

        For GCS-based datasets, the function lists all shard files at the expected
        GCS path and passes them to the dataset constructor.

        Returns:
            Dict[DatasetSplit, IterableDataset]: A dictionary mapping each data split
            (TRAIN, VAL, TEST) to its corresponding IterableDataset instance.
        """

        datasets: dict = dict()
        for split, _ in split_info:
            table_reference = BigqueryEdgeDataReference(
                reference_uri=BqUtils.join_path(
                    BqUtils.format_bq_path(output_bq_dataset),
                    f"{split.value}_edges_{applied_task_identifier}",
                ),
                src_identifier=SRC_FIELD,
                dst_identifier=DST_FIELD,
                edge_type=MIXED_EDGE_TYPE,
            )
            random_column_field = "row_id"
            if format == EdgeDatasetFormat.BIGQUERY:
                datasets[split] = BigQueryHeterogeneousGraphIterableDataset(
                    table=table_reference.reference_uri,
                    random_column=random_column_field,
                    project=get_resource_config().project,
                    **heterogeneous_kwargs,
                )
            elif format in (EdgeDatasetFormat.GCS_JSONL, EdgeDatasetFormat.GCS_PARQUET):
                gcs_target_path = GcsUri.join(
                    gcs_constants.get_edge_dataset_output_path(
                        applied_task_identifier=applied_task_identifier,
                    ),
                    f"{split.value}_edges",
                )
                files_at_glob_path = gcs_utils.list_uris_with_gcs_path_pattern(
                    gcs_path=gcs_target_path, pattern=".*shard-\d+"
                )
                dataset_cls = {
                    EdgeDatasetFormat.GCS_JSONL: GcsJSONLHeterogeneousGraphIterableDataset,
                    EdgeDatasetFormat.GCS_PARQUET: GcsParquetHeterogeneousGraphIterableDataset,
                }[format]
                datasets[split] = dataset_cls(
                    file_uris=files_at_glob_path, **heterogeneous_kwargs
                )
        return datasets

    # Rank 0 will create the resources, and all ranks will wait for it to finish.
    # This is to ensure that resource creation doesn't happen across multiple ranks,
    # since this will create redundant resources and potentially cause issues.
    if distributed_context.global_rank == 0:
        create_resources()
    dist.barrier()  # Ensure all ranks have created the resources
    datasets = instantiate_datasets()
    if we_initialized:
        logger.info(
            f"Finished building edge datasets -- tearing down torch distributed for {distributed_context.global_rank}..."
        )
        dist.destroy_process_group()

    return datasets
