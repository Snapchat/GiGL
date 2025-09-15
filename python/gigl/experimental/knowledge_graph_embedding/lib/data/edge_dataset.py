from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Protocol, Tuple

import torch.distributed as dist
from torch.utils.data import IterableDataset

import gigl.experimental.knowledge_graph_embedding.lib.constants.gcs as gcs_constants
from gigl.common.logger import Logger
from gigl.common.types.uri.gcs_uri import GcsUri
from gigl.common.utils.gcs import GcsUtils
from gigl.common.utils.torch_training import is_distributed_available_and_initialized
from gigl.distributed.dist_context import DistributedContext
from gigl.env.pipelines_config import get_resource_config
from gigl.experimental.knowledge_graph_embedding.common.graph_dataset import (
    CONDENSED_EDGE_TYPE_FIELD,
    DST_FIELD,
    SRC_FIELD,
    BigQueryHeterogeneousGraphIterableDataset,
    GcsJSONLHeterogeneousGraphIterableDataset,
    GcsParquetHeterogeneousGraphIterableDataset,
)
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.dataset_split import DatasetSplit
from gigl.src.common.types.graph_data import EdgeType
from gigl.src.common.types.pb_wrappers.graph_metadata import GraphMetadataPbWrapper
from gigl.src.common.utils.bq import BqUtils
from gigl.src.data_preprocessor.lib.enumerate.utils import EnumeratorEdgeTypeMetadata
from gigl.src.data_preprocessor.lib.ingest.bigquery import BigqueryEdgeDataReference

logger = Logger()


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


@dataclass
class PerSplitFilteredEdgeBigqueryMetadata:
    """Configuration parameters to filter BigQuery tables by split (train/val/test)."""

    split_columns: List[str]
    train_split_clause: str
    val_split_clause: str
    test_split_clause: str

    def clause_per_split(self) -> List[Tuple[DatasetSplit, str]]:
        return [
            (DatasetSplit.TRAIN, self.train_split_clause),
            (DatasetSplit.VAL, self.val_split_clause),
            (DatasetSplit.TEST, self.test_split_clause),
        ]


@dataclass
class PerSplitFilteredEdgeDatasetConfig:
    """Configuration parameters to build filtered datasets by split (train/val/test)."""

    distributed_context: DistributedContext
    enumerated_edge_metadata: List[EnumeratorEdgeTypeMetadata]
    applied_task_identifier: AppliedTaskIdentifier
    output_bq_dataset: str
    graph_metadata: GraphMetadataPbWrapper
    split_dataset_format: EdgeDatasetFormat
    split_config: PerSplitFilteredEdgeBigqueryMetadata


def _get_BigqueryEdgeDataReference_for_split(
    output_bq_dataset: str,
    applied_task_identifier: AppliedTaskIdentifier,
    split: DatasetSplit,
) -> BigqueryEdgeDataReference:
    """Get a BigQuery edge data reference for a given split."""
    return BigqueryEdgeDataReference(
        reference_uri=BqUtils.join_path(
            BqUtils.format_bq_path(output_bq_dataset),
            f"{split.value}_edges_{applied_task_identifier}",
        ),
        src_identifier=SRC_FIELD,
        dst_identifier=DST_FIELD,
        edge_type=EdgeType("mixed", "mixed", "mixed"),
    )


class PerSplitFilteredEdgeDatasetBuilder:
    """Handles creation of edge dataset resources (BigQuery tables and GCS exports)."""

    def __init__(self, config: PerSplitFilteredEdgeDatasetConfig):
        self.config = config
        self.bq_utils = BqUtils(project=get_resource_config().project)
        self.gcs_utils = GcsUtils(project=get_resource_config().project)
        self.split_info = config.split_config.clause_per_split()

    def create_all_resources(self) -> None:
        """Create all required resources for edge datasets."""
        intermediate_table = self._build_intermediate_all_edges_table()
        self._create_filtered_tables_for_each_split(intermediate_table)
        self._export_filtered_tables_to_gcs_if_needed()

    def _build_intermediate_all_edges_table(self) -> str:
        """Build an intermediate edges table by unioning multiple edge tables with split metadata."""
        # Create an intermediate edges table with some split-related metadata.
        has_split_columns = len(self.config.split_config.split_columns) > 0
        split_column_selector = (
            ", ".join(self.config.split_config.split_columns)
            if has_split_columns
            else "RAND() AS rand_split"
        )

        if has_split_columns:
            logger.info(
                f"Using split columns: {self.config.split_config.split_columns}"
            )
        else:
            logger.info("No split columns provided. Using random transductive split.")

        logger.info(
            f"Using train/val/test clauses: '{self.config.split_config.train_split_clause}', "
            f"'{self.config.split_config.val_split_clause}', '{self.config.split_config.test_split_clause}'"
        )

        edge_table_queries: List[str] = []
        for edge_metadata in self.config.enumerated_edge_metadata:
            enumerated_reference = edge_metadata.enumerated_edge_data_reference
            edge_table = BqUtils.format_bq_path(
                bq_path=enumerated_reference.reference_uri
            )
            condensed_edge_type = (
                self.config.graph_metadata.edge_type_to_condensed_edge_type_map[
                    enumerated_reference.edge_type
                ]
            )
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
        logger.info(
            f"Will write train/val/test datasets to BQ dataset {self.config.output_bq_dataset}"
        )

        intermediate_edges_table = BqUtils.join_path(
            BqUtils.format_bq_path(self.config.output_bq_dataset),
            f"intermediate_{self.config.applied_task_identifier}",
        )

        self.bq_utils.run_query(
            query=union_edges_query,
            destination=intermediate_edges_table,
            write_disposition="WRITE_TRUNCATE",
            labels={},
        )

        return intermediate_edges_table

    def _create_filtered_tables_for_each_split(self, intermediate_table: str) -> None:
        """Create separate BigQuery tables for train/validation/test splits."""
        for split, split_clause in self.split_info:
            table_reference = _get_BigqueryEdgeDataReference_for_split(
                self.config.output_bq_dataset,
                self.config.applied_task_identifier,
                split,
            )

            random_column_field = "row_id"
            maybe_extra_field_selector = (
                f", RAND() as {random_column_field}"
                if self.config.split_dataset_format == EdgeDatasetFormat.BIGQUERY
                else ""
            )

            query = f"SELECT * {maybe_extra_field_selector} FROM `{intermediate_table}` WHERE {split_clause} ORDER BY RAND()"

            self.bq_utils.run_query(
                query=query,
                destination=table_reference.reference_uri,
                write_disposition="WRITE_TRUNCATE",
                labels=dict(),
            )

    def _export_filtered_tables_to_gcs_if_needed(self) -> None:
        """Export BigQuery tables to GCS if the format requires it."""
        if self.config.split_dataset_format not in (
            EdgeDatasetFormat.GCS_JSONL,
            EdgeDatasetFormat.GCS_PARQUET,
        ):
            return

        for split, _ in self.split_info:
            table_reference = _get_BigqueryEdgeDataReference_for_split(
                self.config.output_bq_dataset,
                self.config.applied_task_identifier,
                split,
            )

            gcs_target_path = GcsUri.join(
                gcs_constants.get_edge_dataset_output_path(
                    applied_task_identifier=self.config.applied_task_identifier,
                ),
                f"{split.value}_edges",
            )
            destination_glob_path = GcsUri.join(gcs_target_path, "shard-*")

            self.bq_utils.export_to_gcs(
                bq_table_path=table_reference.reference_uri,
                destination_gcs_uri=destination_glob_path,
                destination_format="NEWLINE_DELIMITED_JSON"
                if self.config.split_dataset_format == EdgeDatasetFormat.GCS_JSONL
                else "PARQUET",
            )


class PerSplitIterableDatasetStrategy(Protocol):
    """Protocol for creating different types of iterable datasets with filtered datasets for each split."""

    def create_dataset(
        self,
        config: PerSplitFilteredEdgeDatasetConfig,
        split: DatasetSplit,
        **kwargs,
    ) -> IterableDataset:
        """Create a dataset for the given split."""
        ...


class PerSplitIterableDatasetBigqueryStrategy:
    """Strategy for creating BigQuery-based iterable datasets with filtered datasets for each split."""

    def create_dataset(
        self,
        config: PerSplitFilteredEdgeDatasetConfig,
        split: DatasetSplit,
        **kwargs,
    ) -> IterableDataset:
        # Create table reference specific to BigQuery strategy
        table_reference = _get_BigqueryEdgeDataReference_for_split(
            config.output_bq_dataset,
            config.applied_task_identifier,
            split,
        )

        random_column_field = "row_id"
        return BigQueryHeterogeneousGraphIterableDataset(
            table=table_reference.reference_uri,
            random_column=random_column_field,
            project=get_resource_config().project,
            **kwargs,
        )


class PerSplitIterableDatasetGcsStrategy:
    """Strategy for creating GCS-based edge datasets (JSONL or Parquet)."""

    def __init__(self, format_type: EdgeDatasetFormat):
        self.format_type = format_type
        self.gcs_utils = GcsUtils(project=get_resource_config().project)

    def create_dataset(
        self,
        config: PerSplitFilteredEdgeDatasetConfig,
        split: DatasetSplit,
        **kwargs,
    ) -> IterableDataset:
        gcs_target_path = GcsUri.join(
            gcs_constants.get_edge_dataset_output_path(
                applied_task_identifier=config.applied_task_identifier,
            ),
            f"{split.value}_edges",
        )
        files_at_glob_path = self.gcs_utils.list_uris_with_gcs_path_pattern(
            gcs_path=gcs_target_path, pattern=".*shard-\d+"
        )

        dataset_cls = {
            EdgeDatasetFormat.GCS_JSONL: GcsJSONLHeterogeneousGraphIterableDataset,
            EdgeDatasetFormat.GCS_PARQUET: GcsParquetHeterogeneousGraphIterableDataset,
        }[self.format_type]

        return dataset_cls(file_uris=files_at_glob_path, **kwargs)


class PerSplitIterableDatasetFactory:
    """Factory for creating per-split edge datasets using appropriate strategies."""

    def __init__(self, config: PerSplitFilteredEdgeDatasetConfig):
        self.config = config
        self.strategy_map = {
            EdgeDatasetFormat.BIGQUERY: PerSplitIterableDatasetBigqueryStrategy(),
            EdgeDatasetFormat.GCS_JSONL: PerSplitIterableDatasetGcsStrategy(
                EdgeDatasetFormat.GCS_JSONL
            ),
            EdgeDatasetFormat.GCS_PARQUET: PerSplitIterableDatasetGcsStrategy(
                EdgeDatasetFormat.GCS_PARQUET
            ),
        }
        self.heterogeneous_kwargs = {
            "src_field": SRC_FIELD,
            "dst_field": DST_FIELD,
            "condensed_edge_type_field": CONDENSED_EDGE_TYPE_FIELD,
        }
        self.split_info = config.split_config.clause_per_split()

    def create_datasets(self) -> Dict[DatasetSplit, IterableDataset]:
        """Create and return the edge datasets for each data split."""
        strategy = self.strategy_map[self.config.split_dataset_format]
        datasets: Dict[DatasetSplit, IterableDataset] = {}

        for split, _ in self.split_info:
            datasets[split] = strategy.create_dataset(
                config=self.config,
                split=split,
                **self.heterogeneous_kwargs,
            )

        return datasets


def build_edge_datasets(
    distributed_context: DistributedContext,
    enumerated_edge_metadata: List[EnumeratorEdgeTypeMetadata],
    applied_task_identifier: AppliedTaskIdentifier,
    output_bq_dataset: str,
    graph_metadata: GraphMetadataPbWrapper,
    split_columns: Optional[List[str]] = None,
    train_split_clause: str = "rand_split BETWEEN 0 AND 0.8",
    val_split_clause: str = "rand_split BETWEEN 0.8 AND 0.9",
    test_split_clause: str = "rand_split BETWEEN 0.9 AND 1",
    format: EdgeDatasetFormat = EdgeDatasetFormat.GCS_PARQUET,
) -> Dict[DatasetSplit, IterableDataset]:
    """
    Build edge datasets for training, validation, and testing.  This function
    reads edge data from BigQuery, filters it based on the provided split clauses,
    and writes the filtered data to either BigQuery or GCS in the specified format.

    This function is designed to work in a distributed environment (e.g. at start of training),
    where multiple processes may be running in parallel. It ensures that the resources
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

    if split_columns is None:
        split_columns = list()

    # Create configuration objects
    bq_split_metadata = PerSplitFilteredEdgeBigqueryMetadata(
        split_columns=split_columns,
        train_split_clause=train_split_clause,
        val_split_clause=val_split_clause,
        test_split_clause=test_split_clause,
    )

    config = PerSplitFilteredEdgeDatasetConfig(
        distributed_context=distributed_context,
        enumerated_edge_metadata=enumerated_edge_metadata,
        applied_task_identifier=applied_task_identifier,
        output_bq_dataset=output_bq_dataset,
        graph_metadata=graph_metadata,
        split_dataset_format=format,
        split_config=bq_split_metadata,
    )

    # Handle distributed initialization if needed
    we_initialized_dist = False
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
        we_initialized_dist = True

    try:
        # Run BQ / GCS operations to create filtered datasets
        # Only rank 0 creates these datasets to avoid duplicate operations, all ranks wait for completion
        split_data_builder = PerSplitFilteredEdgeDatasetBuilder(config)
        if distributed_context.global_rank == 0:
            split_data_builder.create_all_resources()
        dist.barrier()  # Ensure all ranks wait for resource creation to complete

        # Create and return torch IterableDatasets for each split using the factory
        factory = PerSplitIterableDatasetFactory(config)
        return factory.create_datasets()
    finally:
        # Cleanup distributed context if we initialized it
        if we_initialized_dist:
            logger.info(
                f"Finished building edge datasets -- tearing down torch distributed for {distributed_context.global_rank}..."
            )
            dist.destroy_process_group()
