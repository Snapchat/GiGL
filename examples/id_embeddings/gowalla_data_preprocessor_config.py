"""
Data Preprocessor Configuration for the Gowalla bipartite graph dataset.

This configuration defines how to preprocess the Gowalla user-item interaction
graph for use with GiGL. The graph consists of:
- User nodes (one node type)
- Item nodes (another node type)
- User-to-item edges (interactions)

The configuration handles creating node tables with degree features from the
raw edge data and defines the preprocessing specs for both nodes and edges.
"""

from typing import Any, Final

from google.cloud.bigquery.job import WriteDisposition

from gigl.common.logger import Logger
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.graph_data import EdgeType, EdgeUsageType, NodeType, Relation
from gigl.src.common.types.pb_wrappers.gigl_resource_config import (
    GiglResourceConfigWrapper,
)
from gigl.src.common.utils.bq import BqUtils
from gigl.src.data_preprocessor.lib.data_preprocessor_config import (
    DataPreprocessorConfig,
    build_ingestion_feature_spec_fn,
    build_passthrough_transform_preprocessing_fn,
)
from gigl.src.data_preprocessor.lib.ingest.bigquery import (
    BigqueryEdgeDataReference,
    BigqueryNodeDataReference,
)
from gigl.src.data_preprocessor.lib.ingest.reference import (
    EdgeDataReference,
    NodeDataReference,
)
from gigl.src.data_preprocessor.lib.types import (
    EdgeDataPreprocessingSpec,
    EdgeOutputIdentifier,
    NodeDataPreprocessingSpec,
    NodeOutputIdentifier,
)

logger = Logger()

# Node type names
USER_NODE_TYPE_NAME: Final[str] = "user"
ITEM_NODE_TYPE_NAME: Final[str] = "item"

# Column names
NODE_ID_COLUMN: Final[str] = "node_id"
DEGREE_COLUMN: Final[str] = "degree"
SRC_COLUMN: Final[str] = "from_user_id"
DST_COLUMN: Final[str] = "to_item_id"


class GowallaDataPreprocessorConfig(DataPreprocessorConfig):
    """
    Data preprocessor configuration for the Gowalla bipartite graph dataset.

    This config handles:
    1. Creating node tables (users and items) with degree features from edge data
    2. Defining preprocessing specs for user and item nodes
    3. Defining preprocessing specs for user-item edges

    Args:
        edge_table (str): Full BigQuery path to the edge table (e.g., "project.dataset.train_edges")
        **kwargs: Additional configuration arguments (currently unused but maintained for extensibility)
    """

    def __init__(self, edge_table: str, **kwargs: Any) -> None:
        super().__init__()

        # Store the edge table path
        self._edge_table = edge_table
        logger.info(f"Initializing Gowalla config with edge table: {self._edge_table}")

        # Define node types
        self._user_node_type = NodeType(USER_NODE_TYPE_NAME)
        self._item_node_type = NodeType(ITEM_NODE_TYPE_NAME)

        # Define edge type (user -> item)
        self._user_item_edge_type = EdgeType(
            src_node_type=self._user_node_type,
            dst_node_type=self._item_node_type,
            relation=Relation("to"),
        )

        # Get resource config
        self._resource_config: GiglResourceConfigWrapper = get_resource_config()

        # Node tables will be created in prepare_for_pipeline
        self._user_node_table: str = ""
        self._item_node_table: str = ""

    def prepare_for_pipeline(
        self, applied_task_identifier: AppliedTaskIdentifier
    ) -> None:
        """
        Prepare node tables before running the preprocessing pipeline.

        This method creates user and item node tables from the edge table,
        with node degree as a simple feature.

        Args:
            applied_task_identifier (AppliedTaskIdentifier): Unique identifier for this pipeline run
        """
        logger.info("Preparing node tables for Gowalla dataset...")

        bq_utils = BqUtils(project=self._resource_config.project)

        # Define table paths
        table_prefix = (
            f"{self._resource_config.project}."
            f"{self._resource_config.temp_assets_bq_dataset_name}."
            f"{applied_task_identifier}"
        )

        self._user_node_table = f"{table_prefix}_user_nodes"
        self._item_node_table = f"{table_prefix}_item_nodes"

        # Create user node table with degree feature
        logger.info(f"Creating user node table: {self._user_node_table}")
        user_node_query = f"""
        SELECT
            {SRC_COLUMN} AS {NODE_ID_COLUMN},
            COUNT(*) AS {DEGREE_COLUMN}
        FROM
            `{self._edge_table}`
        GROUP BY
            {SRC_COLUMN}
        """

        bq_utils.run_query(
            query=user_node_query,
            labels={},
            destination=self._user_node_table,
            write_disposition=WriteDisposition.WRITE_TRUNCATE,
        )
        logger.info(f"Created user node table with degree features")

        # Create item node table with degree feature
        logger.info(f"Creating item node table: {self._item_node_table}")
        item_node_query = f"""
        SELECT
            {DST_COLUMN} AS {NODE_ID_COLUMN},
            COUNT(*) AS {DEGREE_COLUMN}
        FROM
            `{self._edge_table}`
        GROUP BY
            {DST_COLUMN}
        """

        bq_utils.run_query(
            query=item_node_query,
            labels={},
            destination=self._item_node_table,
            write_disposition=WriteDisposition.WRITE_TRUNCATE,
        )
        logger.info(f"Created item node table with degree features")

        # Log statistics
        user_count = bq_utils.count_number_of_rows_in_bq_table(
            bq_table=self._user_node_table, labels={}
        )
        item_count = bq_utils.count_number_of_rows_in_bq_table(
            bq_table=self._item_node_table, labels={}
        )
        logger.info(f"Node tables created: {user_count} users, {item_count} items")

    def get_nodes_preprocessing_spec(
        self,
    ) -> dict[NodeDataReference, NodeDataPreprocessingSpec]:
        """
        Define preprocessing specifications for user and item nodes.

        Returns:
            dict[NodeDataReference, NodeDataPreprocessingSpec]: Mapping of node data
                references to their preprocessing specs.
        """
        node_data_ref_to_preprocessing_specs: dict[
            NodeDataReference, NodeDataPreprocessingSpec
        ] = {}

        # User node preprocessing spec
        logger.info("Defining user node preprocessing spec...")
        user_node_data_ref = BigqueryNodeDataReference(
            reference_uri=self._user_node_table,
            node_type=self._user_node_type,
        )

        user_feature_spec_fn = build_ingestion_feature_spec_fn(
            fixed_int_fields=[NODE_ID_COLUMN, DEGREE_COLUMN],
        )

        user_preprocessing_fn = build_passthrough_transform_preprocessing_fn()

        user_node_output_id = NodeOutputIdentifier(NODE_ID_COLUMN)

        node_data_ref_to_preprocessing_specs[
            user_node_data_ref
        ] = NodeDataPreprocessingSpec(
            identifier_output=user_node_output_id,
            features_outputs=[DEGREE_COLUMN],
            labels_outputs=[],  # No labels for unsupervised tasks
            feature_spec_fn=user_feature_spec_fn,
            preprocessing_fn=user_preprocessing_fn,
        )

        # Item node preprocessing spec
        logger.info("Defining item node preprocessing spec...")
        item_node_data_ref = BigqueryNodeDataReference(
            reference_uri=self._item_node_table,
            node_type=self._item_node_type,
        )

        item_feature_spec_fn = build_ingestion_feature_spec_fn(
            fixed_int_fields=[NODE_ID_COLUMN, DEGREE_COLUMN],
        )

        item_preprocessing_fn = build_passthrough_transform_preprocessing_fn()

        item_node_output_id = NodeOutputIdentifier(NODE_ID_COLUMN)

        node_data_ref_to_preprocessing_specs[
            item_node_data_ref
        ] = NodeDataPreprocessingSpec(
            identifier_output=item_node_output_id,
            features_outputs=[DEGREE_COLUMN],
            labels_outputs=[],  # No labels for unsupervised tasks
            feature_spec_fn=item_feature_spec_fn,
            preprocessing_fn=item_preprocessing_fn,
        )

        logger.info("Node preprocessing specs defined for users and items")
        return node_data_ref_to_preprocessing_specs

    def get_edges_preprocessing_spec(
        self,
    ) -> dict[EdgeDataReference, EdgeDataPreprocessingSpec]:
        """
        Define preprocessing specifications for user-item edges.

        Returns:
            dict[EdgeDataReference, EdgeDataPreprocessingSpec]: Mapping of edge data
                references to their preprocessing specs.
        """
        edge_data_ref_to_preprocessing_specs: dict[
            EdgeDataReference, EdgeDataPreprocessingSpec
        ] = {}

        logger.info("Defining edge preprocessing spec...")

        # Main edge data reference (user -> item interactions)
        main_edge_data_ref = BigqueryEdgeDataReference(
            reference_uri=self._edge_table,
            edge_type=self._user_item_edge_type,
            edge_usage_type=EdgeUsageType.MAIN,
        )

        # Edge output identifier
        edge_output_id = EdgeOutputIdentifier(
            src_node=NodeOutputIdentifier(SRC_COLUMN),
            dst_node=NodeOutputIdentifier(DST_COLUMN),
        )

        # Feature spec (just src and dst, no edge features)
        edge_feature_spec_fn = build_ingestion_feature_spec_fn(
            fixed_int_fields=[SRC_COLUMN, DST_COLUMN],
        )

        # Passthrough preprocessing (no transformations needed)
        edge_preprocessing_fn = build_passthrough_transform_preprocessing_fn()

        edge_data_ref_to_preprocessing_specs[
            main_edge_data_ref
        ] = EdgeDataPreprocessingSpec(
            identifier_output=edge_output_id,
            features_outputs=[],  # No edge features in raw data
            labels_outputs=[],  # No edge labels
            feature_spec_fn=edge_feature_spec_fn,
            preprocessing_fn=edge_preprocessing_fn,
        )

        logger.info("Edge preprocessing spec defined")
        return edge_data_ref_to_preprocessing_specs


def get_gowalla_preprocessor_config(edge_table: str, **kwargs: Any) -> GowallaDataPreprocessorConfig:
    """
    Factory function to create a GowallaDataPreprocessorConfig instance.

    Args:
        edge_table (str): Full BigQuery path to the edge table
        **kwargs: Additional configuration arguments

    Returns:
        GowallaDataPreprocessorConfig: Configured preprocessor instance

    """
    return GowallaDataPreprocessorConfig(edge_table=edge_table, **kwargs)
