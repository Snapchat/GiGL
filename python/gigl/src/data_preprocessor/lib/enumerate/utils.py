import concurrent.futures
import sys
import traceback
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import google.cloud.bigquery as bigquery

from gigl.common.env_config import get_available_cpus
from gigl.common.logger import Logger
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.constants import bq as bq_constants
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.graph_data import EdgeType, EdgeUsageType, NodeType
from gigl.src.common.utils.bq import BqUtils
from gigl.src.data_preprocessor.lib.enumerate import queries as enumeration_queries
from gigl.src.data_preprocessor.lib.ingest.bigquery import (
    BigqueryEdgeDataReference,
    BigqueryNodeDataReference,
)
from gigl.src.data_preprocessor.lib.ingest.reference import (
    EdgeDataReference,
    NodeDataReference,
)

logger = Logger()


def get_enumerated_node_id_map_bq_table_name(
    applied_task_identifier: AppliedTaskIdentifier, node_type: NodeType
) -> str:
    return BqUtils.format_bq_path(
        bq_path=BqUtils.join_path(
            bq_constants.get_embeddings_dataset_bq_path(),
            # applied_task_identifier as suffix so BQ can collapse the table names with same prefix in the UI
            f"enumerated_node_{node_type}_ids_{applied_task_identifier}",
        ),
    )


def get_enumerated_node_features_bq_table_name(
    applied_task_identifier: AppliedTaskIdentifier, node_type: NodeType
) -> str:
    return BqUtils.format_bq_path(
        bq_path=BqUtils.join_path(
            bq_constants.get_embeddings_dataset_bq_path(),
            # applied_task_identifier as suffix so BQ can collapse the table names with same prefix in the UI
            f"enumerated_node_{node_type}_node_features_{applied_task_identifier}",
        ),
    )


def get_enumerated_edge_features_bq_table_name(
    applied_task_identifier: AppliedTaskIdentifier,
    edge_type: EdgeType,
    edge_usage_type: EdgeUsageType,
):
    return BqUtils.format_bq_path(
        bq_path=BqUtils.join_path(
            bq_constants.get_embeddings_dataset_bq_path(),
            # applied_task_identifier as suffix so BQ can collapse the table names with same prefix in the UI
            f"enumerated_edge_{edge_type}_{str(edge_usage_type.value)}_edge_features_{applied_task_identifier}",
        ),
    )


def get_resource_labels() -> Dict[str, str]:
    resource_config = get_resource_config()
    return resource_config.get_resource_labels(
        component=GiGLComponents.DataPreprocessor
    )


@dataclass
class EnumeratorNodeTypeMetadata:
    input_node_data_reference: NodeDataReference
    enumerated_node_data_reference: BigqueryNodeDataReference
    bq_unique_node_ids_enumerated_table_name: str
    num_nodes: int

    def __repr__(self) -> str:
        return f"""EnumeratorNodeTypeMetadata(
            input_node_data_reference={self.input_node_data_reference},
            enumerated_node_data_reference={self.enumerated_node_data_reference},
            bq_unique_node_ids_enumerated_table_name={self.bq_unique_node_ids_enumerated_table_name},
            num_nodes={self.num_nodes})
            """


@dataclass
class EnumeratorEdgeTypeMetadata:
    input_edge_data_reference: EdgeDataReference
    enumerated_edge_data_reference: BigqueryEdgeDataReference
    num_edges: int

    def __repr__(self) -> str:
        return f"""EnumeratorEdgeTypeMetadata(
            input_edge_data_reference={self.input_edge_data_reference},
            enumerated_edge_data_reference={self.enumerated_edge_data_reference},
            num_edges={self.num_edges}
            """


class Enumerator:
    __applied_task_identifier: AppliedTaskIdentifier
    __bq_utils: BqUtils

    def __generate_enumerated_node_id_table_from_src_node_feature_table(
        self,
        bq_source_table_name: str,
        bq_source_table_node_id_col_name: str,
        node_type: NodeType,
    ) -> Tuple[str, int]:
        num_nodes_in_source_table = self.__bq_utils.count_number_of_rows_in_bq_table(
            bq_table=bq_source_table_name, labels=get_resource_labels()
        )

        # Get unique node ids, and store to BQ.
        bq_enumerated_node_id_map_table_name: str = (
            get_enumerated_node_id_map_bq_table_name(
                applied_task_identifier=self.__applied_task_identifier,
                node_type=node_type,
            )
        )
        logger.info(
            f"Will write enumerated node ids to: {bq_enumerated_node_id_map_table_name}"
        )

        unique_node_enumeration_query = enumeration_queries.UNIQUE_NODE_ENUMERATION_QUERY.format(
            bq_source_table_name=bq_source_table_name,
            bq_source_table_node_id_col_name=bq_source_table_node_id_col_name,
            original_node_id_field=enumeration_queries.DEFAULT_ORIGINAL_NODE_ID_FIELD,
            enumerated_int_id_field=enumeration_queries.DEFAULT_ENUMERATED_NODE_ID_FIELD,
        )

        self.__bq_utils.run_query(
            query=unique_node_enumeration_query,
            labels=get_resource_labels(),
            destination=bq_enumerated_node_id_map_table_name,
            write_disposition=bigquery.job.WriteDisposition.WRITE_TRUNCATE,
        )
        num_enumerated_nodes = self.__bq_utils.count_number_of_rows_in_bq_table(
            bq_table=bq_enumerated_node_id_map_table_name, labels=get_resource_labels()
        )

        # Make sure the number of input nodes and output nodes are equivalent.
        # If they are not, it suggests the input table has multiple rows for same node id.
        assert num_nodes_in_source_table == num_enumerated_nodes, (
            f"Number of input nodes not equal to number of enumerated nodes: ({num_nodes_in_source_table} != {num_enumerated_nodes}).  "
            f"This suggests the input table {bq_source_table_name} has multiple rows for the same node, which have been uniquified in "
            f"the enumerated node id table {bq_enumerated_node_id_map_table_name}."
        )
        logger.info(
            f"[Node Type: {node_type}] Finished generating enumerated ids for {num_enumerated_nodes} nodes; mapping written to {bq_enumerated_node_id_map_table_name}."
        )
        return bq_enumerated_node_id_map_table_name, num_enumerated_nodes

    def __generate_enumerated_node_feat_table_using_node_id_map_table(
        self,
        bq_source_table_name: str,
        bq_source_table_node_id_col_name: str,
        bq_enumerated_node_id_map_table_name: str,
        node_type: NodeType,
    ) -> str:
        dst_enumerated_node_features_table_name = (
            get_enumerated_node_features_bq_table_name(
                applied_task_identifier=self.__applied_task_identifier,
                node_type=node_type,
            )
        )
        logger.info(
            f"[Node Type: {node_type}]: Will use enumerated node id map table: {bq_enumerated_node_id_map_table_name} to enumerate nodes in {bq_source_table_name}. Will write resulting table to {dst_enumerated_node_features_table_name}"
        )

        enumerate_node_feature_table_query = enumeration_queries.NODE_FEATURES_ENUMERATION_QUERY.format(
            bq_node_features=bq_source_table_name,
            node_id_col=bq_source_table_node_id_col_name,
            bq_enumerated_node_ids=bq_enumerated_node_id_map_table_name,
            original_node_id_field=enumeration_queries.DEFAULT_ORIGINAL_NODE_ID_FIELD,
            enumerated_int_id_field=enumeration_queries.DEFAULT_ENUMERATED_NODE_ID_FIELD,
        )

        self.__bq_utils.run_query(
            enumerate_node_feature_table_query,
            labels=get_resource_config().get_resource_labels(
                component=GiGLComponents.DataPreprocessor
            ),
            destination=dst_enumerated_node_features_table_name,
            write_disposition=bigquery.job.WriteDisposition.WRITE_TRUNCATE,
        )

        logger.info(
            f"[Node Type: {node_type}]: Finished writing enumerated node features to {dst_enumerated_node_features_table_name}."
        )

        return dst_enumerated_node_features_table_name

    def __enumerate_node_reference(
        self,
        node_data_ref: NodeDataReference,
    ) -> EnumeratorNodeTypeMetadata:
        if not isinstance(node_data_ref, BigqueryNodeDataReference):
            raise NotImplementedError(
                f"Enumeration currently only supported for {BigqueryNodeDataReference.__name__}"
            )

        bq_source_table_name: str = BqUtils.format_bq_path(
            bq_path=node_data_ref.reference_uri,
        )
        logger.info(
            f"[Node Type: {node_data_ref.node_type}]: starting to enumerate node ids from source node table {bq_source_table_name}."
        )
        assert (
            node_data_ref.identifier is not None
        ), f"Missing identifier for node data reference: {node_data_ref}. "

        (
            bq_unique_node_ids_enumerated_table_name,
            num_enumerated_nodes,
        ) = self.__generate_enumerated_node_id_table_from_src_node_feature_table(
            bq_source_table_name=bq_source_table_name,
            bq_source_table_node_id_col_name=node_data_ref.identifier,
            node_type=node_data_ref.node_type,
        )
        bq_destination_enumerated_node_features_table_name: str = self.__generate_enumerated_node_feat_table_using_node_id_map_table(
            bq_source_table_name=bq_source_table_name,
            bq_source_table_node_id_col_name=node_data_ref.identifier,
            bq_enumerated_node_id_map_table_name=bq_unique_node_ids_enumerated_table_name,
            node_type=node_data_ref.node_type,
        )

        return EnumeratorNodeTypeMetadata(
            input_node_data_reference=node_data_ref,
            enumerated_node_data_reference=BigqueryNodeDataReference(
                reference_uri=bq_destination_enumerated_node_features_table_name,
                node_type=node_data_ref.node_type,
                identifier=node_data_ref.identifier,
            ),
            bq_unique_node_ids_enumerated_table_name=bq_unique_node_ids_enumerated_table_name,
            num_nodes=num_enumerated_nodes,
        )

    def __enumerate_all_node_references(
        self,
        node_data_references: Sequence[NodeDataReference],
    ) -> List[EnumeratorNodeTypeMetadata]:
        results: List[EnumeratorNodeTypeMetadata] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=get_available_cpus()
        ) as executor:
            futures: List[concurrent.futures.Future] = list()
            for node_data_ref in node_data_references:
                future = executor.submit(
                    self.__enumerate_node_reference,
                    node_data_ref=node_data_ref,
                )
                futures.append(future)

            for future in futures:
                result: EnumeratorNodeTypeMetadata = future.result()
                results.append(result)

        return results

    def __enumerate_all_edge_references(
        self,
        edge_data_references: Sequence[EdgeDataReference],
        map_enumerator_node_type_metadata: Dict[NodeType, EnumeratorNodeTypeMetadata],
    ) -> List[EnumeratorEdgeTypeMetadata]:
        results: List[EnumeratorEdgeTypeMetadata] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=get_available_cpus()
        ) as executor:
            futures: List[concurrent.futures.Future] = list()
            for edge_data_ref in edge_data_references:
                future = executor.submit(
                    self.__enumerate_edge_reference,
                    edge_data_ref=edge_data_ref,
                    map_enumerator_node_type_metadata=map_enumerator_node_type_metadata,
                )
                futures.append(future)

            for future in futures:
                result: EnumeratorEdgeTypeMetadata = future.result()
                results.append(result)

        return results

    def __generate_enumerated_edge_feat_table_using_node_id_map_tables(
        self,
        edge_type: EdgeType,
        edge_usage_type: EdgeUsageType,
        bq_source_table_name: str,
        bq_source_table_src_node_id_col_name: str,
        bq_source_table_dst_node_id_col_name: str,
        bq_enumerated_src_node_id_map_table_name: str,
        bq_enumerated_dst_node_id_map_table_name: str,
    ) -> Tuple[str, int]:
        dst_enumerated_edge_features_table_name: str = (
            get_enumerated_edge_features_bq_table_name(
                applied_task_identifier=self.__applied_task_identifier,
                edge_type=edge_type,
                edge_usage_type=edge_usage_type,
            )
        )

        num_edges_in_source_table = self.__bq_utils.count_number_of_rows_in_bq_table(
            bq_table=bq_source_table_name, labels=get_resource_labels()
        )

        has_edge_features = (
            self.__bq_utils.count_number_of_columns_in_bq_table(
                bq_table=bq_source_table_name,
            )
            > 2
        )

        graph_edges_enumeration_query = (
            enumeration_queries.EDGE_FEATURES_GRAPH_EDGELIST_ENUMERATION_QUERY
            if has_edge_features
            else enumeration_queries.NO_EDGE_FEATURES_GRAPH_EDGELIST_ENUMERATION_QUERY
        ).format(
            bq_graph=bq_source_table_name,
            src_enumerated_node_ids=bq_enumerated_src_node_id_map_table_name,
            dst_enumerated_node_ids=bq_enumerated_dst_node_id_map_table_name,
            src_node_id_col=bq_source_table_src_node_id_col_name,
            dst_node_id_col=bq_source_table_dst_node_id_col_name,
            original_node_id_field=enumeration_queries.DEFAULT_ORIGINAL_NODE_ID_FIELD,
            enumerated_int_id_field=enumeration_queries.DEFAULT_ENUMERATED_NODE_ID_FIELD,
        )

        self.__bq_utils.run_query(
            query=graph_edges_enumeration_query,
            labels=get_resource_config().get_resource_labels(
                component=GiGLComponents.DataPreprocessor
            ),
            destination=dst_enumerated_edge_features_table_name,
            write_disposition=bigquery.job.WriteDisposition.WRITE_TRUNCATE,
        )

        num_edges_in_enumerated_table = (
            self.__bq_utils.count_number_of_rows_in_bq_table(
                bq_table=dst_enumerated_edge_features_table_name,
                labels=get_resource_labels(),
            )
        )

        # Make sure the number of input edges and output edges are equivalent.
        # If they are not, it suggests there were edges which referenced src or dst nodes
        # that were not in the source or dest node tables.
        assert num_edges_in_source_table == num_edges_in_enumerated_table, (
            f"Number of input edges not equal to number of enumerated edges: ({num_edges_in_source_table} != {num_edges_in_enumerated_table}).  "
            f"This suggests there were edges in {bq_source_table_name} which referenced src nodes not found in {bq_enumerated_src_node_id_map_table_name} "
            f"or dst nodes not found in {bq_enumerated_dst_node_id_map_table_name}."
        )

        return dst_enumerated_edge_features_table_name, num_edges_in_enumerated_table

    def __enumerate_edge_reference(
        self,
        edge_data_ref: EdgeDataReference,
        map_enumerator_node_type_metadata: Dict[NodeType, EnumeratorNodeTypeMetadata],
    ) -> EnumeratorEdgeTypeMetadata:
        if not isinstance(edge_data_ref, BigqueryEdgeDataReference):
            raise NotImplementedError(
                f"Enumeration currently only supported for {BigqueryEdgeDataReference.__name__}"
            )
            # TODO: (svij-sc) Support this use case by dumping data to BQ using a beam pipeline
            # Will follow up in PR

        bq_source_table_name: str = BqUtils.format_bq_path(
            bq_path=edge_data_ref.reference_uri,
        )

        logger.info(
            f"[Edge Type: {edge_data_ref.edge_type} ; Edge Classification: {edge_data_ref.edge_usage_type}]: starting to enumerate node ids from source edge table {bq_source_table_name}."
        )

        # Get source and destination metadata.
        src_enumerated_node_type_metadata = map_enumerator_node_type_metadata[
            edge_data_ref.edge_type.src_node_type
        ]
        dst_enumerated_node_type_metadata = map_enumerator_node_type_metadata[
            edge_data_ref.edge_type.dst_node_type
        ]

        src_enumerated_node_ids = BqUtils.format_bq_path(
            bq_path=src_enumerated_node_type_metadata.bq_unique_node_ids_enumerated_table_name
        )
        dst_enumerated_node_ids = BqUtils.format_bq_path(
            bq_path=dst_enumerated_node_type_metadata.bq_unique_node_ids_enumerated_table_name
        )

        logger.info(
            f"[Edge Type: {edge_data_ref.edge_type} ; Edge Classification: {edge_data_ref.edge_usage_type}]: Started writing enumerated edges (and features)."
        )

        assert (edge_data_ref.src_identifier is not None) and (
            edge_data_ref.dst_identifier is not None
        ), f"Missing identifiers for edge data reference: {edge_data_ref}. "
        (
            bq_enumerated_edge_features_table_name,
            num_enumerated_edges,
        ) = self.__generate_enumerated_edge_feat_table_using_node_id_map_tables(
            edge_type=edge_data_ref.edge_type,
            edge_usage_type=edge_data_ref.edge_usage_type,
            bq_source_table_name=bq_source_table_name,
            bq_source_table_src_node_id_col_name=edge_data_ref.src_identifier,
            bq_source_table_dst_node_id_col_name=edge_data_ref.dst_identifier,
            bq_enumerated_src_node_id_map_table_name=src_enumerated_node_ids,
            bq_enumerated_dst_node_id_map_table_name=dst_enumerated_node_ids,
        )

        logger.info(
            f"[Edge Type: {edge_data_ref.edge_type} ; Edge Classification: {edge_data_ref.edge_usage_type}]: Finished writing enumerated edges (and features) to {bq_enumerated_edge_features_table_name}."
        )

        return EnumeratorEdgeTypeMetadata(
            input_edge_data_reference=edge_data_ref,
            enumerated_edge_data_reference=BigqueryEdgeDataReference(
                reference_uri=bq_enumerated_edge_features_table_name,
                edge_type=edge_data_ref.edge_type,
                edge_usage_type=edge_data_ref.edge_usage_type,
                src_identifier=edge_data_ref.src_identifier,
                dst_identifier=edge_data_ref.dst_identifier,
            ),
            num_edges=num_enumerated_edges,
        )

    def __run(
        self,
        applied_task_identifier: AppliedTaskIdentifier,
        node_data_references: Sequence[NodeDataReference],
        edge_data_references: Sequence[EdgeDataReference],
        gcp_project: str,
    ) -> Tuple[List[EnumeratorNodeTypeMetadata], List[EnumeratorEdgeTypeMetadata]]:
        self.__bq_utils = BqUtils(project=gcp_project)
        self.__applied_task_identifier = applied_task_identifier

        enumerated_node_metadata: List[
            EnumeratorNodeTypeMetadata
        ] = self.__enumerate_all_node_references(
            node_data_references=node_data_references
        )
        map_enumerator_node_type_metadata: Dict[
            NodeType, EnumeratorNodeTypeMetadata
        ] = {
            node_metadata.input_node_data_reference.node_type: node_metadata
            for node_metadata in enumerated_node_metadata
        }
        enumerated_edge_metadata: List[
            EnumeratorEdgeTypeMetadata
        ] = self.__enumerate_all_edge_references(
            edge_data_references=edge_data_references,
            map_enumerator_node_type_metadata=map_enumerator_node_type_metadata,
        )

        logger.info("Finished enumerating all node and edge references.")
        logger.info("Generated the following node enumerations:")
        for node_metadata in enumerated_node_metadata:
            logger.info(node_metadata)
        logger.info("Generated the following edge enumerations:")
        for edge_metadata in enumerated_edge_metadata:
            logger.info(edge_metadata)

        return (enumerated_node_metadata, enumerated_edge_metadata)

    def run(
        self,
        applied_task_identifier: AppliedTaskIdentifier,
        node_data_references: Sequence[NodeDataReference],
        edge_data_references: Sequence[EdgeDataReference],
        gcp_project: str,
    ) -> Tuple[List[EnumeratorNodeTypeMetadata], List[EnumeratorEdgeTypeMetadata]]:
        try:
            return self.__run(
                applied_task_identifier=applied_task_identifier,
                node_data_references=node_data_references,
                edge_data_references=edge_data_references,
                gcp_project=gcp_project,
            )
        except Exception as e:
            logger.error(
                "Enumerator failed due to a raised exception, which will follow"
            )
            logger.error(e)
            logger.error(traceback.format_exc())
            sys.exit(f"System will now exit: {e}")
