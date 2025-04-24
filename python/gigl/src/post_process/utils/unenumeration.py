import concurrent
from typing import List

from google.cloud import bigquery

import gigl.src.data_preprocessor.lib.enumerate.queries as enumeration_queries
import gigl.src.inference.v1.lib.queries as inference_queries
from gigl.common.env_config import get_available_cpus
from gigl.common.logger import Logger
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.types.graph_data import NodeType
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.utils.bq import BqUtils
from gigl.src.inference.lib.assets import InferenceAssets
from gigl.src.inference.v1.lib.inference_output_schema import (
    DEFAULT_EMBEDDINGS_TABLE_SCHEMA,
    DEFAULT_PREDICTIONS_TABLE_SCHEMA,
)
from snapchat.research.gbml import preprocessed_metadata_pb2

logger = Logger()


def _unenumerate_single_inferred_asset(
    inference_output_enumerated_assets_table: str,
    inference_output_node_id_field: str,
    inference_output_unenumerated_assets_table: str,
    enumerator_mapping_table: str,
):
    """Runs un-enumeration query on a single inferred asset (prediction or embedding table).
    Args:
        inference_output_enumerated_assets_table (str): BQ table which contains assets keyed off enumerated node id.
        inference_output_node_id_field (str): Field containing enumerated node ids in the enumerated_assets_table table.
        inference_output_unenumerated_assets_table (str): BQ table which contains "final" unenumerated assets.
        enumerator_mapping_table (str): BQ table which contains mapping between enumerated and original ids.
    """
    # TODO: relevant resource config args should be passed through instead of using global config
    resource_config = get_resource_config()
    bq_utils = BqUtils(project=resource_config.project)
    bq_utils.run_query(
        query=inference_queries.UNENUMERATION_QUERY.format(
            enumerated_assets_table=inference_output_enumerated_assets_table,
            mapping_table=enumerator_mapping_table,
            node_id_field=inference_output_node_id_field,
            original_node_id_field=enumeration_queries.DEFAULT_ORIGINAL_NODE_ID_FIELD,
            enumerated_int_id_field=enumeration_queries.DEFAULT_ENUMERATED_NODE_ID_FIELD,
        ),
        labels=resource_config.get_resource_labels(component=GiGLComponents.Inferencer),
        destination=inference_output_unenumerated_assets_table,
        write_disposition=bigquery.job.WriteDisposition.WRITE_TRUNCATE,
    )


def unenumerate_all_inferred_bq_assets(gbml_config_pb_wrapper: GbmlConfigPbWrapper):
    """Un-enumerates assets that are produced by inference.  These assets include
    embeddings and/or predictions.  The node ids in these outputs are enumerated
    as according to logic specified in the Data Preprocessor component.
    Args:
        gbml_config_pb_wrapper (GbmlConfigPbWrapper): _description_
    """

    # First we need to read all the node types in inferencer output and get their condensed node types.
    inference_output_map = (
        gbml_config_pb_wrapper.shared_config.inference_metadata.node_type_to_inferencer_output_info_map
    )
    node_type_to_condensed_node_type_map = (
        gbml_config_pb_wrapper.graph_metadata_pb_wrapper.node_type_to_condensed_node_type_map
    )

    # We then collect all the assets that need to be un-enumerated and their mapping tables
    enumerated_assets_output_tables: List = list()
    enumerated_node_id_fields: List = list()
    unenumerated_assets_output_tables: List = list()
    mapping_bq_tables: List = list()

    for node_type, inference_output in inference_output_map.items():
        # Get the condensed node type for the inference node type.
        condensed_inference_node_type = node_type_to_condensed_node_type_map[
            NodeType(node_type)
        ]
        logger.info(
            f"Processing node type: {node_type} with condensed node type: {condensed_inference_node_type}"
        )
        preprocessed_metadata_pb: preprocessed_metadata_pb2.PreprocessedMetadata = (
            gbml_config_pb_wrapper.preprocessed_metadata_pb_wrapper.preprocessed_metadata
        )
        node_type_metadata_map = (
            preprocessed_metadata_pb.condensed_node_type_to_preprocessed_metadata
        )
        node_metadata_output = node_type_metadata_map[
            int(condensed_inference_node_type)
        ]
        mapping_bq_table = (
            node_metadata_output.enumerated_node_ids_bq_table
        )  # schema; node_id, int_id (opinionated, specified by enumerator queries)

        if not mapping_bq_table:
            logger.info(
                f"Skipping un-enumeration for node_type={node_type} since no mapping table exists"
            )
            continue

        logger.info(f"Found mapping table to be: {mapping_bq_table}")

        unenumerated_embedding_table_path: str = (
            InferenceAssets.get_unenumerated_embedding_table_path(
                gbml_config_pb_wrapper=gbml_config_pb_wrapper, node_type=node_type
            )
        )
        if unenumerated_embedding_table_path:
            enumerated_assets_output_tables.append(
                InferenceAssets.get_enumerated_embedding_table_path(
                    gbml_config_pb_wrapper=gbml_config_pb_wrapper, node_type=node_type
                )
            )
            enumerated_node_id_fields.append(DEFAULT_EMBEDDINGS_TABLE_SCHEMA.node_field)
            unenumerated_assets_output_tables.append(unenumerated_embedding_table_path)
            mapping_bq_tables.append(mapping_bq_table)

        unenumerated_prediction_table_path: str = (
            InferenceAssets.get_unenumerated_prediction_table_path(
                gbml_config_pb_wrapper=gbml_config_pb_wrapper, node_type=node_type
            )
        )
        if unenumerated_prediction_table_path:
            enumerated_assets_output_tables.append(
                InferenceAssets.get_enumerated_predictions_table_path(
                    gbml_config_pb_wrapper=gbml_config_pb_wrapper, node_type=node_type
                )
            )
            enumerated_node_id_fields.append(
                DEFAULT_PREDICTIONS_TABLE_SCHEMA.node_field
            )
            unenumerated_assets_output_tables.append(unenumerated_prediction_table_path)
            mapping_bq_tables.append(mapping_bq_table)

    # Finally, we un-enumerate all the enumerated assets in parallel.
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=get_available_cpus()
    ) as executor:
        futures: List[concurrent.futures.Future] = list()
        for (
            enumerated_assets_table,
            node_id_field,
            unenumerated_assets_table,
            mapping_bq_table,
        ) in zip(
            enumerated_assets_output_tables,
            enumerated_node_id_fields,
            unenumerated_assets_output_tables,
            mapping_bq_tables,
        ):
            future = executor.submit(
                _unenumerate_single_inferred_asset,
                inference_output_enumerated_assets_table=enumerated_assets_table,
                inference_output_node_id_field=node_id_field,
                inference_output_unenumerated_assets_table=unenumerated_assets_table,
                enumerator_mapping_table=mapping_bq_table,
            )
            futures.append(future)

        for fut in concurrent.futures.as_completed(futures):
            fut.result()  # Rereaise any exceptions

    logger.info(f"Output to tables: {', '.join(unenumerated_assets_output_tables)}")
