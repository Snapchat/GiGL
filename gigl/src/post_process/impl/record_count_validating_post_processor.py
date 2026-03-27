from gigl.common.logger import Logger
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.graph_data import NodeType
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.utils.bq import BqUtils
from gigl.src.common.utils.output_record_validation import validate_node_output_records
from gigl.src.post_process.lib.base_post_processor import BasePostProcessor
from snapchat.research.gbml import gbml_config_pb2

logger = Logger()


class RecordCountValidatingPostProcessor(BasePostProcessor):
    """Post processor that validates output BQ tables have matching record counts.

    For each node type in the inference output, checks that embeddings and/or
    predictions tables have the same row count as the enumerated node IDs table.

    Only applicable for the GLT backend path.
    """

    # We need __init__ as applied_task_identified gets injected PostProcessor._run_post_process
    # But we have no need for it.
    def __init__(self, *, applied_task_identifier: AppliedTaskIdentifier):
        pass

    # TODO: Add edge-level validation support.

    def run_post_process(
        self,
        gbml_config_pb: gbml_config_pb2.GbmlConfig,
    ):
        gbml_config_wrapper = GbmlConfigPbWrapper(gbml_config_pb=gbml_config_pb)
        self._validate_record_counts(gbml_config_wrapper=gbml_config_wrapper)

    def _validate_record_counts(
        self,
        gbml_config_wrapper: GbmlConfigPbWrapper,
        bq_utils: BqUtils | None = None,
    ) -> None:
        """Validates that output BQ tables have matching record counts.

        Extracts inference output metadata from the config wrapper and delegates
        to validate_node_output_records for the actual validation.

        Args:
            gbml_config_wrapper: The GbmlConfig wrapper with access to all metadata.
            bq_utils: Optional BqUtils instance for testing. If None, creates one
                from the resource config.

        Raises:
            ValueError: If any validation errors are found, with all errors listed.
        """
        if bq_utils is None:
            from gigl.env.pipelines_config import get_resource_config

            resource_config = get_resource_config()
            bq_utils = BqUtils(project=resource_config.project)

        inference_output_map = (
            gbml_config_wrapper.shared_config.inference_metadata.node_type_to_inferencer_output_info_map
        )
        node_type_to_condensed = (
            gbml_config_wrapper.graph_metadata_pb_wrapper.node_type_to_condensed_node_type_map
        )
        preprocessed_metadata = (
            gbml_config_wrapper.preprocessed_metadata_pb_wrapper.preprocessed_metadata
        )

        expected_count_tables: dict[str, str] = {}
        embeddings_tables: dict[str, str] = {}
        predictions_tables: dict[str, str] = {}

        for node_type, inference_output in inference_output_map.items():
            condensed_node_type = node_type_to_condensed[NodeType(node_type)]
            node_metadata = (
                preprocessed_metadata.condensed_node_type_to_preprocessed_metadata[
                    int(condensed_node_type)
                ]
            )
            expected_count_tables[
                node_type
            ] = node_metadata.enumerated_node_ids_bq_table

            if inference_output.embeddings_path:
                embeddings_tables[node_type] = inference_output.embeddings_path
            if inference_output.predictions_path:
                predictions_tables[node_type] = inference_output.predictions_path

        validate_node_output_records(
            bq_utils=bq_utils,
            expected_count_tables=expected_count_tables,
            embeddings_tables=embeddings_tables,
            predictions_tables=predictions_tables,
        )
