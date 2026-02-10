from gigl.common import Uri
from gigl.common.logger import Logger
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.graph_data import NodeType
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.utils.bq import BqUtils
from gigl.src.post_process.post_processor import PostProcessor

logger = Logger()


class RecordCountValidatingPostProcessor(PostProcessor):
    """
    Post processor that extends PostProcessor with record count validation.

    Runs all standard PostProcessor logic (unenumeration, user-defined
    post-processing, metric export, cleanup), then validates that for each
    node type, the unenumerated output tables (embeddings, predictions) have
    the same number of rows as the corresponding enumerated_node_ids_bq_table.

    Only applicable for the GLT backend path.
    """

    # TODO: Add edge-level validation support.

    def _run(
        self,
        applied_task_identifier: AppliedTaskIdentifier,
        task_config_uri: Uri,
    ):
        # Run all standard PostProcessor logic first
        super()._run(
            applied_task_identifier=applied_task_identifier,
            task_config_uri=task_config_uri,
        )

        # Then validate record counts
        gbml_config_wrapper = GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
            gbml_config_uri=task_config_uri
        )
        self._validate_record_counts(gbml_config_wrapper=gbml_config_wrapper)

    def _validate_record_counts(
        self,
        gbml_config_wrapper: GbmlConfigPbWrapper,
        bq_utils: BqUtils | None = None,
    ) -> None:
        """Validates that output BQ tables have matching record counts.

        For each node type in the inference output, checks that:
        1. The enumerated_node_ids_bq_table exists.
        2. The embeddings table (if configured) exists and has the same row count.
        3. The predictions table (if configured) exists and has the same row count.
        4. At least one of embeddings or predictions is configured.

        All errors are collected and reported together before raising.

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

        validation_errors: list[str] = []

        inference_output_map = (
            gbml_config_wrapper.shared_config.inference_metadata.node_type_to_inferencer_output_info_map
        )
        node_type_to_condensed = (
            gbml_config_wrapper.graph_metadata_pb_wrapper.node_type_to_condensed_node_type_map
        )
        preprocessed_metadata = (
            gbml_config_wrapper.preprocessed_metadata_pb_wrapper.preprocessed_metadata
        )

        for node_type, inference_output in inference_output_map.items():
            condensed_node_type = node_type_to_condensed[NodeType(node_type)]
            node_metadata = (
                preprocessed_metadata.condensed_node_type_to_preprocessed_metadata[
                    int(condensed_node_type)
                ]
            )
            enumerated_table = node_metadata.enumerated_node_ids_bq_table

            if not enumerated_table:
                validation_errors.append(
                    f"[{node_type}] No enumerated_node_ids_bq_table configured."
                )
                continue

            if not bq_utils.does_bq_table_exist(enumerated_table):
                validation_errors.append(
                    f"[{node_type}] enumerated_node_ids_bq_table does not exist: {enumerated_table}"
                )
                continue

            expected_count = bq_utils.count_number_of_rows_in_bq_table(enumerated_table)
            logger.info(
                f"[{node_type}] enumerated_node_ids_bq_table ({enumerated_table}) has {expected_count} rows."
            )

            # Validate embeddings
            embeddings_path = inference_output.embeddings_path
            if embeddings_path:
                if not bq_utils.does_bq_table_exist(embeddings_path):
                    validation_errors.append(
                        f"[{node_type}] Embeddings table does not exist: {embeddings_path}"
                    )
                else:
                    actual = bq_utils.count_number_of_rows_in_bq_table(embeddings_path)
                    logger.info(
                        f"[{node_type}] Embeddings table ({embeddings_path}) has {actual} rows."
                    )
                    if actual != expected_count:
                        validation_errors.append(
                            f"[{node_type}] Embeddings row count mismatch: "
                            f"expected {expected_count}, got {actual} "
                            f"(table: {embeddings_path})"
                        )

            # Validate predictions
            predictions_path = inference_output.predictions_path
            if predictions_path:
                if not bq_utils.does_bq_table_exist(predictions_path):
                    validation_errors.append(
                        f"[{node_type}] Predictions table does not exist: {predictions_path}"
                    )
                else:
                    actual = bq_utils.count_number_of_rows_in_bq_table(predictions_path)
                    logger.info(
                        f"[{node_type}] Predictions table ({predictions_path}) has {actual} rows."
                    )
                    if actual != expected_count:
                        validation_errors.append(
                            f"[{node_type}] Predictions row count mismatch: "
                            f"expected {expected_count}, got {actual} "
                            f"(table: {predictions_path})"
                        )

            if not embeddings_path and not predictions_path:
                validation_errors.append(
                    f"[{node_type}] Neither embeddings_path nor predictions_path is set."
                )

        if validation_errors:
            error_summary = "\n".join(validation_errors)
            raise ValueError(
                f"Record count validation failed with {len(validation_errors)} error(s):\n"
                f"{error_summary}"
            )

        logger.info("All record count validations passed.")
