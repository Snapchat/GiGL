from gigl.common.logger import Logger
from gigl.src.common.utils.bq import BqUtils

logger = Logger()


def _check_table_row_count(
    bq_utils: BqUtils,
    table_path: str,
    expected_count: int,
    label: str,
) -> str | None:
    """Checks that a BQ table exists and has the expected row count.

    Note: uses BqUtils.does_bq_table_exist() which returns False for any
    exception (not just NotFound). Permission errors and transient failures
    will be reported as "table does not exist."

    Args:
        bq_utils: BqUtils instance for BigQuery operations.
        table_path: Fully qualified BQ table path (project.dataset.table).
        expected_count: Expected number of rows.
        label: Human-readable label for error messages
            (e.g. "[paper] Embeddings").

    Returns:
        An error message string if validation fails, or None if valid.
    """
    if not bq_utils.does_bq_table_exist(table_path):
        return f"{label} table does not exist: {table_path}"

    actual = bq_utils.count_number_of_rows_in_bq_table(table_path)
    logger.info(f"{label} table ({table_path}) has {actual} rows.")
    if actual != expected_count:
        return (
            f"{label} row count mismatch: "
            f"expected {expected_count}, got {actual} "
            f"(table: {table_path})"
        )

    return None


def validate_bq_table_row_count(
    bq_utils: BqUtils,
    table_path: str,
    expected_count: int,
    label: str,
) -> None:
    """Validates that a BQ table exists and has the expected row count.

    This is the atomic building block for single-table validation.
    For batch validation across multiple node types, use
    validate_node_output_records instead.

    Args:
        bq_utils: BqUtils instance for BigQuery operations.
        table_path: Fully qualified BQ table path (project.dataset.table).
        expected_count: Expected number of rows.
        label: Human-readable label for error messages
            (e.g. "[paper] Embeddings").

    Raises:
        ValueError: If the table does not exist or the row count does not
            match.
    """
    error = _check_table_row_count(
        bq_utils=bq_utils,
        table_path=table_path,
        expected_count=expected_count,
        label=label,
    )
    if error is not None:
        raise ValueError(error)


def validate_node_output_records(
    bq_utils: BqUtils,
    expected_count_tables: dict[str, str],
    embeddings_tables: dict[str, str] | None = None,
    predictions_tables: dict[str, str] | None = None,
) -> None:
    """Validates output BQ tables have matching record counts for all node types.

    For each node type in expected_count_tables:
    1. Validates the expected-count (enumerated) table is configured and exists.
    2. Validates the embeddings table (if provided) exists and has matching
       row count.
    3. Validates the predictions table (if provided) exists and has matching
       row count.
    4. Validates at least one of embeddings or predictions is provided.

    All errors are collected and reported together before raising.

    Args:
        bq_utils: BqUtils instance for BigQuery operations.
        expected_count_tables: Mapping of node_type label to source-of-truth
            BQ table path. Values must be non-empty strings.
        embeddings_tables: Mapping of node_type label to embeddings BQ table
            path. All keys must also exist in expected_count_tables.
        predictions_tables: Mapping of node_type label to predictions BQ table
            path. All keys must also exist in expected_count_tables.

    Raises:
        ValueError: If any validation errors are found, with all errors listed.
            Also raised immediately if embeddings_tables or predictions_tables
            contain keys not present in expected_count_tables.
    """
    embeddings_tables = embeddings_tables or {}
    predictions_tables = predictions_tables or {}

    unexpected_keys = (
        embeddings_tables.keys() | predictions_tables.keys()
    ) - expected_count_tables.keys()
    if unexpected_keys:
        raise ValueError(
            f"Output tables reference node types not in expected_count_tables: "
            f"{sorted(unexpected_keys)}"
        )

    validation_errors: list[str] = []

    for node_type, enumerated_table in expected_count_tables.items():
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

        has_embeddings = node_type in embeddings_tables
        has_predictions = node_type in predictions_tables

        if has_embeddings:
            error = _check_table_row_count(
                bq_utils=bq_utils,
                table_path=embeddings_tables[node_type],
                expected_count=expected_count,
                label=f"[{node_type}] Embeddings",
            )
            if error is not None:
                validation_errors.append(error)

        if has_predictions:
            error = _check_table_row_count(
                bq_utils=bq_utils,
                table_path=predictions_tables[node_type],
                expected_count=expected_count,
                label=f"[{node_type}] Predictions",
            )
            if error is not None:
                validation_errors.append(error)

        if not has_embeddings and not has_predictions:
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
