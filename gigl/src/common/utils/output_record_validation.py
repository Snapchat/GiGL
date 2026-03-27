from gigl.common.logger import Logger
from gigl.src.common.utils.bq import BqUtils

logger = Logger()


def _collect_table_row_count_errors(
    bq_utils: BqUtils,
    table_path: str,
    expected_count: int,
    label: str,
) -> list[str]:
    """Checks that a BQ table exists and has the expected row count.

    Returns a list of error strings (empty if valid).

    Same checks as validate_bq_table_row_count but returns errors instead of
    raising, so callers can accumulate errors across multiple tables.

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
        A list of error message strings. Empty if the table exists and has
        the expected row count.
    """
    if not bq_utils.does_bq_table_exist(table_path):
        return [f"{label} table does not exist: {table_path}"]

    actual = bq_utils.count_number_of_rows_in_bq_table(table_path)
    logger.info(f"{label} table ({table_path}) has {actual} rows.")
    if actual != expected_count:
        return [
            f"{label} row count mismatch: "
            f"expected {expected_count}, got {actual} "
            f"(table: {table_path})"
        ]

    return []


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
    errors = _collect_table_row_count_errors(
        bq_utils=bq_utils,
        table_path=table_path,
        expected_count=expected_count,
        label=label,
    )
    if errors:
        raise ValueError(errors[0])


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

    unexpected_keys: set[str] = set()
    unexpected_keys.update(embeddings_tables.keys() - expected_count_tables.keys())
    unexpected_keys.update(predictions_tables.keys() - expected_count_tables.keys())
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
            validation_errors.extend(
                _collect_table_row_count_errors(
                    bq_utils=bq_utils,
                    table_path=embeddings_tables[node_type],
                    expected_count=expected_count,
                    label=f"[{node_type}] Embeddings",
                )
            )

        if has_predictions:
            validation_errors.extend(
                _collect_table_row_count_errors(
                    bq_utils=bq_utils,
                    table_path=predictions_tables[node_type],
                    expected_count=expected_count,
                    label=f"[{node_type}] Predictions",
                )
            )

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
