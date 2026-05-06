"""Schema-aware BQ projection builder for the feature profiler.

Translates a BigQuery table schema into a ``SELECT`` projection that
TFDV can profile. Scalar profileable columns pass through unchanged;
REPEATED ``FLOAT`` / ``FLOAT64`` / ``NUMERIC`` / ``BIGNUMERIC`` columns
(embeddings) expand into four scalar hygiene companions:

* ``<col>_len`` — array length
* ``<col>_has_nan`` — any NaN element
* ``<col>_has_inf`` — any Inf element
* ``<col>_is_all_zero`` — every element equals 0

Structural-sanity (dedup / unique-ratio / top-K) lives in
:mod:`gigl.analytics.data_analyzer.embedding_diagnostics`, which runs its
own aggregate query over ``FARM_FINGERPRINT(TO_JSON_STRING(<col>))``. The
hash is deliberately excluded from this projection so TFDV doesn't render
noisy stats on a 64-bit hash column.
"""

from dataclasses import dataclass

from google.cloud.bigquery import SchemaField

from gigl.common.logger import Logger

logger = Logger()

# BigQuery scalar types TFDV can profile once wrapped as ``list<scalar>`` by
# ``BqTableToRecordBatch``. Matches ``_PROFILEABLE_FIELD_TYPES`` in
# ``feature_profiler.py`` — kept in sync via a single import site.
_SCALAR_PROFILEABLE_TYPES: frozenset[str] = frozenset(
    {
        "STRING",
        "INTEGER",
        "INT64",
        "FLOAT",
        "FLOAT64",
        "NUMERIC",
        "BIGNUMERIC",
        "BOOLEAN",
        "BOOL",
    }
)

# REPEATED types that represent embedding vectors. STRING / INT arrays are
# intentionally excluded — they need different diagnostics (e.g. vocab stats)
# and are out of scope for this pass.
_EMBEDDING_FLOAT_TYPES: frozenset[str] = frozenset(
    {"FLOAT", "FLOAT64", "NUMERIC", "BIGNUMERIC"}
)


@dataclass(frozen=True)
class ProjectionResult:
    """Output of :func:`build_projection`.

    ``projection`` is a list of ``(column_name, sql_expression)`` pairs
    suitable for feeding directly into a
    :class:`~gigl.common.beam.tfdv_transforms.BqTableToRecordBatch`. Each
    entry renders as ``{sql_expression} AS \\`{column_name}\\``` in the
    resulting ``SELECT``.

    ``embedding_columns`` lists the original REPEATED FLOAT column names
    (pre-expansion) in schema order; the dedup pass uses them to locate the
    corresponding ``<col>_hash`` companion.
    """

    projection: list[tuple[str, str]]
    embedding_columns: list[str]


def is_embedding_column(field: SchemaField) -> bool:
    """Return ``True`` for REPEATED FLOAT-family columns (embedding vectors)."""
    return (
        field.mode == "REPEATED" and field.field_type.upper() in _EMBEDDING_FLOAT_TYPES
    )


def detect_embedding_columns(
    schema: dict[str, SchemaField], excluded: set[str]
) -> list[str]:
    """List REPEATED FLOAT-family columns in the schema, in declaration order.

    Excluded columns (typically structural join keys) are dropped.
    """
    return [
        name
        for name, field in schema.items()
        if name not in excluded and is_embedding_column(field)
    ]


def build_projection(
    schema: dict[str, SchemaField], excluded: set[str]
) -> ProjectionResult:
    """Build a TFDV-compatible projection from a BigQuery schema.

    Scalar profileable columns (see :data:`_SCALAR_PROFILEABLE_TYPES`) are
    passed through verbatim, *except* BOOL / BOOLEAN columns are cast to
    INT64. ``BqTableToRecordBatch`` wraps each value in a single-element
    list before emitting an Arrow ``RecordBatch``; TFDV's
    ``get_feature_type_from_arrow_type`` does not accept ``list<bool>``
    (only int / float / string / bytes lists), so a raw BOOL column would
    crash the Dataflow job in ``BasicStatsGenerator.add_input``. Casting
    to INT64 in SQL keeps the BOOL semantics (0/1) profileable as an
    int feature.

    REPEATED FLOAT-family columns are expanded into four scalar hygiene
    companions (see module docstring). The three boolean companions
    (``_has_nan``, ``_has_inf``, ``_is_all_zero``) are likewise cast to
    INT64 for the same reason. REPEATED non-FLOAT columns and
    non-profileable scalar types are skipped with an ``INFO`` log.

    Args:
        schema: Column name → ``SchemaField`` map (as returned by
            ``BqUtils.fetch_bq_table_schema``).
        excluded: Column names to drop entirely (typically structural join
            keys: node ``id_column``; edge ``src_id_column`` +
            ``dst_id_column``).

    Returns:
        :class:`ProjectionResult`. ``projection`` preserves schema order
        with each embedding's hygiene companions appearing in a contiguous
        block.
    """
    projection: list[tuple[str, str]] = []
    embedding_columns: list[str] = []
    for name, field in schema.items():
        if name in excluded:
            continue
        if is_embedding_column(field):
            projection.extend(_embedding_hygiene_projection(name))
            embedding_columns.append(name)
            continue
        if field.mode == "REPEATED":
            logger.info(
                f"skipping REPEATED column {name!r} of type {field.field_type} "
                "(hygiene companions only cover REPEATED FLOAT families)."
            )
            continue
        type_upper = field.field_type.upper()
        if type_upper not in _SCALAR_PROFILEABLE_TYPES:
            logger.info(
                f"skipping column {name!r} of type {field.field_type} "
                "(not TFDV-profileable)."
            )
            continue
        if type_upper in ("BOOL", "BOOLEAN"):
            projection.append((name, f"CAST(`{name}` AS INT64)"))
        else:
            projection.append((name, f"`{name}`"))
    return ProjectionResult(projection=projection, embedding_columns=embedding_columns)


def _embedding_hygiene_projection(column: str) -> list[tuple[str, str]]:
    """Return the four hygiene ``(name, expr)`` entries for one embedding column.

    The three boolean companions are wrapped in ``CAST(... AS INT64)`` so
    the resulting Arrow column is ``list<int64>`` rather than ``list<bool>``;
    see :func:`build_projection` for the TFDV compatibility rationale.
    """
    return [
        (f"{column}_len", f"ARRAY_LENGTH(`{column}`)"),
        (
            f"{column}_has_nan",
            f"CAST(IFNULL((SELECT LOGICAL_OR(IS_NAN(v)) FROM UNNEST(`{column}`) v), "
            "FALSE) AS INT64)",
        ),
        (
            f"{column}_has_inf",
            f"CAST(IFNULL((SELECT LOGICAL_OR(IS_INF(v)) FROM UNNEST(`{column}`) v), "
            "FALSE) AS INT64)",
        ),
        (
            f"{column}_is_all_zero",
            f"CAST(IFNULL((SELECT LOGICAL_AND(v = 0) FROM UNNEST(`{column}`) v), "
            "FALSE) AS INT64)",
        ),
    ]
