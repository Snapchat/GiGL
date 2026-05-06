"""SQL query templates for graph structure analysis.

Each constant is a format-string template parameterized with table names
and column names. Pattern matches gigl/src/data_preprocessor/lib/enumerate/queries.py.
"""

import torch

INT16_MAX = int(torch.iinfo(torch.int16).max)  # 32767

# --- Tier 1: Hard fails ---

DANGLING_EDGES_QUERY = """
SELECT COUNT(*) AS dangling_count
FROM `{table}`
WHERE {src_id_column} IS NULL OR {dst_id_column} IS NULL
"""

EDGE_REFERENTIAL_INTEGRITY_QUERY = """
SELECT
    COUNTIF(src_node.{src_node_id_column} IS NULL) AS missing_src_count,
    COUNTIF(dst_node.{dst_node_id_column} IS NULL) AS missing_dst_count
FROM `{edge_table}` AS e
LEFT JOIN `{src_node_table}` AS src_node
    ON e.{src_id_column} = src_node.{src_node_id_column}
LEFT JOIN `{dst_node_table}` AS dst_node
    ON e.{dst_id_column} = dst_node.{dst_node_id_column}
"""

DUPLICATE_NODE_COUNT_QUERY = """
SELECT COUNT(*) AS duplicate_count FROM (
    SELECT {id_column}
    FROM `{table}`
    GROUP BY {id_column}
    HAVING COUNT(*) > 1
)
"""

# --- Tier 2: Core metrics ---

NODE_COUNT_QUERY = """
SELECT COUNT(*) AS node_count FROM `{table}`
"""

EDGE_COUNT_QUERY = """
SELECT COUNT(*) AS edge_count FROM `{table}`
"""

DUPLICATE_EDGE_COUNT_QUERY = """
SELECT COUNT(*) AS duplicate_count FROM (
    SELECT {src_id_column}, {dst_id_column}
    FROM `{table}`
    GROUP BY {src_id_column}, {dst_id_column}
    HAVING COUNT(*) > 1
)
"""

SELF_LOOP_COUNT_QUERY = """
SELECT COUNT(*) AS self_loop_count
FROM `{table}`
WHERE {src_id_column} = {dst_id_column}
"""

ISOLATED_NODE_COUNT_QUERY = """
SELECT COUNT(*) AS isolated_count FROM (
    SELECT n.{node_id_column}
    FROM `{node_table}` AS n
    LEFT JOIN `{edge_table}` AS e_src
        ON n.{node_id_column} = e_src.{src_id_column}
    LEFT JOIN `{edge_table}` AS e_dst
        ON n.{node_id_column} = e_dst.{dst_id_column}
    WHERE e_src.{src_id_column} IS NULL
        AND e_dst.{dst_id_column} IS NULL
)
"""

DEGREE_DISTRIBUTION_QUERY = """
SELECT
    MIN(degree) AS min_degree,
    MAX(degree) AS max_degree,
    AVG(degree) AS avg_degree,
    APPROX_QUANTILES(degree, 100) AS percentiles
FROM (
    SELECT {id_column}, COUNT(*) AS degree
    FROM `{table}`
    GROUP BY {id_column}
)
"""

DEGREE_BUCKET_QUERY = """
SELECT
    COUNTIF(degree BETWEEN 0 AND 1) AS bucket_0_1,
    COUNTIF(degree BETWEEN 2 AND 10) AS bucket_2_10,
    COUNTIF(degree BETWEEN 11 AND 100) AS bucket_11_100,
    COUNTIF(degree BETWEEN 101 AND 1000) AS bucket_101_1k,
    COUNTIF(degree BETWEEN 1001 AND 10000) AS bucket_1k_10k,
    COUNTIF(degree > 10000) AS bucket_10k_plus
FROM (
    SELECT {id_column}, COUNT(*) AS degree
    FROM `{table}`
    GROUP BY {id_column}
)
"""

TOP_K_HUBS_QUERY = """
SELECT {id_column} AS node_id, COUNT(*) AS degree
FROM `{table}`
GROUP BY {id_column}
ORDER BY degree DESC
LIMIT {k}
"""

SUPER_HUB_INT16_CLAMP_QUERY = f"""
SELECT COUNT(*) AS super_hub_count FROM (
    SELECT {{id_column}}, COUNT(*) AS degree
    FROM `{{table}}`
    GROUP BY {{id_column}}
    HAVING COUNT(*) > {INT16_MAX}
)
"""

COLD_START_NODE_COUNT_QUERY = """
SELECT COUNT(*) AS cold_start_count FROM (
    SELECT n.{node_id_column}, COALESCE(e.degree, 0) AS degree
    FROM `{node_table}` AS n
    LEFT JOIN (
        SELECT nid, COUNT(*) AS degree FROM (
            SELECT {src_id_column} AS nid FROM `{edge_table}`
            UNION ALL
            SELECT {dst_id_column} AS nid FROM `{edge_table}`
        )
        GROUP BY nid
    ) AS e ON n.{node_id_column} = e.nid
    WHERE COALESCE(e.degree, 0) <= 1
)
"""

# --- Tier 3: Label and heterogeneous ---

CLASS_IMBALANCE_QUERY = """
SELECT {label_column} AS label, COUNT(*) AS count
FROM `{table}`
WHERE {label_column} IS NOT NULL
GROUP BY {label_column}
ORDER BY count DESC
"""

LABEL_COVERAGE_QUERY = """
SELECT
    COUNT(*) AS total,
    COUNTIF({label_column} IS NOT NULL) AS labeled,
    SAFE_DIVIDE(COUNTIF({label_column} IS NOT NULL), COUNT(*)) AS coverage
FROM `{table}`
"""

EDGE_TYPE_DISTRIBUTION_QUERY = """
SELECT COUNT(*) AS edge_count FROM `{table}`
"""

EDGE_TYPE_NODE_COVERAGE_QUERY = """
SELECT
    APPROX_COUNT_DISTINCT({src_id_column}) AS distinct_src_count,
    APPROX_COUNT_DISTINCT({dst_id_column}) AS distinct_dst_count
FROM `{table}`
"""


# --- Supervision cross-table analysis ---

SUPERVISION_CROSS_TABLE_QUERY = """
WITH driver_pairs AS (
    SELECT DISTINCT
        {driver_anchor_column} AS anchor,
        {driver_other_column}  AS neighbor
    FROM `{driver_table}`
    WHERE {driver_anchor_column} IS NOT NULL
      AND {driver_other_column}  IS NOT NULL
),
other_pairs AS (
    SELECT DISTINCT
        {other_anchor_column} AS anchor,
        {other_other_column}  AS neighbor
    FROM `{other_table}`
    WHERE {other_anchor_column} IS NOT NULL
      AND {other_other_column}  IS NOT NULL
),
driver_anchors AS (
    SELECT DISTINCT anchor FROM driver_pairs
),
other_per_driver_anchor AS (
    SELECT driver_anchors.anchor,
           COALESCE(other_counts.cnt, 0) AS cnt
    FROM driver_anchors
    LEFT JOIN (
        SELECT anchor, COUNT(*) AS cnt FROM other_pairs GROUP BY anchor
    ) AS other_counts USING (anchor)
)
SELECT
    (SELECT COUNT(*) FROM driver_anchors) AS driver_anchor_count,
    (SELECT COUNT(*) FROM driver_pairs)   AS driver_pair_count,
    (SELECT COUNT(*) FROM other_pairs)    AS other_pair_count,
    (
        SELECT COUNT(*)
        FROM driver_pairs
        INNER JOIN other_pairs USING (anchor, neighbor)
    ) AS overlap_pair_count,
    (SELECT COUNTIF(cnt = 0) FROM other_per_driver_anchor)
        AS driver_anchors_with_zero_other,
    (SELECT AVG(cnt) FROM other_per_driver_anchor)
        AS avg_other_per_driver_anchor,
    (SELECT APPROX_QUANTILES(cnt, 100)[OFFSET(50)] FROM other_per_driver_anchor)
        AS p50_other_per_driver_anchor,
    (SELECT APPROX_QUANTILES(cnt, 100)[OFFSET(90)] FROM other_per_driver_anchor)
        AS p90_other_per_driver_anchor,
    (SELECT APPROX_QUANTILES(cnt, 100)[OFFSET(99)] FROM other_per_driver_anchor)
        AS p99_other_per_driver_anchor,
    (SELECT MAX(cnt) FROM other_per_driver_anchor)
        AS max_other_per_driver_anchor
"""


# --- Node-classification supervision tier ---


def build_label_sentinel_query(
    table: str, label_column: str, sentinel_values: list[str]
) -> str:
    """Build a single-pass query that splits label cells into NULL / sentinel / valid.

    Sentinel values are interpolated as quoted string literals; callers
    must ensure values come from a trusted config (the analyzer config
    is loaded by ``load_analyzer_config`` which already validates the
    structure of the YAML it reads). The label column is cast to STRING
    in the comparison so integer and string sentinels both work.

    Args:
        table: Fully qualified BQ table name.
        label_column: Column whose cells we're bucketing.
        sentinel_values: Strings that should be classified as sentinels
            distinct from SQL NULL.

    Returns:
        SQL query string returning one row with columns ``total_rows``,
        ``null_count``, ``valid_count``, and one ``sentinel_<idx>`` count
        per sentinel value (in declaration order).
    """
    sentinel_clauses = ",\n    ".join(
        f"COUNTIF(CAST({label_column} AS STRING) = "
        f"{_sql_string_literal(sentinel)}) AS sentinel_{idx}"
        for idx, sentinel in enumerate(sentinel_values)
    )
    sentinel_in_list = (
        ", ".join(_sql_string_literal(s) for s in sentinel_values)
        if sentinel_values
        else None
    )
    valid_clause = (
        f"COUNTIF({label_column} IS NOT NULL "
        f"AND CAST({label_column} AS STRING) NOT IN ({sentinel_in_list})) AS valid_count"
        if sentinel_in_list is not None
        else f"COUNTIF({label_column} IS NOT NULL) AS valid_count"
    )
    extra = f",\n    {sentinel_clauses}" if sentinel_clauses else ""
    return f"""
SELECT
    COUNT(*) AS total_rows,
    COUNTIF({label_column} IS NULL) AS null_count,
    {valid_clause}{extra}
FROM `{table}`
"""


def _sql_string_literal(value: str) -> str:
    """Quote a string for safe inline use in BQ SQL.

    Escapes single quotes and backslashes; no other characters are
    transformed. Sentinel values flow into ``IN`` lists so we control
    the surrounding context. Anything more invasive (parameterized
    queries) would require restructuring how every other query in this
    module is built.
    """
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def build_per_class_degree_query(
    node_table: str,
    node_id_column: str,
    label_column: str,
    edge_table: str,
    edge_src_column: str,
    edge_dst_column: str,
) -> str:
    """Per-label-value degree distribution joining labeled nodes to a message-passing edge table.

    Computes for each distinct non-NULL label value: count of class
    members, count with total degree <= 1 (cold-start), and degree
    distribution (mean / median / p90 / p99 / max). NULL labels are
    excluded — they are accounted for separately in
    :class:`LabelSentinelStats`. Sentinel-declared values (e.g. ``-1``)
    are *not* filtered out and surface as their own rows; the caller is
    responsible for partitioning the result into "valid class" vs
    "sentinel" using its own ``label_sentinel_values``.

    Returns one row per distinct non-NULL label value.
    """
    return f"""
WITH node_degrees AS (
    SELECT nid, COUNT(*) AS degree FROM (
        SELECT {edge_src_column} AS nid FROM `{edge_table}`
        UNION ALL
        SELECT {edge_dst_column} AS nid FROM `{edge_table}`
    )
    GROUP BY nid
),
labeled AS (
    SELECT
        CAST(n.{label_column} AS STRING) AS class_value,
        COALESCE(d.degree, 0) AS degree
    FROM `{node_table}` AS n
    LEFT JOIN node_degrees AS d
        ON n.{node_id_column} = d.nid
    WHERE n.{label_column} IS NOT NULL
)
SELECT
    class_value,
    COUNT(*) AS class_count,
    COUNTIF(degree <= 1) AS cold_start_count,
    AVG(degree) AS mean_degree,
    APPROX_QUANTILES(degree, 100) AS percentiles,
    MAX(degree) AS max_degree,
    COUNTIF(degree BETWEEN 0 AND 1) AS bucket_0_1,
    COUNTIF(degree BETWEEN 2 AND 10) AS bucket_2_10,
    COUNTIF(degree BETWEEN 11 AND 100) AS bucket_11_100,
    COUNTIF(degree BETWEEN 101 AND 1000) AS bucket_101_1k,
    COUNTIF(degree BETWEEN 1001 AND 10000) AS bucket_1k_10k,
    COUNTIF(degree > 10000) AS bucket_10k_plus
FROM labeled
GROUP BY class_value
ORDER BY class_count DESC
"""


def build_adjusted_homophily_query(
    node_table: str,
    node_id_column: str,
    label_column: str,
    sentinel_values: list[str],
    edge_table: str,
    edge_src_column: str,
    edge_dst_column: str,
    sample_cap: int,
) -> str:
    """Edge homophily and class-prior-adjusted homophily on a sampled edge set.

    Adjusted homophily is computed per Platonov et al., NeurIPS 2023:

        adjusted = (h_edge - sum_c (D_c / 2|E|)^2)
                   / (1 - sum_c (D_c / 2|E|)^2)

    where ``D_c`` is the sum of degrees of nodes in class ``c`` over the
    sampled edge set. Values near 0 mean "no signal beyond class
    priors"; positive is homophilic, negative heterophilic.

    Edges are sampled by ``MOD(FARM_FINGERPRINT(...), modulus) = 0`` so
    sampling is deterministic and consistent across reruns. ``sample_cap
    = 0`` means full-graph (no sampling).

    Returns one row with: ``edge_homophily``, ``expected_homophily``
    (the class-prior baseline), ``adjusted_homophily`` (computed in
    Python from the two columns above), and ``edge_sample_count``.
    """
    sentinel_filter_src = ""
    sentinel_filter_dst = ""
    if sentinel_values:
        sentinel_in_list = ", ".join(_sql_string_literal(s) for s in sentinel_values)
        sentinel_filter_src = (
            f"AND CAST(s.{label_column} AS STRING) NOT IN ({sentinel_in_list})"
        )
        sentinel_filter_dst = (
            f"AND CAST(d.{label_column} AS STRING) NOT IN ({sentinel_in_list})"
        )

    sample_filter = (
        ""
        if sample_cap <= 0
        else (
            f"WHERE MOD(ABS(FARM_FINGERPRINT(CONCAT("
            f"CAST({edge_src_column} AS STRING), '|', "
            f"CAST({edge_dst_column} AS STRING)))), {{modulus_placeholder}}) = 0"
        )
    )
    # We pass {modulus_placeholder} verbatim and let the caller fill it
    # in based on the cardinality of the edge table, so the same SQL
    # template is used for any sample size.
    return f"""
WITH sampled_edges AS (
    SELECT {edge_src_column} AS src_id, {edge_dst_column} AS dst_id
    FROM `{edge_table}`
    {sample_filter}
),
labeled_pairs AS (
    SELECT
        CAST(s.{label_column} AS STRING) AS src_label,
        CAST(d.{label_column} AS STRING) AS dst_label
    FROM sampled_edges AS e
    JOIN `{node_table}` AS s
        ON e.src_id = s.{node_id_column}
    JOIN `{node_table}` AS d
        ON e.dst_id = d.{node_id_column}
    WHERE s.{label_column} IS NOT NULL
      AND d.{label_column} IS NOT NULL
      {sentinel_filter_src}
      {sentinel_filter_dst}
),
endpoint_classes AS (
    SELECT label, COUNT(*) AS endpoint_count FROM (
        SELECT src_label AS label FROM labeled_pairs
        UNION ALL
        SELECT dst_label AS label FROM labeled_pairs
    )
    GROUP BY label
),
totals AS (
    SELECT SUM(endpoint_count) AS total_endpoints FROM endpoint_classes
)
SELECT
    SAFE_DIVIDE(COUNTIF(src_label = dst_label), COUNT(*)) AS edge_homophily,
    (
        SELECT SUM(POW(SAFE_DIVIDE(endpoint_count, total_endpoints), 2))
        FROM endpoint_classes, totals
    ) AS expected_homophily,
    COUNT(*) AS edge_sample_count
FROM labeled_pairs
"""


CROSS_SPLIT_OVERLAP_QUERY = """
SELECT
    (
        SELECT COUNT(*) FROM (
            SELECT {id_column}
            FROM `{table}`
            WHERE {id_column} IS NOT NULL
              AND {split_column} IS NOT NULL
            GROUP BY {id_column}
            HAVING COUNT(DISTINCT {split_column}) > 1
        )
    ) AS overlap_node_count
"""


SPLIT_VALUE_COUNTS_QUERY = """
SELECT
    CAST({split_column} AS STRING) AS split_value,
    COUNT(*) AS row_count
FROM `{table}`
WHERE {split_column} IS NOT NULL
GROUP BY split_value
ORDER BY row_count DESC
"""


def build_null_rates_query(table: str, columns: list[str]) -> str:
    """Build a batched NULL rates query for multiple columns.

    One query, one table scan, one COUNTIF per column.

    Args:
        table: Fully qualified BQ table name.
        columns: List of column names to check.

    Returns:
        SQL query string.
    """
    countif_clauses = ",\n    ".join(
        f"SAFE_DIVIDE(COUNTIF({col} IS NULL), COUNT(*)) AS {col}_null_rate"
        for col in columns
    )
    return f"""
SELECT
    COUNT(*) AS total_rows,
    {countif_clauses}
FROM `{table}`
"""
