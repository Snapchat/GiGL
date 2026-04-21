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
