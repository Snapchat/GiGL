DEFAULT_ORIGINAL_NODE_ID_FIELD = "node_id"
DEFAULT_ENUMERATED_NODE_ID_FIELD = "int_id"

UNIQUE_NODE_ENUMERATION_QUERY = """
WITH
  source_node_ids AS (
    -- Fail fast if any node id is NULL. Otherwise, NULL would survive SELECT DISTINCT and be
    -- silently enumerated as its own node, and edges referencing it would fail to map to any
    -- enumerated node id downstream. See https://github.com/Snapchat/GiGL/issues/288.
    SELECT
      IF(
        {bq_source_table_node_id_col_name} IS NULL,
        ERROR("Found NULL node id in column `{bq_source_table_node_id_col_name}` of table `{bq_source_table_name}`; node ids must be non-NULL."),
        {bq_source_table_node_id_col_name}
      ) AS {original_node_id_field}
    FROM `{bq_source_table_name}`
  ),
  unique_nodes AS (
    SELECT DISTINCT {original_node_id_field} FROM source_node_ids
  )
SELECT
  {original_node_id_field},
  ROW_NUMBER() OVER(ORDER BY {original_node_id_field}) - 1 AS {enumerated_int_id_field}
FROM
  unique_nodes
"""


NODE_FEATURES_ENUMERATION_QUERY = """
WITH
  unmapped_node_features AS
  (
    SELECT * FROM `{bq_node_features}`
  ),
  enumerated AS
  (
  SELECT
    {original_node_id_field},
    {enumerated_int_id_field}
  FROM
    `{bq_enumerated_node_ids}`
  ),
  mapped_node_features AS (
  SELECT
    enumerated.{enumerated_int_id_field} as {node_id_col},
    unmapped_node_features.* EXCEPT ({node_id_col})
  FROM
    enumerated
  INNER JOIN
    unmapped_node_features
  ON
    enumerated.{original_node_id_field} = unmapped_node_features.{node_id_col})
SELECT
  *
FROM
  mapped_node_features
"""


NO_EDGE_FEATURES_GRAPH_EDGELIST_ENUMERATION_QUERY = """
WITH
  unmapped_graph AS
  (
    SELECT {src_node_id_col}, {dst_node_id_col} FROM `{bq_graph}`
  )
SELECT
  (
    SELECT {enumerated_int_id_field}
    FROM `{src_enumerated_node_ids}`
    WHERE {original_node_id_field} = unmapped_graph.{src_node_id_col}
  ) as {src_node_id_col},
  (
    SELECT {enumerated_int_id_field}
    FROM `{dst_enumerated_node_ids}`
    WHERE {original_node_id_field} = unmapped_graph.{dst_node_id_col}
  ) as {dst_node_id_col},
FROM unmapped_graph
"""

EDGE_FEATURES_GRAPH_EDGELIST_ENUMERATION_QUERY = """
WITH
  unmapped_graph AS
  (
    SELECT
      {src_node_id_col},
      {dst_node_id_col},
      * EXCEPT({src_node_id_col}, {dst_node_id_col})
    FROM
      `{bq_graph}`
  )
SELECT
  (
    SELECT {enumerated_int_id_field}
    FROM `{src_enumerated_node_ids}`
    WHERE {original_node_id_field} = unmapped_graph.{src_node_id_col}
  ) as {src_node_id_col},
  (
    SELECT {enumerated_int_id_field}
    FROM `{dst_enumerated_node_ids}`
    WHERE {original_node_id_field} = unmapped_graph.{dst_node_id_col}
  ) as {dst_node_id_col},
  * EXCEPT({src_node_id_col}, {dst_node_id_col})
FROM unmapped_graph
"""
