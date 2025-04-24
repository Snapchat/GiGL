DEFAULT_ORIGINAL_NODE_ID_FIELD = "node_id"
DEFAULT_ENUMERATED_NODE_ID_FIELD = "int_id"

UNIQUE_NODE_ENUMERATION_QUERY = """
WITH
  unique_nodes AS (
    SELECT DISTINCT {bq_source_table_node_id_col_name} as {original_node_id_field} FROM `{bq_source_table_name}`
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
