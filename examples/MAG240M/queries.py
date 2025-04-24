from examples.MAG240M.common import NUM_PAPER_FEATURES

query_template_reindex_author_writes_paper_table = """
-- Firstly, we reindex the author to the same node space as papers
-- TOTAL_NUM_PAPERS as defined in https://ogb.stanford.edu/docs/lsc/mag240m/
-- The paper node ids are thus: 0 to 121751665; and the author node ids will now start from 121751666
SELECT
    author + {TOTAL_NUM_PAPERS} AS src,
    paper as dst
FROM
    `{author_writes_paper_table}`
"""

query_template_cast_to_homogeneous_edge_table = """
-- Combine the paper cites paper, and the re-indexed author writes paper tables into a single edge table
SELECT
  src,
  dst
FROM
  `{reindexed_author_writes_paper_table}`
UNION ALL
SELECT
  src,
  dst
FROM
  `{paper_cites_paper_table}`
"""

query_template_computed_node_degree_table = """
SELECT
  node_id,
  COUNT(*) AS degree
FROM (
  SELECT
    src AS node_id
  FROM
    `{homogeneous_edge_table}`
  UNION ALL
  SELECT
    dst AS node_id
  FROM
    `{homogeneous_edge_table}`
)
GROUP BY
  node_id
"""

query_template_cast_to_intermediary_homogeneous_node_table = (
    """
WITH authors AS (
    SELECT 
        DISTINCT src as author_id
    FROM 
        `{reindexed_author_writes_paper_table}`
)
SELECT 
    author_id as node_id,
"""
    + ",\n".join([f"    0 AS feat_{i}" for i in range(NUM_PAPER_FEATURES)])
    + """
FROM authors
UNION ALL
SELECT 
    paper as node_id,
"""
    + ",\n".join([f"    feat_{i}" for i in range(NUM_PAPER_FEATURES)])
    + """
FROM
    `{paper_table}`
"""
)


query_template_generate_homogeneous_node_table = (
    """
SELECT
    interim_node_table.node_id as node_id,
    node_degree_table.degree as degree,
"""
    + ",\n".join(
        [
            f"    interim_node_table.feat_{i} as feat_{i}"
            for i in range(NUM_PAPER_FEATURES)
        ]
    )
    + """
FROM
    `{interim_node_table}` AS interim_node_table
JOIN
    `{node_degree_table}` AS node_degree_table
ON
    interim_node_table.node_id = node_degree_table.node_id
"""
)
