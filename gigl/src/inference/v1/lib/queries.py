UNENUMERATION_QUERY = """
SELECT
    mapping.{original_node_id_field},
    * EXCEPT({node_id_field}, {enumerated_int_id_field})
FROM
    `{enumerated_assets_table}` enumerated_assets
INNER JOIN
    `{mapping_table}` mapping
ON
    mapping.int_id = enumerated_assets.{node_id_field}
QUALIFY RANK() OVER (PARTITION BY mapping.{original_node_id_field} ORDER BY RAND()) = 1
"""
