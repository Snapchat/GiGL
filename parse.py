import sqlglot

import sqlparse

query = """
CREATE OR REPLACE TABLE `{edge_table}` AS
WITH agg AS (
    SELECT
        product_id,
        ARRAY_AGG(STRUCT(said)) AS users
    FROM
        `{engagement_table}`
    GROUP BY
        product_id
),
user_pairs AS (
    SELECT
        from_user.said AS from_user_said,
        to_user.said AS to_user_said
    FROM
        agg,
        UNNEST(users) AS from_user WITH OFFSET AS i,
        UNNEST(users) AS to_user WITH OFFSET AS j
    WHERE
        i < j
)
SELECT
    from_user_said,
    to_user_said
FROM
    user_pairs
GROUP BY
    from_user_said,
    to_user_said
"""
res = sqlparse.parse(query)
print(f"Parsed query: {res[0].tokens}")
for token in res[0].tokens:
    print(f"Token: {token.ttype} {token.value}")
exit()

res = sqlglot.parse(query)
print(f"Parsed query: {res}")
for res in res:
    select_expr = res.find(sqlglot.exp.Select)
    print(f"select_expr: {select_expr}")
    if select_expr:
        for expr in select_expr.expressions:
            print(f"expr: {expr}")
    else:
        print(f"No select expression found in {res}")

res_one = sqlglot.parse_one(query).find_all(sqlglot.exp.Select)
print(f"Parsed query: {res_one}")
for res_one in res_one:
    print(f"select_expr: {select_expr}")
    if select_expr:
        for expr in select_expr.expressions:
            print(f"expr: {expr}")
    else:
        print(f"No select expression found in {res_one}")
