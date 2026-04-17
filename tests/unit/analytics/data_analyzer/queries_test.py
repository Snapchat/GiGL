from gigl.analytics.data_analyzer.queries import (
    DANGLING_EDGES_QUERY,
    DEGREE_BUCKET_QUERY,
    DEGREE_DISTRIBUTION_QUERY,
    DUPLICATE_NODE_COUNT_QUERY,
    EDGE_REFERENTIAL_INTEGRITY_QUERY,
    NODE_COUNT_QUERY,
    SUPER_HUB_INT16_CLAMP_QUERY,
    TOP_K_HUBS_QUERY,
    build_null_rates_query,
)
from tests.test_assets.test_case import TestCase

NODE_TABLE = "project.dataset.user_nodes"
EDGE_TABLE = "project.dataset.user_edges"


class NodeCountQueryTest(TestCase):
    def test_contains_table_name(self) -> None:
        sql = NODE_COUNT_QUERY.format(table=NODE_TABLE)
        self.assertIn(f"`{NODE_TABLE}`", sql)
        self.assertIn("COUNT(*)", sql)


class DanglingEdgesQueryTest(TestCase):
    def test_contains_null_checks(self) -> None:
        sql = DANGLING_EDGES_QUERY.format(
            table=EDGE_TABLE, src_id_column="src_uid", dst_id_column="dst_uid"
        )
        self.assertIn("src_uid IS NULL", sql)
        self.assertIn("dst_uid IS NULL", sql)
        self.assertIn(f"`{EDGE_TABLE}`", sql)


class EdgeReferentialIntegrityQueryTest(TestCase):
    def test_contains_left_join(self) -> None:
        sql = EDGE_REFERENTIAL_INTEGRITY_QUERY.format(
            edge_table=EDGE_TABLE,
            node_table=NODE_TABLE,
            src_id_column="src_uid",
            dst_id_column="dst_uid",
            node_id_column="user_id",
        )
        self.assertIn("LEFT JOIN", sql)
        self.assertIn(f"`{NODE_TABLE}`", sql)
        self.assertIn(f"`{EDGE_TABLE}`", sql)
        self.assertIn("IS NULL", sql)


class DuplicateNodeCountQueryTest(TestCase):
    def test_contains_group_by_having(self) -> None:
        sql = DUPLICATE_NODE_COUNT_QUERY.format(table=NODE_TABLE, id_column="user_id")
        self.assertIn("GROUP BY", sql)
        self.assertIn("HAVING", sql)
        self.assertIn("user_id", sql)


class DegreeDistributionQueryTest(TestCase):
    def test_contains_approx_quantiles(self) -> None:
        sql = DEGREE_DISTRIBUTION_QUERY.format(
            table=EDGE_TABLE, id_column="src_uid"
        )
        self.assertIn("APPROX_QUANTILES", sql)
        self.assertIn("src_uid", sql)


class DegreeBucketQueryTest(TestCase):
    def test_contains_countif_buckets(self) -> None:
        sql = DEGREE_BUCKET_QUERY.format(
            table=EDGE_TABLE, id_column="src_uid"
        )
        self.assertIn("COUNTIF", sql)
        self.assertIn("src_uid", sql)


class NullRatesQueryTest(TestCase):
    def test_batches_multiple_columns(self) -> None:
        sql = build_null_rates_query(
            table=NODE_TABLE, columns=["age", "country", "embedding"]
        )
        self.assertIn(f"`{NODE_TABLE}`", sql)
        self.assertEqual(sql.count("COUNTIF"), 3)
        self.assertIn("age", sql)
        self.assertIn("country", sql)
        self.assertIn("embedding", sql)


class SuperHubInt16ClampQueryTest(TestCase):
    def test_contains_32767_threshold(self) -> None:
        sql = SUPER_HUB_INT16_CLAMP_QUERY.format(
            table=EDGE_TABLE, id_column="src_uid"
        )
        self.assertIn("32767", sql)


class TopKHubsQueryTest(TestCase):
    def test_contains_limit(self) -> None:
        sql = TOP_K_HUBS_QUERY.format(
            table=EDGE_TABLE, id_column="src_uid", k=20
        )
        self.assertIn("LIMIT 20", sql)
        self.assertIn("ORDER BY", sql)
        self.assertIn("DESC", sql)
