from gigl.analytics.data_analyzer.queries import (
    COLD_START_NODE_COUNT_QUERY,
    DANGLING_EDGES_QUERY,
    DEGREE_BUCKET_QUERY,
    DEGREE_DISTRIBUTION_QUERY,
    DUPLICATE_NODE_COUNT_QUERY,
    EDGE_REFERENTIAL_INTEGRITY_QUERY,
    NODE_COUNT_QUERY,
    SUPER_HUB_INT16_CLAMP_QUERY,
    SUPERVISION_CROSS_TABLE_QUERY,
    TOP_K_HUBS_QUERY,
    build_null_rates_query,
    build_per_class_degree_query,
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
    def test_contains_left_join_homogeneous(self) -> None:
        """Homogeneous case: src and dst resolve to the same node table."""
        sql = EDGE_REFERENTIAL_INTEGRITY_QUERY.format(
            edge_table=EDGE_TABLE,
            src_node_table=NODE_TABLE,
            dst_node_table=NODE_TABLE,
            src_id_column="src_uid",
            dst_id_column="dst_uid",
            src_node_id_column="user_id",
            dst_node_id_column="user_id",
        )
        self.assertIn("LEFT JOIN", sql)
        self.assertIn(f"`{NODE_TABLE}`", sql)
        self.assertIn(f"`{EDGE_TABLE}`", sql)
        self.assertIn("IS NULL", sql)

    def test_contains_left_join_heterogeneous(self) -> None:
        """Heterogeneous case: src and dst resolve to different node tables.

        Regression test for I3: previously the query took a single node_table
        and always joined both sides against it, producing false-positive
        missing_dst violations on bipartite graphs.
        """
        user_table = "project.dataset.user_nodes"
        content_table = "project.dataset.content_nodes"
        sql = EDGE_REFERENTIAL_INTEGRITY_QUERY.format(
            edge_table=EDGE_TABLE,
            src_node_table=user_table,
            dst_node_table=content_table,
            src_id_column="user_id",
            dst_id_column="content_id",
            src_node_id_column="uid",
            dst_node_id_column="cid",
        )
        self.assertIn(f"`{user_table}`", sql)
        self.assertIn(f"`{content_table}`", sql)
        self.assertIn("e.user_id = src_node.uid", sql)
        self.assertIn("e.content_id = dst_node.cid", sql)


class DuplicateNodeCountQueryTest(TestCase):
    def test_contains_group_by_having(self) -> None:
        sql = DUPLICATE_NODE_COUNT_QUERY.format(table=NODE_TABLE, id_column="user_id")
        self.assertIn("GROUP BY", sql)
        self.assertIn("HAVING", sql)
        self.assertIn("user_id", sql)


class DegreeDistributionQueryTest(TestCase):
    def test_contains_approx_quantiles(self) -> None:
        sql = DEGREE_DISTRIBUTION_QUERY.format(table=EDGE_TABLE, id_column="src_uid")
        self.assertIn("APPROX_QUANTILES", sql)
        self.assertIn("src_uid", sql)


class DegreeBucketQueryTest(TestCase):
    def test_contains_countif_buckets(self) -> None:
        sql = DEGREE_BUCKET_QUERY.format(table=EDGE_TABLE, id_column="src_uid")
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
        sql = SUPER_HUB_INT16_CLAMP_QUERY.format(table=EDGE_TABLE, id_column="src_uid")
        self.assertIn("32767", sql)


class TopKHubsQueryTest(TestCase):
    def test_contains_limit(self) -> None:
        sql = TOP_K_HUBS_QUERY.format(table=EDGE_TABLE, id_column="src_uid", k=20)
        self.assertIn("LIMIT 20", sql)
        self.assertIn("ORDER BY", sql)
        self.assertIn("DESC", sql)


class ColdStartNodeCountQueryTest(TestCase):
    def test_unions_src_and_dst_columns(self) -> None:
        """Cold-start is a property of total degree, not out-degree alone.

        Regression test for C2: previously the query only counted src-side
        edges, which misclassified pure-destination node types (e.g., content
        receiving likes) as cold-start regardless of in-degree.
        """
        sql = COLD_START_NODE_COUNT_QUERY.format(
            node_table=NODE_TABLE,
            edge_table=EDGE_TABLE,
            node_id_column="user_id",
            src_id_column="src_uid",
            dst_id_column="dst_uid",
        )
        self.assertIn("src_uid", sql)
        self.assertIn("dst_uid", sql)
        self.assertIn("UNION ALL", sql)
        self.assertIn(f"`{NODE_TABLE}`", sql)
        self.assertIn(f"`{EDGE_TABLE}`", sql)


class PerClassDegreeQueryTest(TestCase):
    def test_emits_six_log_buckets_for_sparkline(self) -> None:
        """Per-class query carries the same log-bucket counts as the overall
        degree query so the report can render a per-class sparkline next to
        each row using the existing histogram helper.
        """
        sql = build_per_class_degree_query(
            node_table=NODE_TABLE,
            node_id_column="user_id",
            label_column="label",
            edge_table=EDGE_TABLE,
            edge_src_column="src_uid",
            edge_dst_column="dst_uid",
        )
        for column in [
            "bucket_0_1",
            "bucket_2_10",
            "bucket_11_100",
            "bucket_101_1k",
            "bucket_1k_10k",
            "bucket_10k_plus",
        ]:
            self.assertIn(column, sql)
        # And the existing summary projection is unchanged.
        self.assertIn("class_value", sql)
        self.assertIn("APPROX_QUANTILES(degree, 100)", sql)
        self.assertIn("MAX(degree) AS max_degree", sql)
        self.assertIn("GROUP BY class_value", sql)

    def test_does_not_filter_sentinel_values(self) -> None:
        """Sentinel-labeled rows must surface as their own ``class_value`` rows so
        the caller can compute a degree distribution for them. The query no
        longer filters by ``label_sentinel_values``; partitioning happens in
        Python after the rows come back.
        """
        sql = build_per_class_degree_query(
            node_table=NODE_TABLE,
            node_id_column="user_id",
            label_column="label",
            edge_table=EDGE_TABLE,
            edge_src_column="src_uid",
            edge_dst_column="dst_uid",
        )
        self.assertNotIn("NOT IN", sql)
        self.assertNotIn("'-1'", sql)


class SupervisionCrossTableQueryTest(TestCase):
    def test_query_substitutes_all_table_and_column_placeholders(self) -> None:
        sql = SUPERVISION_CROSS_TABLE_QUERY.format(
            driver_table="project.dataset.pos_edges",
            other_table="project.dataset.neg_edges",
            driver_anchor_column="user_id",
            driver_other_column="content_id",
            other_anchor_column="user_id",
            other_other_column="content_id",
        )
        self.assertIn("`project.dataset.pos_edges`", sql)
        self.assertIn("`project.dataset.neg_edges`", sql)
        self.assertIn("user_id AS anchor", sql)
        self.assertIn("content_id  AS neighbor", sql)
        # All 10 returned columns must appear in the projection.
        for column in [
            "driver_anchor_count",
            "driver_pair_count",
            "other_pair_count",
            "overlap_pair_count",
            "driver_anchors_with_zero_other",
            "avg_other_per_driver_anchor",
            "p50_other_per_driver_anchor",
            "p90_other_per_driver_anchor",
            "p99_other_per_driver_anchor",
            "max_other_per_driver_anchor",
        ]:
            self.assertIn(column, sql)
        self.assertIn("INNER JOIN", sql)
