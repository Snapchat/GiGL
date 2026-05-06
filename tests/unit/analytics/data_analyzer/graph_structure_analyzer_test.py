"""Unit tests for GraphStructureAnalyzer.

All BQ calls are mocked via patching BqUtils. The goal is to exercise the
orchestration logic (tier ordering, gating, result population) without hitting
a real BigQuery backend.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

from gigl.analytics.data_analyzer.config import (
    DataAnalyzerConfig,
    EdgeTableSpec,
    NodeTableSpec,
)
from gigl.analytics.data_analyzer.graph_structure_analyzer import (
    DataQualityError,
    GraphStructureAnalyzer,
)
from tests.test_assets.test_case import TestCase


def _make_config(
    label_column: Optional[str] = None,
    compute_reciprocity: bool = False,
    extra_edge: bool = False,
) -> DataAnalyzerConfig:
    edge_tables = [
        EdgeTableSpec(
            bq_table="p.d.edges",
            edge_type="follows",
            src_id_column="src",
            dst_id_column="dst",
            src_node_type="user",
            dst_node_type="user",
        )
    ]
    if extra_edge:
        edge_tables.append(
            EdgeTableSpec(
                bq_table="p.d.edges2",
                edge_type="likes",
                src_id_column="src",
                dst_id_column="dst",
                src_node_type="user",
                dst_node_type="user",
            )
        )
    return DataAnalyzerConfig(
        node_tables=[
            NodeTableSpec(
                bq_table="p.d.nodes",
                node_type="user",
                id_column="uid",
                feature_columns=["f1", "f2"],
                label_column=label_column,
            )
        ],
        edge_tables=edge_tables,
        output_gcs_path="gs://bucket/out/",
        fan_out=[15, 10],
        compute_reciprocity=compute_reciprocity,
    )


def _make_heterogeneous_config() -> DataAnalyzerConfig:
    """User -[viewed]-> content bipartite graph."""
    return DataAnalyzerConfig(
        node_tables=[
            NodeTableSpec(
                bq_table="p.d.users",
                node_type="user",
                id_column="uid",
                feature_columns=["age"],
            ),
            NodeTableSpec(
                bq_table="p.d.content",
                node_type="content",
                id_column="cid",
                feature_columns=["topic"],
            ),
        ],
        edge_tables=[
            EdgeTableSpec(
                bq_table="p.d.viewed",
                edge_type="viewed",
                src_id_column="user_id",
                dst_id_column="content_id",
                src_node_type="user",
                dst_node_type="content",
            )
        ],
        output_gcs_path="gs://bucket/out/",
    )


def _make_supervision_config(anchor_side: str = "src") -> DataAnalyzerConfig:
    """User -[viewed]-> content with one supervision_pos + supervision_neg.

    Args:
        anchor_side: ``"src"`` to anchor on user (default), ``"dst"`` to anchor on content.
    """
    if anchor_side == "src":
        node_anchor = "user"
    elif anchor_side == "dst":
        node_anchor = "content"
    else:
        raise ValueError(f"anchor_side={anchor_side!r} must be 'src' or 'dst'")
    return DataAnalyzerConfig(
        node_tables=[
            NodeTableSpec(bq_table="p.d.users", node_type="user", id_column="uid"),
            NodeTableSpec(bq_table="p.d.content", node_type="content", id_column="cid"),
        ],
        edge_tables=[
            EdgeTableSpec(
                bq_table="p.d.viewed",
                edge_type="viewed",
                role="message_passing",
                src_id_column="user_id",
                dst_id_column="content_id",
                src_node_type="user",
                dst_node_type="content",
            ),
            EdgeTableSpec(
                bq_table="p.d.viewed_pos",
                edge_type="viewed_pos",
                role="supervision_pos",
                node_anchor=node_anchor,
                src_id_column="user_id",
                dst_id_column="content_id",
                src_node_type="user",
                dst_node_type="content",
            ),
            EdgeTableSpec(
                bq_table="p.d.viewed_neg",
                edge_type="viewed_neg",
                role="supervision_neg",
                src_id_column="user_id",
                dst_id_column="content_id",
                src_node_type="user",
                dst_node_type="content",
            ),
        ],
        output_gcs_path="gs://bucket/out/",
    )


def _mock_row(data: dict[str, Any]) -> MagicMock:
    """Mock a BigQuery Row supporting both key and attribute access."""
    row = MagicMock()
    keys = list(data.keys())
    values = list(data.values())
    row.__getitem__ = lambda self, key: (
        data[key] if isinstance(key, str) else values[key]
    )
    row.keys = lambda: keys
    row.values = lambda: values
    for k, v in data.items():
        setattr(row, k, v)
    return row


def _mock_row_iterator(rows: list[dict[str, Any]]) -> MagicMock:
    """Mock a RowIterator yielding the given row dicts."""
    mock = MagicMock()
    mock.__iter__ = lambda self: iter([_mock_row(r) for r in rows])
    return mock


_DEFAULT_SUPERVISION_ROW = {
    "driver_anchor_count": 1000,
    "driver_pair_count": 5000,
    "other_pair_count": 6000,
    "overlap_pair_count": 0,
    "driver_anchors_with_zero_other": 50,
    "avg_other_per_driver_anchor": 4.5,
    "p50_other_per_driver_anchor": 4,
    "p90_other_per_driver_anchor": 12,
    "p99_other_per_driver_anchor": 40,
    "max_other_per_driver_anchor": 200,
}


def _default_row_for_query(query: str) -> dict[str, Any]:
    """Return a reasonable 'zero violation, small graph' row for any query."""
    q = query.lower()
    if "driver_anchor_count" in q:
        return dict(_DEFAULT_SUPERVISION_ROW)
    if "dangling_count" in q:
        return {"dangling_count": 0}
    if "missing_src_count" in q:
        return {"missing_src_count": 0, "missing_dst_count": 0}
    if "duplicate_count" in q:
        return {"duplicate_count": 0}
    if "node_count" in q and "distinct_src_count" not in q:
        return {"node_count": 1000}
    if "edge_count" in q:
        return {"edge_count": 5000}
    if "self_loop_count" in q:
        return {"self_loop_count": 0}
    if "isolated_count" in q:
        return {"isolated_count": 0}
    if "min_degree" in q or "approx_quantiles" in q:
        return {
            "min_degree": 0,
            "max_degree": 100,
            "avg_degree": 5.0,
            "percentiles": list(range(101)),
        }
    if "bucket_0_1" in q:
        return {
            "bucket_0_1": 10,
            "bucket_2_10": 900,
            "bucket_11_100": 80,
            "bucket_101_1k": 10,
            "bucket_1k_10k": 0,
            "bucket_10k_plus": 0,
        }
    if "super_hub_count" in q:
        return {"super_hub_count": 0}
    if "cold_start_count" in q:
        return {"cold_start_count": 50}
    if "null_rate" in q:
        # Include any plausible column name ending in _null_rate with zero default.
        # Extend this list when adding new feature columns to test configs.
        return {
            "total_rows": 1000,
            "f1_null_rate": 0.0,
            "f2_null_rate": 0.01,
            "uid_null_rate": 0.0,
            "cid_null_rate": 0.0,
            "oid_null_rate": 0.0,
            "age_null_rate": 0.0,
            "topic_null_rate": 0.0,
            "is_active_null_rate": 0.0,
        }
    if "distinct_src_count" in q:
        return {"distinct_src_count": 900, "distinct_dst_count": 950}
    # NC supervision tier sentinel-aware label query: presence of
    # ``valid_count`` is the unambiguous discriminator.
    if "valid_count" in q:
        row: dict[str, Any] = {
            "total_rows": 1000,
            "null_count": 50,
            "valid_count": 950,
        }
        for idx in range(5):
            row[f"sentinel_{idx}"] = 0
        return row
    # NC homophily query.
    if "edge_homophily" in q:
        return {
            "edge_homophily": 0.0,
            "expected_homophily": 0.0,
            "edge_sample_count": 0,
        }
    # NC cross-split id-overlap query (matches before generic ``node_count``
    # because ``overlap_node_count`` contains it as a substring).
    if "overlap_node_count" in q:
        return {"overlap_node_count": 0}
    if "labeled" in q:
        return {"total": 1000, "labeled": 800, "coverage": 0.8}
    if "label" in q and "count" in q:
        return {"label": 0, "count": 500}
    # Fallback: one zero-valued scalar
    return {"count": 0}


def _default_rows_for_query(query: str) -> list[dict[str, Any]]:
    q = query.lower()
    if "order by degree desc" in q:
        # Top-K hubs query returns multiple rows
        return [
            {"node_id": "u1", "degree": 500},
            {"node_id": "u2", "degree": 400},
        ]
    if "group by " in q and "label" in q and "order by count" in q:
        return [{"label": 0, "count": 600}, {"label": 1, "count": 400}]
    # NC per-class degree query: emit the same shape graph_structure
    # consumes, with one row per class. Distinguished by ``class_count``
    # + ``cold_start_count`` columns. The six log-bucket counts feed the
    # per-class sparkline column in the report.
    if "class_count" in q and "cold_start_count" in q:
        return [
            {
                "class_value": "0",
                "class_count": 500,
                "cold_start_count": 25,
                "mean_degree": 5.0,
                "percentiles": list(range(101)),
                "max_degree": 100,
                "bucket_0_1": 25,
                "bucket_2_10": 400,
                "bucket_11_100": 65,
                "bucket_101_1k": 10,
                "bucket_1k_10k": 0,
                "bucket_10k_plus": 0,
            },
            {
                "class_value": "1",
                "class_count": 300,
                "cold_start_count": 15,
                "mean_degree": 6.0,
                "percentiles": list(range(101)),
                "max_degree": 110,
                "bucket_0_1": 15,
                "bucket_2_10": 230,
                "bucket_11_100": 50,
                "bucket_101_1k": 5,
                "bucket_1k_10k": 0,
                "bucket_10k_plus": 0,
            },
        ]
    # NC split-value-counts query.
    if "split_value" in q and "row_count" in q:
        return [
            {"split_value": "train", "row_count": 700},
            {"split_value": "val", "row_count": 100},
            {"split_value": "test", "row_count": 100},
        ]
    return [_default_row_for_query(query)]


@patch("gigl.analytics.data_analyzer.graph_structure_analyzer.BqUtils")
class GraphStructureAnalyzerTest(TestCase):
    def test_tier1_passes_when_no_violations(self, mock_bq_cls: MagicMock) -> None:
        """With zero dangling, zero duplicates, zero referential violations, Tier 1 passes."""
        mock_bq = mock_bq_cls.return_value
        mock_bq.run_query.side_effect = lambda query, labels=None: _mock_row_iterator(
            _default_rows_for_query(query)
        )
        analyzer = GraphStructureAnalyzer()
        result = analyzer.analyze(_make_config())
        self.assertIsNotNone(result)
        self.assertEqual(result.dangling_edge_counts["follows"], 0)
        self.assertEqual(result.duplicate_node_counts["user"], 0)
        self.assertEqual(result.node_counts["user"], 1000)

    def test_dangling_edges_raises(self, mock_bq_cls: MagicMock) -> None:
        """If dangling edge query returns > 0, DataQualityError is raised."""
        mock_bq = mock_bq_cls.return_value

        def _side_effect(query: str, labels: Optional[dict] = None) -> MagicMock:
            if "dangling_count" in query:
                return _mock_row_iterator([{"dangling_count": 42}])
            return _mock_row_iterator(_default_rows_for_query(query))

        mock_bq.run_query.side_effect = _side_effect
        analyzer = GraphStructureAnalyzer()
        with self.assertRaises(DataQualityError) as ctx:
            analyzer.analyze(_make_config())
        self.assertEqual(
            ctx.exception.partial_result.dangling_edge_counts["follows"], 42
        )

    def test_duplicate_nodes_raises(self, mock_bq_cls: MagicMock) -> None:
        """If duplicate node query returns > 0, DataQualityError is raised."""
        mock_bq = mock_bq_cls.return_value

        def _side_effect(query: str, labels: Optional[dict] = None) -> MagicMock:
            q = query.lower()
            # The duplicate_node query groups on id_column with HAVING COUNT(*) > 1.
            if "duplicate_count" in q and "having count(*) > 1" in q and "uid" in q:
                return _mock_row_iterator([{"duplicate_count": 5}])
            return _mock_row_iterator(_default_rows_for_query(query))

        mock_bq.run_query.side_effect = _side_effect
        analyzer = GraphStructureAnalyzer()
        with self.assertRaises(DataQualityError):
            analyzer.analyze(_make_config())

    def test_tier3_skipped_without_label(self, mock_bq_cls: MagicMock) -> None:
        """Without label_column, class_imbalance and label_coverage dicts are empty."""
        mock_bq = mock_bq_cls.return_value
        mock_bq.run_query.side_effect = lambda query, labels=None: _mock_row_iterator(
            _default_rows_for_query(query)
        )
        analyzer = GraphStructureAnalyzer()
        result = analyzer.analyze(_make_config(label_column=None))
        self.assertEqual(result.class_imbalance, {})
        self.assertEqual(result.label_coverage, {})

    def test_tier3_populated_with_label(self, mock_bq_cls: MagicMock) -> None:
        """With label_column, class_imbalance and label_coverage are populated."""
        mock_bq = mock_bq_cls.return_value
        mock_bq.run_query.side_effect = lambda query, labels=None: _mock_row_iterator(
            _default_rows_for_query(query)
        )
        analyzer = GraphStructureAnalyzer()
        result = analyzer.analyze(_make_config(label_column="is_active"))
        self.assertIn("user", result.class_imbalance)
        self.assertIn("user", result.label_coverage)
        self.assertAlmostEqual(result.label_coverage["user"], 0.8)

    def test_tier4_skipped_when_flag_false(self, mock_bq_cls: MagicMock) -> None:
        """Without compute_reciprocity flag, reciprocity dict is empty."""
        mock_bq = mock_bq_cls.return_value
        mock_bq.run_query.side_effect = lambda query, labels=None: _mock_row_iterator(
            _default_rows_for_query(query)
        )
        analyzer = GraphStructureAnalyzer()
        result = analyzer.analyze(_make_config(compute_reciprocity=False))
        self.assertEqual(result.reciprocity, {})

    def test_feature_memory_budget_computed(self, mock_bq_cls: MagicMock) -> None:
        """feature_memory_bytes is computed from schema metadata in Python, not a BQ query."""
        mock_bq = mock_bq_cls.return_value
        mock_bq.run_query.side_effect = lambda query, labels=None: _mock_row_iterator(
            _default_rows_for_query(query)
        )
        analyzer = GraphStructureAnalyzer()
        result = analyzer.analyze(_make_config())
        self.assertIn("user", result.feature_memory_bytes)
        # 1000 nodes * 2 features * 8 bytes/float64 = 16000
        self.assertEqual(result.feature_memory_bytes["user"], 1000 * 2 * 8)

    def test_neighbor_explosion_populated(self, mock_bq_cls: MagicMock) -> None:
        """With fan_out=[15,10] and avg degree 5, explosion estimate = 15*10*5."""
        mock_bq = mock_bq_cls.return_value
        mock_bq.run_query.side_effect = lambda query, labels=None: _mock_row_iterator(
            _default_rows_for_query(query)
        )
        analyzer = GraphStructureAnalyzer()
        result = analyzer.analyze(_make_config())
        self.assertIn("follows", result.neighbor_explosion_estimate)
        self.assertGreater(result.neighbor_explosion_estimate["follows"], 0)

    def test_edge_type_distribution_populated_for_multiple_edges(
        self, mock_bq_cls: MagicMock
    ) -> None:
        """edge_type_distribution is populated when there are multiple edge types."""
        mock_bq = mock_bq_cls.return_value
        mock_bq.run_query.side_effect = lambda query, labels=None: _mock_row_iterator(
            _default_rows_for_query(query)
        )
        analyzer = GraphStructureAnalyzer()
        result = analyzer.analyze(_make_config(extra_edge=True))
        self.assertIn("follows", result.edge_type_distribution)
        self.assertIn("likes", result.edge_type_distribution)

    def test_degree_stats_bucket_keys_match_report_bucket_order(
        self, mock_bq_cls: MagicMock
    ) -> None:
        """Bucket keys must exactly match BUCKET_ORDER in report/charts.ai.js.

        Regression test for C1: previously, Python emitted lowercase 'k' keys
        (e.g., '101-1k') while the JS renderer expected uppercase 'K', causing
        the three highest buckets to silently render as zero.
        """
        mock_bq = mock_bq_cls.return_value
        mock_bq.run_query.side_effect = lambda query, labels=None: _mock_row_iterator(
            _default_rows_for_query(query)
        )
        analyzer = GraphStructureAnalyzer()
        result = analyzer.analyze(_make_config())
        self.assertIn("follows_out", result.degree_stats)
        stats = result.degree_stats["follows_out"]
        expected_bucket_keys = ["0-1", "2-10", "11-100", "101-1K", "1K-10K", "10K+"]
        self.assertEqual(list(stats.buckets.keys()), expected_bucket_keys)

    def test_cold_start_query_includes_both_src_and_dst_columns(
        self, mock_bq_cls: MagicMock
    ) -> None:
        """Cold-start must be computed from total degree (src + dst).

        Regression test for C2: previously only src-side edges were counted,
        misclassifying pure-destination nodes as cold-start.
        """
        mock_bq = mock_bq_cls.return_value
        mock_bq.run_query.side_effect = lambda query, labels=None: _mock_row_iterator(
            _default_rows_for_query(query)
        )
        analyzer = GraphStructureAnalyzer()
        analyzer.analyze(_make_config())
        cold_start_queries = [
            call.kwargs["query"]
            for call in mock_bq.run_query.call_args_list
            if "cold_start_count" in call.kwargs.get("query", "")
        ]
        self.assertGreaterEqual(len(cold_start_queries), 1)
        for sql in cold_start_queries:
            self.assertIn("src", sql)
            self.assertIn("dst", sql)
            self.assertIn("UNION ALL", sql)

    def test_query_scalar_raises_on_empty_rows(self, mock_bq_cls: MagicMock) -> None:
        """Scalar queries must fail loudly on unexpected empty results.

        Regression test for I2: previously _query_scalar silently returned 0
        when BQ returned no rows, hiding driver/auth/schema issues.
        """
        mock_bq = mock_bq_cls.return_value
        mock_bq.run_query.side_effect = lambda query, labels=None: _mock_row_iterator(
            []
        )
        analyzer = GraphStructureAnalyzer()
        with self.assertRaises(RuntimeError) as ctx:
            analyzer.analyze(_make_config())
        self.assertIn("expected exactly 1 row", str(ctx.exception))

    def test_writes_graph_structure_sidecar_on_success(
        self, mock_bq_cls: MagicMock
    ) -> None:
        """analyze() writes a Pydantic JSON sidecar to output_gcs_path."""
        mock_bq = mock_bq_cls.return_value
        mock_bq.run_query.side_effect = lambda query, labels=None: _mock_row_iterator(
            _default_rows_for_query(query)
        )
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config()
            config.output_gcs_path = tmp
            analyzer = GraphStructureAnalyzer()
            analyzer.analyze(config)

            sidecar = Path(tmp) / "graph_structure.json"
            self.assertTrue(sidecar.exists())
            payload = json.loads(sidecar.read_text())
            self.assertEqual(payload["schema_version"], "1")
            self.assertEqual(payload["component"], "graph_structure")
            self.assertIn("data", payload)
            self.assertEqual(payload["data"]["node_counts"], {"user": 1000})

    def test_writes_graph_structure_sidecar_on_tier1_failure(
        self, mock_bq_cls: MagicMock
    ) -> None:
        """On DataQualityError, the partial_result is persisted to the sidecar."""
        mock_bq = mock_bq_cls.return_value

        def _side_effect(query: str, labels: Optional[dict] = None) -> MagicMock:
            if "dangling_count" in query:
                return _mock_row_iterator([{"dangling_count": 7}])
            return _mock_row_iterator(_default_rows_for_query(query))

        mock_bq.run_query.side_effect = _side_effect
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config()
            config.output_gcs_path = tmp
            analyzer = GraphStructureAnalyzer()
            with self.assertRaises(DataQualityError):
                analyzer.analyze(config)

            sidecar = Path(tmp) / "graph_structure.json"
            self.assertTrue(sidecar.exists())
            payload = json.loads(sidecar.read_text())
            self.assertEqual(payload["data"]["dangling_edge_counts"], {"follows": 7})

    def test_supervision_cross_table_empty_when_no_pos_table(
        self, mock_bq_cls: MagicMock
    ) -> None:
        """No supervision_pos / supervision_neg → list is empty, no extra queries."""
        mock_bq = mock_bq_cls.return_value
        mock_bq.run_query.side_effect = lambda query, labels=None: _mock_row_iterator(
            _default_rows_for_query(query)
        )
        analyzer = GraphStructureAnalyzer()
        result = analyzer.analyze(_make_config())
        self.assertEqual(result.supervision_cross_table_stats, [])
        cross_table_calls = [
            call.kwargs["query"]
            for call in mock_bq.run_query.call_args_list
            if "driver_anchor_count" in call.kwargs.get("query", "")
        ]
        self.assertEqual(cross_table_calls, [])

    def test_supervision_cross_table_pairs_pos_with_neg_and_mp(
        self, mock_bq_cls: MagicMock
    ) -> None:
        """One pos + one neg + one mp → 3 jobs (pos×neg, pos×mp, neg×mp)."""
        mock_bq = mock_bq_cls.return_value
        mock_bq.run_query.side_effect = lambda query, labels=None: _mock_row_iterator(
            _default_rows_for_query(query)
        )
        config = _make_supervision_config()
        analyzer = GraphStructureAnalyzer()
        result = analyzer.analyze(config)
        # 3 pairs: pos×neg, pos×mp, neg×mp.
        self.assertEqual(len(result.supervision_cross_table_stats), 3)
        pairs = {
            (s.driver_edge_type, s.other_edge_type)
            for s in result.supervision_cross_table_stats
        }
        self.assertEqual(
            pairs,
            {
                ("viewed_pos", "viewed_neg"),
                ("viewed_pos", "viewed"),
                ("viewed_neg", "viewed"),
            },
        )
        for stats in result.supervision_cross_table_stats:
            self.assertEqual(stats.node_anchor, "user")
            self.assertEqual(stats.driver_anchor_count, 1000)
            self.assertEqual(stats.avg_other_per_driver_anchor, 4.5)

    def test_supervision_cross_table_skips_mismatched_node_types(
        self, mock_bq_cls: MagicMock
    ) -> None:
        """A pos table is only paired with neg/mp tables that share its node types."""
        mock_bq = mock_bq_cls.return_value
        mock_bq.run_query.side_effect = lambda query, labels=None: _mock_row_iterator(
            _default_rows_for_query(query)
        )
        config = _make_supervision_config()
        config.node_tables.append(
            NodeTableSpec(bq_table="p.d.other", node_type="other", id_column="oid")
        )
        config.edge_tables.append(
            EdgeTableSpec(
                bq_table="p.d.unrelated",
                edge_type="unrelated",
                role="message_passing",
                src_id_column="src",
                dst_id_column="dst",
                src_node_type="other",
                dst_node_type="other",
            )
        )
        analyzer = GraphStructureAnalyzer()
        result = analyzer.analyze(config)
        edge_types_in_results = {
            s.other_edge_type for s in result.supervision_cross_table_stats
        }
        self.assertNotIn("unrelated", edge_types_in_results)

    def test_supervision_cross_table_overlap_flagged(
        self, mock_bq_cls: MagicMock
    ) -> None:
        """When the cross-table query reports overlap > 0, the field is preserved."""
        mock_bq = mock_bq_cls.return_value

        def _side_effect(query: str, labels: Optional[dict] = None) -> MagicMock:
            if "driver_anchor_count" in query:
                row = dict(_DEFAULT_SUPERVISION_ROW)
                row["overlap_pair_count"] = 17
                return _mock_row_iterator([row])
            return _mock_row_iterator(_default_rows_for_query(query))

        mock_bq.run_query.side_effect = _side_effect
        analyzer = GraphStructureAnalyzer()
        result = analyzer.analyze(_make_supervision_config())
        for stats in result.supervision_cross_table_stats:
            self.assertEqual(stats.overlap_pair_count, 17)

    def test_supervision_cross_table_resolves_dst_anchor(
        self, mock_bq_cls: MagicMock
    ) -> None:
        """When node_anchor == dst_node_type, query uses dst_id_column for the anchor."""
        mock_bq = mock_bq_cls.return_value
        mock_bq.run_query.side_effect = lambda query, labels=None: _mock_row_iterator(
            _default_rows_for_query(query)
        )
        config = _make_supervision_config(anchor_side="dst")
        analyzer = GraphStructureAnalyzer()
        analyzer.analyze(config)
        cross_table_queries = [
            call.kwargs["query"]
            for call in mock_bq.run_query.call_args_list
            if "driver_anchor_count" in call.kwargs.get("query", "")
        ]
        self.assertGreaterEqual(len(cross_table_queries), 1)
        for sql in cross_table_queries:
            # Anchor column is content_id (dst side); neighbor column is user_id.
            self.assertIn("content_id AS anchor", sql)
            self.assertIn("user_id  AS neighbor", sql)

    def test_queries_log_populated_across_sections(
        self, mock_bq_cls: MagicMock
    ) -> None:
        """``result.queries`` carries one entry per BQ-backed report block.

        The report renderer keys disclosures off this dict, so the analyzer
        must populate it with the conventional ``<section>:<metric>:<scope>``
        block IDs the JS expects.
        """
        mock_bq = mock_bq_cls.return_value
        mock_bq.run_query.side_effect = lambda query, labels=None: _mock_row_iterator(
            _default_rows_for_query(query)
        )
        analyzer = GraphStructureAnalyzer()
        result = analyzer.analyze(_make_config(label_column="is_active"))

        # At least one block_id from each major section should be recorded.
        keys = result.queries.keys()
        self.assertTrue(
            any(k.startswith("data_quality:dangling_edges:") for k in keys),
            f"missing data_quality block_ids; got {sorted(keys)}",
        )
        self.assertTrue(
            any(k.startswith("graph_structure:degree:") for k in keys),
            f"missing graph_structure block_ids; got {sorted(keys)}",
        )
        # The label_sentinel sub-block runs for any labeled node type even
        # when no edge is explicitly tagged as message-passing — assert on
        # the broader prefix so this stays robust across fixture variations.
        self.assertTrue(
            any(k.startswith("nc_supervision:") for k in keys),
            f"missing nc_supervision block_ids; got {sorted(keys)}",
        )
        self.assertTrue(
            any(k.startswith("advanced:class_imbalance:") for k in keys),
            f"missing advanced block_ids; got {sorted(keys)}",
        )

        # Values are non-empty SQL strings.
        for sql_list in result.queries.values():
            self.assertGreater(len(sql_list), 0)
            for sql in sql_list:
                self.assertIsInstance(sql, str)
                self.assertGreater(len(sql.strip()), 0)

    def test_queries_log_persisted_on_tier1_failure(
        self, mock_bq_cls: MagicMock
    ) -> None:
        """When Tier 1 fails, the partial result must still carry recorded queries.

        The sidecar written for failed runs is the user's only debugging
        artifact, so it has to retain the rendered SQL behind the failed
        check.
        """
        mock_bq = mock_bq_cls.return_value

        def _side_effect(query: str, labels: Optional[dict] = None) -> MagicMock:
            if "dangling_count" in query:
                return _mock_row_iterator([{"dangling_count": 7}])
            return _mock_row_iterator(_default_rows_for_query(query))

        mock_bq.run_query.side_effect = _side_effect
        analyzer = GraphStructureAnalyzer()
        with self.assertRaises(DataQualityError) as ctx:
            analyzer.analyze(_make_config())
        partial = ctx.exception.partial_result
        self.assertTrue(
            any(k.startswith("data_quality:dangling_edges:") for k in partial.queries),
            f"missing dangling_edges block_id; got {sorted(partial.queries)}",
        )

    def test_heterogeneous_tier1_joins_correct_node_tables(
        self, mock_bq_cls: MagicMock
    ) -> None:
        """For hetero edges, src and dst must join against their own node tables.

        Regression test for I3: previously every edge table was joined against
        node_tables[0] on both sides, producing false-positive missing_dst
        violations for bipartite edges like user->content.
        """
        mock_bq = mock_bq_cls.return_value
        mock_bq.run_query.side_effect = lambda query, labels=None: _mock_row_iterator(
            _default_rows_for_query(query)
        )
        analyzer = GraphStructureAnalyzer()
        result = analyzer.analyze(_make_heterogeneous_config())
        self.assertEqual(result.referential_integrity_violations["viewed"], 0)
        # Inspect the referential integrity query: src joins user_nodes, dst joins content_nodes.
        ref_queries = [
            call.kwargs["query"]
            for call in mock_bq.run_query.call_args_list
            if "missing_src_count" in call.kwargs.get("query", "")
        ]
        self.assertGreaterEqual(len(ref_queries), 1)
        ref_sql = ref_queries[0]
        self.assertIn("`p.d.users`", ref_sql)
        self.assertIn("`p.d.content`", ref_sql)
        self.assertIn("e.user_id = src_node.uid", ref_sql)
        self.assertIn("e.content_id = dst_node.cid", ref_sql)
