"""Unit tests for the NC supervision tier in GraphStructureAnalyzer.

Mocks BqUtils to exercise the orchestration logic (label sentinel
accounting, per-class degree, adjusted homophily, cross-split id-overlap
hard fail) without hitting a real BigQuery backend.
"""

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


def _make_nc_config(
    label_sentinel_values: Optional[list[str]] = None,
    split_column: Optional[str] = None,
    homophily_sample_cap: int = 0,
) -> DataAnalyzerConfig:
    return DataAnalyzerConfig(
        node_tables=[
            NodeTableSpec(
                bq_table="p.d.users",
                node_type="user",
                id_column="uid",
                feature_columns=["f1"],
                label_column="node_label",
                label_sentinel_values=label_sentinel_values or [],
                split_column=split_column,
            )
        ],
        edge_tables=[
            EdgeTableSpec(
                bq_table="p.d.edges",
                edge_type="to",
                role="message_passing",
                src_id_column="src",
                dst_id_column="dst",
                src_node_type="user",
                dst_node_type="user",
            )
        ],
        output_gcs_path="gs://bucket/out/",
        label_homophily_edge_sample_cap=homophily_sample_cap,
    )


def _mock_row(data: dict[str, Any]) -> MagicMock:
    row = MagicMock()
    row.__getitem__ = lambda self, key: data[key]
    row.keys = lambda: list(data.keys())
    for k, v in data.items():
        setattr(row, k, v)
    return row


def _mock_row_iterator(rows: list[dict[str, Any]]) -> MagicMock:
    mock = MagicMock()
    mock.__iter__ = lambda self: iter([_mock_row(r) for r in rows])
    return mock


def _default_rows_for_query(
    query: str,
    sentinel_count: int = 0,
    overlap_count: int = 0,
    include_sentinel_class_row: bool = False,
) -> list[dict[str, Any]]:
    """Reasonable fixture rows for any query the analyzer issues.

    Returns an NC-friendly fixture: balanced classes, mild homophily,
    no cross-split overlap unless ``overlap_count > 0``.

    When ``include_sentinel_class_row`` is True, the per-class degree
    fixture includes an additional ``class_value="-1"`` row so tests can
    exercise the sentinel-vs-class partition in
    ``_compute_per_class_degree``.
    """
    q = query.lower()
    if "dangling_count" in q:
        return [{"dangling_count": 0}]
    if "missing_src_count" in q:
        return [{"missing_src_count": 0, "missing_dst_count": 0}]
    if "duplicate_count" in q:
        return [{"duplicate_count": 0}]
    if "self_loop_count" in q:
        return [{"self_loop_count": 0}]
    if "isolated_count" in q:
        return [{"isolated_count": 0}]
    if "super_hub_count" in q:
        return [{"super_hub_count": 0}]
    if "cold_start_count" in q and "class_count" not in q:
        return [{"cold_start_count": 50}]
    # Per-class degree (the SQL now also carries bucket_* columns) must be
    # matched before the standalone Tier-2 bucket branch below — both share
    # ``bucket_0_1`` but only the per-class query has ``class_count``.
    if "class_count" in q and "approx_quantiles" in q:
        rows: list[dict[str, Any]] = [
            {
                "class_value": "0",
                "class_count": 600,
                "cold_start_count": 30,
                "mean_degree": 5.0,
                "percentiles": list(range(101)),
                "max_degree": 100,
                "bucket_0_1": 30,
                "bucket_2_10": 500,
                "bucket_11_100": 60,
                "bucket_101_1k": 10,
                "bucket_1k_10k": 0,
                "bucket_10k_plus": 0,
            },
            {
                "class_value": "1",
                "class_count": 200,
                "cold_start_count": 5,
                "mean_degree": 7.0,
                "percentiles": list(range(101)),
                "max_degree": 120,
                "bucket_0_1": 5,
                "bucket_2_10": 150,
                "bucket_11_100": 40,
                "bucket_101_1k": 5,
                "bucket_1k_10k": 0,
                "bucket_10k_plus": 0,
            },
        ]
        if include_sentinel_class_row:
            # Mirrors a real "-1" sentinel row coming back from BQ alongside
            # the valid classes; the parser must route it to
            # ``sentinel_degree_stats`` (not ``per_class_degree``) when "-1"
            # is declared in ``label_sentinel_values``. Concentrated
            # cold-start with a long tail mirrors the typical
            # missing-label pool.
            rows.append(
                {
                    "class_value": "-1",
                    "class_count": 50,
                    "cold_start_count": 40,
                    "mean_degree": 1.5,
                    "percentiles": list(range(101)),
                    "max_degree": 80,
                    "bucket_0_1": 40,
                    "bucket_2_10": 8,
                    "bucket_11_100": 2,
                    "bucket_101_1k": 0,
                    "bucket_1k_10k": 0,
                    "bucket_10k_plus": 0,
                }
            )
        return rows
    if "min_degree" in q or "approx_quantiles" in q and "class_count" not in q:
        # Tier 2 degree distribution.
        if "edge_homophily" in q or "labeled_pairs" in q:
            pass  # fall through — handled below
        else:
            return [
                {
                    "min_degree": 0,
                    "max_degree": 100,
                    "avg_degree": 5.0,
                    "percentiles": list(range(101)),
                }
            ]
    if "bucket_0_1" in q:
        return [
            {
                "bucket_0_1": 10,
                "bucket_2_10": 900,
                "bucket_11_100": 80,
                "bucket_101_1k": 10,
                "bucket_1k_10k": 0,
                "bucket_10k_plus": 0,
            }
        ]
    if "null_rate" in q:
        return [
            {
                "total_rows": 1000,
                "f1_null_rate": 0.0,
                "node_label_null_rate": 0.0,
                "uid_null_rate": 0.0,
            }
        ]
    if "distinct_src_count" in q:
        return [{"distinct_src_count": 900, "distinct_dst_count": 950}]
    # ``overlap_node_count`` is a substring of ``node_count`` so it must
    # be matched before the generic node-count branch below.
    if "overlap_node_count" in q:
        return [{"overlap_node_count": overlap_count}]
    if "node_count" in q:
        return [{"node_count": 1000}]
    if "edge_count" in q:
        return [{"edge_count": 5000}]
    # NC-specific queries below.
    if "valid_count" in q and "null_count" in q:
        # Sentinel-aware label query.
        row = {
            "total_rows": 1000,
            "null_count": 100,
            "valid_count": 800 - sentinel_count,
        }
        # Add up to a few sentinel_<idx> entries; tests that need them
        # fill them in via dedicated side-effects rather than this default.
        for idx in range(5):
            row[f"sentinel_{idx}"] = sentinel_count if idx == 0 else 0
        return [row]
    if "edge_homophily" in q or "labeled_pairs" in q:
        return [
            {
                "edge_homophily": 0.7,
                "expected_homophily": 0.5,
                "edge_sample_count": 4500,
            }
        ]
    if "overlap_node_count" in q:
        return [{"overlap_node_count": overlap_count}]
    if "split_value" in q and "row_count" in q:
        return [
            {"split_value": "train", "row_count": 700},
            {"split_value": "val", "row_count": 100},
            {"split_value": "test", "row_count": 100},
        ]
    if "order by degree desc" in q:
        return [
            {"node_id": "u1", "degree": 500},
            {"node_id": "u2", "degree": 400},
        ]
    if "label" in q and "count" in q and "order by count" in q:
        return [{"label": 0, "count": 600}, {"label": 1, "count": 400}]
    if "labeled" in q and "coverage" in q:
        return [{"total": 1000, "labeled": 800, "coverage": 0.8}]
    return [{"count": 0}]


@patch("gigl.analytics.data_analyzer.graph_structure_analyzer.BqUtils")
class NodeClassificationSupervisionTierTest(TestCase):
    def test_skipped_when_no_label_column(self, mock_bq_cls: MagicMock) -> None:
        config = DataAnalyzerConfig(
            node_tables=[
                NodeTableSpec(
                    bq_table="p.d.users",
                    node_type="user",
                    id_column="uid",
                )
            ],
            edge_tables=[
                EdgeTableSpec(
                    bq_table="p.d.edges",
                    edge_type="to",
                    role="message_passing",
                    src_id_column="src",
                    dst_id_column="dst",
                    src_node_type="user",
                    dst_node_type="user",
                )
            ],
            output_gcs_path="gs://bucket/out/",
        )
        mock_bq = mock_bq_cls.return_value
        mock_bq.run_query.side_effect = lambda query, labels=None: _mock_row_iterator(
            _default_rows_for_query(query)
        )
        analyzer = GraphStructureAnalyzer()
        result = analyzer.analyze(config)
        self.assertEqual(result.node_classification_supervision_stats, [])

    def test_populated_when_label_column_set(self, mock_bq_cls: MagicMock) -> None:
        mock_bq = mock_bq_cls.return_value
        mock_bq.run_query.side_effect = lambda query, labels=None: _mock_row_iterator(
            _default_rows_for_query(query)
        )
        analyzer = GraphStructureAnalyzer()
        result = analyzer.analyze(_make_nc_config())
        stats_list = result.node_classification_supervision_stats
        self.assertEqual(len(stats_list), 1)
        stats = stats_list[0]
        self.assertEqual(stats.node_type, "user")
        self.assertEqual(stats.label_column, "node_label")
        # Sentinel accounting.
        self.assertEqual(stats.sentinel_stats.total_rows, 1000)
        self.assertEqual(stats.sentinel_stats.null_count, 100)
        self.assertEqual(stats.sentinel_stats.valid_label_count, 800)
        self.assertAlmostEqual(stats.sentinel_stats.valid_label_coverage, 0.8)
        # Per-class degree.
        self.assertEqual(len(stats.per_class_degree), 2)
        self.assertEqual(stats.per_class_degree[0].class_value, "0")
        self.assertEqual(stats.per_class_degree[0].count, 600)
        # Per-class log buckets are populated for the report sparkline.
        self.assertEqual(
            sorted(stats.per_class_degree[0].buckets.keys()),
            sorted(["0-1", "2-10", "11-100", "101-1K", "1K-10K", "10K+"]),
        )
        self.assertEqual(stats.per_class_degree[0].buckets["2-10"], 500)
        self.assertEqual(stats.per_class_degree[1].buckets["11-100"], 40)
        # Rendered SQL was captured for the report's per-class disclosure.
        self.assertIn("nc_supervision:per_class_degree:user", result.queries)
        nc_sql_list = result.queries["nc_supervision:per_class_degree:user"]
        self.assertTrue(any("class_value" in q for q in nc_sql_list))
        # Homophily.
        self.assertEqual(len(stats.homophily), 1)
        self.assertAlmostEqual(stats.homophily[0].edge_homophily, 0.7)
        # adjusted = (0.7 - 0.5) / (1 - 0.5) = 0.4
        self.assertAlmostEqual(stats.homophily[0].adjusted_homophily, 0.4)
        # No split column → no cross_split_overlap.
        self.assertIsNone(stats.cross_split_overlap)

    def test_sentinel_counts_surface(self, mock_bq_cls: MagicMock) -> None:
        mock_bq = mock_bq_cls.return_value
        mock_bq.run_query.side_effect = lambda query, labels=None: _mock_row_iterator(
            _default_rows_for_query(query, sentinel_count=42)
        )
        analyzer = GraphStructureAnalyzer()
        result = analyzer.analyze(
            _make_nc_config(label_sentinel_values=["-1", "unknown"])
        )
        sentinel_stats = result.node_classification_supervision_stats[0].sentinel_stats
        # First sentinel "-1" gets sentinel_0 = 42 from the fixture; the rest are 0.
        self.assertEqual(sentinel_stats.sentinel_counts["-1"], 42)
        self.assertEqual(sentinel_stats.sentinel_counts["unknown"], 0)

    def test_sentinel_degree_stats_partitioned_from_per_class(
        self, mock_bq_cls: MagicMock
    ) -> None:
        """Rows whose label matches a declared sentinel get routed to
        ``sentinel_degree_stats``; valid-class rows stay in
        ``per_class_degree``. Both share the same ``PerClassDegreeStats``
        shape so the sentinel pool's degree distribution is read end-to-end
        the same way a "real" class is.
        """
        mock_bq = mock_bq_cls.return_value
        mock_bq.run_query.side_effect = lambda query, labels=None: _mock_row_iterator(
            _default_rows_for_query(query, include_sentinel_class_row=True)
        )
        analyzer = GraphStructureAnalyzer()
        result = analyzer.analyze(_make_nc_config(label_sentinel_values=["-1"]))
        stats = result.node_classification_supervision_stats[0]

        # Valid classes only — "-1" must NOT appear here.
        self.assertEqual(len(stats.per_class_degree), 2)
        self.assertEqual(
            sorted(c.class_value for c in stats.per_class_degree), ["0", "1"]
        )

        # Sentinel "-1" lands in its own list with full degree distribution.
        self.assertEqual(len(stats.sentinel_degree_stats), 1)
        sentinel_row = stats.sentinel_degree_stats[0]
        self.assertEqual(sentinel_row.class_value, "-1")
        self.assertEqual(sentinel_row.count, 50)
        self.assertEqual(sentinel_row.cold_start_count, 40)
        self.assertAlmostEqual(sentinel_row.mean_degree, 1.5)
        self.assertEqual(sentinel_row.max_degree, 80)
        # Buckets carry the same log-bucket keys as per_class_degree so the
        # report sparkline renders identically.
        self.assertEqual(
            sorted(sentinel_row.buckets.keys()),
            sorted(["0-1", "2-10", "11-100", "101-1K", "1K-10K", "10K+"]),
        )
        self.assertEqual(sentinel_row.buckets["0-1"], 40)

    def test_sentinel_degree_stats_empty_when_no_sentinels_declared(
        self, mock_bq_cls: MagicMock
    ) -> None:
        """When ``label_sentinel_values`` is empty, every non-NULL label row
        is treated as a real class and ``sentinel_degree_stats`` stays empty
        even if a "-1" row happens to come back from BQ.
        """
        mock_bq = mock_bq_cls.return_value
        mock_bq.run_query.side_effect = lambda query, labels=None: _mock_row_iterator(
            _default_rows_for_query(query, include_sentinel_class_row=True)
        )
        analyzer = GraphStructureAnalyzer()
        result = analyzer.analyze(_make_nc_config())
        stats = result.node_classification_supervision_stats[0]
        self.assertEqual(len(stats.sentinel_degree_stats), 0)
        # "-1" appears as a regular class row when not declared as sentinel.
        self.assertIn("-1", [c.class_value for c in stats.per_class_degree])

    def test_split_column_no_overlap_passes(self, mock_bq_cls: MagicMock) -> None:
        mock_bq = mock_bq_cls.return_value
        mock_bq.run_query.side_effect = lambda query, labels=None: _mock_row_iterator(
            _default_rows_for_query(query, overlap_count=0)
        )
        analyzer = GraphStructureAnalyzer()
        result = analyzer.analyze(_make_nc_config(split_column="split"))
        overlap = result.node_classification_supervision_stats[0].cross_split_overlap
        self.assertIsNotNone(overlap)
        assert overlap is not None
        self.assertEqual(overlap.overlap_node_count, 0)
        self.assertEqual(overlap.split_value_counts.get("train"), 700)

    def test_split_column_overlap_raises(self, mock_bq_cls: MagicMock) -> None:
        mock_bq = mock_bq_cls.return_value
        mock_bq.run_query.side_effect = lambda query, labels=None: _mock_row_iterator(
            _default_rows_for_query(query, overlap_count=17)
        )
        analyzer = GraphStructureAnalyzer()
        with self.assertRaises(DataQualityError) as ctx:
            analyzer.analyze(_make_nc_config(split_column="split"))
        partial = ctx.exception.partial_result
        self.assertEqual(len(partial.node_classification_supervision_stats), 1)
        overlap = partial.node_classification_supervision_stats[0].cross_split_overlap
        assert overlap is not None
        self.assertEqual(overlap.overlap_node_count, 17)

    def test_homophily_query_includes_sample_filter_when_capped(
        self, mock_bq_cls: MagicMock
    ) -> None:
        mock_bq = mock_bq_cls.return_value

        def _side_effect(query: str, labels: Optional[dict] = None) -> MagicMock:
            q = query.lower()
            if "edge_count" in q and "duplicate_count" not in q:
                # Force edge_count above the cap so sampling activates.
                return _mock_row_iterator([{"edge_count": 200_000}])
            return _mock_row_iterator(_default_rows_for_query(query))

        mock_bq.run_query.side_effect = _side_effect
        analyzer = GraphStructureAnalyzer()
        analyzer.analyze(_make_nc_config(homophily_sample_cap=10_000))

        homophily_queries = [
            call.kwargs["query"]
            for call in mock_bq.run_query.call_args_list
            if "edge_homophily" in call.kwargs.get("query", "")
        ]
        self.assertGreaterEqual(len(homophily_queries), 1)
        # When sample_cap > 0 and edge_count > cap, the query carries
        # a MOD(ABS(FARM_FINGERPRINT(...))) sample filter.
        for sql in homophily_queries:
            self.assertIn("FARM_FINGERPRINT", sql)
            self.assertIn("MOD(", sql)

    def test_homophily_skips_sampling_when_below_cap(
        self, mock_bq_cls: MagicMock
    ) -> None:
        mock_bq = mock_bq_cls.return_value
        mock_bq.run_query.side_effect = lambda query, labels=None: _mock_row_iterator(
            _default_rows_for_query(query)
        )
        analyzer = GraphStructureAnalyzer()
        analyzer.analyze(_make_nc_config(homophily_sample_cap=1_000_000))
        homophily_queries = [
            call.kwargs["query"]
            for call in mock_bq.run_query.call_args_list
            if "edge_homophily" in call.kwargs.get("query", "")
        ]
        self.assertGreaterEqual(len(homophily_queries), 1)
        # edge_count fixture is 5000 which is < the 1M cap; no sampling.
        for sql in homophily_queries:
            self.assertNotIn("FARM_FINGERPRINT", sql)

    def test_existing_lp_path_byte_compatible(self, mock_bq_cls: MagicMock) -> None:
        """An LP-only config (no label_column) leaves the new tier empty.

        Sanity check that adding the NC tier didn't accidentally fire on
        configs that should remain LP-only. The supervision_cross_table
        path remains driven by edge ``role`` and is exercised by
        ``graph_structure_analyzer_test.py``.
        """
        mock_bq = mock_bq_cls.return_value
        mock_bq.run_query.side_effect = lambda query, labels=None: _mock_row_iterator(
            _default_rows_for_query(query)
        )
        config = DataAnalyzerConfig(
            node_tables=[
                NodeTableSpec(
                    bq_table="p.d.users",
                    node_type="user",
                    id_column="uid",
                )
            ],
            edge_tables=[
                EdgeTableSpec(
                    bq_table="p.d.edges",
                    edge_type="to",
                    role="message_passing",
                    src_id_column="src",
                    dst_id_column="dst",
                    src_node_type="user",
                    dst_node_type="user",
                )
            ],
            output_gcs_path="gs://bucket/out/",
        )
        analyzer = GraphStructureAnalyzer()
        result = analyzer.analyze(config)
        self.assertEqual(result.node_classification_supervision_stats, [])
