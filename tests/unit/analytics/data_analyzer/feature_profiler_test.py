"""Unit tests for the FeatureProfiler.

Dataflow job execution is mocked: ``beam.Pipeline`` is replaced with a
dummy that records construction, and ``init_beam_pipeline_options`` is
patched so tests don't touch real GCP resources.

Type-specific projection logic (scalar-type filtering, embedding
expansion) is exercised in ``embedding_projection_test.py``; this file
focuses on how ``FeatureProfiler`` wires projection → Beam → diagnostics
→ sidecar.
"""
import itertools
import tempfile
from typing import Optional
from unittest.mock import MagicMock, patch

import apache_beam as beam
from google.cloud.bigquery import SchemaField

from gigl.analytics.data_analyzer.config import (
    DataAnalyzerConfig,
    EdgeTableSpec,
    NodeTableSpec,
)
from gigl.analytics.data_analyzer.feature_profiler import (
    FeatureProfiler,
    _collect_profile_tasks,
    _resolve_projection,
)
from gigl.analytics.data_analyzer.types import EmbeddingDiagnosticsResult, TopKEntry
from gigl.env.pipelines_config import GiglResourceConfigWrapper
from gigl.src.common.constants.components import GiGLComponents
from tests.test_assets.test_case import TestCase

# Fixed values used wherever ``FeatureProfiler.profile`` is called from these
# tests. The exact strings show up in assertions for the Dataflow job-name
# suffix, so they're constants rather than per-test fixtures.
_TEST_JOB_NAME_PREFIX = "tp"
_TEST_RUN_TIMESTAMP = "20260101-0000"


def _schema(
    fields: list[tuple[str, str, str]],
) -> dict[str, SchemaField]:
    """Build a schema dict from ``(name, field_type, mode)`` tuples."""
    return {
        name: SchemaField(name=name, field_type=field_type, mode=mode)
        for name, field_type, mode in fields
    }


def _run_profile(prof: FeatureProfiler, **kwargs):
    """Invoke ``FeatureProfiler.profile`` with the test prefix and timestamp.

    Centralizes the new required kwargs (``job_name_prefix``, ``run_timestamp``)
    so individual tests stay focused on their own setup.
    """
    return prof.profile(
        job_name_prefix=_TEST_JOB_NAME_PREFIX,
        run_timestamp=_TEST_RUN_TIMESTAMP,
        **kwargs,
    )


def _make_config(
    node_specs: Optional[list[NodeTableSpec]] = None,
    edge_specs: Optional[list[EdgeTableSpec]] = None,
    output_gcs_path: Optional[str] = None,
) -> DataAnalyzerConfig:
    # A temp dir keeps the sidecar write side-effect local to the test.
    out_path = (
        output_gcs_path
        if output_gcs_path is not None
        else (tempfile.mkdtemp(prefix="feature_profiler_test_"))
    )
    return DataAnalyzerConfig(
        node_tables=node_specs
        if node_specs is not None
        else [
            NodeTableSpec(
                bq_table="p.d.users",
                node_type="user",
                id_column="uid",
                feature_columns=["age", "country"],
            )
        ],
        edge_tables=edge_specs
        if edge_specs is not None
        else [
            EdgeTableSpec(
                bq_table="p.d.follows",
                edge_type="follows",
                src_id_column="src",
                dst_id_column="dst",
                src_node_type="user",
                dst_node_type="user",
                feature_columns=["weight"],
            )
        ],
        output_gcs_path=out_path,
    )


class ResolveProjectionTest(TestCase):
    def test_auto_infers_from_schema_when_explicit_is_empty(self) -> None:
        bq_utils = MagicMock()
        bq_utils.fetch_bq_table_schema.return_value = _schema(
            [
                ("uid", "STRING", "REQUIRED"),
                ("age", "INT64", "NULLABLE"),
                ("country", "STRING", "NULLABLE"),
            ]
        )
        result, schema_error = _resolve_projection(
            bq_table="p.d.users",
            explicit=[],
            excluded={"uid"},
            bq_utils=bq_utils,
        )
        self.assertIsNone(schema_error)
        self.assertEqual([name for name, _ in result.projection], ["age", "country"])
        self.assertEqual(result.embedding_columns, [])

    def test_honors_explicit_feature_columns(self) -> None:
        bq_utils = MagicMock()
        bq_utils.fetch_bq_table_schema.return_value = _schema(
            [
                ("uid", "STRING", "REQUIRED"),
                ("age", "INT64", "NULLABLE"),
                ("country", "STRING", "NULLABLE"),
                ("extra_feature", "FLOAT64", "NULLABLE"),
            ]
        )
        result, schema_error = _resolve_projection(
            bq_table="p.d.users",
            explicit=["age", "country"],
            excluded={"uid"},
            bq_utils=bq_utils,
        )
        self.assertIsNone(schema_error)
        self.assertEqual([name for name, _ in result.projection], ["age", "country"])

    def test_logs_and_drops_explicit_columns_not_in_schema(self) -> None:
        bq_utils = MagicMock()
        bq_utils.fetch_bq_table_schema.return_value = _schema(
            [("age", "INT64", "NULLABLE")]
        )
        with self.assertLogs(level="WARNING") as cap:
            result, schema_error = _resolve_projection(
                bq_table="p.d.users",
                explicit=["age", "phantom"],
                excluded=set(),
                bq_utils=bq_utils,
            )
        self.assertIsNone(schema_error)
        self.assertEqual([name for name, _ in result.projection], ["age"])
        self.assertTrue(
            any("phantom" in msg for msg in cap.output),
            f"expected warning about phantom, got {cap.output}",
        )

    def test_schema_fetch_failure_returns_empty_projection(self) -> None:
        bq_utils = MagicMock()
        bq_utils.fetch_bq_table_schema.side_effect = RuntimeError("permission denied")
        with self.assertLogs(level="WARNING"):
            result, schema_error = _resolve_projection(
                bq_table="p.d.missing",
                explicit=[],
                excluded=set(),
                bq_utils=bq_utils,
            )
        self.assertEqual(result.projection, [])
        self.assertEqual(result.embedding_columns, [])
        self.assertIsNotNone(schema_error)
        self.assertIn("permission denied", schema_error)
        self.assertIn("p.d.missing", schema_error)

    def test_bool_scalar_columns_are_cast_to_int64(self) -> None:
        """BOOL/BOOLEAN scalars must be CAST to INT64 in the projection.

        ``BqTableToRecordBatch`` wraps each scalar value in a single-element
        list before TFDV consumes it. TFDV's
        ``get_feature_type_from_arrow_type`` rejects ``list<bool>``; passing
        a raw BOOL crashes the Dataflow job in BasicStatsGenerator. Casting
        to INT64 keeps the BOOL semantics profileable as an int feature.
        """
        bq_utils = MagicMock()
        bq_utils.fetch_bq_table_schema.return_value = _schema(
            [
                ("uid", "STRING", "REQUIRED"),
                ("is_active", "BOOL", "NULLABLE"),
                ("flagged", "BOOLEAN", "NULLABLE"),
                ("age", "INT64", "NULLABLE"),
            ]
        )
        result, schema_error = _resolve_projection(
            bq_table="p.d.users",
            explicit=[],
            excluded={"uid"},
            bq_utils=bq_utils,
        )
        self.assertIsNone(schema_error)
        by_name = dict(result.projection)
        self.assertEqual(by_name["is_active"], "CAST(`is_active` AS INT64)")
        self.assertEqual(by_name["flagged"], "CAST(`flagged` AS INT64)")
        # Non-BOOL scalars retain their pass-through expression.
        self.assertEqual(by_name["age"], "`age`")

    def test_embedding_column_expands_into_hygiene_companions(self) -> None:
        bq_utils = MagicMock()
        bq_utils.fetch_bq_table_schema.return_value = _schema(
            [
                ("uid", "STRING", "REQUIRED"),
                ("emb", "FLOAT64", "REPEATED"),
            ]
        )
        result, schema_error = _resolve_projection(
            bq_table="p.d.users",
            explicit=[],
            excluded={"uid"},
            bq_utils=bq_utils,
        )
        self.assertIsNone(schema_error)
        self.assertEqual(
            [name for name, _ in result.projection],
            ["emb_len", "emb_has_nan", "emb_has_inf", "emb_is_all_zero"],
        )
        self.assertEqual(result.embedding_columns, ["emb"])
        # The three boolean hygiene companions must be CAST to INT64;
        # otherwise TFDV crashes on list<bool>.
        by_name = dict(result.projection)
        self.assertIn("CAST", by_name["emb_has_nan"])
        self.assertIn("AS INT64", by_name["emb_has_nan"])
        self.assertIn("CAST", by_name["emb_has_inf"])
        self.assertIn("AS INT64", by_name["emb_has_inf"])
        self.assertIn("CAST", by_name["emb_is_all_zero"])
        self.assertIn("AS INT64", by_name["emb_is_all_zero"])
        # _len stays as a plain ARRAY_LENGTH (already INT64).
        self.assertNotIn("CAST", by_name["emb_len"])


class CollectProfileTasksTest(TestCase):
    def test_preserves_explicit_feature_columns_and_skips_inference(self) -> None:
        config = _make_config()
        bq_utils = MagicMock()
        bq_utils.fetch_bq_table_schema.side_effect = lambda table: _schema(
            [
                ("uid", "STRING", "REQUIRED"),
                ("age", "INT64", "NULLABLE"),
                ("country", "STRING", "NULLABLE"),
                ("src", "STRING", "REQUIRED"),
                ("dst", "STRING", "REQUIRED"),
                ("weight", "FLOAT64", "NULLABLE"),
            ]
        )
        tasks, errors = _collect_profile_tasks(config, bq_utils)
        self.assertEqual(errors, [])
        by_key = {t.result_key: t for t in tasks}
        self.assertEqual(
            [name for name, _ in by_key["node:user"].projection], ["age", "country"]
        )
        self.assertEqual(
            [name for name, _ in by_key["edge:follows"].projection], ["weight"]
        )
        self.assertEqual(by_key["node:user"].embedding_columns, [])

    def test_infers_columns_when_feature_columns_empty(self) -> None:
        config = _make_config(
            node_specs=[
                NodeTableSpec(
                    bq_table="p.d.users",
                    node_type="user",
                    id_column="uid",
                    feature_columns=[],
                )
            ],
            edge_specs=[
                EdgeTableSpec(
                    bq_table="p.d.follows",
                    edge_type="follows",
                    src_id_column="src",
                    dst_id_column="dst",
                    src_node_type="user",
                    dst_node_type="user",
                    feature_columns=[],
                )
            ],
        )
        bq_utils = MagicMock()
        bq_utils.fetch_bq_table_schema.side_effect = lambda table: {
            "p.d.users": _schema(
                [
                    ("uid", "STRING", "REQUIRED"),
                    ("age", "INT64", "NULLABLE"),
                    ("country", "STRING", "NULLABLE"),
                ]
            ),
            "p.d.follows": _schema(
                [
                    ("src", "STRING", "REQUIRED"),
                    ("dst", "STRING", "REQUIRED"),
                    ("weight", "FLOAT64", "NULLABLE"),
                ]
            ),
        }[table]
        tasks, errors = _collect_profile_tasks(config, bq_utils)
        self.assertEqual(errors, [])
        by_key = {t.result_key: t for t in tasks}
        self.assertEqual(
            [name for name, _ in by_key["node:user"].projection], ["age", "country"]
        )
        self.assertEqual(
            [name for name, _ in by_key["edge:follows"].projection], ["weight"]
        )

    def test_embedding_columns_surface_on_task(self) -> None:
        config = _make_config(
            node_specs=[
                NodeTableSpec(
                    bq_table="p.d.users",
                    node_type="user",
                    id_column="uid",
                    feature_columns=[],
                )
            ],
            edge_specs=[],
        )
        bq_utils = MagicMock()
        bq_utils.fetch_bq_table_schema.return_value = _schema(
            [
                ("uid", "STRING", "REQUIRED"),
                ("age", "INT64", "NULLABLE"),
                ("emb", "FLOAT64", "REPEATED"),
            ]
        )
        tasks, errors = _collect_profile_tasks(config, bq_utils)
        self.assertEqual(errors, [])
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task.embedding_columns, ["emb"])
        # age + four hygiene companions for emb.
        self.assertEqual(
            [name for name, _ in task.projection],
            ["age", "emb_len", "emb_has_nan", "emb_has_inf", "emb_is_all_zero"],
        )

    def test_skips_table_when_resolved_projection_is_empty(self) -> None:
        config = _make_config(
            node_specs=[
                NodeTableSpec(
                    bq_table="p.d.users",
                    node_type="user",
                    id_column="uid",
                    feature_columns=[],
                )
            ],
            edge_specs=[],
        )
        bq_utils = MagicMock()
        bq_utils.fetch_bq_table_schema.return_value = _schema(
            [("uid", "STRING", "REQUIRED")]
        )
        with self.assertLogs(level="WARNING") as log_capture:
            tasks, errors = _collect_profile_tasks(config, bq_utils)
        self.assertEqual(tasks, [])
        self.assertTrue(
            any("node:user" in msg for msg in log_capture.output),
            f"expected warning mentioning node:user, got {log_capture.output}",
        )
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].result_key, "node:user")
        self.assertEqual(errors[0].stage, "empty_projection")
        self.assertEqual(errors[0].bq_table, "p.d.users")

    def test_schema_fetch_failure_skips_table_without_crashing(self) -> None:
        config = _make_config(
            node_specs=[
                NodeTableSpec(
                    bq_table="p.d.broken",
                    node_type="broken",
                    id_column="uid",
                    feature_columns=[],
                ),
                NodeTableSpec(
                    bq_table="p.d.users",
                    node_type="user",
                    id_column="uid",
                    feature_columns=[],
                ),
            ],
            edge_specs=[],
        )
        bq_utils = MagicMock()

        def _maybe_raise(table: str):
            if table == "p.d.broken":
                raise RuntimeError("permission denied")
            return _schema(
                [
                    ("uid", "STRING", "REQUIRED"),
                    ("age", "INT64", "NULLABLE"),
                ]
            )

        bq_utils.fetch_bq_table_schema.side_effect = _maybe_raise
        with self.assertLogs(level="WARNING") as log_capture:
            tasks, errors = _collect_profile_tasks(config, bq_utils)
        self.assertEqual([t.result_key for t in tasks], ["node:user"])
        self.assertTrue(
            any("broken" in msg for msg in log_capture.output),
            f"expected warning mentioning broken table, got {log_capture.output}",
        )
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].result_key, "node:broken")
        self.assertEqual(errors[0].stage, "schema_fetch")
        self.assertIn("permission denied", errors[0].message)


class FeatureProfilerRunTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._resource_config = MagicMock(spec=GiglResourceConfigWrapper)
        self._resource_config.project = "test-project"

        self._gcp_options = MagicMock(name="GoogleCloudOptions")
        self._gcp_options.job_name = "test-job-name"
        self._gcp_options.project = "test-project"
        self._gcp_options.region = "us-central1"

        pipeline_options = MagicMock(name="PipelineOptions")
        pipeline_options.view_as = MagicMock(return_value=self._gcp_options)

        self._init_beam_pipeline_options = patch(
            "gigl.analytics.data_analyzer.feature_profiler.init_beam_pipeline_options",
            return_value=pipeline_options,
        ).start()

        self._bq_utils_cls = patch(
            "gigl.analytics.data_analyzer.feature_profiler.BqUtils",
        ).start()
        self._bq_utils_cls.return_value.fetch_bq_table_schema.return_value = {}

        self._diagnostics_cls = patch(
            "gigl.analytics.data_analyzer.feature_profiler.EmbeddingDiagnostics",
        ).start()
        self._diagnostics_cls.return_value.analyze.return_value = {}

        self._pipelines: list[MagicMock] = []

        def _make_pipeline(*args, **kwargs):
            pipeline = MagicMock(name="Pipeline")
            pipeline_result = MagicMock(name="PipelineResult")
            pipeline_result.wait_until_finish = MagicMock(return_value=None)
            pipeline_result.job_id = MagicMock(return_value="test-job-id")
            pipeline.run = MagicMock(return_value=pipeline_result)
            self._pipelines.append(pipeline)
            return pipeline

        self._pipeline_ctor = patch(
            "gigl.analytics.data_analyzer.feature_profiler.beam.Pipeline",
            side_effect=_make_pipeline,
        ).start()

        self.addCleanup(patch.stopall)

    def test_returns_empty_when_inferred_columns_are_all_ids(self) -> None:
        config = _make_config(
            node_specs=[
                NodeTableSpec(
                    bq_table="p.d.users",
                    node_type="user",
                    id_column="uid",
                    feature_columns=[],
                )
            ],
            edge_specs=[
                EdgeTableSpec(
                    bq_table="p.d.follows",
                    edge_type="follows",
                    src_id_column="src",
                    dst_id_column="dst",
                    src_node_type="user",
                    dst_node_type="user",
                    feature_columns=[],
                )
            ],
        )
        self._bq_utils_cls.return_value.fetch_bq_table_schema.side_effect = (
            lambda table: {
                "p.d.users": _schema([("uid", "STRING", "REQUIRED")]),
                "p.d.follows": _schema(
                    [
                        ("src", "STRING", "REQUIRED"),
                        ("dst", "STRING", "REQUIRED"),
                    ]
                ),
            }[table]
        )
        profiler = FeatureProfiler()
        result = _run_profile(
            profiler, config=config, resource_config=self._resource_config
        )
        self.assertEqual(result.facets_html_paths, {})
        self.assertEqual(result.stats_paths, {})
        self.assertEqual(len(self._pipelines), 0)

    def test_launches_one_pipeline_per_feature_table(self) -> None:
        config = _make_config()
        self._bq_utils_cls.return_value.fetch_bq_table_schema.side_effect = (
            lambda table: _schema(
                [
                    ("uid", "STRING", "REQUIRED"),
                    ("age", "INT64", "NULLABLE"),
                    ("country", "STRING", "NULLABLE"),
                    ("src", "STRING", "REQUIRED"),
                    ("dst", "STRING", "REQUIRED"),
                    ("weight", "FLOAT64", "NULLABLE"),
                ]
            )
        )
        profiler = FeatureProfiler()
        result = _run_profile(
            profiler, config=config, resource_config=self._resource_config
        )
        self.assertEqual(len(self._pipelines), 2)
        self.assertEqual(
            sorted(result.facets_html_paths.keys()),
            ["edge:follows", "node:user"],
        )
        self.assertEqual(
            sorted(result.stats_paths.keys()),
            ["edge:follows", "node:user"],
        )
        component_kwargs = [
            call.kwargs.get("component")
            for call in self._init_beam_pipeline_options.call_args_list
        ]
        self.assertTrue(all(c == GiGLComponents.DataAnalyzer for c in component_kwargs))

    def test_gcs_paths_use_expected_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._bq_utils_cls.return_value.fetch_bq_table_schema.side_effect = (
                lambda table: _schema(
                    [
                        ("uid", "STRING", "REQUIRED"),
                        ("age", "INT64", "NULLABLE"),
                        ("country", "STRING", "NULLABLE"),
                        ("src", "STRING", "REQUIRED"),
                        ("dst", "STRING", "REQUIRED"),
                        ("weight", "FLOAT64", "NULLABLE"),
                    ]
                )
            )
            profiler = FeatureProfiler()
            result = _run_profile(
                profiler,
                config=_make_config(output_gcs_path=f"{tmp}/run1/"),
                resource_config=self._resource_config,
            )
        # Single-chunk tables produce a list of length 1 with the historical
        # flat path (no chunk_NN/ subdir).
        self.assertEqual(
            result.facets_html_paths["node:user"],
            [f"{tmp}/run1/feature_profiler/nodes/user/facets.html"],
        )
        self.assertEqual(
            result.stats_paths["node:user"],
            [f"{tmp}/run1/feature_profiler/nodes/user/stats.tfrecord"],
        )
        self.assertEqual(
            result.facets_html_paths["edge:follows"],
            [f"{tmp}/run1/feature_profiler/edges/follows/facets.html"],
        )

    def test_individual_pipeline_failure_is_caught(self) -> None:
        self._bq_utils_cls.return_value.fetch_bq_table_schema.side_effect = (
            lambda table: _schema(
                [
                    ("uid", "STRING", "REQUIRED"),
                    ("age", "INT64", "NULLABLE"),
                    ("country", "STRING", "NULLABLE"),
                    ("src", "STRING", "REQUIRED"),
                    ("dst", "STRING", "REQUIRED"),
                    ("weight", "FLOAT64", "NULLABLE"),
                ]
            )
        )
        counter = itertools.count(1)

        def _make_pipeline_fail_second(*args, **kwargs):
            pipeline = MagicMock(name="Pipeline")
            pipeline_result = MagicMock(name="PipelineResult")
            pipeline_result.job_id = MagicMock(return_value="test-job-id")
            if next(counter) == 2:
                pipeline_result.wait_until_finish = MagicMock(
                    side_effect=RuntimeError("Dataflow boom")
                )
            else:
                pipeline_result.wait_until_finish = MagicMock(return_value=None)
            pipeline.run = MagicMock(return_value=pipeline_result)
            self._pipelines.append(pipeline)
            return pipeline

        self._pipeline_ctor.side_effect = _make_pipeline_fail_second

        profiler = FeatureProfiler()
        result = _run_profile(
            profiler,
            config=_make_config(),
            resource_config=self._resource_config,
        )
        self.assertEqual(len(self._pipelines), 2)
        total_keys = set(result.facets_html_paths.keys())
        self.assertEqual(len(total_keys), 1)
        self.assertLessEqual(total_keys, {"node:user", "edge:follows"})
        # The failed pipeline shows up as a structured error, so the report can
        # surface it instead of silently dropping the table.
        self.assertEqual(len(result.errors), 1)
        err = result.errors[0]
        self.assertIn(err.result_key, {"node:user", "edge:follows"})
        self.assertEqual(err.stage, "dataflow")
        self.assertIn("Dataflow boom", err.message)
        # The Dataflow job id and console URL should be captured so the report
        # can deep-link to the failed job's logs.
        self.assertEqual(err.job_id, "test-job-id")
        self.assertEqual(err.job_name, "test-job-name")
        self.assertIsNotNone(err.console_url)
        self.assertIn("test-job-id", err.console_url)
        self.assertIn("us-central1", err.console_url)
        self.assertIn("test-project", err.console_url)

    def test_embedding_diagnostics_failure_is_recorded(self) -> None:
        """When the diagnostics pass raises, every requesting table gets an error."""
        self._bq_utils_cls.return_value.fetch_bq_table_schema.side_effect = (
            lambda table: _schema(
                [
                    ("uid", "STRING", "REQUIRED"),
                    ("emb", "FLOAT64", "REPEATED"),
                    ("src", "STRING", "REQUIRED"),
                    ("dst", "STRING", "REQUIRED"),
                    ("weight_emb", "FLOAT64", "REPEATED"),
                ]
            )
        )
        self._diagnostics_cls.return_value.analyze.side_effect = RuntimeError(
            "diagnostics down"
        )
        profiler = FeatureProfiler()
        result = _run_profile(
            profiler,
            config=_make_config(
                node_specs=[
                    NodeTableSpec(
                        bq_table="p.d.users",
                        node_type="user",
                        id_column="uid",
                        feature_columns=[],
                    )
                ],
                edge_specs=[
                    EdgeTableSpec(
                        bq_table="p.d.follows",
                        edge_type="follows",
                        src_id_column="src",
                        dst_id_column="dst",
                        src_node_type="user",
                        dst_node_type="user",
                        feature_columns=[],
                    )
                ],
            ),
            resource_config=self._resource_config,
        )
        diagnostics_errors = [
            e for e in result.errors if e.stage == "embedding_diagnostics"
        ]
        self.assertEqual(
            sorted(e.result_key for e in diagnostics_errors),
            ["edge:follows", "node:user"],
        )
        for e in diagnostics_errors:
            self.assertIn("diagnostics down", e.message)

    def test_uses_data_analyzer_job_name_suffix(self) -> None:
        self._bq_utils_cls.return_value.fetch_bq_table_schema.side_effect = (
            lambda table: _schema(
                [
                    ("uid", "STRING", "REQUIRED"),
                    ("age", "INT64", "NULLABLE"),
                    ("country", "STRING", "NULLABLE"),
                    ("src", "STRING", "REQUIRED"),
                    ("dst", "STRING", "REQUIRED"),
                    ("weight", "FLOAT64", "NULLABLE"),
                ]
            )
        )
        profiler = FeatureProfiler()
        _run_profile(
            profiler,
            config=_make_config(),
            resource_config=self._resource_config,
        )
        suffixes = {
            call.kwargs.get("job_name_suffix")
            for call in self._init_beam_pipeline_options.call_args_list
        }
        self.assertEqual(
            suffixes,
            {
                f"{_TEST_JOB_NAME_PREFIX}-{_TEST_RUN_TIMESTAMP}-profile-node-user",
                f"{_TEST_JOB_NAME_PREFIX}-{_TEST_RUN_TIMESTAMP}-profile-edge-follows",
            },
        )
        # Regression-protect the rename from "data-analyzer" → "analyzer";
        # the static prefix lives in the applied_task_identifier kwarg.
        identifiers = {
            call.kwargs.get("applied_task_identifier")
            for call in self._init_beam_pipeline_options.call_args_list
        }
        self.assertEqual(identifiers, {"analyzer"})

    def test_runs_embedding_diagnostics_for_tables_with_embeddings(self) -> None:
        config = _make_config(
            node_specs=[
                NodeTableSpec(
                    bq_table="p.d.users",
                    node_type="user",
                    id_column="uid",
                    feature_columns=[],
                )
            ],
            edge_specs=[],
        )
        self._bq_utils_cls.return_value.fetch_bq_table_schema.return_value = _schema(
            [
                ("uid", "STRING", "REQUIRED"),
                ("age", "INT64", "NULLABLE"),
                ("emb", "FLOAT64", "REPEATED"),
            ]
        )
        self._diagnostics_cls.return_value.analyze.return_value = {
            "node:user": {
                "emb": EmbeddingDiagnosticsResult(
                    total=100,
                    unique_count=98,
                    unique_ratio=0.98,
                    top_k=[TopKEntry(hash=1, count=2, fraction=0.02)],
                )
            }
        }
        profiler = FeatureProfiler()
        result = _run_profile(
            profiler, config=config, resource_config=self._resource_config
        )

        # Diagnostics invoked once with a single request for the embedding column.
        self._diagnostics_cls.return_value.analyze.assert_called_once()
        call_arg = self._diagnostics_cls.return_value.analyze.call_args[0][0]
        self.assertEqual(len(call_arg), 1)
        self.assertEqual(call_arg[0].result_key, "node:user")
        self.assertEqual(call_arg[0].embedding_columns, ["emb"])

        self.assertIn("node:user", result.embedding_diagnostics)
        self.assertEqual(
            result.embedding_diagnostics["node:user"]["emb"].unique_ratio, 0.98
        )

    def test_skips_embedding_diagnostics_when_no_embeddings(self) -> None:
        config = _make_config()
        self._bq_utils_cls.return_value.fetch_bq_table_schema.side_effect = (
            lambda table: _schema(
                [
                    ("uid", "STRING", "REQUIRED"),
                    ("age", "INT64", "NULLABLE"),
                    ("country", "STRING", "NULLABLE"),
                    ("src", "STRING", "REQUIRED"),
                    ("dst", "STRING", "REQUIRED"),
                    ("weight", "FLOAT64", "NULLABLE"),
                ]
            )
        )
        profiler = FeatureProfiler()
        _run_profile(profiler, config=config, resource_config=self._resource_config)
        self._diagnostics_cls.return_value.analyze.assert_not_called()

    def test_writes_feature_profile_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._bq_utils_cls.return_value.fetch_bq_table_schema.side_effect = (
                lambda table: _schema(
                    [
                        ("uid", "STRING", "REQUIRED"),
                        ("age", "INT64", "NULLABLE"),
                        ("country", "STRING", "NULLABLE"),
                        ("src", "STRING", "REQUIRED"),
                        ("dst", "STRING", "REQUIRED"),
                        ("weight", "FLOAT64", "NULLABLE"),
                    ]
                )
            )
            profiler = FeatureProfiler()
            _run_profile(
                profiler,
                config=_make_config(output_gcs_path=tmp),
                resource_config=self._resource_config,
            )
            from pathlib import Path

            sidecar = Path(tmp) / "feature_profile.json"
            self.assertTrue(sidecar.exists(), f"{sidecar} was not written")
            import json

            payload = json.loads(sidecar.read_text())
            self.assertEqual(payload["schema_version"], "1")
            self.assertEqual(payload["component"], "feature_profile")
            self.assertIn("data", payload)

    def test_forwards_preprocessor_sizing_to_beam_options(self) -> None:
        """Node tasks pull sizing from ``node_preprocessor_config``; edge tasks from ``edge_preprocessor_config``.

        The analyzer reuses the preprocessor's Dataflow sizing on the
        same kind of table (node vs edge) rather than declaring its own
        block. Falls out of mirroring the
        ``data_preprocessor.lib.transform.utils.transform_features``
        pattern; verified by checking the kwargs passed to
        ``init_beam_pipeline_options`` for each profiled table.
        """
        node_block = self._resource_config.preprocessor_config.node_preprocessor_config
        node_block.machine_type = "n2d-highmem-64"
        node_block.num_workers = 4
        node_block.max_num_workers = 128
        node_block.disk_size_gb = 300
        node_block.timeout = 10800

        edge_block = self._resource_config.preprocessor_config.edge_preprocessor_config
        edge_block.machine_type = "n2d-highmem-32"
        edge_block.num_workers = 1
        edge_block.max_num_workers = 64
        edge_block.disk_size_gb = 200
        edge_block.timeout = 0  # falsy timeout collapses to None

        self._bq_utils_cls.return_value.fetch_bq_table_schema.side_effect = (
            lambda table: _schema(
                [
                    ("uid", "STRING", "REQUIRED"),
                    ("age", "INT64", "NULLABLE"),
                    ("country", "STRING", "NULLABLE"),
                    ("src", "STRING", "REQUIRED"),
                    ("dst", "STRING", "REQUIRED"),
                    ("weight", "FLOAT64", "NULLABLE"),
                ]
            )
        )

        profiler = FeatureProfiler()
        _run_profile(
            profiler, config=_make_config(), resource_config=self._resource_config
        )

        node_suffix = f"{_TEST_JOB_NAME_PREFIX}-{_TEST_RUN_TIMESTAMP}-profile-node-user"
        edge_suffix = (
            f"{_TEST_JOB_NAME_PREFIX}-{_TEST_RUN_TIMESTAMP}-profile-edge-follows"
        )
        calls_by_suffix = {
            call.kwargs.get("job_name_suffix"): call
            for call in self._init_beam_pipeline_options.call_args_list
        }
        self.assertIn(node_suffix, calls_by_suffix)
        self.assertIn(edge_suffix, calls_by_suffix)

        node_kwargs = calls_by_suffix[node_suffix].kwargs
        self.assertEqual(node_kwargs["machine_type"], "n2d-highmem-64")
        self.assertEqual(node_kwargs["num_workers"], 4)
        self.assertEqual(node_kwargs["max_num_workers"], 128)
        self.assertEqual(node_kwargs["disk_size_gb"], 300)
        self.assertEqual(node_kwargs["timeout_seconds"], 10800)

        edge_kwargs = calls_by_suffix[edge_suffix].kwargs
        self.assertEqual(edge_kwargs["machine_type"], "n2d-highmem-32")
        self.assertEqual(edge_kwargs["num_workers"], 1)
        self.assertEqual(edge_kwargs["max_num_workers"], 64)
        self.assertEqual(edge_kwargs["disk_size_gb"], 200)
        self.assertIsNone(edge_kwargs["timeout_seconds"])

    def test_passes_sharded_read_config_keyed_per_table_kind(self) -> None:
        """Node tasks shard on ``id_column``; edge tasks shard on ``src_id_column``.

        The ``BigQueryShardedReadConfig`` carries the shard key and points
        at ``resource_config.temp_assets_bq_dataset_name`` for the BQ
        export temp dataset, mirroring the data_preprocessor's
        ``ShardedExportRead`` pattern.
        """
        self._resource_config.temp_assets_bq_dataset_name = "gigl_temp_assets"

        self._bq_utils_cls.return_value.fetch_bq_table_schema.side_effect = (
            lambda table: _schema(
                [
                    ("uid", "STRING", "REQUIRED"),
                    ("age", "INT64", "NULLABLE"),
                    ("country", "STRING", "NULLABLE"),
                    ("src", "STRING", "REQUIRED"),
                    ("dst", "STRING", "REQUIRED"),
                    ("weight", "FLOAT64", "NULLABLE"),
                ]
            )
        )

        bq_to_record_batch_calls: list[dict] = []

        class _FakeBqTableToRecordBatch(beam.PTransform):
            def __init__(self, **kwargs):
                super().__init__()
                bq_to_record_batch_calls.append(kwargs)

            def expand(
                self, pbegin
            ):  # pragma: no cover - never expanded; pipeline is mocked
                return pbegin

        with patch(
            "gigl.analytics.data_analyzer.feature_profiler.BqTableToRecordBatch",
            _FakeBqTableToRecordBatch,
        ):
            profiler = FeatureProfiler()
            _run_profile(
                profiler, config=_make_config(), resource_config=self._resource_config
            )

        calls_by_table = {
            kwargs["bq_table"]: kwargs for kwargs in bq_to_record_batch_calls
        }
        self.assertIn("p.d.users", calls_by_table)
        self.assertIn("p.d.follows", calls_by_table)

        node_config = calls_by_table["p.d.users"]["sharded_read_config"]
        self.assertEqual(node_config.shard_key, "uid")
        self.assertEqual(node_config.project_id, "test-project")
        self.assertEqual(node_config.temp_dataset_name, "gigl_temp_assets")
        self.assertEqual(node_config.num_shards, 20)

        edge_config = calls_by_table["p.d.follows"]["sharded_read_config"]
        self.assertEqual(edge_config.shard_key, "src")
        self.assertEqual(edge_config.project_id, "test-project")
        self.assertEqual(edge_config.temp_dataset_name, "gigl_temp_assets")
        self.assertEqual(edge_config.num_shards, 20)


class ChunkedProjectionTest(TestCase):
    """Wide projections split into multiple per-chunk Dataflow pipelines.

    Beam 2.56's runner-v2 cannot reliably iterate the per-key state TFDV's
    ``CombinePerKey(PreCombineFn)`` accumulates over very wide projections.
    The profiler chunks each table's projection into ≤ ``max_features_per_chunk``
    pieces and emits one ``_ProfileTask`` per chunk, with ``chunk_NN/`` GCS
    artifact subdirs and chunk-aware Dataflow job names.
    """

    def setUp(self) -> None:
        super().setUp()
        self._resource_config = MagicMock(spec=GiglResourceConfigWrapper)
        self._resource_config.project = "test-project"
        self._resource_config.temp_assets_bq_dataset_name = "gigl_temp_assets"

        self._gcp_options = MagicMock(name="GoogleCloudOptions")
        self._gcp_options.job_name = "test-job-name"
        self._gcp_options.project = "test-project"
        self._gcp_options.region = "us-central1"

        pipeline_options = MagicMock(name="PipelineOptions")
        pipeline_options.view_as = MagicMock(return_value=self._gcp_options)

        self._init_beam_pipeline_options = patch(
            "gigl.analytics.data_analyzer.feature_profiler.init_beam_pipeline_options",
            return_value=pipeline_options,
        ).start()

        self._bq_utils_cls = patch(
            "gigl.analytics.data_analyzer.feature_profiler.BqUtils",
        ).start()
        self._bq_utils_cls.return_value.fetch_bq_table_schema.return_value = {}

        self._diagnostics_cls = patch(
            "gigl.analytics.data_analyzer.feature_profiler.EmbeddingDiagnostics",
        ).start()
        self._diagnostics_cls.return_value.analyze.return_value = {}

        self._pipelines: list[MagicMock] = []

        def _make_pipeline(*args, **kwargs):
            pipeline = MagicMock(name="Pipeline")
            pipeline_result = MagicMock(name="PipelineResult")
            pipeline_result.wait_until_finish = MagicMock(return_value=None)
            pipeline_result.job_id = MagicMock(return_value="test-job-id")
            pipeline.run = MagicMock(return_value=pipeline_result)
            self._pipelines.append(pipeline)
            return pipeline

        self._pipeline_ctor = patch(
            "gigl.analytics.data_analyzer.feature_profiler.beam.Pipeline",
            side_effect=_make_pipeline,
        ).start()

        self.addCleanup(patch.stopall)

    def _config_with_chunk_cap(
        self, max_features_per_chunk: int, output_gcs_path: str
    ) -> DataAnalyzerConfig:
        return DataAnalyzerConfig(
            node_tables=[
                NodeTableSpec(
                    bq_table="p.d.users",
                    node_type="user",
                    id_column="uid",
                    feature_columns=["age", "country", "city", "lang"],
                )
            ],
            edge_tables=[],
            output_gcs_path=output_gcs_path,
            max_features_per_chunk=max_features_per_chunk,
            compute_per_class_feature_stats=False,
        )

    def test_chunks_wide_projection_into_multiple_pipelines(self) -> None:
        self._bq_utils_cls.return_value.fetch_bq_table_schema.return_value = _schema(
            [
                ("uid", "STRING", "REQUIRED"),
                ("age", "INT64", "NULLABLE"),
                ("country", "STRING", "NULLABLE"),
                ("city", "STRING", "NULLABLE"),
                ("lang", "STRING", "NULLABLE"),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config_with_chunk_cap(2, output_gcs_path=f"{tmp}/run/")
            profiler = FeatureProfiler()
            result = _run_profile(
                profiler, config=config, resource_config=self._resource_config
            )

        # 4 explicit feature columns, cap of 2 → 2 chunks (2+2).
        self.assertEqual(len(self._pipelines), 2)
        self.assertEqual(len(result.facets_html_paths["node:user"]), 2)
        self.assertEqual(len(result.stats_paths["node:user"]), 2)

        suffixes = sorted(
            call.kwargs["job_name_suffix"]
            for call in self._init_beam_pipeline_options.call_args_list
        )
        # Multi-chunk runs append "-chunk-NN-of-NN" to disambiguate Dataflow jobs.
        self.assertTrue(
            all("-chunk-00-of-02" in s or "-chunk-01-of-02" in s for s in suffixes)
        )

        # Per-chunk GCS artifacts land under chunk_NN/.
        sorted_facets = sorted(result.facets_html_paths["node:user"])
        self.assertTrue(
            sorted_facets[0].endswith(
                "/feature_profiler/nodes/user/chunk_00/facets.html"
            ),
            f"got {sorted_facets[0]!r}",
        )
        self.assertTrue(
            sorted_facets[1].endswith(
                "/feature_profiler/nodes/user/chunk_01/facets.html"
            ),
            f"got {sorted_facets[1]!r}",
        )

    def test_single_chunk_path_is_flat(self) -> None:
        """When the projection fits in one chunk, the historical flat path is preserved."""
        self._bq_utils_cls.return_value.fetch_bq_table_schema.return_value = _schema(
            [
                ("uid", "STRING", "REQUIRED"),
                ("age", "INT64", "NULLABLE"),
                ("country", "STRING", "NULLABLE"),
                ("city", "STRING", "NULLABLE"),
                ("lang", "STRING", "NULLABLE"),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            # Cap >> projection → exactly one chunk.
            config = self._config_with_chunk_cap(100, output_gcs_path=f"{tmp}/run/")
            profiler = FeatureProfiler()
            result = _run_profile(
                profiler, config=config, resource_config=self._resource_config
            )

        self.assertEqual(len(self._pipelines), 1)
        paths = result.facets_html_paths["node:user"]
        self.assertEqual(len(paths), 1)
        self.assertTrue(
            paths[0].endswith("/feature_profiler/nodes/user/facets.html"),
            f"single-chunk path should stay flat; got {paths[0]!r}",
        )
        # No chunk suffix in the Dataflow job-name suffix for single-chunk runs.
        suffix = self._init_beam_pipeline_options.call_args_list[0].kwargs[
            "job_name_suffix"
        ]
        self.assertNotIn("-chunk-", suffix)

    def test_slice_columns_force_included_in_every_chunk(self) -> None:
        """Slice columns must appear in every chunk so TFDV slicing applies uniformly."""
        self._bq_utils_cls.return_value.fetch_bq_table_schema.return_value = _schema(
            [
                ("uid", "STRING", "REQUIRED"),
                ("age", "INT64", "NULLABLE"),
                ("country", "STRING", "NULLABLE"),
                ("city", "STRING", "NULLABLE"),
                ("lang", "STRING", "NULLABLE"),
                ("node_label", "INT64", "NULLABLE"),
            ]
        )
        bq_to_record_batch_calls: list[dict] = []

        class _FakeBqTableToRecordBatch(beam.PTransform):
            def __init__(self, **kwargs):
                super().__init__()
                bq_to_record_batch_calls.append(kwargs)

            def expand(self, pbegin):  # pragma: no cover
                return pbegin

        with tempfile.TemporaryDirectory() as tmp:
            config = DataAnalyzerConfig(
                node_tables=[
                    NodeTableSpec(
                        bq_table="p.d.users",
                        node_type="user",
                        id_column="uid",
                        feature_columns=["age", "country", "city", "lang"],
                        label_column="node_label",
                    )
                ],
                edge_tables=[],
                output_gcs_path=f"{tmp}/run/",
                max_features_per_chunk=3,  # 4 features + label slice → 2 chunks
                compute_per_class_feature_stats=True,
            )
            with patch(
                "gigl.analytics.data_analyzer.feature_profiler.BqTableToRecordBatch",
                _FakeBqTableToRecordBatch,
            ):
                profiler = FeatureProfiler()
                _run_profile(
                    profiler, config=config, resource_config=self._resource_config
                )

        # Every chunk's projection should include node_label so TFDV slicing
        # works on each chunk independently.
        self.assertEqual(len(bq_to_record_batch_calls), 2)
        for call in bq_to_record_batch_calls:
            projected_names = {name for name, _ in call["projection"]}
            self.assertIn("node_label", projected_names, msg=str(projected_names))

    def test_embedding_diagnostics_runs_once_per_table_across_chunks(self) -> None:
        """The embedding-diagnostics BQ aggregate runs once per table, not once per chunk."""
        self._bq_utils_cls.return_value.fetch_bq_table_schema.return_value = _schema(
            [
                ("uid", "STRING", "REQUIRED"),
                ("age", "INT64", "NULLABLE"),
                ("country", "STRING", "NULLABLE"),
                ("emb", "FLOAT64", "REPEATED"),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            config = DataAnalyzerConfig(
                node_tables=[
                    NodeTableSpec(
                        bq_table="p.d.users",
                        node_type="user",
                        id_column="uid",
                        feature_columns=[],  # auto-infer
                    )
                ],
                edge_tables=[],
                output_gcs_path=f"{tmp}/run/",
                max_features_per_chunk=2,
                compute_per_class_feature_stats=False,
            )
            profiler = FeatureProfiler()
            _run_profile(profiler, config=config, resource_config=self._resource_config)

        # Wide-enough table to chunk → multiple Dataflow pipelines launched.
        self.assertGreater(len(self._pipelines), 1)
        # ...but EmbeddingDiagnostics.analyze fires exactly once for the whole table.
        self._diagnostics_cls.return_value.analyze.assert_called_once()
        request_list = self._diagnostics_cls.return_value.analyze.call_args[0][0]
        self.assertEqual(len(request_list), 1)
        self.assertEqual(request_list[0].result_key, "node:user")
        self.assertEqual(request_list[0].embedding_columns, ["emb"])
