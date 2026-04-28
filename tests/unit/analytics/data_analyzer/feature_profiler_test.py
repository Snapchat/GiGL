"""Unit tests for the FeatureProfiler.

Dataflow job execution is mocked: ``beam.Pipeline`` is replaced with a
dummy that records construction, and ``init_beam_pipeline_options`` /
``get_resource_config`` are patched so tests don't touch real GCP
resources.
"""
import itertools
from typing import Optional
from unittest.mock import MagicMock, patch

from gigl.analytics.data_analyzer.config import (
    DataAnalyzerConfig,
    EdgeTableSpec,
    NodeTableSpec,
)
from gigl.analytics.data_analyzer.feature_profiler import (
    FeatureProfiler,
    _collect_profile_tasks,
)
from gigl.common import LocalUri
from gigl.src.common.constants.components import GiGLComponents
from tests.test_assets.test_case import TestCase


def _make_config(
    node_specs: Optional[list[NodeTableSpec]] = None,
    edge_specs: Optional[list[EdgeTableSpec]] = None,
    output_gcs_path: str = "gs://bucket/out",
) -> DataAnalyzerConfig:
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
        output_gcs_path=output_gcs_path,
    )


class CollectProfileTasksTest(TestCase):
    def test_skips_tables_without_feature_columns(self) -> None:
        config = _make_config(
            node_specs=[
                NodeTableSpec(
                    bq_table="p.d.a",
                    node_type="a",
                    id_column="id",
                    feature_columns=["f1"],
                ),
                NodeTableSpec(
                    bq_table="p.d.b",
                    node_type="b",
                    id_column="id",
                    feature_columns=[],
                ),
            ],
            edge_specs=[
                EdgeTableSpec(
                    bq_table="p.d.e1",
                    edge_type="e1",
                    src_id_column="s",
                    dst_id_column="d",
                    src_node_type="a",
                    dst_node_type="a",
                    feature_columns=["w"],
                ),
                EdgeTableSpec(
                    bq_table="p.d.e2",
                    edge_type="e2",
                    src_id_column="s",
                    dst_id_column="d",
                    src_node_type="a",
                    dst_node_type="a",
                    feature_columns=[],
                ),
            ],
        )
        tasks = _collect_profile_tasks(config)
        keys = sorted(t.result_key for t in tasks)
        self.assertEqual(keys, ["edge:e1", "node:a"])

    def test_preserves_feature_columns(self) -> None:
        config = _make_config()
        tasks = _collect_profile_tasks(config)
        by_key = {t.result_key: t for t in tasks}
        self.assertEqual(by_key["node:user"].feature_columns, ["age", "country"])
        self.assertEqual(by_key["edge:follows"].feature_columns, ["weight"])


class FeatureProfilerRaisesTest(TestCase):
    def test_raises_when_resource_config_uri_missing(self) -> None:
        profiler = FeatureProfiler()
        with self.assertRaises(ValueError):
            profiler.profile(config=_make_config(), resource_config_uri=None)


class FeatureProfilerRunTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._resource_config_uri = LocalUri("/tmp/fake_resource_config.yaml")

        self._get_resource_config = patch(
            "gigl.analytics.data_analyzer.feature_profiler.get_resource_config"
        ).start()
        self._init_beam_pipeline_options = patch(
            "gigl.analytics.data_analyzer.feature_profiler.init_beam_pipeline_options",
            return_value=MagicMock(name="PipelineOptions"),
        ).start()

        self._pipelines: list[MagicMock] = []

        def _make_pipeline(*args, **kwargs):
            pipeline = MagicMock(name="Pipeline")
            pipeline.__enter__ = MagicMock(return_value=pipeline)
            pipeline.__exit__ = MagicMock(return_value=False)
            self._pipelines.append(pipeline)
            return pipeline

        self._pipeline_ctor = patch(
            "gigl.analytics.data_analyzer.feature_profiler.beam.Pipeline",
            side_effect=_make_pipeline,
        ).start()

        self.addCleanup(patch.stopall)

    def test_returns_empty_when_no_tables_have_features(self) -> None:
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
        profiler = FeatureProfiler()
        result = profiler.profile(
            config=config, resource_config_uri=self._resource_config_uri
        )
        self.assertEqual(result.facets_html_paths, {})
        self.assertEqual(result.stats_paths, {})
        self.assertEqual(len(self._pipelines), 0)

    def test_launches_one_pipeline_per_feature_table(self) -> None:
        config = _make_config()
        profiler = FeatureProfiler()
        result = profiler.profile(
            config=config, resource_config_uri=self._resource_config_uri
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
        self._get_resource_config.assert_called_once_with(
            resource_config_uri=self._resource_config_uri
        )
        component_kwargs = [
            call.kwargs.get("component")
            for call in self._init_beam_pipeline_options.call_args_list
        ]
        self.assertTrue(all(c == GiGLComponents.DataAnalyzer for c in component_kwargs))

    def test_gcs_paths_use_expected_layout(self) -> None:
        profiler = FeatureProfiler()
        result = profiler.profile(
            config=_make_config(output_gcs_path="gs://bucket/run1/"),
            resource_config_uri=self._resource_config_uri,
        )
        self.assertEqual(
            result.facets_html_paths["node:user"],
            "gs://bucket/run1/feature_profiler/nodes/user/facets.html",
        )
        self.assertEqual(
            result.stats_paths["node:user"],
            "gs://bucket/run1/feature_profiler/nodes/user/stats.tfrecord",
        )
        self.assertEqual(
            result.facets_html_paths["edge:follows"],
            "gs://bucket/run1/feature_profiler/edges/follows/facets.html",
        )

    def test_individual_pipeline_failure_is_caught(self) -> None:
        counter = itertools.count(1)

        def _make_pipeline_fail_second(*args, **kwargs):
            pipeline = MagicMock(name="Pipeline")
            pipeline.__enter__ = MagicMock(return_value=pipeline)
            if next(counter) == 2:
                pipeline.__exit__ = MagicMock(side_effect=RuntimeError("Dataflow boom"))
            else:
                pipeline.__exit__ = MagicMock(return_value=False)
            self._pipelines.append(pipeline)
            return pipeline

        self._pipeline_ctor.side_effect = _make_pipeline_fail_second

        profiler = FeatureProfiler()
        result = profiler.profile(
            config=_make_config(),
            resource_config_uri=self._resource_config_uri,
        )
        self.assertEqual(len(self._pipelines), 2)
        total_keys = set(result.facets_html_paths.keys())
        self.assertEqual(len(total_keys), 1)
        self.assertLessEqual(total_keys, {"node:user", "edge:follows"})

    def test_uses_data_analyzer_job_name_suffix(self) -> None:
        profiler = FeatureProfiler()
        profiler.profile(
            config=_make_config(),
            resource_config_uri=self._resource_config_uri,
        )
        suffixes = {
            call.kwargs.get("job_name_suffix")
            for call in self._init_beam_pipeline_options.call_args_list
        }
        self.assertEqual(
            suffixes,
            {"profile-node-user", "profile-edge-follows"},
        )
