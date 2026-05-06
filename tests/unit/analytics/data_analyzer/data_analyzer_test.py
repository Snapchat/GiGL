import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from gigl.analytics.data_analyzer.config import (
    DataAnalyzerConfig,
    EdgeTableSpec,
    NodeTableSpec,
)
from gigl.analytics.data_analyzer.data_analyzer import (
    DataAnalyzer,
    _resolve_job_name_prefix,
    _write_report,
)
from gigl.analytics.data_analyzer.graph_structure_analyzer import DataQualityError
from gigl.analytics.data_analyzer.types import FeatureProfileResult, GraphAnalysisResult
from tests.test_assets.test_case import TestCase

HTML = "<html><body>report</body></html>"
_TEST_JOB_NAME_PREFIX = "tp"
_TEST_RUN_TIMESTAMP = "20260101-0000"


def _run(analyzer: DataAnalyzer, **kwargs) -> str:
    """Invoke ``DataAnalyzer.run`` with the test prefix and timestamp.

    Centralizes the new required kwargs so individual tests stay focused.
    """
    return analyzer.run(
        job_name_prefix=_TEST_JOB_NAME_PREFIX,
        run_timestamp=_TEST_RUN_TIMESTAMP,
        **kwargs,
    )


def _make_config(output_gcs_path: str) -> DataAnalyzerConfig:
    return DataAnalyzerConfig(
        node_tables=[
            NodeTableSpec(
                bq_table="p.d.users",
                node_type="user",
                id_column="uid",
                feature_columns=["age"],
            )
        ],
        edge_tables=[
            EdgeTableSpec(
                bq_table="p.d.follows",
                edge_type="follows",
                src_id_column="src",
                dst_id_column="dst",
                src_node_type="user",
                dst_node_type="user",
            )
        ],
        output_gcs_path=output_gcs_path,
    )


class WriteReportLocalTest(TestCase):
    def test_writes_to_local_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_report(HTML, tmpdir)
            report = Path(path)
            self.assertTrue(report.exists())
            self.assertEqual(report.read_text(), HTML)
            self.assertEqual(report.name, "report.html")

    def test_creates_missing_parent_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "nested" / "path"
            path = _write_report(HTML, str(nested))
            self.assertTrue(Path(path).exists())


@patch("gigl.analytics.data_analyzer.data_analyzer.GcsUtils")
class WriteReportGcsTest(TestCase):
    def test_uploads_to_gcs(self, mock_gcs_cls: MagicMock) -> None:
        path = _write_report(HTML, "gs://my-bucket/output/")
        self.assertEqual(path, "gs://my-bucket/output/report.html")
        mock_gcs_cls.return_value.upload_from_string.assert_called_once()

    def test_handles_trailing_slash(self, mock_gcs_cls: MagicMock) -> None:
        path_with = _write_report(HTML, "gs://my-bucket/output/")
        path_without = _write_report(HTML, "gs://my-bucket/output")
        self.assertEqual(path_with, path_without)


class DataAnalyzerRunTest(TestCase):
    """Orchestrator tests: structure analyzer and feature profiler run
    concurrently, their results both reach ``generate_report``, and failures
    in either are handled independently without blocking the other.
    """

    def setUp(self) -> None:
        super().setUp()
        self._generate_report = patch(
            "gigl.analytics.data_analyzer.data_analyzer.generate_report",
            return_value=HTML,
        ).start()
        self._analyze = patch(
            "gigl.analytics.data_analyzer.data_analyzer.GraphStructureAnalyzer.analyze",
        ).start()
        self._profile = patch(
            "gigl.analytics.data_analyzer.data_analyzer.FeatureProfiler.profile",
        ).start()
        self.addCleanup(patch.stopall)

    def test_invokes_both_analyzer_and_profiler(self) -> None:
        analysis = GraphAnalysisResult()
        profile = FeatureProfileResult(
            facets_html_paths={"node:user": ["gs://b/facets.html"]}
        )
        self._analyze.return_value = analysis
        self._profile.return_value = profile

        with tempfile.TemporaryDirectory() as tmpdir:
            _run(
                DataAnalyzer(),
                config=_make_config(tmpdir),
                resource_config=MagicMock(),
            )

        self.assertEqual(self._analyze.call_count, 1)
        self.assertEqual(self._profile.call_count, 1)
        _, call_kwargs = self._generate_report.call_args
        self.assertIs(call_kwargs["analysis_result"], analysis)
        self.assertIs(call_kwargs["profile_result"], profile)

    def test_profiler_failure_does_not_block_report(self) -> None:
        self._analyze.return_value = GraphAnalysisResult()
        self._profile.side_effect = RuntimeError("Dataflow went boom")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = _run(
                DataAnalyzer(),
                config=_make_config(tmpdir),
                resource_config=MagicMock(),
            )
            _, call_kwargs = self._generate_report.call_args
            self.assertIsInstance(call_kwargs["profile_result"], FeatureProfileResult)
            self.assertEqual(call_kwargs["profile_result"].facets_html_paths, {})
            self.assertTrue(Path(path).exists())

    def test_data_quality_error_uses_partial_result_and_still_runs_profiler(
        self,
    ) -> None:
        partial = GraphAnalysisResult(dangling_edge_counts={"follows": 1})
        self._analyze.side_effect = DataQualityError(
            "Tier 1 failure", partial_result=partial
        )
        profile = FeatureProfileResult(
            facets_html_paths={"node:user": ["gs://b/facets.html"]}
        )
        self._profile.return_value = profile

        with tempfile.TemporaryDirectory() as tmpdir:
            _run(
                DataAnalyzer(),
                config=_make_config(tmpdir),
                resource_config=MagicMock(),
            )

        _, call_kwargs = self._generate_report.call_args
        self.assertIs(call_kwargs["analysis_result"], partial)
        self.assertIs(call_kwargs["profile_result"], profile)

    def test_passes_resource_config_to_profiler(self) -> None:
        self._analyze.return_value = GraphAnalysisResult()
        self._profile.return_value = FeatureProfileResult()
        resource_config = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            _run(
                DataAnalyzer(),
                config=_make_config(tmpdir),
                resource_config=resource_config,
            )

        self.assertIs(self._profile.call_args.args[1], resource_config)

    def test_components_structure_only_skips_profiler(self) -> None:
        analysis = GraphAnalysisResult(node_counts={"user": 100})
        self._analyze.return_value = analysis

        with tempfile.TemporaryDirectory() as tmpdir:
            _run(
                DataAnalyzer(),
                config=_make_config(tmpdir),
                resource_config=MagicMock(),
                components="structure",
            )

        self.assertEqual(self._analyze.call_count, 1)
        self.assertEqual(self._profile.call_count, 0)
        _, call_kwargs = self._generate_report.call_args
        self.assertIs(call_kwargs["analysis_result"], analysis)
        self.assertIsInstance(call_kwargs["profile_result"], FeatureProfileResult)
        self.assertEqual(call_kwargs["profile_result"].facets_html_paths, {})

    def test_components_feature_only_skips_analyzer(self) -> None:
        profile = FeatureProfileResult(
            facets_html_paths={"node:user": ["gs://b/facets.html"]}
        )
        self._profile.return_value = profile

        with tempfile.TemporaryDirectory() as tmpdir:
            _run(
                DataAnalyzer(),
                config=_make_config(tmpdir),
                resource_config=MagicMock(),
                components="feature",
            )

        self.assertEqual(self._analyze.call_count, 0)
        self.assertEqual(self._profile.call_count, 1)
        _, call_kwargs = self._generate_report.call_args
        self.assertIs(call_kwargs["profile_result"], profile)
        self.assertIsInstance(call_kwargs["analysis_result"], GraphAnalysisResult)
        self.assertEqual(call_kwargs["analysis_result"].node_counts, {})

    def test_custom_worker_image_uri_passed_through_feature_only(self) -> None:
        self._profile.return_value = FeatureProfileResult()
        image_uri = "gcr.io/proj/gbml_dataflow_runtime:20260422-000000"

        with tempfile.TemporaryDirectory() as tmpdir:
            _run(
                DataAnalyzer(),
                config=_make_config(tmpdir),
                resource_config=MagicMock(),
                components="feature",
                custom_worker_image_uri=image_uri,
            )

        self.assertEqual(
            self._profile.call_args.kwargs["custom_worker_image_uri"], image_uri
        )

    def test_custom_worker_image_uri_passed_through_both(self) -> None:
        self._analyze.return_value = GraphAnalysisResult()
        self._profile.return_value = FeatureProfileResult()
        image_uri = "gcr.io/proj/gbml_dataflow_runtime:20260422-000000"

        with tempfile.TemporaryDirectory() as tmpdir:
            _run(
                DataAnalyzer(),
                config=_make_config(tmpdir),
                resource_config=MagicMock(),
                components="both",
                custom_worker_image_uri=image_uri,
            )

        # "both" path submits FeatureProfiler.profile via ThreadPoolExecutor
        # with (config, resource_config, job_name_prefix, run_timestamp,
        # custom_worker_image_uri) as positional args.
        positional = self._profile.call_args.args
        self.assertEqual(positional[2], _TEST_JOB_NAME_PREFIX)
        self.assertEqual(positional[3], _TEST_RUN_TIMESTAMP)
        self.assertEqual(positional[4], image_uri)

    def test_custom_worker_image_uri_defaults_to_none(self) -> None:
        self._profile.return_value = FeatureProfileResult()

        with tempfile.TemporaryDirectory() as tmpdir:
            _run(
                DataAnalyzer(),
                config=_make_config(tmpdir),
                resource_config=MagicMock(),
                components="feature",
            )

        self.assertIsNone(self._profile.call_args.kwargs["custom_worker_image_uri"])


class ResolveJobNamePrefixTest(TestCase):
    """The resolver enforces 'set somewhere' + lightweight shape checks."""

    def test_uses_yaml_when_cli_unset(self) -> None:
        self.assertEqual(
            _resolve_job_name_prefix(cli_value=None, yaml_value="cd-content"),
            "cd-content",
        )

    def test_uses_cli_when_yaml_unset(self) -> None:
        self.assertEqual(
            _resolve_job_name_prefix(cli_value="svij-test", yaml_value=None),
            "svij-test",
        )

    def test_cli_overrides_yaml_when_both_set(self) -> None:
        with self.assertLogs(level="INFO") as cap:
            result = _resolve_job_name_prefix(
                cli_value="svij-test", yaml_value="cd-content"
            )
        self.assertEqual(result, "svij-test")
        self.assertTrue(
            any("overrides YAML" in msg for msg in cap.output),
            f"expected override log, got {cap.output}",
        )

    def test_raises_when_neither_set(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_job_name_prefix(cli_value=None, yaml_value=None)

    def test_raises_when_both_empty_strings(self) -> None:
        # argparse with required=False yields None for absent flags, but
        # an empty YAML value would slip through without this check.
        with self.assertRaises(ValueError):
            _resolve_job_name_prefix(cli_value=None, yaml_value="")

    def test_rejects_uppercase(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_job_name_prefix(cli_value="SvijTest", yaml_value=None)

    def test_rejects_underscore(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_job_name_prefix(cli_value="svij_test", yaml_value=None)

    def test_rejects_too_long(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_job_name_prefix(
                cli_value="a" * 21,  # 21 chars exceeds the 20-char cap
                yaml_value=None,
            )

    def test_accepts_at_length_cap(self) -> None:
        prefix = "a" + "b" * 19  # exactly 20 chars
        self.assertEqual(
            _resolve_job_name_prefix(cli_value=prefix, yaml_value=None), prefix
        )
