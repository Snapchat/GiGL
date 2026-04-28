import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from gigl.analytics.data_analyzer.config import (
    DataAnalyzerConfig,
    EdgeTableSpec,
    NodeTableSpec,
)
from gigl.analytics.data_analyzer.data_analyzer import DataAnalyzer, _write_report
from gigl.analytics.data_analyzer.graph_structure_analyzer import DataQualityError
from gigl.analytics.data_analyzer.types import FeatureProfileResult, GraphAnalysisResult
from gigl.common import LocalUri
from tests.test_assets.test_case import TestCase

HTML = "<html><body>report</body></html>"


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
            facets_html_paths={"node:user": "gs://b/facets.html"}
        )
        self._analyze.return_value = analysis
        self._profile.return_value = profile

        with tempfile.TemporaryDirectory() as tmpdir:
            DataAnalyzer().run(
                config=_make_config(tmpdir),
                resource_config_uri=LocalUri("/tmp/fake.yaml"),
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
            path = DataAnalyzer().run(
                config=_make_config(tmpdir),
                resource_config_uri=LocalUri("/tmp/fake.yaml"),
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
            facets_html_paths={"node:user": "gs://b/facets.html"}
        )
        self._profile.return_value = profile

        with tempfile.TemporaryDirectory() as tmpdir:
            DataAnalyzer().run(
                config=_make_config(tmpdir),
                resource_config_uri=LocalUri("/tmp/fake.yaml"),
            )

        _, call_kwargs = self._generate_report.call_args
        self.assertIs(call_kwargs["analysis_result"], partial)
        self.assertIs(call_kwargs["profile_result"], profile)

    def test_passes_resource_config_uri_to_profiler(self) -> None:
        self._analyze.return_value = GraphAnalysisResult()
        self._profile.return_value = FeatureProfileResult()
        resource_config_uri = LocalUri("/tmp/fake.yaml")

        with tempfile.TemporaryDirectory() as tmpdir:
            DataAnalyzer().run(
                config=_make_config(tmpdir),
                resource_config_uri=resource_config_uri,
            )

        self.assertEqual(self._profile.call_args.args[1], resource_config_uri)
