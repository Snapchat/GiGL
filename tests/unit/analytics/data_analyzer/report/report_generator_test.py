from pathlib import Path

from gigl.analytics.data_analyzer.report.report_generator import generate_report
from gigl.analytics.data_analyzer.types import DegreeStats, GraphAnalysisResult
from tests.test_assets.test_case import TestCase

GOLDEN_REPORT_PATH = (
    Path(__file__).parents[4] / "test_assets" / "analytics" / "golden_report.html"
)


def _make_test_result() -> GraphAnalysisResult:
    """Deterministic test data for snapshot testing."""
    return GraphAnalysisResult(
        duplicate_node_counts={"user": 0},
        dangling_edge_counts={"follows": 0},
        referential_integrity_violations={"follows": 0},
        node_counts={"user": 1000000},
        edge_counts={"follows": 5000000},
        null_rates={"p.d.nodes": {"age": 0.05, "country": 0.12}},
        duplicate_edge_counts={"follows": 150},
        self_loop_counts={"follows": 0},
        isolated_node_counts={"user": 8000},
        degree_stats={
            "follows_out": DegreeStats(
                min=0,
                max=50000,
                mean=10.0,
                median=5,
                p90=25,
                p99=200,
                p999=5000,
                percentiles=list(range(101)),
                buckets={
                    "0-1": 100000,
                    "2-10": 600000,
                    "11-100": 250000,
                    "101-1K": 45000,
                    "1K-10K": 4500,
                    "10K+": 500,
                },
            )
        },
        top_hubs={"follows_out": [("hub_1", 50000), ("hub_2", 35000)]},
        super_hub_int16_clamp_count={"follows_out": 2},
        cold_start_node_counts={"user": 100000},
        feature_memory_bytes={"user": 8000000000},
        neighbor_explosion_estimate={"follows": 75000},
    )


class ReportGeneratorStructuralTest(TestCase):
    """Structural assertions on the generated HTML."""

    def test_output_is_non_empty_html(self) -> None:
        html = generate_report(
            analysis_result=_make_test_result(),
            profile_result=None,
            config=None,
        )
        self.assertIsInstance(html, str)
        self.assertGreater(len(html), 1000)
        self.assertIn("<html", html)
        self.assertIn("GiGL Data Analysis Report", html)

    def test_placeholders_all_replaced(self) -> None:
        html = generate_report(
            analysis_result=_make_test_result(),
            profile_result=None,
            config=None,
        )
        # None of the injection placeholders should remain in the output.
        self.assertNotIn("/* INJECT_STYLES */", html)
        self.assertNotIn("/* INJECT_SCRIPTS */", html)
        self.assertNotIn("/* INJECT_ANALYSIS_DATA */", html)
        self.assertNotIn("/* INJECT_PROFILE_DATA */", html)

    def test_injected_data_present(self) -> None:
        html = generate_report(
            analysis_result=_make_test_result(),
            profile_result=None,
            config=None,
        )
        # The JSON data lives inside a hidden script tag.
        self.assertIn('"node_counts"', html)
        # Either 1000000 (int) or "1000000" (str) is acceptable depending on serialization.
        self.assertTrue('"1000000"' in html or "1000000" in html)

    def test_empty_profile_serializes_as_empty_object(self) -> None:
        html = generate_report(
            analysis_result=_make_test_result(),
            profile_result=None,
            config=None,
        )
        # When profile_result is None, we inject an empty JSON object.
        self.assertIn('id="profile-data"', html)


class ReportGeneratorSnapshotTest(TestCase):
    """Golden-file snapshot test to catch structural regressions."""

    def test_snapshot_matches_golden(self) -> None:
        html = generate_report(
            analysis_result=_make_test_result(),
            profile_result=None,
            config=None,
        )
        if not GOLDEN_REPORT_PATH.exists():
            self.fail(
                f"Golden file missing: {GOLDEN_REPORT_PATH}. "
                f"Create it by writing the current output of generate_report "
                f"with _make_test_result() as input."
            )
        golden = GOLDEN_REPORT_PATH.read_text()
        self.assertEqual(
            html,
            golden,
            msg=(
                "HTML output changed. If this is intentional, regenerate the "
                f"golden file at {GOLDEN_REPORT_PATH}."
            ),
        )
