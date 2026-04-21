"""Main orchestrator and CLI entry point for the BQ Data Analyzer."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from gigl.analytics.data_analyzer.config import DataAnalyzerConfig, load_analyzer_config
from gigl.analytics.data_analyzer.feature_profiler import FeatureProfiler
from gigl.analytics.data_analyzer.graph_structure_analyzer import (
    DataQualityError,
    GraphStructureAnalyzer,
)
from gigl.analytics.data_analyzer.report.report_generator import generate_report
from gigl.analytics.data_analyzer.types import FeatureProfileResult, GraphAnalysisResult
from gigl.common import GcsUri, Uri, UriFactory
from gigl.common.logger import Logger
from gigl.common.utils.gcs import GcsUtils

logger = Logger()


def _write_report(html: str, output_gcs_path: str) -> str:
    """Write the HTML report to a GCS URI or local path.

    Args:
        html: Rendered HTML string.
        output_gcs_path: Output directory. If it starts with ``gs://`` the
            report is uploaded via ``GcsUtils``. Otherwise it is written to
            the local filesystem (the directory is created if missing).

    Returns:
        The full path to the written ``report.html`` file.
    """
    trimmed = output_gcs_path.rstrip("/")
    report_path = f"{trimmed}/report.html"
    if trimmed.startswith("gs://"):
        GcsUtils().upload_from_string(GcsUri(report_path), html)
    else:
        local_path = Path(report_path).expanduser().resolve()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(html)
        report_path = str(local_path)
    return report_path


class DataAnalyzer:
    """Orchestrates graph structure analysis, feature profiling, and report generation.

    Example:
        >>> from gigl.analytics.data_analyzer.config import load_analyzer_config
        >>> config = load_analyzer_config("gs://bucket/config.yaml")
        >>> analyzer = DataAnalyzer()
        >>> report_path = analyzer.run(config=config)
    """

    def run(
        self,
        config: DataAnalyzerConfig,
        resource_config_uri: Optional[Uri] = None,
    ) -> str:
        """Run the full analysis pipeline and write an HTML report.

        The report is written to ``{config.output_gcs_path}/report.html`` via
        ``GcsUtils`` when the output path is a ``gs://`` URI, or to the local
        filesystem otherwise (the parent directory is created if missing).

        Args:
            config: Analyzer configuration.
            resource_config_uri: Optional resource config for Dataflow sizing.

        Returns:
            The path to the written ``report.html`` (GCS URI or local path).
        """
        structure_analyzer = GraphStructureAnalyzer()
        feature_profiler = FeatureProfiler()

        with ThreadPoolExecutor(max_workers=2) as executor:
            structure_future = executor.submit(structure_analyzer.analyze, config)
            profile_future = executor.submit(
                feature_profiler.profile, config, resource_config_uri
            )

            analysis_result: GraphAnalysisResult
            try:
                analysis_result = structure_future.result()
            except DataQualityError as e:
                logger.error(f"Tier 1 data quality failure: {e}")
                analysis_result = e.partial_result

            profile_result: FeatureProfileResult
            try:
                profile_result = profile_future.result()
            except Exception as e:
                logger.exception(f"Feature profiler failed: {e}")
                profile_result = FeatureProfileResult()

        html = generate_report(
            analysis_result=analysis_result,
            profile_result=profile_result,
            config=config,
        )

        report_path = _write_report(html, config.output_gcs_path)
        logger.info(f"Report written to {report_path}")
        return report_path


def main() -> None:
    """CLI entry point for the BQ Data Analyzer."""
    parser = argparse.ArgumentParser(
        description="BQ Data Analyzer: analyze graph data in BigQuery before GNN training"
    )
    parser.add_argument(
        "--analyzer_config_uri",
        required=True,
        help="Path or GCS URI to the analyzer YAML config",
    )
    parser.add_argument(
        "--resource_config_uri",
        required=False,
        help="Path or GCS URI to the resource config for Dataflow sizing",
    )
    args = parser.parse_args()

    config = load_analyzer_config(args.analyzer_config_uri)
    resource_config_uri: Optional[Uri] = (
        UriFactory.create_uri(args.resource_config_uri)
        if args.resource_config_uri
        else None
    )
    analyzer = DataAnalyzer()
    report_path = analyzer.run(config=config, resource_config_uri=resource_config_uri)
    logger.info(f"Report generated at: {report_path}")


if __name__ == "__main__":
    main()
