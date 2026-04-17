"""Main orchestrator and CLI entry point for the BQ Data Analyzer."""
import argparse
from typing import Optional

from gigl.analytics.data_analyzer.config import DataAnalyzerConfig, load_analyzer_config
from gigl.analytics.data_analyzer.graph_structure_analyzer import (
    DataQualityError,
    GraphStructureAnalyzer,
)
from gigl.analytics.data_analyzer.report.report_generator import generate_report
from gigl.analytics.data_analyzer.types import (
    FeatureProfileResult,
    GraphAnalysisResult,
)
from gigl.common import Uri
from gigl.common.logger import Logger

logger = Logger()


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
        """Run the full analysis pipeline and generate an HTML report.

        Args:
            config: Analyzer configuration.
            resource_config_uri: Optional resource config for Dataflow sizing.

        Returns:
            GCS path to the generated HTML report.
        """
        analysis_result: GraphAnalysisResult
        profile_result: Optional[FeatureProfileResult] = None

        structure_analyzer = GraphStructureAnalyzer()
        try:
            analysis_result = structure_analyzer.analyze(config)
        except DataQualityError as e:
            logger.error(f"Tier 1 data quality failure: {e}")
            analysis_result = e.partial_result

        # TODO: run feature profiler (TFDV/Dataflow) in parallel once implemented.

        html = generate_report(
            analysis_result=analysis_result,
            profile_result=profile_result,
            config=config,
        )

        report_gcs_path = f"{config.output_gcs_path.rstrip('/')}/report.html"
        logger.info(f"Generated report; would upload to {report_gcs_path}")
        # TODO: wire up GCS upload via gigl.common.utils.gcs.GcsUtils

        return report_gcs_path


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
    analyzer = DataAnalyzer()
    report_path = analyzer.run(config=config)
    logger.info(f"Report generated at: {report_path}")


if __name__ == "__main__":
    main()
