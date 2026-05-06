"""Main orchestrator and CLI entry point for the BQ Data Analyzer."""
import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, Optional

from gigl.analytics.data_analyzer.config import DataAnalyzerConfig, load_analyzer_config
from gigl.analytics.data_analyzer.feature_profiler import FeatureProfiler
from gigl.analytics.data_analyzer.graph_structure_analyzer import (
    DataQualityError,
    GraphStructureAnalyzer,
)
from gigl.analytics.data_analyzer.report.report_generator import generate_report
from gigl.analytics.data_analyzer.types import FeatureProfileResult, GraphAnalysisResult
from gigl.common import GcsUri, UriFactory
from gigl.common.logger import Logger
from gigl.common.utils.gcs import GcsUtils
from gigl.env.pipelines_config import GiglResourceConfigWrapper, get_resource_config
from gigl.src.common.utils.time import current_formatted_datetime

logger = Logger()

# Lowercase, hyphen-safe, ≤20 chars. Composes cleanly with
# ``get_sanitized_dataflow_job_name`` and keeps the final Dataflow job
# name (``gigl-analyzer-{prefix}-{ts}-profile-{kind}-{type}``) inside
# Dataflow's ~63-char budget for typical type_name lengths.
_JOB_NAME_PREFIX_REGEX = re.compile(r"^[a-z][a-z0-9-]{0,19}$")
_RUN_TIMESTAMP_FORMAT = "%Y%m%d-%H%M"


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
        resource_config: GiglResourceConfigWrapper,
        job_name_prefix: str,
        run_timestamp: str,
        components: Literal["structure", "feature", "both"] = "both",
        custom_worker_image_uri: Optional[str] = None,
    ) -> str:
        """Run the analysis pipeline and write an HTML report.

        The report is written to ``{config.output_gcs_path}/report.html`` via
        ``GcsUtils`` when the output path is a ``gs://`` URI, or to the local
        filesystem otherwise (the parent directory is created if missing).

        Args:
            config: Analyzer configuration.
            resource_config: Resource config for Dataflow sizing.
            job_name_prefix: Prefix mixed into every per-table Dataflow job
                name (resolved at the entry point from CLI flag or YAML).
            run_timestamp: Per-run timestamp shared by every per-table job in
                this invocation (computed once at the entry point).
            components: Which components to run. ``"both"`` (default) runs the
                structure analyzer and feature profiler concurrently.
                ``"structure"`` runs only the graph structure analyzer.
                ``"feature"`` runs only the feature profiler. The skipped
                component is represented in the report by an empty result.
            custom_worker_image_uri: Optional Docker image URI for the Dataflow
                worker harness used by the feature profiler. When ``None``, the
                profiler falls back to ``DEFAULT_GIGL_RELEASE_SRC_IMAGE_DATAFLOW_CPU``.

        Returns:
            The path to the written ``report.html`` (GCS URI or local path).
        """
        analysis_result: GraphAnalysisResult
        profile_result: FeatureProfileResult

        if components == "both":
            with ThreadPoolExecutor(max_workers=2) as executor:
                structure_future = executor.submit(
                    GraphStructureAnalyzer().analyze, config
                )
                profile_future = executor.submit(
                    FeatureProfiler().profile,
                    config,
                    resource_config,
                    job_name_prefix,
                    run_timestamp,
                    custom_worker_image_uri,
                )

                try:
                    analysis_result = structure_future.result()
                except DataQualityError as e:
                    logger.error(f"Tier 1 data quality failure: {e}")
                    analysis_result = e.partial_result

                try:
                    profile_result = profile_future.result()
                except Exception as e:
                    logger.exception(f"Feature profiler failed: {e}")
                    profile_result = FeatureProfileResult()
        elif components == "structure":
            try:
                analysis_result = GraphStructureAnalyzer().analyze(config)
            except DataQualityError as e:
                logger.error(f"Tier 1 data quality failure: {e}")
                analysis_result = e.partial_result
            profile_result = FeatureProfileResult()
        elif components == "feature":
            analysis_result = GraphAnalysisResult()
            profile_result = FeatureProfiler().profile(
                config,
                resource_config,
                job_name_prefix=job_name_prefix,
                run_timestamp=run_timestamp,
                custom_worker_image_uri=custom_worker_image_uri,
            )
        else:
            raise ValueError(
                f"components={components!r} must be one of 'structure', 'feature', 'both'"
            )

        html = generate_report(
            analysis_result=analysis_result,
            profile_result=profile_result,
        )

        report_path = _write_report(html, config.output_gcs_path)
        logger.info(f"Report written to {report_path}")
        return report_path


def _resolve_job_name_prefix(
    cli_value: Optional[str], yaml_value: Optional[str]
) -> str:
    """Pick the effective ``job_name_prefix`` from CLI flag or YAML field.

    CLI takes precedence; if both are set and differ the override is logged.
    Raises ``ValueError`` if neither source supplies a value, or if the
    chosen value doesn't match the lowercase / hyphen / ≤20-char shape we
    require to keep the final Dataflow job name within Dataflow's ~63-char
    cap.
    """
    if cli_value and yaml_value and cli_value != yaml_value:
        logger.info(
            f"--job_name_prefix={cli_value!r} overrides YAML "
            f"job_name_prefix={yaml_value!r}."
        )
    effective = cli_value or yaml_value
    if not effective:
        raise ValueError(
            "job_name_prefix is required: pass --job_name_prefix on the CLI "
            "or set job_name_prefix in the analyzer YAML."
        )
    if not _JOB_NAME_PREFIX_REGEX.fullmatch(effective):
        raise ValueError(
            f"job_name_prefix={effective!r} is invalid. Expected lowercase "
            "letters, digits, and hyphens, starting with a letter, ≤20 chars."
        )
    return effective


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
    parser.add_argument(
        "--only",
        choices=["structure", "feature", "both"],
        default="both",
        help=(
            "Run only the graph structure analyzer, only the feature profiler, "
            "or both (default: both)."
        ),
    )
    parser.add_argument(
        "--custom_worker_image_uri",
        type=str,
        required=False,
        help=(
            "Docker image URI to use for the Dataflow worker harness in the "
            "feature profiler. When omitted, falls back to "
            "DEFAULT_GIGL_RELEASE_SRC_IMAGE_DATAFLOW_CPU."
        ),
    )
    parser.add_argument(
        "--job_name_prefix",
        type=str,
        required=False,
        help=(
            "Prefix mixed into every per-table Dataflow job name to "
            "disambiguate concurrent / repeat runs. Required, but may be "
            "set in the analyzer YAML instead. CLI overrides YAML. Lowercase "
            "letters, digits, and hyphens, starting with a letter, ≤20 chars."
        ),
    )
    args = parser.parse_args()
    resource_config = get_resource_config(
        UriFactory.create_uri(args.resource_config_uri)
    )
    config = load_analyzer_config(args.analyzer_config_uri)
    job_name_prefix = _resolve_job_name_prefix(
        cli_value=args.job_name_prefix, yaml_value=config.job_name_prefix
    )
    run_timestamp = current_formatted_datetime(_RUN_TIMESTAMP_FORMAT)
    logger.info(
        f"Using job_name_prefix={job_name_prefix!r}, run_timestamp={run_timestamp!r}."
    )

    analyzer = DataAnalyzer()
    report_path = analyzer.run(
        config=config,
        resource_config=resource_config,
        job_name_prefix=job_name_prefix,
        run_timestamp=run_timestamp,
        components=args.only,
        custom_worker_image_uri=args.custom_worker_image_uri,
    )
    logger.info(f"Report generated at: {report_path}")


if __name__ == "__main__":
    main()
