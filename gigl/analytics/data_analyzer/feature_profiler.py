"""TFDV feature profiling via Beam/Dataflow.

Launches one Dataflow pipeline per (node or edge) table that declares
``feature_columns`` in the analyzer config. Each pipeline reads the
selected columns from BigQuery, emits ``pa.RecordBatch`` batches, and
runs ``tfdv.GenerateStatistics`` to write a Facets HTML visualization
plus a TFDV stats TFRecord to GCS.

Pipelines are launched concurrently using an internal
``ThreadPoolExecutor``; each worker blocks on
``p.run().wait_until_finish()`` for its table. Per-table exceptions are
logged and the failed table is omitted from the returned
``FeatureProfileResult`` - callers (and the HTML report) already handle
missing keys.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import apache_beam as beam

from gigl.analytics.data_analyzer.config import DataAnalyzerConfig
from gigl.analytics.data_analyzer.types import FeatureProfileResult
from gigl.common import Uri, UriFactory
from gigl.common.beam.tfdv_transforms import (
    BqTableToRecordBatch,
    GenerateAndVisualizeStats,
)
from gigl.common.logger import Logger
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.utils.dataflow import init_beam_pipeline_options

logger = Logger()

_PARALLEL_DATAFLOW_WORKERS = 10
_APPLIED_TASK_IDENTIFIER = AppliedTaskIdentifier("data-analyzer")


@dataclass(frozen=True)
class _ProfileTask:
    """One profiling unit: all features of a single node or edge table.

    ``kind`` is ``"node"`` or ``"edge"`` (singular) and is used to build
    the GCS output path and the result key (``"node:user"``, etc.).
    """

    kind: str
    type_name: str
    bq_table: str
    feature_columns: list[str]

    @property
    def result_key(self) -> str:
        return f"{self.kind}:{self.type_name}"


class FeatureProfiler:
    """Runs TFDV feature profiling on BQ tables via Dataflow.

    Example:
        >>> profiler = FeatureProfiler()
        >>> result = profiler.profile(config, resource_config_uri=uri)
        >>> result.facets_html_paths["node:user"]
        'gs://bucket/analyzer/feature_profiler/nodes/user/facets.html'
    """

    def profile(
        self,
        config: DataAnalyzerConfig,
        resource_config_uri: Optional[Uri] = None,
    ) -> FeatureProfileResult:
        """Run TFDV profiling on all tables with declared feature columns.

        Launches one Dataflow pipeline per table concurrently. Tables with
        no ``feature_columns`` are skipped. Per-table failures are logged
        and omitted from the result.

        Args:
            config: Analyzer configuration with node and edge table specs.
            resource_config_uri: Resource config for Dataflow sizing.
                Required - TFDV profiling needs Dataflow.

        Returns:
            ``FeatureProfileResult`` with GCS paths keyed by
            ``"node:{type}"`` / ``"edge:{type}"``. Empty if no tables
            declared feature columns.

        Raises:
            ValueError: If ``resource_config_uri`` is None.
        """
        if resource_config_uri is None:
            raise ValueError(
                "FeatureProfiler requires a resource_config_uri for Dataflow sizing. "
                "Pass --resource_config_uri when invoking the DataAnalyzer CLI."
            )
        # Eagerly populate the process-global resource config so that
        # `init_beam_pipeline_options` (called on worker threads below)
        # can resolve it without args.
        get_resource_config(resource_config_uri=resource_config_uri)

        tasks = _collect_profile_tasks(config)
        if not tasks:
            logger.info("No tables declared feature_columns; returning empty result.")
            return FeatureProfileResult()

        logger.info(f"Launching {len(tasks)} Dataflow feature-profile job(s).")
        result = FeatureProfileResult()
        with ThreadPoolExecutor(max_workers=_PARALLEL_DATAFLOW_WORKERS) as executor:
            future_to_task = {
                executor.submit(
                    self._run_single_pipeline, task, config.output_gcs_path
                ): task
                for task in tasks
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    facets_uri, stats_uri = future.result()
                    result.facets_html_paths[task.result_key] = facets_uri
                    result.stats_paths[task.result_key] = stats_uri
                except Exception as exc:
                    logger.exception(
                        f"Feature profiling failed for {task.result_key} "
                        f"(table={task.bq_table}): {exc}"
                    )
        return result

    def _run_single_pipeline(
        self, task: _ProfileTask, output_gcs_path: str
    ) -> tuple[str, str]:
        """Build, run, and block on a single table's Dataflow pipeline.

        Returns the ``(facets_uri, stats_uri)`` strings on success.
        """
        base = f"{output_gcs_path.rstrip('/')}/feature_profiler/{task.kind}s/{task.type_name}"
        facets_uri = UriFactory.create_uri(f"{base}/facets.html")
        stats_uri = UriFactory.create_uri(f"{base}/stats.tfrecord")

        options = init_beam_pipeline_options(
            applied_task_identifier=_APPLIED_TASK_IDENTIFIER,
            job_name_suffix=f"profile-{task.kind}-{task.type_name}",
            component=GiGLComponents.DataAnalyzer,
        )
        with beam.Pipeline(options=options) as p:
            _ = (
                p
                | f"Read {task.result_key} from BQ"
                >> BqTableToRecordBatch(
                    bq_table=task.bq_table,
                    feature_columns=task.feature_columns,
                )
                | f"Generate TFDV stats for {task.result_key}"
                >> GenerateAndVisualizeStats(
                    facets_report_uri=facets_uri,
                    stats_output_uri=stats_uri,
                )
            )
        logger.info(f"Finished feature profiling for {task.result_key}.")
        return facets_uri.uri, stats_uri.uri


def _collect_profile_tasks(config: DataAnalyzerConfig) -> list[_ProfileTask]:
    """Flatten the analyzer config into one ``_ProfileTask`` per table that
    has non-empty ``feature_columns``. Tables without features are skipped.
    """
    tasks: list[_ProfileTask] = []
    for node_table in config.node_tables:
        if node_table.feature_columns:
            tasks.append(
                _ProfileTask(
                    kind="node",
                    type_name=node_table.node_type,
                    bq_table=node_table.bq_table,
                    feature_columns=list(node_table.feature_columns),
                )
            )
    for edge_table in config.edge_tables:
        if edge_table.feature_columns:
            tasks.append(
                _ProfileTask(
                    kind="edge",
                    type_name=edge_table.edge_type,
                    bq_table=edge_table.bq_table,
                    feature_columns=list(edge_table.feature_columns),
                )
            )
    return tasks
