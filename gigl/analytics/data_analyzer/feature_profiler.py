"""TFDV feature profiling via Beam/Dataflow.

Launches one Dataflow pipeline per node and edge table in the analyzer
config. For each table, the BQ projection is built from the table schema
via :func:`~gigl.analytics.data_analyzer.embedding_projection.build_projection`:
scalar profileable columns pass through, REPEATED FLOAT-family columns
(embeddings) expand into four hygiene companions
(``<col>_len``/``_has_nan``/``_has_inf``/``_is_all_zero``). Each pipeline
reads the resulting columns from BigQuery, emits ``pa.RecordBatch``
batches, and runs ``tfdv.GenerateStatistics`` to write a Facets HTML
visualization plus a TFDV stats TFRecord to GCS.

After all Dataflow pipelines finish, one aggregate BigQuery query per
embedding column runs via
:class:`~gigl.analytics.data_analyzer.embedding_diagnostics.EmbeddingDiagnostics`
to compute structural sanity (unique ratio + top-K most-frequent hashes).
The final :class:`FeatureProfileResult` is serialized to
``{output_gcs_path}/feature_profile.json`` via :func:`write_artifact` so
external consumers can parse it without scraping HTML.

Tables whose final projection is empty (e.g. only ID columns, or a schema
fetch failed) are skipped with a warning. Per-table Beam failures, the
diagnostics pass, and the sidecar write are all best-effort: the TFDV
artifacts remain valuable even if one downstream step fails.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Optional

import apache_beam as beam
import tensorflow_data_validation as tfdv
from apache_beam.options.pipeline_options import GoogleCloudOptions
from tensorflow_data_validation.utils import slicing_util

from gigl.analytics.data_analyzer.config import DataAnalyzerConfig
from gigl.analytics.data_analyzer.embedding_diagnostics import (
    EmbeddingDiagnostics,
    EmbeddingDiagnosticsRequest,
)
from gigl.analytics.data_analyzer.embedding_projection import (
    ProjectionResult,
    build_projection,
)
from gigl.analytics.data_analyzer.types import (
    FeatureProfileError,
    FeatureProfileResult,
    write_artifact,
)
from gigl.common import UriFactory
from gigl.common.beam.sharded_read import BigQueryShardedReadConfig
from gigl.common.beam.tfdv_transforms import (
    BqTableToRecordBatch,
    GenerateAndVisualizeStats,
)
from gigl.common.logger import Logger
from gigl.env.pipelines_config import GiglResourceConfigWrapper
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.utils.bq import BqUtils
from gigl.src.common.utils.dataflow import init_beam_pipeline_options

logger = Logger()

_PARALLEL_DATAFLOW_WORKERS = 10
# Kept short to leave room for the per-run prefix and timestamp inside
# the Dataflow job-name budget (~63 chars).
_APPLIED_TASK_IDENTIFIER = AppliedTaskIdentifier("analyzer")


def _safe_dataflow_job_id(result: Any) -> Optional[str]:
    """Return ``result.job_id()`` if present, else ``None``.

    The DataflowRunner returns a ``DataflowPipelineResult`` whose
    ``job_id()`` method exposes the submitted job's UUID. Other runners
    (DirectRunner, etc.) don't have this attribute; we degrade silently
    instead of raising so callers can keep an unrelated failure path
    clean.
    """
    job_id_attr = getattr(result, "job_id", None)
    if job_id_attr is None:
        return None
    try:
        if callable(job_id_attr):
            value = job_id_attr()
        else:
            value = job_id_attr
    except Exception:
        return None
    return str(value) if value else None


def _build_dataflow_console_url(
    project: Optional[str], region: Optional[str], job_id: Optional[str]
) -> Optional[str]:
    """Compose the Cloud Console URL for a Dataflow job.

    Returns ``None`` if any of project / region / job_id is missing,
    rather than producing a malformed URL.
    """
    if not project or not region or not job_id:
        return None
    return (
        f"https://console.cloud.google.com/dataflow/jobs/{region}/{job_id}"
        f"?project={project}"
    )


def _resolve_projection(
    bq_table: str,
    explicit: list[str],
    excluded: set[str],
    bq_utils: BqUtils,
    extra_columns: Optional[list[str]] = None,
) -> tuple[ProjectionResult, Optional[str]]:
    """Build the projection for one table, honoring an explicit override.

    If ``explicit`` is non-empty, the schema is still fetched but only
    those columns are considered (minus ``excluded``). Explicit names not
    present in the schema are logged and dropped rather than raising.
    Otherwise every non-excluded column is routed through
    :func:`build_projection`.

    ``extra_columns`` are appended to the resulting projection unconditionally
    if they exist in the schema (e.g. label / split columns the analyzer
    needs available for TFDV slicing even when the user's explicit
    ``feature_columns`` doesn't list them). Extras already present in the
    base projection are skipped to avoid duplicate SELECT entries; extras
    missing from the schema are warned about and dropped.

    Returns ``(projection_result, error_message_or_none)``. A non-None
    second element means the schema fetch failed; the caller should
    surface that as a structured error instead of just silently skipping
    the table.
    """
    try:
        schema = bq_utils.fetch_bq_table_schema(bq_table)
    except Exception as exc:
        message = f"Schema fetch failed for {bq_table}: {exc}"
        logger.warning(message)
        return ProjectionResult(projection=[], embedding_columns=[]), message

    if explicit:
        unknown = [c for c in explicit if c not in schema]
        if unknown:
            logger.warning(
                f"{bq_table}: explicit feature_columns {unknown} not in "
                f"schema; ignoring."
            )
        filtered_schema = {
            name: field
            for name, field in schema.items()
            if name in explicit and name not in excluded
        }
        base = build_projection(filtered_schema, excluded=set())
    else:
        base = build_projection(schema, excluded=excluded)

    if extra_columns:
        existing_names = {name for name, _ in base.projection}
        extras_schema = {}
        for column in extra_columns:
            if column in existing_names:
                continue
            if column not in schema:
                logger.warning(
                    f"{bq_table}: extra projection column {column!r} not in "
                    f"schema; ignoring."
                )
                continue
            extras_schema[column] = schema[column]
        if extras_schema:
            extras_projection = build_projection(extras_schema, excluded=set())
            base = ProjectionResult(
                projection=list(base.projection) + list(extras_projection.projection),
                embedding_columns=list(base.embedding_columns),
            )

    return base, None


@dataclass(frozen=True)
class _ProfileTask:
    """One profiling unit: all columns of a single node or edge table.

    ``kind`` is ``"node"`` or ``"edge"`` (singular) and is used to build
    the GCS output path and the result key (``"node:user"``, etc.).

    ``shard_key`` is the column the BQ read fans out on (hash-mod-N) to
    avoid the single-giant-export pattern that hangs ``SplitWithSizing``
    on very large tables. Sourced from ``NodeTableSpec.id_column`` for
    node tables and ``EdgeTableSpec.src_id_column`` for edge tables —
    both are guaranteed present and uniformly distributed enough for a
    FARM_FINGERPRINT-based mod split.

    ``slice_columns`` lists columns whose distinct values should each
    produce a slice of the TFDV stats. The values come from
    ``NodeTableSpec.label_column`` / ``NodeTableSpec.split_column`` —
    when set, the profiler routes them through ``slicing_util`` so the
    resulting TFDV stats include per-slice ``DatasetFeatureStatistics``
    entries (per-class label histograms, per-class feature null-rate,
    per-split distributions). Empty for edge tables and for node tables
    that don't activate NC supervision.
    """

    kind: str
    type_name: str
    bq_table: str
    projection: list[tuple[str, str]]
    embedding_columns: list[str]
    shard_key: str
    slice_columns: list[str] = field(default_factory=list)
    chunk_index: int = 0
    total_chunks: int = 1

    @property
    def result_key(self) -> str:
        return f"{self.kind}:{self.type_name}"

    @property
    def artifact_subdir(self) -> str:
        """Empty for single-chunk tables; ``chunk_NN/`` for multi-chunk tables.

        Multi-chunk tables write each chunk's Facets HTML + stats TFRecord
        under their own ``chunk_NN/`` subdir to avoid collisions; single-chunk
        tables keep the historical flat layout for backward-compatible URLs.
        """
        if self.total_chunks <= 1:
            return ""
        return f"chunk_{self.chunk_index:02d}/"


class FeatureProfiler:
    """Runs TFDV feature profiling + embedding diagnostics on BQ tables via Dataflow.

    Example:
        >>> profiler = FeatureProfiler()
        >>> result = profiler.profile(config, resource_config=config)
        >>> result.facets_html_paths["node:user"]
        'gs://bucket/analyzer/feature_profiler/nodes/user/facets.html'
    """

    def profile(
        self,
        config: DataAnalyzerConfig,
        resource_config: GiglResourceConfigWrapper,
        job_name_prefix: str,
        run_timestamp: str,
        custom_worker_image_uri: Optional[str] = None,
    ) -> FeatureProfileResult:
        """Run TFDV profiling + embedding diagnostics for every table in the config.

        For each table, the BQ projection is built via
        :func:`_resolve_projection` (explicit ``feature_columns`` narrow the
        schema; otherwise every non-excluded column is considered).
        Embedding columns (REPEATED FLOAT families) expand into hygiene
        companions in the projection and trigger a post-Dataflow structural
        diagnostics pass.

        Tables whose final projection is empty are skipped with a warning.
        Per-table Dataflow failures are logged and omitted. The embedding
        diagnostics pass and JSON sidecar write are best-effort.

        Args:
            config: Analyzer configuration with node and edge table specs.
            resource_config: Resource config; its ``.project`` is used for
                BigQuery schema lookups and diagnostics queries.
            job_name_prefix: User-supplied prefix mixed into every per-table
                Dataflow job name to disambiguate concurrent / repeat runs.
            run_timestamp: Per-run timestamp string mixed into every per-table
                Dataflow job name. Computed once at the entry point so all
                jobs from one analyzer invocation share the same value.
            custom_worker_image_uri: Optional Docker image URI for the
                Dataflow worker harness. When ``None``, falls back to
                ``DEFAULT_GIGL_RELEASE_SRC_IMAGE_DATAFLOW_CPU``.

        Returns:
            :class:`FeatureProfileResult` with GCS paths keyed by
            ``"node:{type}"`` / ``"edge:{type}"`` plus any embedding
            diagnostics that succeeded. Empty facets / stats paths indicate
            a skipped or failed table.
        """
        bq_utils = BqUtils(project=resource_config.project)
        tasks, collection_errors = _collect_profile_tasks(config, bq_utils)
        result = FeatureProfileResult()
        result.errors.extend(collection_errors)
        if not tasks:
            logger.info("No tables have profileable columns; returning empty result.")
            self._maybe_write_sidecar(result, config.output_gcs_path)
            return result

        logger.info(f"Launching {len(tasks)} Dataflow feature-profile job(s).")
        with ThreadPoolExecutor(max_workers=_PARALLEL_DATAFLOW_WORKERS) as executor:
            future_to_task = {
                executor.submit(
                    self._run_single_pipeline,
                    task,
                    config.output_gcs_path,
                    resource_config,
                    job_name_prefix,
                    run_timestamp,
                    custom_worker_image_uri,
                ): task
                for task in tasks
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    facets_uri, stats_uri = future.result()
                    # ``setdefault`` keeps multi-chunk per-table aggregation safe
                    # under the unordered ``as_completed`` iteration: each chunk
                    # lands as a list entry under the table-level result_key.
                    result.facets_html_paths.setdefault(task.result_key, []).append(
                        facets_uri
                    )
                    result.stats_paths.setdefault(task.result_key, []).append(stats_uri)
                    if task.slice_columns:
                        result.slice_columns_by_result_key[task.result_key] = list(
                            task.slice_columns
                        )
                except Exception as exc:
                    logger.exception(
                        f"Feature profiling failed for {task.result_key} "
                        f"(table={task.bq_table}): {exc}"
                    )
                    result.errors.append(
                        FeatureProfileError(
                            result_key=task.result_key,
                            bq_table=task.bq_table,
                            stage="dataflow",
                            message=f"{type(exc).__name__}: {exc}",
                            job_id=getattr(exc, "_gigl_job_id", None),
                            job_name=getattr(exc, "_gigl_job_name", None),
                            console_url=getattr(exc, "_gigl_console_url", None),
                        )
                    )

        self._run_embedding_diagnostics(tasks, bq_utils, result)
        self._maybe_write_sidecar(result, config.output_gcs_path)
        return result

    def _run_single_pipeline(
        self,
        task: _ProfileTask,
        output_gcs_path: str,
        resource_config: GiglResourceConfigWrapper,
        job_name_prefix: str,
        run_timestamp: str,
        custom_worker_image_uri: Optional[str] = None,
    ) -> tuple[str, str]:
        """Build, run, and block on a single table's Dataflow pipeline.

        Returns the ``(facets_uri, stats_uri)`` strings on success.

        Worker sizing (machine_type / num_workers / max_num_workers /
        disk_size_gb / timeout) is read from
        ``resource_config.preprocessor_config.node_preprocessor_config`` for
        node tasks and ``.edge_preprocessor_config`` for edge tasks. The
        analyzer reuses the preprocessor's Dataflow sizing on the same
        kind of table rather than declaring its own block, mirroring the
        pattern in
        :func:`gigl.src.data_preprocessor.lib.transform.utils.transform_features`.

        Captures the Dataflow ``job_id`` / ``job_name`` / console URL on the
        raised exception (as ``_gigl_*`` attributes) when the pipeline fails
        on a Dataflow runner. The caller reads those off the exception and
        promotes them into a :class:`FeatureProfileError` so the HTML report
        can deep-link to the failed job's logs. Best-effort: a non-Dataflow
        runner (e.g. DirectRunner in tests) yields ``None`` for job_id.
        """
        # Single-chunk tables keep the historical flat layout
        # (``.../{type}/facets.html``); multi-chunk tables write each chunk
        # under its own ``chunk_NN/`` subdir so the stats / Facets per chunk
        # don't collide.
        base = (
            f"{output_gcs_path.rstrip('/')}/feature_profiler/"
            f"{task.kind}s/{task.type_name}/{task.artifact_subdir}"
        ).rstrip("/")
        facets_uri = UriFactory.create_uri(f"{base}/facets.html")
        stats_uri = UriFactory.create_uri(f"{base}/stats.tfrecord")

        if task.kind == "node":
            dataflow_config = (
                resource_config.preprocessor_config.node_preprocessor_config
            )
        elif task.kind == "edge":
            dataflow_config = (
                resource_config.preprocessor_config.edge_preprocessor_config
            )
        else:
            raise ValueError(
                f"Unexpected task.kind={task.kind!r}; expected 'node' or 'edge'."
            )

        # Append a chunk suffix to the Dataflow job-name only when the table
        # is actually being chunked, to keep single-chunk job names stable
        # and within Dataflow's 63-char job-name budget for the common case.
        chunk_suffix = (
            f"-chunk-{task.chunk_index:02d}-of-{task.total_chunks:02d}"
            if task.total_chunks > 1
            else ""
        )
        options = init_beam_pipeline_options(
            applied_task_identifier=_APPLIED_TASK_IDENTIFIER,
            job_name_suffix=(
                f"{job_name_prefix}-{run_timestamp}-profile-"
                f"{task.kind}-{task.type_name}{chunk_suffix}"
            ),
            component=GiGLComponents.DataAnalyzer,
            custom_worker_image_uri=custom_worker_image_uri,
            timeout_seconds=dataflow_config.timeout
            if dataflow_config.timeout
            else None,
            num_workers=dataflow_config.num_workers,
            max_num_workers=dataflow_config.max_num_workers,
            machine_type=dataflow_config.machine_type,
            disk_size_gb=dataflow_config.disk_size_gb,
        )
        gcp_opts = options.view_as(GoogleCloudOptions)
        job_name = gcp_opts.job_name
        project = gcp_opts.project
        region = gcp_opts.region

        stats_options = _build_slice_stats_options(task.slice_columns)

        # Shard the BQ read on the natural per-table key (id_column for nodes,
        # src_id_column for edges). Mirrors the data_preprocessor's
        # ShardedExportRead pattern; without it, a single giant ReadFromBigQuery
        # on a large user/edge table hangs Dataflow's SplitWithSizing on
        # oversized GCS Avro reads. ``num_shards`` defaults to 20 inside the
        # config dataclass (matches the preprocessor default).
        sharded_read_config = BigQueryShardedReadConfig(
            shard_key=task.shard_key,
            project_id=resource_config.project,
            temp_dataset_name=resource_config.temp_assets_bq_dataset_name,
        )

        pipeline = beam.Pipeline(options=options)
        _ = (
            pipeline
            | f"Read {task.result_key} from BQ"
            >> BqTableToRecordBatch(
                bq_table=task.bq_table,
                projection=task.projection,
                sharded_read_config=sharded_read_config,
            )
            | f"Generate TFDV stats for {task.result_key}"
            >> GenerateAndVisualizeStats(
                facets_report_uri=facets_uri,
                stats_output_uri=stats_uri,
                stats_options=stats_options,
            )
        )
        result = pipeline.run()
        try:
            result.wait_until_finish()
        except Exception as exc:
            job_id = _safe_dataflow_job_id(result)
            console_url = _build_dataflow_console_url(
                project=project, region=region, job_id=job_id
            )
            exc._gigl_job_id = job_id  # type: ignore[attr-defined]
            exc._gigl_job_name = job_name  # type: ignore[attr-defined]
            exc._gigl_console_url = console_url  # type: ignore[attr-defined]
            raise
        logger.info(f"Finished feature profiling for {task.result_key}.")
        return facets_uri.uri, stats_uri.uri

    def _run_embedding_diagnostics(
        self,
        tasks: list[_ProfileTask],
        bq_utils: BqUtils,
        result: FeatureProfileResult,
    ) -> None:
        """Run structural diagnostics for every task with embedding columns.

        Best-effort: any exception is caught so the sidecar write and the
        already-produced TFDV artifacts remain valuable.

        Multi-chunk tables emit multiple ``_ProfileTask``s with the same
        ``result_key`` and ``embedding_columns`` (table-level). We dedupe
        per ``result_key`` so the embedding-diagnostics BQ aggregate runs
        once per table, not once per chunk.
        """
        deduped: dict[str, EmbeddingDiagnosticsRequest] = {}
        for task in tasks:
            if not task.embedding_columns:
                continue
            existing = deduped.get(task.result_key)
            if existing is None:
                deduped[task.result_key] = EmbeddingDiagnosticsRequest(
                    result_key=task.result_key,
                    bq_table=task.bq_table,
                    embedding_columns=list(task.embedding_columns),
                )
                continue
            # Same result_key seen on a previous chunk — union the embedding
            # columns to be safe against any chunk that happens to carry a
            # narrower embedding subset (chunks share table-level
            # embedding_columns today, but defensive).
            seen = set(existing.embedding_columns)
            extra = [c for c in task.embedding_columns if c not in seen]
            if extra:
                deduped[task.result_key] = EmbeddingDiagnosticsRequest(
                    result_key=existing.result_key,
                    bq_table=existing.bq_table,
                    embedding_columns=existing.embedding_columns + extra,
                )
        requests = list(deduped.values())
        if not requests:
            return
        try:
            diagnostics = EmbeddingDiagnostics(bq_utils=bq_utils).analyze(requests)
        except Exception as exc:
            logger.exception(f"Embedding diagnostics pass failed: {exc}")
            message = f"{type(exc).__name__}: {exc}"
            for request in requests:
                result.errors.append(
                    FeatureProfileError(
                        result_key=request.result_key,
                        bq_table=request.bq_table,
                        stage="embedding_diagnostics",
                        message=message,
                    )
                )
            return
        for result_key, per_column in diagnostics.items():
            result.embedding_diagnostics[result_key] = per_column

    def _maybe_write_sidecar(
        self, result: FeatureProfileResult, output_gcs_path: str
    ) -> None:
        """Best-effort write of the Pydantic JSON sidecar."""
        try:
            write_artifact(
                result=result,
                component="feature_profile",
                output_gcs_path=output_gcs_path,
            )
        except Exception as exc:
            logger.exception(f"Failed to write feature_profile.json sidecar: {exc}")


def _build_slice_stats_options(
    slice_columns: list[str],
) -> Optional[tfdv.StatsOptions]:
    """Build a ``tfdv.StatsOptions`` configured to slice on the given columns.

    Returns ``None`` when no slice columns are requested so callers can
    cheaply pass through to TFDV's defaults. Each entry produces a
    standard "feature value slicer" that emits one slice per distinct
    value of the column. The unsliced ("Overall") stats are always
    emitted by TFDV in addition to the per-slice stats, so existing
    consumers continue to see the same top-level stats they did before
    slicing was enabled.
    """
    if not slice_columns:
        return None
    slice_functions = [
        slicing_util.get_feature_value_slicer({column: None})
        for column in slice_columns
    ]
    return tfdv.StatsOptions(slice_functions=slice_functions)


def _chunk_projection(
    projection: list[tuple[str, str]],
    max_features: int,
    forced_columns: set[str],
) -> list[list[tuple[str, str]]]:
    """Slice a projection into ``ceil(len/max_features)`` ≤``max_features``-sized chunks.

    Beam 2.56's runner-v2 cannot reliably iterate the per-key state TFDV's
    ``CombinePerKey(PreCombineFn)`` accumulates over very wide projections
    (work items time out on ``Instruction id ... was not registered``).
    Splitting the projection across multiple Dataflow pipelines keeps
    every per-key partition small enough for the runner to iterate.

    ``forced_columns`` (typically slice columns: ``label_column`` /
    ``split_column``) are present in **every** chunk so TFDV slicing
    applies uniformly across chunks. Each chunk's effective non-forced
    budget is ``max_features - len(forced_pairs)`` (clamped to ≥1).

    Args:
        projection: ``(column_name, sql_expression)`` pairs from
            :func:`_resolve_projection`. Slice columns are already in here
            (via that function's ``extra_columns``).
        max_features: Target per-chunk column cap. The actual chunk size
            is ``max_features`` for non-forced columns plus the forced
            columns appended.
        forced_columns: Names that must appear in every chunk.

    Returns:
        Non-empty list of chunks. Empty input returns ``[]``.
    """
    forced_pairs = [(n, e) for n, e in projection if n in forced_columns]
    rest = [(n, e) for n, e in projection if n not in forced_columns]
    if not rest:
        return [list(forced_pairs)] if forced_pairs else []
    budget_per_chunk = max(1, max_features - len(forced_pairs))
    chunks: list[list[tuple[str, str]]] = []
    for start in range(0, len(rest), budget_per_chunk):
        chunks.append(list(forced_pairs) + rest[start : start + budget_per_chunk])
    return chunks


def _collect_profile_tasks(
    config: DataAnalyzerConfig, bq_utils: BqUtils
) -> tuple[list[_ProfileTask], list[FeatureProfileError]]:
    """Flatten the analyzer config into one ``_ProfileTask`` per table.

    Resolves the projection for each node/edge spec by either restricting
    to explicit ``feature_columns`` or auto-inferring from the BQ table
    schema (excluding structural join keys). Tables whose resolved
    projection is empty (e.g. only ID columns, or the schema fetch failed)
    are logged, recorded as a structured ``FeatureProfileError`` so the
    HTML report can surface them, and skipped.
    """
    tasks: list[_ProfileTask] = []
    errors: list[FeatureProfileError] = []
    for node_table in config.node_tables:
        result_key = f"node:{node_table.node_type}"
        # Slice columns must be in the projection so TFDV can read them.
        # ``compute_per_class_feature_stats`` opts out of the label slice
        # without forcing the user to drop ``label_column`` itself (the
        # graph_structure_analyzer NC tier still needs the column there).
        slice_columns: list[str] = []
        if (
            node_table.label_column is not None
            and config.compute_per_class_feature_stats
        ):
            slice_columns.append(node_table.label_column)
        if node_table.split_column is not None:
            slice_columns.append(node_table.split_column)

        projection, schema_error = _resolve_projection(
            bq_table=node_table.bq_table,
            explicit=node_table.feature_columns,
            excluded={node_table.id_column},
            bq_utils=bq_utils,
            extra_columns=slice_columns,
        )
        if schema_error is not None:
            errors.append(
                FeatureProfileError(
                    result_key=result_key,
                    bq_table=node_table.bq_table,
                    stage="schema_fetch",
                    message=schema_error,
                )
            )
            continue
        if not projection.projection:
            message = (
                f"No profileable columns after projection "
                f"(id_column={node_table.id_column!r}, "
                f"explicit feature_columns={node_table.feature_columns})."
            )
            logger.warning(f"Skipping {result_key}: {message}")
            errors.append(
                FeatureProfileError(
                    result_key=result_key,
                    bq_table=node_table.bq_table,
                    stage="empty_projection",
                    message=message,
                )
            )
            continue
        # Slice columns that didn't make it into the projection (missing
        # from schema) are dropped; ``_resolve_projection`` already logged.
        projected_names = {name for name, _ in projection.projection}
        active_slice_columns = [
            column for column in slice_columns if column in projected_names
        ]
        chunks = _chunk_projection(
            projection.projection,
            max_features=config.max_features_per_chunk,
            forced_columns=set(active_slice_columns),
        )
        for chunk_index, chunk_projection in enumerate(chunks):
            tasks.append(
                _ProfileTask(
                    kind="node",
                    type_name=node_table.node_type,
                    bq_table=node_table.bq_table,
                    projection=chunk_projection,
                    embedding_columns=projection.embedding_columns,
                    shard_key=node_table.id_column,
                    slice_columns=active_slice_columns,
                    chunk_index=chunk_index,
                    total_chunks=len(chunks),
                )
            )
    for edge_table in config.edge_tables:
        result_key = f"edge:{edge_table.edge_type}"
        projection, schema_error = _resolve_projection(
            bq_table=edge_table.bq_table,
            explicit=edge_table.feature_columns,
            excluded={
                edge_table.src_id_column,
                edge_table.dst_id_column,
            },
            bq_utils=bq_utils,
        )
        if schema_error is not None:
            errors.append(
                FeatureProfileError(
                    result_key=result_key,
                    bq_table=edge_table.bq_table,
                    stage="schema_fetch",
                    message=schema_error,
                )
            )
            continue
        if not projection.projection:
            message = (
                f"No profileable columns after projection "
                f"(src_id_column={edge_table.src_id_column!r}, "
                f"dst_id_column={edge_table.dst_id_column!r}, "
                f"explicit feature_columns={edge_table.feature_columns})."
            )
            logger.warning(f"Skipping {result_key}: {message}")
            errors.append(
                FeatureProfileError(
                    result_key=result_key,
                    bq_table=edge_table.bq_table,
                    stage="empty_projection",
                    message=message,
                )
            )
            continue
        chunks = _chunk_projection(
            projection.projection,
            max_features=config.max_features_per_chunk,
            forced_columns=set(),
        )
        for chunk_index, chunk_projection in enumerate(chunks):
            tasks.append(
                _ProfileTask(
                    kind="edge",
                    type_name=edge_table.edge_type,
                    bq_table=edge_table.bq_table,
                    projection=chunk_projection,
                    embedding_columns=projection.embedding_columns,
                    shard_key=edge_table.src_id_column,
                    chunk_index=chunk_index,
                    total_chunks=len(chunks),
                )
            )
    return tasks, errors
