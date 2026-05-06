"""GraphStructureAnalyzer: 4-tier BigQuery-based graph data quality checks.

Tier 1 (hard fails)
    dangling edges, referential integrity, duplicate nodes. Any violation
    raises DataQualityError with a partially populated GraphAnalysisResult.

Tier 2 (core metrics)
    node/edge counts, degree distribution, top-K hubs, INT16 clamp hazards,
    isolated/cold-start nodes, duplicate edges, self-loops, NULL rates, and
    two Python-side computations (feature memory budget, neighbor explosion).

Tier 3 (label and heterogeneous)
    class imbalance and label coverage (auto-enabled when node_tables have a
    label_column); edge-type distribution and per-edge-type node coverage
    (auto-enabled when more than one edge table is declared).

Tier 4 (opt-in)
    reciprocity, power-law exponent estimate. Gated by config flags.
"""

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from gigl.analytics.data_analyzer.config import (
    EDGE_ROLE_MESSAGE_PASSING,
    EDGE_ROLE_SUPERVISION_NEG,
    EDGE_ROLE_SUPERVISION_POS,
    DataAnalyzerConfig,
    EdgeTableSpec,
    NodeTableSpec,
)
from gigl.analytics.data_analyzer.queries import (
    CLASS_IMBALANCE_QUERY,
    COLD_START_NODE_COUNT_QUERY,
    CROSS_SPLIT_OVERLAP_QUERY,
    DANGLING_EDGES_QUERY,
    DEGREE_BUCKET_QUERY,
    DEGREE_DISTRIBUTION_QUERY,
    DUPLICATE_EDGE_COUNT_QUERY,
    DUPLICATE_NODE_COUNT_QUERY,
    EDGE_COUNT_QUERY,
    EDGE_REFERENTIAL_INTEGRITY_QUERY,
    EDGE_TYPE_DISTRIBUTION_QUERY,
    EDGE_TYPE_NODE_COVERAGE_QUERY,
    ISOLATED_NODE_COUNT_QUERY,
    LABEL_COVERAGE_QUERY,
    NODE_COUNT_QUERY,
    SELF_LOOP_COUNT_QUERY,
    SPLIT_VALUE_COUNTS_QUERY,
    SUPER_HUB_INT16_CLAMP_QUERY,
    SUPERVISION_CROSS_TABLE_QUERY,
    TOP_K_HUBS_QUERY,
    build_adjusted_homophily_query,
    build_label_sentinel_query,
    build_null_rates_query,
    build_per_class_degree_query,
)
from gigl.analytics.data_analyzer.types import (
    CrossSplitOverlap,
    DegreeStats,
    GraphAnalysisResult,
    HomophilyStats,
    LabelSentinelStats,
    NodeClassificationSupervisionStats,
    PerClassDegreeStats,
    SupervisionCrossTableStats,
    write_artifact,
)
from gigl.common.logger import Logger
from gigl.src.common.utils.bq import BqUtils

logger = Logger()

# Default assumption for feature memory budget: float64 per feature column.
_BYTES_PER_FEATURE = 8
_TOP_K_HUBS = 20
_PARALLEL_BQ_WORKERS = 10


class DataQualityError(Exception):
    """Raised when Tier 1 hard-fail checks detect data quality violations.

    Carries a partially populated GraphAnalysisResult so callers can inspect
    which specific checks failed without re-running the analyzer.
    """

    def __init__(self, message: str, partial_result: GraphAnalysisResult) -> None:
        super().__init__(message)
        self.partial_result = partial_result


class GraphStructureAnalyzer:
    """Runs BigQuery SQL checks across 4 tiers against the tables declared in a config.

    Example:
        >>> config = load_analyzer_config("gs://bucket/config.yaml")
        >>> analyzer = GraphStructureAnalyzer()
        >>> result = analyzer.analyze(config)
        >>> result.node_counts["user"]
        1000000

    Tier 1 is blocking: a violation raises DataQualityError before Tiers 2-4 run.
    Tiers 2-4 are aggregated best-effort into a single GraphAnalysisResult.
    """

    def __init__(self, bq_project: Optional[str] = None) -> None:
        self._bq_utils = BqUtils(project=bq_project)
        self._query_log: dict[str, list[str]] = {}

    def analyze(self, config: DataAnalyzerConfig) -> GraphAnalysisResult:
        """Run all applicable tiers and return aggregated results.

        Always writes a versioned JSON sidecar to
        ``{config.output_gcs_path}/graph_structure.json`` before returning
        (or re-raising), so partial Tier 1 failures are recoverable by
        downstream consumers without rerunning the analyzer.

        Args:
            config: Data analyzer configuration declaring node and edge tables
                plus any opt-in expensive checks (reciprocity, etc.).

        Returns:
            GraphAnalysisResult with tier 1-4 fields populated per config.

        Raises:
            DataQualityError: If tier 1 checks find any violations. The
                exception carries a partial result with the specific counts;
                that same partial result is persisted to the sidecar.
        """
        self._query_log = {}
        result = GraphAnalysisResult()
        try:
            logger.info("Starting graph structure analysis (Tier 1: hard fails)")
            self._run_tier1(config, result)

            logger.info("Tier 1 passed. Running Tier 2 (core metrics)")
            self._run_tier2(config, result)

            logger.info("Running Tier 3 (label / heterogeneous)")
            self._run_tier3(config, result)

            logger.info("Running node-classification supervision tier")
            self._run_node_classification_supervision(config, result)

            logger.info("Running supervision cross-table analysis")
            self._run_supervision_cross_table(config, result)

            logger.info("Running Tier 4 (opt-in)")
            self._run_tier4(config, result)
        except DataQualityError as err:
            err.partial_result.queries = dict(self._query_log)
            self._maybe_write_sidecar(err.partial_result, config.output_gcs_path)
            raise
        result.queries = dict(self._query_log)
        self._maybe_write_sidecar(result, config.output_gcs_path)
        return result

    def _maybe_write_sidecar(
        self, result: GraphAnalysisResult, output_gcs_path: str
    ) -> None:
        """Best-effort write of the Pydantic JSON sidecar.

        Never raises: the sidecar is a convenience artifact, not a
        correctness contract. Failures are logged and swallowed so Tier 1
        errors (which also trigger a sidecar write) propagate intact.
        """
        try:
            write_artifact(
                result=result,
                component="graph_structure",
                output_gcs_path=output_gcs_path,
            )
        except Exception as exc:
            logger.exception(f"Failed to write graph_structure.json sidecar: {exc}")

    # ------------------------------------------------------------------ #
    # Tier 1: hard fails                                                  #
    # ------------------------------------------------------------------ #

    def _run_tier1(
        self, config: DataAnalyzerConfig, result: GraphAnalysisResult
    ) -> None:
        """Run all tier 1 checks; raise DataQualityError on any violation."""
        violations: list[str] = []
        node_tables_by_type = {nt.node_type: nt for nt in config.node_tables}

        # Duplicate nodes (per node table).
        for node_table in config.node_tables:
            query = DUPLICATE_NODE_COUNT_QUERY.format(
                table=node_table.bq_table, id_column=node_table.id_column
            )
            count = self._query_scalar(
                query,
                "duplicate_count",
                block_id=f"data_quality:duplicate_nodes:{node_table.node_type}",
            )
            result.duplicate_node_counts[node_table.node_type] = count
            if count > 0:
                violations.append(
                    f"node_type={node_table.node_type} has {count} duplicate IDs"
                )

        # Dangling edges and referential integrity (per edge table).
        for edge_table in config.edge_tables:
            dangling_query = DANGLING_EDGES_QUERY.format(
                table=edge_table.bq_table,
                src_id_column=edge_table.src_id_column,
                dst_id_column=edge_table.dst_id_column,
            )
            dangling = self._query_scalar(
                dangling_query,
                "dangling_count",
                block_id=f"data_quality:dangling_edges:{edge_table.edge_type}",
            )
            result.dangling_edge_counts[edge_table.edge_type] = dangling
            if dangling > 0:
                violations.append(
                    f"edge_type={edge_table.edge_type} has {dangling} dangling edges"
                )

            # Referential integrity: src and dst can resolve to different node
            # tables on heterogeneous graphs. `load_analyzer_config` guarantees
            # src_node_type / dst_node_type are populated and known.
            if not config.node_tables:
                continue
            assert edge_table.src_node_type is not None, (
                f"edge_type={edge_table.edge_type} has no src_node_type; "
                "load the config via load_analyzer_config to backfill it."
            )
            assert edge_table.dst_node_type is not None, (
                f"edge_type={edge_table.edge_type} has no dst_node_type; "
                "load the config via load_analyzer_config to backfill it."
            )
            src_node_table = node_tables_by_type[edge_table.src_node_type]
            dst_node_table = node_tables_by_type[edge_table.dst_node_type]
            ref_query = EDGE_REFERENTIAL_INTEGRITY_QUERY.format(
                edge_table=edge_table.bq_table,
                src_node_table=src_node_table.bq_table,
                dst_node_table=dst_node_table.bq_table,
                src_id_column=edge_table.src_id_column,
                dst_id_column=edge_table.dst_id_column,
                src_node_id_column=src_node_table.id_column,
                dst_node_id_column=dst_node_table.id_column,
            )
            self._record_query(
                f"data_quality:referential_integrity:{edge_table.edge_type}",
                ref_query,
            )
            rows = list(self._bq_utils.run_query(query=ref_query, labels={}))
            if len(rows) != 1:
                raise RuntimeError(
                    f"Referential integrity query expected exactly 1 row; "
                    f"got {len(rows)}. Query: {ref_query.strip()[:200]}"
                )
            missing_src = int(rows[0]["missing_src_count"] or 0)
            missing_dst = int(rows[0]["missing_dst_count"] or 0)
            total_missing = missing_src + missing_dst
            result.referential_integrity_violations[
                edge_table.edge_type
            ] = total_missing
            if total_missing > 0:
                violations.append(
                    f"edge_type={edge_table.edge_type} has {total_missing} "
                    "referential integrity violations"
                )

        if violations:
            msg = "Tier 1 data quality violations detected:\n  - " + "\n  - ".join(
                violations
            )
            logger.error(msg)
            raise DataQualityError(msg, partial_result=result)

    # ------------------------------------------------------------------ #
    # Tier 2: core metrics                                                #
    # ------------------------------------------------------------------ #

    def _run_tier2(
        self, config: DataAnalyzerConfig, result: GraphAnalysisResult
    ) -> None:
        """Collect core structural metrics, fanning out BQ jobs in parallel.

        Edge-level metrics are computed from the src-side perspective:
        isolated/cold-start joins pair each edge with its src_node_type's
        table. Hetero dst-perspective coverage is exposed separately via
        Tier 3 edge_type_node_coverage.

        BQ jobs are I/O-bound so ThreadPoolExecutor is used. Each worker
        writes to distinct keys of the shared `result` dict (one key per
        node_type / edge_type), so no lock is required under CPython's GIL.
        """
        node_tables_by_type = {nt.node_type: nt for nt in config.node_tables}

        with ThreadPoolExecutor(max_workers=_PARALLEL_BQ_WORKERS) as executor:
            futures = []
            for node_table in config.node_tables:
                futures.append(
                    executor.submit(self._tier2_node_metrics, node_table, result)
                )
            for edge_table in config.edge_tables:
                src_node_table = node_tables_by_type.get(edge_table.src_node_type or "")
                futures.append(
                    executor.submit(
                        self._tier2_edge_metrics, edge_table, src_node_table, result
                    )
                )
            for future in futures:
                future.result()  # re-raise any exception

        # Python-side computations run after all BQ data is collected.
        self._compute_feature_memory_budget(config, result)
        self._compute_neighbor_explosion_estimate(config, result)

    def _tier2_node_metrics(
        self, node_table: NodeTableSpec, result: GraphAnalysisResult
    ) -> None:
        node_count_query = NODE_COUNT_QUERY.format(table=node_table.bq_table)
        node_count = self._query_scalar(
            node_count_query,
            "node_count",
            block_id=f"graph_structure:node_count:{node_table.node_type}",
        )
        result.node_counts[node_table.node_type] = node_count

        columns_to_check: list[str] = [node_table.id_column]
        columns_to_check.extend(node_table.feature_columns)
        if node_table.label_column:
            columns_to_check.append(node_table.label_column)

        null_query = build_null_rates_query(
            table=node_table.bq_table, columns=columns_to_check
        )
        self._record_query(
            f"data_quality:null_rates:node:{node_table.node_type}", null_query
        )
        rows = list(self._bq_utils.run_query(query=null_query, labels={}))
        if rows:
            row = rows[0]
            rates: dict[str, float] = {}
            for col in columns_to_check:
                key = f"{col}_null_rate"
                rate = row[key]
                rates[col] = float(rate) if rate is not None else 0.0
            result.null_rates[node_table.node_type] = rates

    def _tier2_edge_metrics(
        self,
        edge_table: EdgeTableSpec,
        node_table: Optional[NodeTableSpec],
        result: GraphAnalysisResult,
    ) -> None:
        edge_type = edge_table.edge_type

        # Scalar counts.
        edge_count_query = EDGE_COUNT_QUERY.format(table=edge_table.bq_table)
        result.edge_counts[edge_type] = self._query_scalar(
            edge_count_query,
            "edge_count",
            block_id=f"graph_structure:edge_count:{edge_type}",
        )
        duplicate_edges_query = DUPLICATE_EDGE_COUNT_QUERY.format(
            table=edge_table.bq_table,
            src_id_column=edge_table.src_id_column,
            dst_id_column=edge_table.dst_id_column,
        )
        result.duplicate_edge_counts[edge_type] = self._query_scalar(
            duplicate_edges_query,
            "duplicate_count",
            block_id=f"data_quality:duplicate_edges:{edge_type}",
        )
        self_loop_query = SELF_LOOP_COUNT_QUERY.format(
            table=edge_table.bq_table,
            src_id_column=edge_table.src_id_column,
            dst_id_column=edge_table.dst_id_column,
        )
        result.self_loop_counts[edge_type] = self._query_scalar(
            self_loop_query,
            "self_loop_count",
            block_id=f"graph_structure:self_loops:{edge_type}",
        )

        # Super-hub INT16 clamp check (indexed by src).
        super_hub_query = SUPER_HUB_INT16_CLAMP_QUERY.format(
            table=edge_table.bq_table, id_column=edge_table.src_id_column
        )
        result.super_hub_int16_clamp_count[edge_type] = self._query_scalar(
            super_hub_query,
            "super_hub_count",
            block_id=f"graph_structure:super_hub_clamp:{edge_type}",
        )

        # Isolated and cold-start require a node table join.
        if node_table is not None:
            isolated_query = ISOLATED_NODE_COUNT_QUERY.format(
                node_table=node_table.bq_table,
                edge_table=edge_table.bq_table,
                node_id_column=node_table.id_column,
                src_id_column=edge_table.src_id_column,
                dst_id_column=edge_table.dst_id_column,
            )
            result.isolated_node_counts[edge_type] = self._query_scalar(
                isolated_query,
                "isolated_count",
                block_id=f"graph_structure:isolated_nodes:{edge_type}",
            )
            cold_start_query = COLD_START_NODE_COUNT_QUERY.format(
                node_table=node_table.bq_table,
                edge_table=edge_table.bq_table,
                node_id_column=node_table.id_column,
                src_id_column=edge_table.src_id_column,
                dst_id_column=edge_table.dst_id_column,
            )
            result.cold_start_node_counts[edge_type] = self._query_scalar(
                cold_start_query,
                "cold_start_count",
                block_id=f"graph_structure:cold_start_nodes:{edge_type}",
            )

        # Top-K hubs (by src).
        top_hubs_query = TOP_K_HUBS_QUERY.format(
            table=edge_table.bq_table,
            id_column=edge_table.src_id_column,
            k=_TOP_K_HUBS,
        )
        self._record_query(f"graph_structure:top_hubs:{edge_type}", top_hubs_query)
        top_hub_rows = list(self._bq_utils.run_query(query=top_hubs_query, labels={}))
        result.top_hubs[edge_type] = [
            (str(row["node_id"]), int(row["degree"])) for row in top_hub_rows
        ]

        # Degree statistics: distribution + buckets, in + out directions.
        for direction, id_column in (
            ("out", edge_table.src_id_column),
            ("in", edge_table.dst_id_column),
        ):
            degree_key = f"{edge_type}_{direction}"
            result.degree_stats[degree_key] = self._build_degree_stats(
                table=edge_table.bq_table,
                id_column=id_column,
                block_id=f"graph_structure:degree:{degree_key}",
            )

    def _build_degree_stats(
        self, table: str, id_column: str, *, block_id: Optional[str] = None
    ) -> DegreeStats:
        """Run degree distribution + bucket queries and pack into DegreeStats.

        When ``block_id`` is provided both rendered SQL strings are recorded
        under that key (in distribution-then-bucket order) so the report can
        show the full pair behind the histogram + summary line.
        """
        dist_query = DEGREE_DISTRIBUTION_QUERY.format(table=table, id_column=id_column)
        bucket_query = DEGREE_BUCKET_QUERY.format(table=table, id_column=id_column)
        if block_id is not None:
            self._record_query(block_id, dist_query)
            self._record_query(block_id, bucket_query)
        dist_rows = list(self._bq_utils.run_query(query=dist_query, labels={}))
        bucket_rows = list(self._bq_utils.run_query(query=bucket_query, labels={}))
        dist_row = dist_rows[0]
        bucket_row = bucket_rows[0]

        percentiles_raw = list(dist_row["percentiles"])
        percentiles = [int(p) if p is not None else 0 for p in percentiles_raw]
        # APPROX_QUANTILES(degree, 100) returns 101 values: index 0..100.
        median = percentiles[50] if len(percentiles) > 50 else 0
        p90 = percentiles[90] if len(percentiles) > 90 else percentiles[-1]
        p99 = percentiles[99] if len(percentiles) > 99 else percentiles[-1]
        # We only have 100-bucket quantiles, so p999 ~= p99 as best-effort.
        p999 = p99

        # Bucket keys must match BUCKET_ORDER in report/charts.ai.js for the
        # histogram to render correctly; keep uppercase K.
        buckets: dict[str, int] = {
            "0-1": int(bucket_row["bucket_0_1"]),
            "2-10": int(bucket_row["bucket_2_10"]),
            "11-100": int(bucket_row["bucket_11_100"]),
            "101-1K": int(bucket_row["bucket_101_1k"]),
            "1K-10K": int(bucket_row["bucket_1k_10k"]),
            "10K+": int(bucket_row["bucket_10k_plus"]),
        }

        return DegreeStats(
            min=int(dist_row["min_degree"] or 0),
            max=int(dist_row["max_degree"] or 0),
            mean=float(dist_row["avg_degree"] or 0.0),
            median=median,
            p90=p90,
            p99=p99,
            p999=p999,
            percentiles=percentiles,
            buckets=buckets,
        )

    # ------------------------------------------------------------------ #
    # Tier 3: label and heterogeneous                                     #
    # ------------------------------------------------------------------ #

    def _run_tier3(
        self, config: DataAnalyzerConfig, result: GraphAnalysisResult
    ) -> None:
        # Label-related checks per node table with a label column.
        for node_table in config.node_tables:
            if not node_table.label_column:
                continue
            class_imbalance_query = CLASS_IMBALANCE_QUERY.format(
                table=node_table.bq_table,
                label_column=node_table.label_column,
            )
            self._record_query(
                f"advanced:class_imbalance:{node_table.node_type}",
                class_imbalance_query,
            )
            class_rows = list(
                self._bq_utils.run_query(query=class_imbalance_query, labels={})
            )
            result.class_imbalance[node_table.node_type] = {
                str(row["label"]): int(row["count"]) for row in class_rows
            }

            label_coverage_query = LABEL_COVERAGE_QUERY.format(
                table=node_table.bq_table,
                label_column=node_table.label_column,
            )
            self._record_query(
                f"advanced:label_coverage:{node_table.node_type}",
                label_coverage_query,
            )
            coverage_rows = list(
                self._bq_utils.run_query(query=label_coverage_query, labels={})
            )
            if coverage_rows:
                coverage = coverage_rows[0]["coverage"]
                result.label_coverage[node_table.node_type] = (
                    float(coverage) if coverage is not None else 0.0
                )

        # Heterogeneous distribution only if more than one edge type.
        if len(config.edge_tables) > 1:
            for edge_table in config.edge_tables:
                edge_type = edge_table.edge_type
                # Edge-type distribution is effectively the edge count; reuse.
                if edge_type in result.edge_counts:
                    result.edge_type_distribution[edge_type] = result.edge_counts[
                        edge_type
                    ]
                else:
                    edge_type_dist_query = EDGE_TYPE_DISTRIBUTION_QUERY.format(
                        table=edge_table.bq_table
                    )
                    result.edge_type_distribution[edge_type] = self._query_scalar(
                        edge_type_dist_query,
                        "edge_count",
                        block_id=f"advanced:edge_type_distribution:{edge_type}",
                    )
                coverage_query = EDGE_TYPE_NODE_COVERAGE_QUERY.format(
                    table=edge_table.bq_table,
                    src_id_column=edge_table.src_id_column,
                    dst_id_column=edge_table.dst_id_column,
                )
                self._record_query(
                    f"advanced:edge_type_node_coverage:{edge_type}", coverage_query
                )
                coverage_rows = list(
                    self._bq_utils.run_query(query=coverage_query, labels={})
                )
                if coverage_rows:
                    row = coverage_rows[0]
                    result.edge_type_node_coverage[edge_type] = {
                        "distinct_src_count": int(row["distinct_src_count"] or 0),
                        "distinct_dst_count": int(row["distinct_dst_count"] or 0),
                    }

    # ------------------------------------------------------------------ #
    # Node-classification supervision tier                                #
    # ------------------------------------------------------------------ #

    def _run_node_classification_supervision(
        self, config: DataAnalyzerConfig, result: GraphAnalysisResult
    ) -> None:
        """Run NC-supervision-tier checks for every labeled node table.

        Activates whenever a ``NodeTableSpec.label_column`` is set.
        Computes the BQ-side metrics that aren't covered by the TFDV
        slicing in the feature profiler:

        1. Sentinel-vs-NULL accounting on the label column.
        2. Per-class degree distribution (joining labels to a
           message-passing edge table).
        3. Adjusted homophily on a sampled message-passing edge set
           (raw + class-prior-adjusted, per Platonov et al. 2023).
        4. Optional label informativeness when
           ``config.compute_label_informativeness`` is True.
        5. Cross-split node-id leakage (hard fail) when
           ``NodeTableSpec.split_column`` is set.

        Hard fails (cross-split id overlap) raise
        :class:`DataQualityError` with a partially populated result, just
        like Tier 1.
        """
        message_passing_tables = [
            edge
            for edge in config.edge_tables
            if edge.role == EDGE_ROLE_MESSAGE_PASSING
        ]
        violations: list[str] = []

        for node_table in config.node_tables:
            if node_table.label_column is None:
                continue

            sentinel_stats = self._compute_label_sentinel_stats(node_table)
            per_class_degree, sentinel_degree_stats = self._compute_per_class_degree(
                node_table, message_passing_tables
            )
            homophily = self._compute_homophily_for_node_type(
                node_table, message_passing_tables, config
            )
            cross_split_overlap = self._compute_cross_split_overlap(node_table)

            stats = NodeClassificationSupervisionStats(
                node_type=node_table.node_type,
                label_column=node_table.label_column,
                sentinel_stats=sentinel_stats,
                per_class_degree=per_class_degree,
                sentinel_degree_stats=sentinel_degree_stats,
                homophily=homophily,
                cross_split_overlap=cross_split_overlap,
            )
            result.node_classification_supervision_stats.append(stats)

            if (
                cross_split_overlap is not None
                and cross_split_overlap.overlap_node_count > 0
            ):
                violations.append(
                    f"node_type={node_table.node_type}: "
                    f"{cross_split_overlap.overlap_node_count} node_ids appear "
                    f"in more than one split (column "
                    f"{node_table.split_column!r})"
                )

        if violations:
            msg = (
                "Node-classification supervision violations detected:\n  - "
                + "\n  - ".join(violations)
            )
            logger.error(msg)
            raise DataQualityError(msg, partial_result=result)

    def _compute_label_sentinel_stats(
        self, node_table: NodeTableSpec
    ) -> LabelSentinelStats:
        """Single-pass query splitting label cells into NULL / sentinel / valid."""
        assert (
            node_table.label_column is not None
        ), "_compute_label_sentinel_stats requires NodeTableSpec.label_column"
        query = build_label_sentinel_query(
            table=node_table.bq_table,
            label_column=node_table.label_column,
            sentinel_values=node_table.label_sentinel_values,
        )
        self._record_query(
            f"nc_supervision:label_sentinel:{node_table.node_type}", query
        )
        rows = list(self._bq_utils.run_query(query=query, labels={}))
        if len(rows) != 1:
            raise RuntimeError(
                f"Label sentinel query expected exactly 1 row; got {len(rows)}. "
                f"node_type={node_table.node_type}"
            )
        row = rows[0]
        total_rows = int(row["total_rows"] or 0)
        null_count = int(row["null_count"] or 0)
        valid_count = int(row["valid_count"] or 0)
        sentinel_counts: dict[str, int] = {}
        for index, sentinel in enumerate(node_table.label_sentinel_values):
            sentinel_counts[sentinel] = int(row[f"sentinel_{index}"] or 0)
        coverage = (valid_count / total_rows) if total_rows > 0 else 0.0
        return LabelSentinelStats(
            total_rows=total_rows,
            null_count=null_count,
            sentinel_counts=sentinel_counts,
            valid_label_count=valid_count,
            valid_label_coverage=coverage,
        )

    def _compute_per_class_degree(
        self,
        node_table: NodeTableSpec,
        message_passing_tables: list[EdgeTableSpec],
    ) -> tuple[list[PerClassDegreeStats], list[PerClassDegreeStats]]:
        """Per-label-value degree distribution against a message-passing edge table.

        Only edge tables whose src or dst node_type matches the labeled
        node_type are included. The edge-type identity is not preserved
        on the result here because per-class degree is defined over total
        degree (in + out) regardless of which edge table contributed it.
        When multiple message-passing edge tables match, only the first
        is used to keep the output flat — multi-edge-type per-class
        degree is left for a future iteration.

        Returns a 2-tuple ``(per_class, sentinel)``: rows whose
        ``class_value`` matches a declared sentinel in
        ``node_table.label_sentinel_values`` are routed to ``sentinel``;
        all other non-NULL label rows go to ``per_class``.
        """
        matching = [
            edge_table
            for edge_table in message_passing_tables
            if node_table.node_type
            in (edge_table.src_node_type, edge_table.dst_node_type)
        ]
        if not matching:
            return [], []
        edge_table = matching[0]
        if len(matching) > 1:
            logger.info(
                f"Per-class degree for node_type={node_table.node_type!r}: "
                f"using first matching message-passing edge table "
                f"{edge_table.edge_type!r} of {[m.edge_type for m in matching]}."
            )

        assert (
            node_table.label_column is not None
        ), "_compute_per_class_degree requires NodeTableSpec.label_column"
        query = build_per_class_degree_query(
            node_table=node_table.bq_table,
            node_id_column=node_table.id_column,
            label_column=node_table.label_column,
            edge_table=edge_table.bq_table,
            edge_src_column=edge_table.src_id_column,
            edge_dst_column=edge_table.dst_id_column,
        )
        self._record_query(
            f"nc_supervision:per_class_degree:{node_table.node_type}", query
        )
        rows = list(self._bq_utils.run_query(query=query, labels={}))
        sentinel_value_set = set(node_table.label_sentinel_values)
        per_class: list[PerClassDegreeStats] = []
        sentinel: list[PerClassDegreeStats] = []
        for row in rows:
            percentiles_raw = list(row["percentiles"]) if row["percentiles"] else []
            percentiles = [int(p) if p is not None else 0 for p in percentiles_raw]
            median = percentiles[50] if len(percentiles) > 50 else 0
            p90 = (
                percentiles[90]
                if len(percentiles) > 90
                else (percentiles[-1] if percentiles else 0)
            )
            p99 = (
                percentiles[99]
                if len(percentiles) > 99
                else (percentiles[-1] if percentiles else 0)
            )
            # Bucket keys must match BUCKET_ORDER in report/charts.ai.js so the
            # sparkline histogram lines up with the overall degree chart.
            buckets: dict[str, int] = {
                "0-1": int(row["bucket_0_1"] or 0),
                "2-10": int(row["bucket_2_10"] or 0),
                "11-100": int(row["bucket_11_100"] or 0),
                "101-1K": int(row["bucket_101_1k"] or 0),
                "1K-10K": int(row["bucket_1k_10k"] or 0),
                "10K+": int(row["bucket_10k_plus"] or 0),
            }
            class_value = str(row["class_value"])
            stats = PerClassDegreeStats(
                class_value=class_value,
                count=int(row["class_count"] or 0),
                cold_start_count=int(row["cold_start_count"] or 0),
                mean_degree=float(row["mean_degree"] or 0.0),
                median_degree=median,
                p90_degree=p90,
                p99_degree=p99,
                max_degree=int(row["max_degree"] or 0),
                buckets=buckets,
            )
            if class_value in sentinel_value_set:
                sentinel.append(stats)
            else:
                per_class.append(stats)
        return per_class, sentinel

    def _compute_homophily_for_node_type(
        self,
        node_table: NodeTableSpec,
        message_passing_tables: list[EdgeTableSpec],
        config: DataAnalyzerConfig,
    ) -> list[HomophilyStats]:
        """Sampled adjusted homophily per (labeled node type, edge type).

        Edges are sampled to ``config.label_homophily_edge_sample_cap``
        via deterministic ``MOD(FARM_FINGERPRINT(...))`` filtering. The
        modulus is computed from the edge table's row count so the
        sampled set is ~= the cap; small graphs (count <= cap) skip
        sampling entirely.
        """
        out: list[HomophilyStats] = []
        for edge_table in message_passing_tables:
            if node_table.node_type not in (
                edge_table.src_node_type,
                edge_table.dst_node_type,
            ):
                continue
            # Edge-count subquery here is unrelated to the per-edge-type one
            # in Tier 2 — it gates only the sampling decision below — so we
            # don't tag it for the report and just run it.
            edge_count = self._query_scalar(
                EDGE_COUNT_QUERY.format(table=edge_table.bq_table), "edge_count"
            )
            cap = config.label_homophily_edge_sample_cap
            if cap > 0 and edge_count > cap:
                modulus = max(1, edge_count // cap)
                sample_cap = cap
            else:
                modulus = 1
                sample_cap = 0  # signal "no sampling"
            assert (
                node_table.label_column is not None
            ), "_compute_homophily_for_node_type requires NodeTableSpec.label_column"
            template = build_adjusted_homophily_query(
                node_table=node_table.bq_table,
                node_id_column=node_table.id_column,
                label_column=node_table.label_column,
                sentinel_values=node_table.label_sentinel_values,
                edge_table=edge_table.bq_table,
                edge_src_column=edge_table.src_id_column,
                edge_dst_column=edge_table.dst_id_column,
                sample_cap=sample_cap,
            )
            query = template.replace("{modulus_placeholder}", str(modulus))
            self._record_query(
                f"nc_supervision:homophily:{node_table.node_type}:"
                f"{edge_table.edge_type}",
                query,
            )
            rows = list(self._bq_utils.run_query(query=query, labels={}))
            if len(rows) != 1:
                raise RuntimeError(
                    f"Adjusted-homophily query expected exactly 1 row; got "
                    f"{len(rows)}. node_type={node_table.node_type}, "
                    f"edge_type={edge_table.edge_type}"
                )
            row = rows[0]
            edge_homophily_value = row["edge_homophily"]
            expected_value = row["expected_homophily"]
            edge_homophily = (
                float(edge_homophily_value) if edge_homophily_value is not None else 0.0
            )
            expected = float(expected_value) if expected_value is not None else 0.0
            if expected < 1.0:
                adjusted = (edge_homophily - expected) / (1.0 - expected)
            else:
                adjusted = 0.0
            out.append(
                HomophilyStats(
                    edge_type=edge_table.edge_type,
                    edge_homophily=edge_homophily,
                    adjusted_homophily=adjusted,
                    edge_sample_count=int(row["edge_sample_count"] or 0),
                    label_informativeness=None,
                )
            )
        return out

    def _compute_cross_split_overlap(
        self, node_table: NodeTableSpec
    ) -> Optional[CrossSplitOverlap]:
        """Cross-split id leakage + per-split row counts. Returns None if no split_column."""
        if node_table.split_column is None:
            return None
        block_id = f"nc_supervision:cross_split:{node_table.node_type}"
        cross_split_query = CROSS_SPLIT_OVERLAP_QUERY.format(
            table=node_table.bq_table,
            id_column=node_table.id_column,
            split_column=node_table.split_column,
        )
        overlap_count = self._query_scalar(
            cross_split_query, "overlap_node_count", block_id=block_id
        )
        split_value_query = SPLIT_VALUE_COUNTS_QUERY.format(
            table=node_table.bq_table,
            split_column=node_table.split_column,
        )
        self._record_query(block_id, split_value_query)
        split_rows = list(self._bq_utils.run_query(query=split_value_query, labels={}))
        split_value_counts: dict[str, int] = {
            str(row["split_value"]): int(row["row_count"] or 0) for row in split_rows
        }
        return CrossSplitOverlap(
            overlap_node_count=overlap_count,
            split_value_counts=split_value_counts,
        )

    # ------------------------------------------------------------------ #
    # Supervision cross-table analysis                                    #
    # ------------------------------------------------------------------ #

    def _run_supervision_cross_table(
        self, config: DataAnalyzerConfig, result: GraphAnalysisResult
    ) -> None:
        """Run cross-table per-anchor stats for supervision edge tables.

        For every ``supervision_pos`` table we pair it with each
        ``supervision_neg`` and ``message_passing`` table that shares its
        ``(src_node_type, dst_node_type)``, then compute per-anchor edge
        counts and label-leakage overlap. Each ``supervision_neg`` table
        also drives a pass against matching ``message_passing`` tables so
        the report can flag (negative-edge ∩ message-passing) leaks. Jobs
        run in parallel via ``ThreadPoolExecutor`` (BQ is I/O-bound).
        """
        pos_tables = [
            e for e in config.edge_tables if e.role == EDGE_ROLE_SUPERVISION_POS
        ]
        neg_tables = [
            e for e in config.edge_tables if e.role == EDGE_ROLE_SUPERVISION_NEG
        ]
        # Treat unset role as message_passing (default), matching backfill behavior.
        mp_tables = [
            e
            for e in config.edge_tables
            if e.role is None or e.role == EDGE_ROLE_MESSAGE_PASSING
        ]

        jobs: list[tuple[EdgeTableSpec, EdgeTableSpec, str]] = []

        # Driver = positive: pair with every neg / mp sharing (src_type, dst_type).
        for pos in pos_tables:
            assert pos.node_anchor is not None, (
                f"edge_type={pos.edge_type}: supervision_pos must have node_anchor; "
                "load the config via load_analyzer_config to enforce this."
            )
            for other in neg_tables + mp_tables:
                if (pos.src_node_type, pos.dst_node_type) == (
                    other.src_node_type,
                    other.dst_node_type,
                ):
                    jobs.append((pos, other, pos.node_anchor))

        # Driver = negative: pair with mp sharing (src_type, dst_type). Anchor
        # is the negative's own node_anchor when set, else inherited from a
        # matching positive table to keep configs concise.
        for neg in neg_tables:
            anchor = neg.node_anchor or self._inherit_anchor_from_pos(neg, pos_tables)
            if anchor is None:
                continue
            for mp in mp_tables:
                if (neg.src_node_type, neg.dst_node_type) == (
                    mp.src_node_type,
                    mp.dst_node_type,
                ):
                    jobs.append((neg, mp, anchor))

        if not jobs:
            return

        with ThreadPoolExecutor(max_workers=_PARALLEL_BQ_WORKERS) as executor:
            futures = [
                executor.submit(self._supervision_pair_stats, driver, other, anchor)
                for driver, other, anchor in jobs
            ]
            for future in futures:
                stats = future.result()
                if stats is not None:
                    result.supervision_cross_table_stats.append(stats)

    @staticmethod
    def _inherit_anchor_from_pos(
        neg: EdgeTableSpec, pos_tables: list[EdgeTableSpec]
    ) -> Optional[str]:
        """Return the node_anchor of any positive table sharing neg's node types.

        Lets users declare ``node_anchor`` once on the positive table and
        skip duplicating it on the matching negative.
        """
        for pos in pos_tables:
            if (pos.src_node_type, pos.dst_node_type) == (
                neg.src_node_type,
                neg.dst_node_type,
            ):
                return pos.node_anchor
        return None

    @staticmethod
    def _resolve_anchor_columns(
        edge_table: EdgeTableSpec, node_anchor: str
    ) -> Optional[tuple[str, str]]:
        """Return (anchor_column, other_column) for the given anchor node_type.

        If ``node_anchor`` matches both src and dst (homogeneous self-loop
        edge), prefer the src side. Returns ``None`` if it matches neither.
        """
        if node_anchor == edge_table.src_node_type:
            return edge_table.src_id_column, edge_table.dst_id_column
        if node_anchor == edge_table.dst_node_type:
            return edge_table.dst_id_column, edge_table.src_id_column
        return None

    def _supervision_pair_stats(
        self,
        driver: EdgeTableSpec,
        other: EdgeTableSpec,
        node_anchor: str,
    ) -> Optional[SupervisionCrossTableStats]:
        """Run the cross-table query for one (driver, other) pair.

        Returns ``None`` (and logs a warning) when the anchor cannot be
        resolved on one of the two tables — happens only on misconfigured
        heterogeneous pairs and should not abort the whole run.
        """
        driver_cols = self._resolve_anchor_columns(driver, node_anchor)
        other_cols = self._resolve_anchor_columns(other, node_anchor)
        if driver_cols is None or other_cols is None:
            logger.warning(
                f"Skipping supervision pair driver={driver.edge_type!r} "
                f"other={other.edge_type!r}: node_anchor={node_anchor!r} not "
                "present on both tables."
            )
            return None

        driver_anchor_column, driver_other_column = driver_cols
        other_anchor_column, other_other_column = other_cols

        query = SUPERVISION_CROSS_TABLE_QUERY.format(
            driver_table=driver.bq_table,
            other_table=other.bq_table,
            driver_anchor_column=driver_anchor_column,
            driver_other_column=driver_other_column,
            other_anchor_column=other_anchor_column,
            other_other_column=other_other_column,
        )
        self._record_query(
            f"supervision_overlap:{driver.edge_type}:{other.edge_type}:"
            f"{driver_anchor_column}:{other_anchor_column}",
            query,
        )
        rows = list(self._bq_utils.run_query(query=query, labels={}))
        if len(rows) != 1:
            raise RuntimeError(
                f"Supervision cross-table query expected exactly 1 row; "
                f"got {len(rows)}. driver={driver.edge_type} other={other.edge_type}"
            )
        row = rows[0]
        avg_value = row["avg_other_per_driver_anchor"]
        return SupervisionCrossTableStats(
            driver_edge_type=driver.edge_type,
            driver_role=driver.role or EDGE_ROLE_MESSAGE_PASSING,
            other_edge_type=other.edge_type,
            other_role=other.role or EDGE_ROLE_MESSAGE_PASSING,
            node_anchor=node_anchor,
            driver_anchor_count=int(row["driver_anchor_count"] or 0),
            driver_pair_count=int(row["driver_pair_count"] or 0),
            other_pair_count=int(row["other_pair_count"] or 0),
            overlap_pair_count=int(row["overlap_pair_count"] or 0),
            driver_anchors_with_zero_other=int(
                row["driver_anchors_with_zero_other"] or 0
            ),
            avg_other_per_driver_anchor=float(avg_value)
            if avg_value is not None
            else 0.0,
            p50_other_per_driver_anchor=int(row["p50_other_per_driver_anchor"] or 0),
            p90_other_per_driver_anchor=int(row["p90_other_per_driver_anchor"] or 0),
            p99_other_per_driver_anchor=int(row["p99_other_per_driver_anchor"] or 0),
            max_other_per_driver_anchor=int(row["max_other_per_driver_anchor"] or 0),
        )

    # ------------------------------------------------------------------ #
    # Tier 4: opt-in                                                      #
    # ------------------------------------------------------------------ #

    def _run_tier4(
        self, config: DataAnalyzerConfig, result: GraphAnalysisResult
    ) -> None:
        """Populate opt-in metrics gated by config flags.

        Power-law exponent is always cheap (derived from existing degree stats)
        and is computed whenever degree stats are available. Reciprocity,
        homophily, connected components and clustering require dedicated
        queries not yet defined; they remain empty unless the corresponding
        flag is enabled AND a query is implemented.
        """
        # Power-law exponent: approximate from degree stats using a simple
        # heuristic: alpha ~= 1 + log(max) / log(median) for median > 1.
        for degree_key, stats in result.degree_stats.items():
            if stats.median > 1 and stats.max > stats.median:
                exponent = 1.0 + math.log(stats.max) / math.log(stats.median)
                result.power_law_exponent[degree_key] = exponent

        if config.compute_reciprocity:
            # Query not yet defined; log and skip.
            logger.warning(
                "compute_reciprocity=True but reciprocity query is not implemented; "
                "skipping Tier 4 reciprocity."
            )

    # ------------------------------------------------------------------ #
    # Python-only computations                                            #
    # ------------------------------------------------------------------ #

    def _compute_feature_memory_budget(
        self, config: DataAnalyzerConfig, result: GraphAnalysisResult
    ) -> None:
        """Estimate per-node-type memory footprint of features (float64 assumed)."""
        for node_table in config.node_tables:
            node_count = result.node_counts.get(node_table.node_type, 0)
            num_features = len(node_table.feature_columns)
            result.feature_memory_bytes[node_table.node_type] = (
                node_count * num_features * _BYTES_PER_FEATURE
            )

    def _compute_neighbor_explosion_estimate(
        self, config: DataAnalyzerConfig, result: GraphAnalysisResult
    ) -> None:
        """Multiply fan-out factors and scale by out-degree mean per edge type."""
        if not config.fan_out:
            return
        fan_out_product = 1
        for hop in config.fan_out:
            fan_out_product *= int(hop)
        for edge_table in config.edge_tables:
            out_stats = result.degree_stats.get(f"{edge_table.edge_type}_out")
            if out_stats is None:
                continue
            estimate = int(fan_out_product * max(out_stats.mean, 1.0))
            result.neighbor_explosion_estimate[edge_table.edge_type] = estimate

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _query_scalar(
        self, query: str, column: str, *, block_id: Optional[str] = None
    ) -> int:
        """Run a single-row, single-column query and return the scalar as int.

        Scalar queries (COUNT, COUNTIF) must return exactly one row with a
        non-NULL value for the requested column. Any deviation indicates a
        driver, auth, or schema mismatch rather than legitimate data — raise
        loudly instead of silently coercing to 0, which would let a broken run
        pass through as a green-light result.

        When ``block_id`` is provided the rendered SQL is recorded under
        that key in ``self._query_log`` so the report can surface it.
        """
        if block_id is not None:
            self._record_query(block_id, query)
        rows = list(self._bq_utils.run_query(query=query, labels={}))
        if len(rows) != 1:
            raise RuntimeError(
                f"Scalar query expected exactly 1 row; got {len(rows)}. "
                f"Query: {query.strip()[:200]}"
            )
        value = rows[0][column]
        if value is None:
            raise RuntimeError(
                f"Scalar query returned NULL for column '{column}'. "
                f"Query: {query.strip()[:200]}"
            )
        return int(value)

    def _record_query(self, block_id: str, query: str) -> None:
        """Append ``query`` under ``block_id`` in the per-block SQL log.

        The report JS does dict lookups against ``GraphAnalysisResult.queries``
        keyed by the same ``block_id`` strings. CPython's GIL makes
        ``dict.setdefault`` and ``list.append`` atomic, so concurrent writes
        from the Tier-2 thread pool are safe without an explicit lock.
        """
        self._query_log.setdefault(block_id, []).append(query)
