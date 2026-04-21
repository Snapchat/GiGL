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
    DataAnalyzerConfig,
    EdgeTableSpec,
    NodeTableSpec,
)
from gigl.analytics.data_analyzer.queries import (
    CLASS_IMBALANCE_QUERY,
    COLD_START_NODE_COUNT_QUERY,
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
    SUPER_HUB_INT16_CLAMP_QUERY,
    TOP_K_HUBS_QUERY,
    build_null_rates_query,
)
from gigl.analytics.data_analyzer.types import DegreeStats, GraphAnalysisResult
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

    def analyze(self, config: DataAnalyzerConfig) -> GraphAnalysisResult:
        """Run all applicable tiers and return aggregated results.

        Args:
            config: Data analyzer configuration declaring node and edge tables
                plus any opt-in expensive checks (reciprocity, etc.).

        Returns:
            GraphAnalysisResult with tier 1-4 fields populated per config.

        Raises:
            DataQualityError: If tier 1 checks find any violations. The
                exception carries a partial result with the specific counts.
        """
        result = GraphAnalysisResult()
        logger.info("Starting graph structure analysis (Tier 1: hard fails)")
        self._run_tier1(config, result)

        logger.info("Tier 1 passed. Running Tier 2 (core metrics)")
        self._run_tier2(config, result)

        logger.info("Running Tier 3 (label / heterogeneous)")
        self._run_tier3(config, result)

        logger.info("Running Tier 4 (opt-in)")
        self._run_tier4(config, result)
        return result

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
            count = self._query_scalar(query, "duplicate_count")
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
            dangling = self._query_scalar(dangling_query, "dangling_count")
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
        node_count = self._query_scalar(
            NODE_COUNT_QUERY.format(table=node_table.bq_table), "node_count"
        )
        result.node_counts[node_table.node_type] = node_count

        columns_to_check: list[str] = [node_table.id_column]
        columns_to_check.extend(node_table.feature_columns)
        if node_table.label_column:
            columns_to_check.append(node_table.label_column)

        null_query = build_null_rates_query(
            table=node_table.bq_table, columns=columns_to_check
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
        result.edge_counts[edge_type] = self._query_scalar(
            EDGE_COUNT_QUERY.format(table=edge_table.bq_table), "edge_count"
        )
        result.duplicate_edge_counts[edge_type] = self._query_scalar(
            DUPLICATE_EDGE_COUNT_QUERY.format(
                table=edge_table.bq_table,
                src_id_column=edge_table.src_id_column,
                dst_id_column=edge_table.dst_id_column,
            ),
            "duplicate_count",
        )
        result.self_loop_counts[edge_type] = self._query_scalar(
            SELF_LOOP_COUNT_QUERY.format(
                table=edge_table.bq_table,
                src_id_column=edge_table.src_id_column,
                dst_id_column=edge_table.dst_id_column,
            ),
            "self_loop_count",
        )

        # Super-hub INT16 clamp check (indexed by src).
        result.super_hub_int16_clamp_count[edge_type] = self._query_scalar(
            SUPER_HUB_INT16_CLAMP_QUERY.format(
                table=edge_table.bq_table, id_column=edge_table.src_id_column
            ),
            "super_hub_count",
        )

        # Isolated and cold-start require a node table join.
        if node_table is not None:
            result.isolated_node_counts[edge_type] = self._query_scalar(
                ISOLATED_NODE_COUNT_QUERY.format(
                    node_table=node_table.bq_table,
                    edge_table=edge_table.bq_table,
                    node_id_column=node_table.id_column,
                    src_id_column=edge_table.src_id_column,
                    dst_id_column=edge_table.dst_id_column,
                ),
                "isolated_count",
            )
            result.cold_start_node_counts[edge_type] = self._query_scalar(
                COLD_START_NODE_COUNT_QUERY.format(
                    node_table=node_table.bq_table,
                    edge_table=edge_table.bq_table,
                    node_id_column=node_table.id_column,
                    src_id_column=edge_table.src_id_column,
                    dst_id_column=edge_table.dst_id_column,
                ),
                "cold_start_count",
            )

        # Top-K hubs (by src).
        top_hub_rows = list(
            self._bq_utils.run_query(
                query=TOP_K_HUBS_QUERY.format(
                    table=edge_table.bq_table,
                    id_column=edge_table.src_id_column,
                    k=_TOP_K_HUBS,
                ),
                labels={},
            )
        )
        result.top_hubs[edge_type] = [
            (str(row["node_id"]), int(row["degree"])) for row in top_hub_rows
        ]

        # Degree statistics: distribution + buckets, in + out directions.
        for direction, id_column in (
            ("out", edge_table.src_id_column),
            ("in", edge_table.dst_id_column),
        ):
            result.degree_stats[f"{edge_type}_{direction}"] = self._build_degree_stats(
                table=edge_table.bq_table, id_column=id_column
            )

    def _build_degree_stats(self, table: str, id_column: str) -> DegreeStats:
        """Run degree distribution + bucket queries and pack into DegreeStats."""
        dist_rows = list(
            self._bq_utils.run_query(
                query=DEGREE_DISTRIBUTION_QUERY.format(
                    table=table, id_column=id_column
                ),
                labels={},
            )
        )
        bucket_rows = list(
            self._bq_utils.run_query(
                query=DEGREE_BUCKET_QUERY.format(table=table, id_column=id_column),
                labels={},
            )
        )
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
            class_rows = list(
                self._bq_utils.run_query(
                    query=CLASS_IMBALANCE_QUERY.format(
                        table=node_table.bq_table,
                        label_column=node_table.label_column,
                    ),
                    labels={},
                )
            )
            result.class_imbalance[node_table.node_type] = {
                str(row["label"]): int(row["count"]) for row in class_rows
            }

            coverage_rows = list(
                self._bq_utils.run_query(
                    query=LABEL_COVERAGE_QUERY.format(
                        table=node_table.bq_table,
                        label_column=node_table.label_column,
                    ),
                    labels={},
                )
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
                    result.edge_type_distribution[edge_type] = self._query_scalar(
                        EDGE_TYPE_DISTRIBUTION_QUERY.format(table=edge_table.bq_table),
                        "edge_count",
                    )
                coverage_rows = list(
                    self._bq_utils.run_query(
                        query=EDGE_TYPE_NODE_COVERAGE_QUERY.format(
                            table=edge_table.bq_table,
                            src_id_column=edge_table.src_id_column,
                            dst_id_column=edge_table.dst_id_column,
                        ),
                        labels={},
                    )
                )
                if coverage_rows:
                    row = coverage_rows[0]
                    result.edge_type_node_coverage[edge_type] = {
                        "distinct_src_count": int(row["distinct_src_count"] or 0),
                        "distinct_dst_count": int(row["distinct_dst_count"] or 0),
                    }

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

    def _query_scalar(self, query: str, column: str) -> int:
        """Run a single-row, single-column query and return the scalar as int.

        Scalar queries (COUNT, COUNTIF) must return exactly one row with a
        non-NULL value for the requested column. Any deviation indicates a
        driver, auth, or schema mismatch rather than legitimate data — raise
        loudly instead of silently coercing to 0, which would let a broken run
        pass through as a green-light result.
        """
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
