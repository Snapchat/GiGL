"""Structural-sanity diagnostics for REPEATED FLOAT (embedding) columns.

Runs one BigQuery aggregate per (table, embedding column) to compute
``total`` rows, ``unique_count`` of distinct vectors, ``unique_ratio``,
and the top-K most-frequent hash clusters. Uses
``FARM_FINGERPRINT(TO_JSON_STRING(<col>))`` as the deduplication key —
cheap, deterministic, and exact for equality (not similarity).

A low ``unique_ratio`` or a heavily-weighted top entry indicates upstream
degeneracy (many rows emitting the same embedding — often a zero-padded
placeholder for missing data).

The component is best-effort: a failure on one column logs a warning and
is skipped; callers receive an empty mapping for that column rather than
an exception.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

from gigl.analytics.data_analyzer.types import EmbeddingDiagnosticsResult, TopKEntry
from gigl.common.logger import Logger
from gigl.src.common.utils.bq import BqUtils

logger = Logger()

_PARALLEL_DIAGNOSTICS_QUERIES = 8
_DEFAULT_TOP_K = 20


@dataclass(frozen=True)
class EmbeddingDiagnosticsRequest:
    """One (table, embedding columns, result key) triple to analyze.

    ``result_key`` is the per-table analyzer key (``"node:{type}"`` or
    ``"edge:{type}"``) used to organize outputs into the
    ``FeatureProfileResult.embedding_diagnostics`` two-level dict.
    """

    result_key: str
    bq_table: str
    embedding_columns: list[str]


class EmbeddingDiagnostics:
    """Compute structural diagnostics for embedding columns via BigQuery."""

    def __init__(
        self,
        bq_utils: BqUtils,
        top_k: int = _DEFAULT_TOP_K,
        max_workers: int = _PARALLEL_DIAGNOSTICS_QUERIES,
    ) -> None:
        self._bq_utils = bq_utils
        self._top_k = top_k
        self._max_workers = max_workers

    def analyze(
        self, requests: list[EmbeddingDiagnosticsRequest]
    ) -> dict[str, dict[str, EmbeddingDiagnosticsResult]]:
        """Run one aggregate query per (table, column) and collect results.

        Per-column failures are logged and skipped; one bad column does not
        sink other columns in the same request or other requests. A request
        whose every column failed produces an empty inner dict, which is
        omitted from the output.

        Args:
            requests: One entry per table with at least one embedding column.

        Returns:
            ``{result_key: {column_name: EmbeddingDiagnosticsResult}}``.
            Missing keys indicate the column's query failed.
        """
        jobs: list[tuple[str, str, str]] = []
        for request in requests:
            for column in request.embedding_columns:
                jobs.append((request.result_key, request.bq_table, column))
        if not jobs:
            return {}

        logger.info(
            f"Running {len(jobs)} embedding diagnostic query(ies) across "
            f"{len(requests)} table(s)."
        )
        out: dict[str, dict[str, EmbeddingDiagnosticsResult]] = {}
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_key = {
                executor.submit(
                    self._analyze_column, bq_table=bq_table, column=column
                ): (result_key, column)
                for result_key, bq_table, column in jobs
            }
            for future in as_completed(future_to_key):
                result_key, column = future_to_key[future]
                try:
                    diagnostics = future.result()
                except Exception as exc:
                    logger.exception(
                        f"Embedding diagnostics failed for "
                        f"{result_key}:{column}: {exc}"
                    )
                    continue
                if diagnostics is None:
                    continue
                out.setdefault(result_key, {})[column] = diagnostics
        return out

    def _analyze_column(
        self, bq_table: str, column: str
    ) -> Optional[EmbeddingDiagnosticsResult]:
        """Run the dedup aggregate for one column; return its result."""
        query = _build_dedup_query(bq_table=bq_table, column=column, top_k=self._top_k)
        rows = list(self._bq_utils.run_query(query=query, labels={}))
        if len(rows) != 1:
            raise RuntimeError(
                f"Embedding diagnostics query expected exactly 1 row for "
                f"{bq_table}.{column}; got {len(rows)}."
            )
        row = rows[0]
        total = int(row["total"] or 0)
        unique_count = int(row["unique_count"] or 0)
        unique_ratio = float(row["unique_ratio"] or 0.0)
        top_k_rows = row["top_k"] or []
        top_k = [
            TopKEntry(
                hash=int(entry["hash_value"]),
                count=int(entry["count_value"]),
                fraction=float(entry["fraction"] or 0.0),
            )
            for entry in top_k_rows
        ]
        return EmbeddingDiagnosticsResult(
            total=total,
            unique_count=unique_count,
            unique_ratio=unique_ratio,
            top_k=top_k,
        )


def _build_dedup_query(bq_table: str, column: str, top_k: int) -> str:
    """Render the per-column dedup aggregate.

    ``FARM_FINGERPRINT(TO_JSON_STRING(<col>))`` is deterministic and
    collision-resistant enough for this purpose — we're looking for
    unusually clumped clusters, not cryptographic uniqueness.
    """
    return f"""
WITH hashes AS (
  SELECT FARM_FINGERPRINT(TO_JSON_STRING(`{column}`)) AS h
  FROM `{bq_table}`
),
counts AS (
  SELECT h, COUNT(*) AS n FROM hashes GROUP BY h
),
agg AS (
  SELECT SUM(n) AS total, COUNT(*) AS unique_count FROM counts
)
SELECT
  agg.total,
  agg.unique_count,
  SAFE_DIVIDE(agg.unique_count, agg.total) AS unique_ratio,
  ARRAY(
    SELECT AS STRUCT
      h AS hash_value,
      n AS count_value,
      SAFE_DIVIDE(n, agg.total) AS fraction
    FROM counts
    ORDER BY n DESC
    LIMIT {top_k}
  ) AS top_k
FROM agg
""".strip()
