"""Unit tests for embedding_diagnostics.

BQ calls are mocked via patched ``BqUtils.run_query``.
"""
from typing import Any
from unittest.mock import MagicMock

from gigl.analytics.data_analyzer.embedding_diagnostics import (
    EmbeddingDiagnostics,
    EmbeddingDiagnosticsRequest,
)
from tests.test_assets.test_case import TestCase


def _mock_row(data: dict[str, Any]) -> MagicMock:
    row = MagicMock()
    row.__getitem__ = lambda self, key: data[key]
    return row


def _mock_rows(rows: list[dict[str, Any]]) -> MagicMock:
    iterator = MagicMock()
    iterator.__iter__ = lambda self: iter([_mock_row(r) for r in rows])
    return iterator


def _success_row(
    total: int = 100,
    unique_count: int = 90,
    unique_ratio: float = 0.9,
    top_k: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "total": total,
        "unique_count": unique_count,
        "unique_ratio": unique_ratio,
        "top_k": top_k if top_k is not None else [],
    }


class EmbeddingDiagnosticsAnalyzeTest(TestCase):
    def test_empty_requests_returns_empty_mapping(self) -> None:
        bq_utils = MagicMock()
        diagnostics = EmbeddingDiagnostics(bq_utils=bq_utils)
        self.assertEqual(diagnostics.analyze([]), {})
        bq_utils.run_query.assert_not_called()

    def test_one_query_per_embedding_column(self) -> None:
        bq_utils = MagicMock()
        bq_utils.run_query.side_effect = lambda query, labels=None: _mock_rows(
            [_success_row()]
        )
        requests = [
            EmbeddingDiagnosticsRequest(
                result_key="node:user",
                bq_table="p.d.users",
                embedding_columns=["emb_a", "emb_b"],
            ),
            EmbeddingDiagnosticsRequest(
                result_key="node:content",
                bq_table="p.d.content",
                embedding_columns=["emb_c"],
            ),
        ]
        EmbeddingDiagnostics(bq_utils=bq_utils).analyze(requests)
        self.assertEqual(bq_utils.run_query.call_count, 3)

    def test_result_is_populated_from_row(self) -> None:
        bq_utils = MagicMock()
        bq_utils.run_query.return_value = _mock_rows(
            [
                _success_row(
                    total=1000,
                    unique_count=980,
                    unique_ratio=0.98,
                    top_k=[
                        {"hash_value": 1, "count_value": 10, "fraction": 0.01},
                        {"hash_value": 2, "count_value": 5, "fraction": 0.005},
                    ],
                )
            ]
        )
        requests = [
            EmbeddingDiagnosticsRequest(
                result_key="node:user",
                bq_table="p.d.users",
                embedding_columns=["emb"],
            )
        ]
        out = EmbeddingDiagnostics(bq_utils=bq_utils).analyze(requests)
        result = out["node:user"]["emb"]
        self.assertEqual(result.total, 1000)
        self.assertEqual(result.unique_count, 980)
        self.assertAlmostEqual(result.unique_ratio, 0.98)
        self.assertEqual(len(result.top_k), 2)
        self.assertEqual(result.top_k[0].hash, 1)
        self.assertEqual(result.top_k[0].count, 10)
        self.assertAlmostEqual(result.top_k[0].fraction, 0.01)

    def test_per_column_failure_is_logged_and_skipped(self) -> None:
        bq_utils = MagicMock()

        def _side_effect(query: str, labels: Any = None) -> MagicMock:
            if "emb_bad" in query:
                raise RuntimeError("BQ permission denied")
            return _mock_rows([_success_row()])

        bq_utils.run_query.side_effect = _side_effect
        requests = [
            EmbeddingDiagnosticsRequest(
                result_key="node:user",
                bq_table="p.d.users",
                embedding_columns=["emb_good", "emb_bad"],
            )
        ]
        with self.assertLogs(level="ERROR") as cap:
            out = EmbeddingDiagnostics(bq_utils=bq_utils).analyze(requests)
        self.assertIn("emb_good", out["node:user"])
        self.assertNotIn("emb_bad", out["node:user"])
        self.assertTrue(
            any("emb_bad" in msg for msg in cap.output),
            f"expected error mentioning emb_bad, got {cap.output}",
        )

    def test_query_uses_farm_fingerprint_and_table(self) -> None:
        bq_utils = MagicMock()
        bq_utils.run_query.return_value = _mock_rows([_success_row()])
        requests = [
            EmbeddingDiagnosticsRequest(
                result_key="node:user",
                bq_table="proj.ds.users",
                embedding_columns=["emb"],
            )
        ]
        EmbeddingDiagnostics(bq_utils=bq_utils).analyze(requests)
        query = bq_utils.run_query.call_args_list[0].kwargs["query"]
        self.assertIn("FARM_FINGERPRINT(TO_JSON_STRING(`emb`))", query)
        self.assertIn("`proj.ds.users`", query)
        self.assertIn("LIMIT 20", query)

    def test_top_k_limit_is_configurable(self) -> None:
        bq_utils = MagicMock()
        bq_utils.run_query.return_value = _mock_rows([_success_row()])
        requests = [
            EmbeddingDiagnosticsRequest(
                result_key="node:user",
                bq_table="p.d.users",
                embedding_columns=["emb"],
            )
        ]
        EmbeddingDiagnostics(bq_utils=bq_utils, top_k=5).analyze(requests)
        query = bq_utils.run_query.call_args_list[0].kwargs["query"]
        self.assertIn("LIMIT 5", query)

    def test_empty_row_result_raises(self) -> None:
        bq_utils = MagicMock()
        bq_utils.run_query.return_value = _mock_rows([])
        requests = [
            EmbeddingDiagnosticsRequest(
                result_key="node:user",
                bq_table="p.d.users",
                embedding_columns=["emb"],
            )
        ]
        with self.assertLogs(level="ERROR"):
            out = EmbeddingDiagnostics(bq_utils=bq_utils).analyze(requests)
        # Failure is caught, result_key is omitted.
        self.assertNotIn("node:user", out)
