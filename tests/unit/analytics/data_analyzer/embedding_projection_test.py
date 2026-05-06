"""Unit tests for embedding_projection."""
from google.cloud.bigquery import SchemaField

from gigl.analytics.data_analyzer.embedding_projection import (
    build_projection,
    detect_embedding_columns,
    is_embedding_column,
)
from tests.test_assets.test_case import TestCase


def _schema(
    fields: list[tuple[str, str, str]],
) -> dict[str, SchemaField]:
    return {
        name: SchemaField(name=name, field_type=field_type, mode=mode)
        for name, field_type, mode in fields
    }


class IsEmbeddingColumnTest(TestCase):
    def test_repeated_float64_is_embedding(self) -> None:
        field = SchemaField(name="emb", field_type="FLOAT64", mode="REPEATED")
        self.assertTrue(is_embedding_column(field))

    def test_repeated_float_is_embedding(self) -> None:
        field = SchemaField(name="emb", field_type="FLOAT", mode="REPEATED")
        self.assertTrue(is_embedding_column(field))

    def test_repeated_numeric_is_embedding(self) -> None:
        field = SchemaField(name="weights", field_type="NUMERIC", mode="REPEATED")
        self.assertTrue(is_embedding_column(field))

    def test_repeated_string_is_not_embedding(self) -> None:
        field = SchemaField(name="tags", field_type="STRING", mode="REPEATED")
        self.assertFalse(is_embedding_column(field))

    def test_scalar_float_is_not_embedding(self) -> None:
        field = SchemaField(name="weight", field_type="FLOAT64", mode="NULLABLE")
        self.assertFalse(is_embedding_column(field))


class DetectEmbeddingColumnsTest(TestCase):
    def test_returns_repeated_float_family_in_schema_order(self) -> None:
        schema = _schema(
            [
                ("age", "INT64", "NULLABLE"),
                ("emb_a", "FLOAT64", "REPEATED"),
                ("country", "STRING", "NULLABLE"),
                ("emb_b", "NUMERIC", "REPEATED"),
                ("tags", "STRING", "REPEATED"),
            ]
        )
        self.assertEqual(
            detect_embedding_columns(schema, excluded=set()),
            ["emb_a", "emb_b"],
        )

    def test_excluded_columns_dropped(self) -> None:
        schema = _schema(
            [
                ("emb_a", "FLOAT64", "REPEATED"),
                ("emb_b", "FLOAT64", "REPEATED"),
            ]
        )
        self.assertEqual(
            detect_embedding_columns(schema, excluded={"emb_a"}),
            ["emb_b"],
        )


class BuildProjectionTest(TestCase):
    def test_scalar_columns_pass_through_backtick_quoted(self) -> None:
        schema = _schema(
            [
                ("age", "INT64", "NULLABLE"),
                ("country", "STRING", "NULLABLE"),
            ]
        )
        result = build_projection(schema, excluded=set())
        self.assertEqual(
            result.projection,
            [("age", "`age`"), ("country", "`country`")],
        )
        self.assertEqual(result.embedding_columns, [])

    def test_excluded_columns_dropped(self) -> None:
        schema = _schema(
            [
                ("uid", "STRING", "REQUIRED"),
                ("age", "INT64", "NULLABLE"),
            ]
        )
        result = build_projection(schema, excluded={"uid"})
        self.assertEqual(result.projection, [("age", "`age`")])

    def test_embedding_column_expands_to_four_hygiene_entries(self) -> None:
        schema = _schema([("emb", "FLOAT64", "REPEATED")])
        result = build_projection(schema, excluded=set())
        self.assertEqual(
            [name for name, _ in result.projection],
            ["emb_len", "emb_has_nan", "emb_has_inf", "emb_is_all_zero"],
        )
        self.assertEqual(result.embedding_columns, ["emb"])

    def test_embedding_hygiene_expressions_match_expected_sql(self) -> None:
        schema = _schema([("emb", "FLOAT64", "REPEATED")])
        result = build_projection(schema, excluded=set())
        by_name = dict(result.projection)
        self.assertEqual(by_name["emb_len"], "ARRAY_LENGTH(`emb`)")
        self.assertEqual(
            by_name["emb_has_nan"],
            "IFNULL((SELECT LOGICAL_OR(IS_NAN(v)) FROM UNNEST(`emb`) v), FALSE)",
        )
        self.assertEqual(
            by_name["emb_has_inf"],
            "IFNULL((SELECT LOGICAL_OR(IS_INF(v)) FROM UNNEST(`emb`) v), FALSE)",
        )
        self.assertEqual(
            by_name["emb_is_all_zero"],
            "IFNULL((SELECT LOGICAL_AND(v = 0) FROM UNNEST(`emb`) v), FALSE)",
        )

    def test_hash_column_is_not_in_projection(self) -> None:
        # Hash column lives in the diagnostics pass, not the TFDV projection.
        schema = _schema([("emb", "FLOAT64", "REPEATED")])
        result = build_projection(schema, excluded=set())
        names = {name for name, _ in result.projection}
        self.assertNotIn("emb_hash", names)

    def test_mixed_scalar_and_embedding_preserves_schema_order(self) -> None:
        schema = _schema(
            [
                ("age", "INT64", "NULLABLE"),
                ("emb", "FLOAT64", "REPEATED"),
                ("country", "STRING", "NULLABLE"),
            ]
        )
        result = build_projection(schema, excluded=set())
        self.assertEqual(
            [name for name, _ in result.projection],
            [
                "age",
                "emb_len",
                "emb_has_nan",
                "emb_has_inf",
                "emb_is_all_zero",
                "country",
            ],
        )

    def test_repeated_non_float_columns_are_skipped(self) -> None:
        schema = _schema(
            [
                ("age", "INT64", "NULLABLE"),
                ("tags", "STRING", "REPEATED"),
                ("ids", "INT64", "REPEATED"),
            ]
        )
        with self.assertLogs(level="INFO") as cap:
            result = build_projection(schema, excluded=set())
        self.assertEqual([name for name, _ in result.projection], ["age"])
        skip_log = " ".join(cap.output)
        self.assertIn("tags", skip_log)
        self.assertIn("ids", skip_log)

    def test_non_profileable_scalar_types_are_skipped(self) -> None:
        schema = _schema(
            [
                ("age", "INT64", "NULLABLE"),
                ("extras", "RECORD", "NULLABLE"),
                ("location", "GEOGRAPHY", "NULLABLE"),
                ("event_time", "TIMESTAMP", "NULLABLE"),
            ]
        )
        with self.assertLogs(level="INFO") as cap:
            result = build_projection(schema, excluded=set())
        self.assertEqual(result.projection, [("age", "`age`")])
        skip_log = " ".join(cap.output)
        self.assertIn("RECORD", skip_log)
        self.assertIn("GEOGRAPHY", skip_log)
        self.assertIn("TIMESTAMP", skip_log)
