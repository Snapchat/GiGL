"""Unit tests for the shared TFDV/Beam PTransforms."""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import apache_beam as beam
import pyarrow as pa
from apache_beam.testing.util import assert_that, equal_to

from gigl.common import LocalUri
from gigl.common.beam.tfdv_transforms import (
    BqTableToRecordBatch,
    GenerateAndVisualizeStats,
)
from tests.test_assets.test_case import TestCase


class BqTableToRecordBatchTest(TestCase):
    def test_raises_on_empty_feature_columns(self) -> None:
        with self.assertRaises(ValueError):
            BqTableToRecordBatch(bq_table="p.d.t", feature_columns=[])

    def test_query_uses_backtick_quoted_columns_and_table(self) -> None:
        transform = BqTableToRecordBatch(
            bq_table="proj.ds.users",
            feature_columns=["age", "country"],
        )
        captured_kwargs: dict = {}

        def _fake_read(**kwargs):
            captured_kwargs.update(kwargs)
            return beam.Create([{"age": 1, "country": "US"}])

        with patch(
            "gigl.common.beam.tfdv_transforms.beam.io.ReadFromBigQuery",
            side_effect=_fake_read,
        ):
            with beam.Pipeline() as p:
                _ = p | transform

        self.assertEqual(
            captured_kwargs["query"],
            "SELECT `age`, `country` FROM `proj.ds.users`",
        )
        self.assertTrue(captured_kwargs["use_standard_sql"])
        self.assertNotIn("project", captured_kwargs)

    def test_passes_bq_project_when_given(self) -> None:
        transform = BqTableToRecordBatch(
            bq_table="proj.ds.users",
            feature_columns=["age"],
            bq_project="billing-project",
        )
        captured_kwargs: dict = {}

        def _fake_read(**kwargs):
            captured_kwargs.update(kwargs)
            return beam.Create([{"age": 1}])

        with patch(
            "gigl.common.beam.tfdv_transforms.beam.io.ReadFromBigQuery",
            side_effect=_fake_read,
        ):
            with beam.Pipeline() as p:
                _ = p | transform

        self.assertEqual(captured_kwargs["project"], "billing-project")

    def test_emits_record_batches_with_list_typed_columns(self) -> None:
        rows = [
            {"age": 30, "country": "US"},
            {"age": 25, "country": "CA"},
            {"age": None, "country": "US"},
        ]

        def _fake_read(**kwargs):
            return beam.Create(rows)

        def _extract(batch: pa.RecordBatch) -> tuple:
            age_type = batch.schema.field("age").type
            country_type = batch.schema.field("country").type
            return (
                batch.num_rows,
                tuple(sorted(batch.schema.names)),
                pa.types.is_list(age_type),
                pa.types.is_list(country_type),
                tuple(batch.column("age").to_pylist()),
                tuple(batch.column("country").to_pylist()),
            )

        with patch(
            "gigl.common.beam.tfdv_transforms.beam.io.ReadFromBigQuery",
            side_effect=_fake_read,
        ):
            with beam.Pipeline() as p:
                batches = p | BqTableToRecordBatch(
                    bq_table="p.d.t",
                    feature_columns=["age", "country"],
                    batch_size=10,
                )
                summaries = batches | "Summarize batch" >> beam.Map(_extract)
                assert_that(
                    summaries,
                    equal_to(
                        [
                            (
                                3,
                                ("age", "country"),
                                True,
                                True,
                                ([30], [25], None),
                                (["US"], ["CA"], ["US"]),
                            )
                        ]
                    ),
                )


class GenerateAndVisualizeStatsTest(TestCase):
    def test_runs_and_writes_artifacts(self) -> None:
        """Smoke test: runs the PTransform on a tiny in-memory RecordBatch and
        verifies that both the Facets HTML and the stats TFRecord are written.
        """
        batch = pa.RecordBatch.from_pydict(
            {
                "age": pa.array([[30], [25], [40]], type=pa.list_(pa.int64())),
                "country": pa.array(
                    [["US"], ["CA"], ["US"]], type=pa.list_(pa.string())
                ),
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            facets_path = os.path.join(tmpdir, "facets.html")
            stats_path = os.path.join(tmpdir, "stats.tfrecord")
            with beam.Pipeline() as p:
                _ = (
                    p
                    | "Create a single record batch" >> beam.Create([batch])
                    | GenerateAndVisualizeStats(
                        facets_report_uri=LocalUri(facets_path),
                        stats_output_uri=LocalUri(stats_path),
                    )
                )

            self.assertTrue(
                Path(facets_path).exists(),
                f"Facets HTML not written at {facets_path}",
            )
            self.assertGreater(Path(facets_path).stat().st_size, 0)
            written = list(Path(tmpdir).glob("stats.tfrecord*"))
            self.assertTrue(
                written, f"No stats TFRecord written under prefix {stats_path}"
            )
