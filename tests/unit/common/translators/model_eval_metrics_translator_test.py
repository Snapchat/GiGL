import json
import os
import tempfile

from gigl.common import LocalUri
from gigl.src.common.translators.model_eval_metrics_translator import (
    write_eval_metrics_to_uri,
)
from gigl.src.common.types.model_eval_metrics import (
    EvalMetric,
    EvalMetricsCollection,
    EvalMetricType,
)
from tests.test_assets.test_case import TestCase


class TestWriteEvalMetricsToUri(TestCase):
    def test_write_eval_metrics_to_uri_writes_kfp_compatible_json(self) -> None:
        """Verifies that write_eval_metrics_to_uri produces a JSON file in
        the format expected by the KFP metric logger."""
        eval_metrics = EvalMetricsCollection(
            metrics=[EvalMetric.from_eval_metric_type(EvalMetricType.loss, 0.42)]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "eval_metrics.json")
            output_uri = LocalUri(output_path)

            write_eval_metrics_to_uri(
                eval_metrics=eval_metrics, eval_metrics_uri=output_uri
            )

            self.assertTrue(os.path.exists(output_path))
            with open(output_path) as f:
                written = json.load(f)

            self.assertIn("metrics", written)
            metrics_list = written["metrics"]
            self.assertEqual(len(metrics_list), 1)
            self.assertEqual(metrics_list[0]["name"], "loss")
            self.assertEqual(metrics_list[0]["numberValue"], "0.42")
            self.assertEqual(metrics_list[0]["format"], "RAW")

    def test_write_eval_metrics_to_uri_with_multiple_metrics(self) -> None:
        """Verifies that multiple metrics are written correctly."""
        eval_metrics = EvalMetricsCollection(
            metrics=[
                EvalMetric.from_eval_metric_type(EvalMetricType.loss, 0.42),
                EvalMetric.from_eval_metric_type(EvalMetricType.acc, 0.95),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "eval_metrics.json")
            output_uri = LocalUri(output_path)

            write_eval_metrics_to_uri(
                eval_metrics=eval_metrics, eval_metrics_uri=output_uri
            )

            with open(output_path) as f:
                written = json.load(f)

            metrics_list = written["metrics"]
            self.assertEqual(len(metrics_list), 2)
            metric_names = {m["name"] for m in metrics_list}
            self.assertEqual(metric_names, {"loss", "acc"})
