import unittest

from gigl.src.common.types.model_eval_metrics import (
    EvalMetric,
    EvalMetricsCollection,
    EvalMetricType,
)


class EvalMetricsCollectionTest(unittest.TestCase):
    def test_eval_metrics_constructor(self):
        mrr_model_metric = EvalMetric.from_eval_metric_type(
            eval_metric_type=EvalMetricType.mrr, value=0.8
        )
        some_other_metric_name = "foo"
        auc_model_metric = EvalMetric(
            name=some_other_metric_name,
            value=0.9,
        )
        m = EvalMetricsCollection(metrics=[mrr_model_metric, auc_model_metric])
        self.assertEqual(m.metrics[EvalMetricType.mrr.value].value, 0.8)
        self.assertEqual(m.metrics[some_other_metric_name].value, 0.9)

    def test_eval_metrics_add(self):
        m = EvalMetricsCollection()
        m.add_metric(
            model_metric=EvalMetric.from_eval_metric_type(
                eval_metric_type=EvalMetricType.mrr, value=0.8
            )
        )
        self.assertEqual(m.metrics[EvalMetricType.mrr.value].value, 0.8)
