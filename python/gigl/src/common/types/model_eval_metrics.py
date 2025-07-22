from dataclasses import dataclass
from enum import Enum


class EvalMetricType(Enum):
    mrr = "mrr"
    loss = "loss"
    hits = "hits"
    acc = "acc"

    @classmethod
    def get_all_criteria(cls) -> list[str]:
        return [m.name for m in cls]


@dataclass
class EvalMetric:
    name: str
    value: float

    @classmethod
    def from_eval_metric_type(cls, eval_metric_type: EvalMetricType, value: float):
        return cls(
            name=eval_metric_type.name,
            value=value,
        )

    def __post_init__(self):
        self.value = float(self.value)


class EvalMetricsCollection:
    def __init__(self, metrics: list[EvalMetric] = []):
        self._metrics: dict[str, EvalMetric] = dict()
        self.add_metrics(metrics=metrics)

    @property
    def metrics(self) -> dict[str, EvalMetric]:
        return self._metrics

    def add_metric(self, model_metric: EvalMetric):
        self._metrics[model_metric.name] = model_metric

    def add_metrics(self, metrics: list[EvalMetric]):
        for model_metric in metrics:
            self.add_metric(model_metric)

    def __repr__(self):
        metrics_str_lst: list[str] = [
            f"{model_metric}" for _, model_metric in self._metrics.items()
        ]
        metrics_str = f"{self.__class__.__name__}({', '.join(metrics_str_lst)})"
        return metrics_str
