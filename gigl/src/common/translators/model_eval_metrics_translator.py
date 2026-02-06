import json
from pathlib import Path

from gigl.common import LocalUri
from gigl.src.common.types.model_eval_metrics import EvalMetricsCollection


class EvalMetricsCollectionTranslator:
    @classmethod
    def write_kfp_metrics_to_pipeline_metric_path(
        cls, eval_metrics: EvalMetricsCollection, path: LocalUri
    ):
        kfp_metrics_list = []

        for metric in eval_metrics.metrics.values():
            kfp_metrics_list.append(
                {
                    "name": metric.name,
                    "numberValue": f"{metric.value}",
                    # KFP api specific; v2 will deprecate this format (alternative is percentage in v1 - which v2 has removed)
                    "format": "RAW",
                }
            )

        metrics = {"metrics": kfp_metrics_list}
        Path(path.uri).parent.mkdir(parents=True, exist_ok=True)
        with open(path.uri, "w") as f:
            json.dump(metrics, f)
