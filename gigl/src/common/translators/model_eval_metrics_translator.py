import json
import os
import tempfile
from pathlib import Path

from gigl.common import LocalUri, Uri
from gigl.common.logger import Logger
from gigl.src.common.types.model_eval_metrics import EvalMetricsCollection
from gigl.src.common.utils.file_loader import FileLoader

logger = Logger()


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


def write_eval_metrics_to_uri(
    eval_metrics: EvalMetricsCollection, eval_metrics_uri: Uri
) -> None:
    """Writes evaluation metrics to a URI in KFP-compatible JSON format.

    Args:
        eval_metrics: Collection of evaluation metrics to write.
        eval_metrics_uri: Destination URI for the metrics JSON file.
    """
    with tempfile.NamedTemporaryFile(delete=False) as tfh:
        local_tfh_uri = LocalUri(tfh.name)
    try:
        EvalMetricsCollectionTranslator.write_kfp_metrics_to_pipeline_metric_path(
            eval_metrics=eval_metrics, path=local_tfh_uri
        )
        FileLoader().load_file(
            file_uri_src=local_tfh_uri,
            file_uri_dst=eval_metrics_uri,
            should_create_symlinks_if_possible=False,
        )
        logger.info(f"Wrote eval metrics to {eval_metrics_uri.uri}.")
    finally:
        os.remove(local_tfh_uri.uri)
