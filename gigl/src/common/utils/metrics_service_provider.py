import os
from typing import Optional

from gigl.common import Uri
from gigl.common.logger import Logger
from gigl.common.metrics.base_metrics import NopMetricsPublisher
from gigl.common.metrics.metrics_interface import OpsMetricPublisher
from gigl.common.utils import os_utils
from gigl.common.utils.proto_utils import ProtoUtils
from snapchat.research.gbml.gbml_config_pb2 import GbmlConfig

logger = Logger()

_metrics_instance: Optional[OpsMetricPublisher] = None
JOB_NAME_GROUPING_ENV_KEY = "GBML_JOB_NAME"


def initialize_metrics(task_config_uri: Uri, service_name: str):
    global _metrics_instance
    os.environ[JOB_NAME_GROUPING_ENV_KEY] = service_name
    proto_utils = ProtoUtils()
    task_config: GbmlConfig = proto_utils.read_proto_from_yaml(
        uri=task_config_uri, proto_cls=GbmlConfig
    )

    metrics_cls_path = task_config.metrics_config.metrics_cls_path
    metrics_args = task_config.metrics_config.metrics_args

    if not metrics_cls_path:
        logger.info("Custom metrics class not provided. Using No-op metrics")
        _metrics_instance = NopMetricsPublisher()
        return

    metrics_cls = os_utils.import_obj(metrics_cls_path)
    try:
        metrics_cls_instance: OpsMetricPublisher = metrics_cls(**metrics_args)
        assert isinstance(metrics_cls_instance, OpsMetricPublisher)
        _metrics_instance = metrics_cls_instance
        logger.info(f"Instantiated Custom Metrics Class from: {metrics_cls_path}")
    except Exception as e:
        logger.error(f"Could not instantiate class {metrics_cls_path}: {e}")
        raise e


def get_metrics_service_instance() -> Optional[OpsMetricPublisher]:
    if _metrics_instance is None:
        logger.warning(
            "initialize_metrics() was not called, using NopMetricsPulisher as default"
        )
        raise RuntimeError(
            "Metrics instance is not initialized. Call initialize_metrics() before getting the instance."
        )

    return _metrics_instance


def init_metrics_publisher_grouping_for_job(service_name: str) -> None:
    os.environ[JOB_NAME_GROUPING_ENV_KEY] = service_name
