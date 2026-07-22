import os
from typing import Optional

from gigl.common import Uri, UriFactory
from gigl.common.logger import Logger
from gigl.common.metrics.base_metrics import NopMetricsPublisher
from gigl.common.metrics.metrics_interface import OpsMetricPublisher
from gigl.common.utils import os_utils
from gigl.common.utils.proto_utils import ProtoUtils
from snapchat.research.gbml.gbml_config_pb2 import GbmlConfig

logger = Logger()

_metrics_instance: Optional[OpsMetricPublisher] = None
JOB_NAME_GROUPING_ENV_KEY = "GBML_JOB_NAME"
TASK_CONFIG_URI_ENV_KEY = "GIGL_TASK_CONFIG_URI"


def initialize_metrics(task_config_uri: Uri, service_name: str) -> bool:
    """Initialize the global metrics publisher from the task config.

    Reads the metrics configuration from the task config YAML. If a custom
    metrics class is specified, attempts to instantiate it. Falls back to
    ``NopMetricsPublisher`` if no class is configured or instantiation fails.

    Args:
        task_config_uri: URI to the task config YAML file.
        service_name: Name of the service, used for metric grouping.

    Returns:
        ``True`` if metrics were initialized successfully (including the
        no-op default when no custom class is configured), ``False`` if a
        custom metrics class was specified but could not be loaded or instantiated.
    """
    global _metrics_instance
    os.environ[JOB_NAME_GROUPING_ENV_KEY] = service_name
    os.environ[TASK_CONFIG_URI_ENV_KEY] = str(task_config_uri)
    proto_utils = ProtoUtils()
    task_config: GbmlConfig = proto_utils.read_proto_from_yaml(
        uri=task_config_uri, proto_cls=GbmlConfig
    )

    metrics_cls_path = task_config.metrics_config.metrics_cls_path
    metrics_args = task_config.metrics_config.metrics_args

    if not metrics_cls_path:
        logger.info("Custom metrics class not provided. Using No-op metrics")
        _metrics_instance = NopMetricsPublisher()
        return True

    try:
        metrics_cls = os_utils.import_obj(metrics_cls_path)
        metrics_cls_instance: OpsMetricPublisher = metrics_cls(**metrics_args)
        assert isinstance(metrics_cls_instance, OpsMetricPublisher)
        _metrics_instance = metrics_cls_instance
        logger.info(f"Instantiated Custom Metrics Class from: {metrics_cls_path}")
        return True
    except Exception as e:
        logger.error(
            f"Could not instantiate class {metrics_cls_path}: {e}. Falling back to No-op metrics."
        )
        _metrics_instance = NopMetricsPublisher()
        return False


def get_metrics_service_instance() -> OpsMetricPublisher:
    if _metrics_instance is None:
        env_task_uri = os.environ.get(TASK_CONFIG_URI_ENV_KEY)
        env_service_name = os.environ.get(JOB_NAME_GROUPING_ENV_KEY) or os.environ.get(
            "GIGL_APPLIED_TASK_IDENTIFIER"
        )
        logger.debug(
            f"Detected task_config_uri={env_task_uri}, service_name={env_service_name} from env vars"
        )

        if env_task_uri and env_service_name:
            logger.info(
                f"Uninitialized process detected, initializing metrics instance from env vars (task_config_uri={env_task_uri}, service_name={env_service_name})"
            )
            initialize_metrics(
                task_config_uri=UriFactory.create_uri(env_task_uri),
                service_name=env_service_name,
            )

    if _metrics_instance is None:
        raise RuntimeError(
            "Metrics instance is not initialized. Call initialize_metrics() before getting the instance."
        )

    return _metrics_instance


def init_metrics_publisher_grouping_for_job(service_name: str) -> None:
    os.environ[JOB_NAME_GROUPING_ENV_KEY] = service_name
