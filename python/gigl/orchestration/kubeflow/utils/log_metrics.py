import kfp


def log_metrics_to_ui(
    task_config_uri: str,
    component_name: str,
    base_image: str,
) -> kfp.components.BaseComponent:
    """Publishes metrics for components to the Vertex AI Pipeline UI.
    Args:
        task_config_uri (str): URI to the task config.
        component (str): Name of the component to log metrics for.
        base_image: The Docker image to be used as the base image for the component.

    Returns:
        kfp.components.BaseComponent: The component to log metrics.
    """
    kfp_component = kfp.dsl.component(_log_eval_metrics_to_ui, base_image=base_image)
    return kfp_component(task_config_uri=task_config_uri, component=component_name)


def _log_eval_metrics_to_ui(
    task_config_uri: str,
    component: str,
    metrics: kfp.dsl.Output[kfp.dsl.Metrics],
) -> None:
    """Publishes metrics for components to the Vertex AI Pipeline UI.
    Args:
        task_config_uri (str): URI to the task config.
        component (str): Name of the component to log metrics for.
        metrics (Output[Metrics]): Metrics object to log metrics. Populated by the KFP SDK.
    """

    # This is required to resolve below packages when containerized by KFP.
    import json
    import os
    import sys

    from google.api_core.exceptions import NotFound

    sys.path.append(os.getcwd())

    from gigl.common import UriFactory
    from gigl.common.logger import Logger
    from gigl.common.utils.proto_utils import ProtoUtils
    from gigl.src.common.constants.components import GiGLComponents
    from gigl.src.common.utils.file_loader import FileLoader
    from snapchat.research.gbml import gbml_config_pb2

    logger = Logger()
    proto_utils = ProtoUtils()
    gbml_config_pb = proto_utils.read_proto_from_yaml(
        uri=UriFactory.create_uri(task_config_uri), proto_cls=gbml_config_pb2.GbmlConfig
    )
    logger.info(f"Read gbml config pb from {task_config_uri}")
    if component == GiGLComponents.Trainer.value:
        eval_metrics_uri = UriFactory.create_uri(
            uri=gbml_config_pb.shared_config.trained_model_metadata.eval_metrics_uri
        )
    elif component == GiGLComponents.PostProcessor.value:
        eval_metrics_uri = UriFactory.create_uri(
            uri=gbml_config_pb.shared_config.postprocessed_metadata.post_processor_log_metrics_uri
        )
    else:
        raise ValueError(f"Unknown component: {component}")

    logger.info(f"Fetching eval metrics from: {eval_metrics_uri.uri}")

    file_loader = FileLoader()
    metrics_str: str

    try:
        tfh = file_loader.load_to_temp_file(file_uri_src=eval_metrics_uri)
        with open(tfh.name, "r") as f:
            metrics_str = f.read()
    except (NotFound, FileNotFoundError) as e:
        logger.warning(
            f"Error loading metrics file: {e}, evaluation could have been skipped"
        )
        return

    logger.info(f"Got metrics_str: {metrics_str}")
    j = json.loads(metrics_str)
    if "metrics" in j:
        for metric in j["metrics"]:
            metrics.log_metric(metric["name"], metric["numberValue"])
