"""Runtime initialization helpers for GiGL component entrypoints."""

import os
from typing import Optional

from gigl.common import Uri
from gigl.env.constants import (
    GIGL_CPU_DOCKER_URI_ENV_KEY,
    GIGL_CUDA_DOCKER_URI_ENV_KEY,
)
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.utils.gigl_env import get_gigl_runtime_env_vars
from gigl.src.common.utils.metrics_service_provider import initialize_metrics


def initialize_gigl_runtime(
    applied_task_identifier: str,
    task_config_uri: Uri,
    resource_config_uri: Uri,
    service_name: str,
    component: GiGLComponents,
    cpu_docker_uri: Optional[str] = None,
    cuda_docker_uri: Optional[str] = None,
) -> None:
    """Initialize GiGL runtime environment and metrics for a component.

    For ``SubgraphSampler`` and ``SplitGenerator`` only metrics are initialized;
    runtime env vars are not set, since these legacy (Scala/Spark) components do
    not consume the GiGL Python runtime.

    Args:
        applied_task_identifier: Unique identifier for the GiGL job.
        task_config_uri: URI to the task config YAML file.
        resource_config_uri: URI to the resource config YAML file.
        service_name: Name of the service, used for metric grouping.
        component: GiGL component being initialized.
        cpu_docker_uri: CPU source image URI. Defaults to the release CPU image.
        cuda_docker_uri: CUDA source image URI. Defaults to the release CUDA image.
    """
    if component in {GiGLComponents.SubgraphSampler, GiGLComponents.SplitGenerator}:
        initialize_metrics(task_config_uri=task_config_uri, service_name=service_name)
        return

    # TODO(kmonte): Also expose the dataflow docker URI (used as custom_worker_image_uri by
    # DataPreprocessor/Inferencer) as a GIGL_DATAFLOW_DOCKER_URI env var for parity with the
    # CPU/CUDA docker URIs. Requires a new key in gigl/env/constants.py and threading it
    # through get_gigl_runtime_env_vars.
    resolved_cpu_docker_uri = (
        os.environ.get(GIGL_CPU_DOCKER_URI_ENV_KEY)
        if cpu_docker_uri is None
        else cpu_docker_uri
    )
    resolved_cuda_docker_uri = (
        os.environ.get(GIGL_CUDA_DOCKER_URI_ENV_KEY)
        if cuda_docker_uri is None
        else cuda_docker_uri
    )
    os.environ.update(
        get_gigl_runtime_env_vars(
            applied_task_identifier=applied_task_identifier,
            task_config_uri=task_config_uri,
            resource_config_uri=resource_config_uri,
            component=component,
            cpu_docker_uri=resolved_cpu_docker_uri,
            cuda_docker_uri=resolved_cuda_docker_uri,
        )
    )
    initialize_metrics(task_config_uri=task_config_uri, service_name=service_name)
