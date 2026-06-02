"""Runtime initialization helpers for GiGL component entrypoints."""

import os
from typing import Optional

from gigl.common import Uri
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

    Args:
        applied_task_identifier: Unique identifier for the GiGL job.
        task_config_uri: URI to the task config YAML file.
        resource_config_uri: URI to the resource config YAML file.
        service_name: Name of the service, used for metric grouping.
        component: GiGL component being initialized.
        cpu_docker_uri: CPU source image URI. Defaults to the release CPU image.
        cuda_docker_uri: CUDA source image URI. Defaults to the release CUDA image.
    """
    os.environ.update(
        get_gigl_runtime_env_vars(
            applied_task_identifier=applied_task_identifier,
            task_config_uri=task_config_uri,
            resource_config_uri=resource_config_uri,
            component=component,
            cpu_docker_uri=cpu_docker_uri,
            cuda_docker_uri=cuda_docker_uri,
        )
    )
    initialize_metrics(task_config_uri=task_config_uri, service_name=service_name)
