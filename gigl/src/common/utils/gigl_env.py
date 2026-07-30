"""Helpers for constructing GiGL runtime environment variables."""

import os
from typing import Optional

from gigl.common import Uri
from gigl.common.constants import (
    DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU,
    DEFAULT_GIGL_RELEASE_SRC_IMAGE_CUDA,
)
from gigl.env.constants import (
    GIGL_APPLIED_TASK_IDENTIFIER_ENV_KEY,
    GIGL_BIGQUERY_QUOTA_PROJECT_ENV_KEY,
    GIGL_COMPONENT_ENV_KEY,
    GIGL_CPU_DOCKER_URI_ENV_KEY,
    GIGL_CUDA_DOCKER_URI_ENV_KEY,
    GIGL_RESOURCE_CONFIG_URI_ENV_KEY,
    GIGL_TASK_CONFIG_URI_ENV_KEY,
)
from gigl.src.common.constants.components import GiGLComponents


def get_gigl_runtime_env_vars(
    applied_task_identifier: str,
    task_config_uri: Uri,
    resource_config_uri: Uri,
    component: GiGLComponents,
    cpu_docker_uri: Optional[str] = None,
    cuda_docker_uri: Optional[str] = None,
) -> dict[str, str]:
    """Build standard GiGL runtime environment variables.

    Args:
        applied_task_identifier: Unique identifier for the GiGL job.
        task_config_uri: URI to the task config YAML file.
        resource_config_uri: URI to the resource config YAML file.
        component: GiGL component being initialized.
        cpu_docker_uri: CPU source image URI. Defaults to the release CPU image.
        cuda_docker_uri: CUDA source image URI. Defaults to the release CUDA image.

    Returns:
        A mapping of standard GiGL environment variable names to values. A
        non-empty BigQuery quota-project override from the launcher environment
        is included when present.
    """
    resolved_cpu_docker_uri = cpu_docker_uri or DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU
    resolved_cuda_docker_uri = cuda_docker_uri or DEFAULT_GIGL_RELEASE_SRC_IMAGE_CUDA

    runtime_env_vars = {
        GIGL_APPLIED_TASK_IDENTIFIER_ENV_KEY: str(applied_task_identifier),
        GIGL_TASK_CONFIG_URI_ENV_KEY: str(task_config_uri),
        GIGL_RESOURCE_CONFIG_URI_ENV_KEY: str(resource_config_uri),
        GIGL_COMPONENT_ENV_KEY: component.name,
        GIGL_CPU_DOCKER_URI_ENV_KEY: resolved_cpu_docker_uri,
        GIGL_CUDA_DOCKER_URI_ENV_KEY: resolved_cuda_docker_uri,
    }
    bigquery_quota_project = os.environ.get(GIGL_BIGQUERY_QUOTA_PROJECT_ENV_KEY)
    if bigquery_quota_project:
        runtime_env_vars[GIGL_BIGQUERY_QUOTA_PROJECT_ENV_KEY] = bigquery_quota_project
    return runtime_env_vars
