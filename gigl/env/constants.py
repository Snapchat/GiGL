"""Environment-variable keys used across GiGL.

These keys are set either on component-process ``os.environ`` by
``gigl.src.common.utils.gigl_runtime.initialize_gigl_runtime`` or on
launched process envs by ``gigl.src.common.custom_launcher.launch_custom``
and ``gigl.src.common.vertex_ai_launcher`` so that receiving CLIs can
``os.environ.get(...)`` their runtime context.

``GIGL_RESOURCE_CONFIG_URI`` is also written to the parent ``os.environ`` by
``gigl.env.pipelines_config.get_resource_config`` so that downstream readers
(e.g. ``GiglResourceConfigWrapper.get_resource_config_uri``) can recover the
value within the same process. Use :func:`read_resource_config_uri_from_env`
to read it.
"""

import os
from typing import Final, Optional

GIGL_APPLIED_TASK_IDENTIFIER_ENV_KEY: Final[str] = "GIGL_APPLIED_TASK_IDENTIFIER"
GIGL_TASK_CONFIG_URI_ENV_KEY: Final[str] = "GIGL_TASK_CONFIG_URI"
GIGL_RESOURCE_CONFIG_URI_ENV_KEY: Final[str] = "GIGL_RESOURCE_CONFIG_URI"
GIGL_CPU_DOCKER_URI_ENV_KEY: Final[str] = "GIGL_CPU_DOCKER_URI"
GIGL_CUDA_DOCKER_URI_ENV_KEY: Final[str] = "GIGL_CUDA_DOCKER_URI"
GIGL_COMPONENT_ENV_KEY: Final[str] = "GIGL_COMPONENT"
GIGL_BIGQUERY_QUOTA_PROJECT_ENV_KEY: Final[str] = "GIGL_BIGQUERY_QUOTA_PROJECT"


def read_resource_config_uri_from_env() -> Optional[str]:
    """Read the resource-config URI from ``GIGL_RESOURCE_CONFIG_URI``.

    Returns:
        The URI string if the variable is set, else ``None``.
    """
    return os.environ.get(GIGL_RESOURCE_CONFIG_URI_ENV_KEY)
