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
to read it; that helper also falls back to the deprecated
``RESOURCE_CONFIG_PATH`` with a one-time warning.
"""

import os
from typing import Final, Optional

from gigl.common.logger import Logger

GIGL_APPLIED_TASK_IDENTIFIER_ENV_KEY: Final[str] = "GIGL_APPLIED_TASK_IDENTIFIER"
GIGL_TASK_CONFIG_URI_ENV_KEY: Final[str] = "GIGL_TASK_CONFIG_URI"
GIGL_RESOURCE_CONFIG_URI_ENV_KEY: Final[str] = "GIGL_RESOURCE_CONFIG_URI"
GIGL_CPU_DOCKER_URI_ENV_KEY: Final[str] = "GIGL_CPU_DOCKER_URI"
GIGL_CUDA_DOCKER_URI_ENV_KEY: Final[str] = "GIGL_CUDA_DOCKER_URI"
GIGL_COMPONENT_ENV_KEY: Final[str] = "GIGL_COMPONENT"

# Optional override for the quota / billing project used by GiGL's BigQuery
# clients. When set to a non-empty value, every ``BqUtils`` BigQuery client
# bills quota to this project. The override is scoped to BigQuery clients only,
# leaving other Google clients (e.g. Cloud Storage) on their default quota
# project. Unset (or empty) means no override.
GIGL_BIGQUERY_QUOTA_PROJECT_ENV_KEY: Final[str] = "GIGL_BIGQUERY_QUOTA_PROJECT"

_LEGACY_RESOURCE_CONFIG_ENV_KEY: Final[str] = "RESOURCE_CONFIG_PATH"
_legacy_resource_config_env_warned: bool = False
_logger = Logger()


def read_resource_config_uri_from_env() -> Optional[str]:
    """Read the resource-config URI from the environment.

    Prefers ``GIGL_RESOURCE_CONFIG_URI``. Falls back to the deprecated
    ``RESOURCE_CONFIG_PATH`` and emits a one-time warning if that path is taken.

    Returns:
        The URI string if set under either name, else ``None``.
    """
    global _legacy_resource_config_env_warned
    value = os.environ.get(GIGL_RESOURCE_CONFIG_URI_ENV_KEY)
    if value is not None:
        return value

    legacy_value = os.environ.get(_LEGACY_RESOURCE_CONFIG_ENV_KEY)
    if legacy_value is not None and not _legacy_resource_config_env_warned:
        _logger.warning(
            f"Environment variable {_LEGACY_RESOURCE_CONFIG_ENV_KEY!r} is deprecated; "
            f"use {GIGL_RESOURCE_CONFIG_URI_ENV_KEY!r} instead. "
            "Support for the legacy name will be removed in a future release."
        )
        _legacy_resource_config_env_warned = True
    return legacy_value
