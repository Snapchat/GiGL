"""Environment-variable keys used across GiGL.

Most of these keys are set on subprocess env (never on the parent
``os.environ``) by ``gigl.src.common.custom_launcher.launch_custom`` so that
receiving CLIs can ``os.environ.get(...)`` their runtime context.

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
