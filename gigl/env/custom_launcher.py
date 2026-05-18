"""Environment-variable keys exported by ``launch_custom``.

These keys are set on the subprocess env (never on the parent
``os.environ``) by ``gigl.src.common.custom_launcher.launch_custom`` so
that receiving CLIs can ``os.environ.get(...)`` their runtime context.
"""

from typing import Final

GIGL_APPLIED_TASK_IDENTIFIER_ENV_KEY: Final[str] = "GIGL_APPLIED_TASK_IDENTIFIER"
GIGL_TASK_CONFIG_URI_ENV_KEY: Final[str] = "GIGL_TASK_CONFIG_URI"
GIGL_RESOURCE_CONFIG_URI_ENV_KEY: Final[str] = "GIGL_RESOURCE_CONFIG_URI"
GIGL_PROCESS_COMMAND_ENV_KEY: Final[str] = "GIGL_PROCESS_COMMAND"
GIGL_CPU_DOCKER_URI_ENV_KEY: Final[str] = "GIGL_CPU_DOCKER_URI"
GIGL_CUDA_DOCKER_URI_ENV_KEY: Final[str] = "GIGL_CUDA_DOCKER_URI"
GIGL_COMPONENT_ENV_KEY: Final[str] = "GIGL_COMPONENT"
