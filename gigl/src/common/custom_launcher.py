"""Subprocess dispatch for ``CustomLauncherConfig``-backed launchers.

Takes ``CustomLauncherConfig.command`` and ``CustomLauncherConfig.args``
verbatim and shells out via ``subprocess.run(shell_line, shell=True)``.
The shell-style invocation honors leading ``KEY=VALUE`` env-var
assignments in ``command`` so callers can self-document required env
without forcing the dispatcher to parse env separately.

The receiving subprocess has no special protocol — it is expected to be
a plain CLI that argparses whatever flags the YAML wires up via
``args[]``. The dispatcher performs no template substitution; any
dynamic content (runtime URIs, image refs, etc.) is the caller's
responsibility — typically resolved at YAML-load time before the
proto reaches this module.

The dispatcher exports its context args as ``GIGL_*`` environment
variables on the subprocess env (see ``gigl.env.custom_launcher``) so
receiving CLIs can ``os.environ.get(...)`` whatever runtime context
they need. The parent process's ``os.environ`` is never mutated; the
``GIGL_*`` keys live only in the per-call env passed to
``subprocess.run``.
"""

import os
import shlex
import subprocess
from collections.abc import Mapping
from typing import Optional

from gigl.common import Uri
from gigl.common.logger import Logger
from gigl.env.custom_launcher import (
    GIGL_APPLIED_TASK_IDENTIFIER_ENV_KEY,
    GIGL_COMPONENT_ENV_KEY,
    GIGL_CPU_DOCKER_URI_ENV_KEY,
    GIGL_CUDA_DOCKER_URI_ENV_KEY,
    GIGL_PROCESS_COMMAND_ENV_KEY,
    GIGL_RESOURCE_CONFIG_URI_ENV_KEY,
    GIGL_TASK_CONFIG_URI_ENV_KEY,
)
from gigl.src.common.constants.components import GiGLComponents
from snapchat.research.gbml.gigl_resource_config_pb2 import CustomLauncherConfig

logger = Logger()

_LAUNCHABLE_COMPONENTS: frozenset[GiGLComponents] = frozenset(
    {GiGLComponents.Trainer, GiGLComponents.Inferencer}
)


def launch_custom(
    custom_launcher_config: CustomLauncherConfig,
    applied_task_identifier: str,
    task_config_uri: Uri,
    resource_config_uri: Uri,
    process_command: str,
    process_runtime_args: Mapping[str, str],
    cpu_docker_uri: Optional[str],
    cuda_docker_uri: Optional[str],
    component: GiGLComponents,
) -> None:
    """Shell out to ``custom_launcher_config.command`` with ``args[]`` appended.

    Composes a shell line as ``command`` followed by each ``args[]``
    element passed through ``shlex.quote``, then invokes
    ``subprocess.run(shell_line, shell=True, check=True, env=env)``.

    The dispatcher takes ``command`` and ``args[]`` verbatim — no
    template substitution of any kind. Any placeholder text in those
    fields reaches ``subprocess.run`` literally; consumers that want
    substitution should resolve it at YAML-load time before the proto
    reaches this module.

    The subprocess env is built per-call from ``os.environ.copy()`` plus
    the ``GIGL_*`` keys defined in :mod:`gigl.env.custom_launcher`. The
    parent process's ``os.environ`` is never mutated. Optional URI args
    (``cpu_docker_uri``, ``cuda_docker_uri``) are omitted from the env
    when ``None`` so the receiver's ``os.environ.get(KEY)`` returns
    ``None`` and preserves the original ``Optional[str]`` semantics.

    Args:
        custom_launcher_config: Proto whose ``command`` is the shell
            snippet to execute and whose ``args`` are positional
            arguments appended verbatim.
        applied_task_identifier: Exported to the subprocess as
            ``GIGL_APPLIED_TASK_IDENTIFIER``.
        task_config_uri: Exported to the subprocess as
            ``GIGL_TASK_CONFIG_URI`` (stringified).
        resource_config_uri: Exported to the subprocess as
            ``GIGL_RESOURCE_CONFIG_URI`` (stringified).
        process_command: Exported to the subprocess as
            ``GIGL_PROCESS_COMMAND``.
        process_runtime_args: Accepted for API symmetry with the
            GLT-side Vertex AI launchers but not currently exported —
            there is no clean single-env-var encoding for a dict.
        cpu_docker_uri: Exported as ``GIGL_CPU_DOCKER_URI`` when set;
            the env var is omitted entirely when ``None``.
        cuda_docker_uri: Exported as ``GIGL_CUDA_DOCKER_URI`` when set;
            the env var is omitted entirely when ``None``.
        component: Which GiGL component is being launched. Must be in
            ``_LAUNCHABLE_COMPONENTS``. Exported as ``GIGL_COMPONENT``
            using ``component.name`` (e.g. ``"Trainer"``).

    Raises:
        ValueError: If ``component`` is not Trainer or Inferencer, or if
            ``custom_launcher_config.command`` is empty.
        subprocess.CalledProcessError: If the spawned subprocess exits
            non-zero.
    """
    if component not in _LAUNCHABLE_COMPONENTS:
        raise ValueError(f"Invalid component: {component}")
    if not custom_launcher_config.command:
        raise ValueError("CustomLauncherConfig.command must be set")

    command: str = custom_launcher_config.command
    args: list[str] = list(custom_launcher_config.args)

    env: dict[str, str] = os.environ.copy()
    env[GIGL_APPLIED_TASK_IDENTIFIER_ENV_KEY] = applied_task_identifier
    env[GIGL_TASK_CONFIG_URI_ENV_KEY] = str(task_config_uri)
    env[GIGL_RESOURCE_CONFIG_URI_ENV_KEY] = str(resource_config_uri)
    env[GIGL_PROCESS_COMMAND_ENV_KEY] = process_command
    env[GIGL_COMPONENT_ENV_KEY] = component.name
    if cpu_docker_uri is not None:
        env[GIGL_CPU_DOCKER_URI_ENV_KEY] = cpu_docker_uri
    if cuda_docker_uri is not None:
        env[GIGL_CUDA_DOCKER_URI_ENV_KEY] = cuda_docker_uri

    shell_line = " ".join([command, *(shlex.quote(a) for a in args)])
    logger.info(f"Launching {component.name} via subprocess: {shell_line!r}")
    subprocess.run(shell_line, shell=True, check=True, env=env)
