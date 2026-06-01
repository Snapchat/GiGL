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
variables on the subprocess env (see ``gigl.env.constants``) so
receiving CLIs can ``os.environ.get(...)`` whatever runtime context
they need. The parent process's ``os.environ`` is never mutated; the
``GIGL_*`` keys live only in the per-call env passed to
``subprocess.run``.
"""

import os
import shlex
import subprocess
from typing import Optional

from gigl.common import Uri
from gigl.common.logger import Logger
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.utils.gigl_runtime import get_gigl_runtime_env_vars
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
    the ``GIGL_*`` keys defined in :mod:`gigl.env.constants`. The
    parent process's ``os.environ`` is never mutated. When ``None`` is
    passed for ``cpu_docker_uri`` / ``cuda_docker_uri``, the
    corresponding env var falls back to
    :data:`gigl.common.constants.DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU` /
    :data:`gigl.common.constants.DEFAULT_GIGL_RELEASE_SRC_IMAGE_CUDA`
    so receivers always observe a usable image URI.

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
        cpu_docker_uri: Exported as ``GIGL_CPU_DOCKER_URI``. Falls back
            to ``DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU`` when ``None``.
        cuda_docker_uri: Exported as ``GIGL_CUDA_DOCKER_URI``. Falls
            back to ``DEFAULT_GIGL_RELEASE_SRC_IMAGE_CUDA`` when
            ``None``.
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
    env.update(
        get_gigl_runtime_env_vars(
            applied_task_identifier=applied_task_identifier,
            task_config_uri=task_config_uri,
            resource_config_uri=resource_config_uri,
            component=component,
            cpu_docker_uri=cpu_docker_uri,
            cuda_docker_uri=cuda_docker_uri,
        )
    )

    shell_line = " ".join([command, *(shlex.quote(a) for a in args)])
    logger.info(f"Launching {component.name} via subprocess: {shell_line!r}")
    subprocess.run(shell_line, shell=True, check=True, env=env)
