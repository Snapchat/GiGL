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
"""

import shlex
import subprocess
from collections.abc import Mapping
from typing import Optional

from gigl.common import Uri
from gigl.common.logger import Logger
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
    ``subprocess.run(shell_line, shell=True, check=True)``.

    The dispatcher takes ``command`` and ``args[]`` verbatim — no
    template substitution of any kind. Any placeholder text in those
    fields reaches ``subprocess.run`` literally; consumers that want
    substitution should resolve it at YAML-load time before the proto
    reaches this module.

    ``applied_task_identifier``, ``task_config_uri``,
    ``resource_config_uri``, ``process_command``,
    ``process_runtime_args``, ``cpu_docker_uri``, and ``cuda_docker_uri``
    are accepted for API symmetry with the GLT-side Vertex AI launchers
    but are intentionally not plumbed into the subprocess — the
    receiving CLI is expected to source whatever context it needs from
    the resource config it gets handed (or from env vars inherited from
    the parent process).

    Args:
        custom_launcher_config: Proto whose ``command`` is the shell
            snippet to execute and whose ``args`` are positional
            arguments appended verbatim.
        applied_task_identifier: Accepted for back-compat; ignored.
        task_config_uri: Accepted for back-compat; ignored.
        resource_config_uri: Accepted for back-compat; ignored.
        process_command: Accepted for back-compat; ignored.
        process_runtime_args: Accepted for back-compat; ignored.
        cpu_docker_uri: Accepted for back-compat; ignored.
        cuda_docker_uri: Accepted for back-compat; ignored.
        component: Which GiGL component is being launched. Must be in
            ``_LAUNCHABLE_COMPONENTS``.

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

    shell_line = " ".join([command, *(shlex.quote(a) for a in args)])
    logger.info(f"Launching {component.name} via subprocess: {shell_line!r}")
    subprocess.run(shell_line, shell=True, check=True)
