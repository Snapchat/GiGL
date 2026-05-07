"""Subprocess dispatch for ``CustomResourceConfig``-backed launchers.

Takes ``CustomResourceConfig.command`` and ``CustomResourceConfig.args``
verbatim and shells out via ``subprocess.run(shell_line, shell=True)``.
The shell-style invocation honors leading ``KEY=VALUE`` env-var
assignments in ``command`` so callers can self-document required env
without forcing the dispatcher to parse env separately.

Before invoking the subprocess, the dispatcher composes a per-call
``env`` dict from ``os.environ.copy()`` plus the launcher-managed
``GIGL_*`` keys below, and passes it via ``subprocess.run(..., env=...)``.
The parent process's ``os.environ`` is **not** mutated, so concurrent
``launch_custom`` calls in the same process do not race or leak
context across launches.

Env-var contract (set on every call):

- ``GIGL_TASK_CONFIG_URI``         — ``str(task_config_uri)``
- ``GIGL_RESOURCE_CONFIG_URI``     — ``str(resource_config_uri)``
- ``GIGL_COMPONENT``               — ``component.name`` (``"Trainer"`` / ``"Inferencer"``)
- ``GIGL_APPLIED_TASK_IDENTIFIER`` — ``str(applied_task_identifier)``
- ``GIGL_CUDA_DOCKER_IMAGE``       — ``cuda_docker_uri or ""``
- ``GIGL_CPU_DOCKER_IMAGE``        — ``cpu_docker_uri or ""``

``GIGL_*`` is a reserved prefix for launcher-managed context. Custom-
launcher YAML authors must not collide with these names: a leading
``KEY=VALUE python ...`` shell prefix on ``command`` overrides the
inherited env *for that command*, so a stray ``GIGL_FOO=...`` prefix
would silently shadow what the dispatcher set.

The receiving subprocess is otherwise a plain CLI that argparses
whatever flags the YAML wires up via ``args[]``, plus reads any
``GIGL_*`` keys it cares about from ``os.environ``.
"""

import os
import shlex
import subprocess
from collections.abc import Mapping
from typing import Optional

from gigl.common import Uri
from gigl.common.logger import Logger
from gigl.src.common.constants.components import GiGLComponents
from snapchat.research.gbml.gigl_resource_config_pb2 import CustomResourceConfig

logger = Logger()

_LAUNCHABLE_COMPONENTS: frozenset[GiGLComponents] = frozenset(
    {GiGLComponents.Trainer, GiGLComponents.Inferencer}
)


def launch_custom(
    custom_resource_config: CustomResourceConfig,
    applied_task_identifier: str,
    task_config_uri: Uri,
    resource_config_uri: Uri,
    process_command: str,
    process_runtime_args: Mapping[str, str],
    cpu_docker_uri: Optional[str],
    cuda_docker_uri: Optional[str],
    component: GiGLComponents,
) -> None:
    """Shell out to ``custom_resource_config.command`` with ``args[]`` appended.

    Composes a shell line as ``command`` followed by each ``args[]``
    element passed through ``shlex.quote``, then invokes
    ``subprocess.run(shell_line, shell=True, check=True, env=...)``.

    The dispatcher takes ``command`` and ``args[]`` verbatim — no
    template substitution of any kind. Any placeholder text in those
    fields reaches ``subprocess.run`` literally; consumers that want
    runtime context should read it from the ``GIGL_*`` env vars the
    dispatcher injects on the subprocess (see module docstring).

    The subprocess's env is built as ``os.environ.copy()`` plus the
    six launcher-managed ``GIGL_*`` keys; the parent process's
    ``os.environ`` is not mutated.

    Args:
        custom_resource_config: Proto whose ``command`` is the shell
            snippet to execute and whose ``args`` are positional
            arguments appended verbatim.
        applied_task_identifier: Stable job identifier; exposed to the
            subprocess as ``GIGL_APPLIED_TASK_IDENTIFIER``.
        task_config_uri: Frozen GbmlConfig URI; exposed as
            ``GIGL_TASK_CONFIG_URI``.
        resource_config_uri: GiglResourceConfig URI; exposed as
            ``GIGL_RESOURCE_CONFIG_URI``.
        process_command: Accepted for API symmetry with the GLT-side
            Vertex AI launchers; not plumbed to the subprocess.
        process_runtime_args: Accepted for API symmetry; not plumbed.
        cpu_docker_uri: CPU image URI; exposed as
            ``GIGL_CPU_DOCKER_IMAGE`` (empty string when ``None``).
        cuda_docker_uri: CUDA image URI; exposed as
            ``GIGL_CUDA_DOCKER_IMAGE`` (empty string when ``None``).
        component: Which GiGL component is being launched. Must be in
            ``_LAUNCHABLE_COMPONENTS``. Exposed as ``GIGL_COMPONENT``.

    Raises:
        ValueError: If ``component`` is not Trainer or Inferencer, or if
            ``custom_resource_config.command`` is empty.
        subprocess.CalledProcessError: If the spawned subprocess exits
            non-zero.
    """
    if component not in _LAUNCHABLE_COMPONENTS:
        raise ValueError(f"Invalid component: {component}")
    if not custom_resource_config.command:
        raise ValueError("CustomResourceConfig.command must be set")

    command: str = custom_resource_config.command
    args: list[str] = list(custom_resource_config.args)

    shell_line = " ".join([command, *(shlex.quote(a) for a in args)])

    # Per-call env dict: copy the parent env so the subprocess inherits
    # PATH / GCP creds / etc., then overlay the launcher-managed
    # GIGL_* keys. Parent ``os.environ`` is not mutated.
    env = os.environ.copy()
    env.update(
        {
            "GIGL_TASK_CONFIG_URI": str(task_config_uri),
            "GIGL_RESOURCE_CONFIG_URI": str(resource_config_uri),
            "GIGL_COMPONENT": component.name,
            "GIGL_APPLIED_TASK_IDENTIFIER": str(applied_task_identifier),
            "GIGL_CUDA_DOCKER_IMAGE": cuda_docker_uri or "",
            "GIGL_CPU_DOCKER_IMAGE": cpu_docker_uri or "",
        }
    )

    logger.info(f"Launching {component.name} via subprocess: {shell_line!r}")
    subprocess.run(shell_line, shell=True, check=True, env=env)
