"""Subprocess dispatch for ``CustomResourceConfig``-backed launchers.

Resolves ``${gigl:*}`` placeholders in ``CustomResourceConfig.command`` /
``CustomResourceConfig.args`` against a runtime context (task config URI,
applied task identifier, component, …), then shells out via
``subprocess.run(shell_line, shell=True)``. The shell-style invocation
honors leading ``KEY=VALUE`` env-var assignments in ``command`` so
callers can self-document required env without forcing the dispatcher to
parse env separately.

The receiving subprocess has no special protocol — it is expected to be
a plain CLI that argparses whatever flags the YAML wires up via
``args[]``. When more context is needed than ``CustomResourceConfig`` can
carry directly, the YAML embeds ``${gigl:<key>}`` placeholders; this
module populates the values just before exec via the ``gigl`` resolver
(see ``gigl.common.omegaconf_resolvers``).
"""

import shlex
import subprocess
from collections.abc import Mapping
from typing import Optional

from omegaconf import OmegaConf

from gigl.common import Uri
from gigl.common.logger import Logger
from gigl.common.omegaconf_resolvers import (
    register_resolvers,
    set_gigl_resolver_values,
)
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
    is_dry_run: bool = False,
) -> None:
    """Resolve ``custom_resource_config`` and shell out to the configured command.

    Populates the ``gigl`` OmegaConf resolver from the runtime kwargs,
    re-resolves ``custom_resource_config.command`` / ``.args`` so any
    ``${gigl:*}`` placeholders bind to runtime values, then invokes the
    composed shell line via ``subprocess.run(shell=True, check=True)``.

    ``process_command`` and ``process_runtime_args`` are accepted for
    back-compat with the existing GLT trainer / inferencer call sites
    but are intentionally NOT plumbed through to the subprocess —
    consumers re-derive them from the ``--gbml_uri`` (or equivalent)
    they receive.

    Args:
        custom_resource_config: Proto whose ``command`` is the shell
            snippet to execute and whose ``args`` are positional
            arguments. Both fields support ``${gigl:<key>}``
            interpolation.
        applied_task_identifier: Stable identifier for the job; exposed
            as ``${gigl:applied_task_identifier}``.
        task_config_uri: URI of the GbmlConfig serialized as YAML;
            exposed as ``${gigl:task_config_uri}``.
        resource_config_uri: URI of the GiglResourceConfig serialized as
            YAML; exposed as ``${gigl:resource_config_uri}``.
        process_command: Accepted for back-compat; ignored.
        process_runtime_args: Accepted for back-compat; ignored.
        cpu_docker_uri: Optional CPU Docker image URI; exposed as
            ``${gigl:cpu_docker_image}`` (empty string when ``None``).
        cuda_docker_uri: Optional CUDA Docker image URI; exposed as
            ``${gigl:cuda_docker_image}`` (empty string when ``None``).
        component: Which GiGL component is being launched. Must be in
            ``_LAUNCHABLE_COMPONENTS``. Exposed as ``${gigl:component}``
            (Title-case ``.name``, e.g. ``"Trainer"``).
        is_dry_run: If True, the resolved shell line is logged and the
            function returns without spawning a subprocess. Exposed as
            ``${gigl:is_dry_run}`` (string ``"1"`` or ``"0"``).

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

    # Defensive registration — direct callers (tests, scripts) may build
    # a CustomResourceConfig programmatically without going through
    # ProtoUtils, which is the usual registration path.
    register_resolvers()

    set_gigl_resolver_values(
        {
            "applied_task_identifier": applied_task_identifier,
            "task_config_uri": str(task_config_uri),
            "resource_config_uri": str(resource_config_uri),
            # Title-case component name so the receiving CLI's argparse
            # ``choices=["Trainer", "Inferencer"]`` accepts it. ``.value``
            # is lowercase and would mismatch.
            "component": component.name,
            "cpu_docker_image": cpu_docker_uri or "",
            "cuda_docker_image": cuda_docker_uri or "",
            "is_dry_run": "1" if is_dry_run else "0",
        }
    )

    # Re-resolve via OmegaConf so any ${gigl:*} placeholder in the
    # proto's command/args strings binds to the now-populated runtime
    # value.
    container = OmegaConf.create(
        {
            "command": custom_resource_config.command,
            "args": list(custom_resource_config.args),
        }
    )
    resolved_command: str = container.command  # type: ignore[assignment]
    resolved_args: list[str] = list(container.args)  # type: ignore[arg-type]

    shell_line = " ".join(
        [resolved_command, *(shlex.quote(a) for a in resolved_args)]
    )
    logger.info(
        f"Launching {component.name} via subprocess: {shell_line!r} "
        f"dry_run={is_dry_run}"
    )
    if is_dry_run:
        return
    subprocess.run(shell_line, shell=True, check=True)
