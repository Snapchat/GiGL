"""Dispatch for `CustomResourceConfig`-backed, pluggable launchers.

Resolves `CustomResourceConfig.launcher_fn` at runtime via
`gigl.common.utils.os_utils.import_obj` and invokes the resulting callable
with the same kwargs GiGL's built-in Vertex AI launchers receive, plus the
opaque `launcher_args` map. This lets downstream consumers plug in their own
cluster-management logic (for example, a managed Ray platform) without
forking GiGL.
"""

from collections.abc import Mapping
from typing import Optional

from gigl.common import Uri
from gigl.common.logger import Logger
from gigl.common.utils.os_utils import import_obj
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
    """Resolve `custom_resource_config.launcher_fn` and invoke it.

    The resolved callable receives all of the positional context the built-in
    Vertex AI launchers receive (task/resource URIs, docker URIs, process
    command/args, component, dry-run flag) plus the opaque `launcher_args`
    dict materialized from the proto map.

    Args:
        custom_resource_config: Proto whose `launcher_fn` is a dotted import
            path to a callable and whose `launcher_args` is an opaque map
            forwarded verbatim to the callable.
        applied_task_identifier: Stable identifier for the job.
        task_config_uri: URI of the GbmlConfig serialized as YAML.
        resource_config_uri: URI of the GiglResourceConfig serialized as YAML.
        process_command: Command (module path) the launcher should execute.
        process_runtime_args: Flag map appended to the process command.
        cpu_docker_uri: Optional CPU Docker image URI.
        cuda_docker_uri: Optional CUDA Docker image URI.
        component: Which GiGL component is being launched. Must be in
            `_LAUNCHABLE_COMPONENTS`.
        is_dry_run: If True, the launcher should validate inputs and return
            without actually spawning remote jobs.

    Raises:
        ValueError: If `component` is not Trainer or Inferencer, or if
            `custom_resource_config.launcher_fn` is empty.
        TypeError: If the resolved `launcher_fn` is not callable.
    """
    if component not in _LAUNCHABLE_COMPONENTS:
        raise ValueError(f"Invalid component: {component}")
    fn_path = custom_resource_config.launcher_fn
    if not fn_path:
        raise ValueError("CustomResourceConfig.launcher_fn must be set")
    launcher = import_obj(fn_path)
    if not callable(launcher):
        raise TypeError(f"{fn_path} resolved to non-callable: {type(launcher)}")
    launcher_args = dict(custom_resource_config.launcher_args)
    logger.info(
        f"Dispatching {component.value} to custom launcher {fn_path} "
        f"with keys {sorted(launcher_args.keys())} dry_run={is_dry_run}"
    )
    launcher(
        applied_task_identifier=applied_task_identifier,
        task_config_uri=task_config_uri,
        resource_config_uri=resource_config_uri,
        process_command=process_command,
        process_runtime_args=process_runtime_args,
        launcher_args=launcher_args,
        cpu_docker_uri=cpu_docker_uri,
        cuda_docker_uri=cuda_docker_uri,
        component=component,
        is_dry_run=is_dry_run,
    )
