import os

from kfp.dsl import PipelineTask

from gigl.common.types.resource_config import CommonPipelineComponentConfigs

# Environment variables that select loader behaviour at runtime and must be
# propagated from the pipeline-compilation environment into each component's
# container. A value set when the pipeline is compiled/submitted (for example
# ``GIGL_COLLATE_IMPL=vectorized``) would otherwise never reach the remote
# component pod: the component container's environment is fixed at compile time
# and does not inherit the submitter's shell. Once the value is on the component
# pod, the launcher forwards it on to the spawned worker pool (see
# ``vertex_ai_launcher._collate_impl_passthrough_env_vars``). Kept as literals
# here to avoid importing the distributed loader package into the orchestration
# layer; mirror the canonical names in ``gigl/distributed/utils/neighborloader.py``.
_FORWARDED_LOADER_ENV_VARS = ("GIGL_COLLATE_IMPL", "GIGL_ABLP_LABEL_FORMAT")


def _propagate_loader_selection_env_vars(task: PipelineTask) -> None:
    """Copy any set loader-selection env vars from the compile environment onto ``task``.

    Only variables that are present and non-empty in ``os.environ`` at
    compile time are set, so an unset selector leaves the component to use its
    own default.
    """
    for name in _FORWARDED_LOADER_ENV_VARS:
        value = os.environ.get(name)
        if value:
            task.set_env_variable(name=name, value=value)


def add_task_resource_requirements(
    task: PipelineTask,
    common_pipeline_component_configs: CommonPipelineComponentConfigs,
):
    """
    Adds resource requirements to a the Kubeflow Pipeline (KFP) Task (ContainerOp)

    Args:
        task (ContainerOp): The task to add resource requirements to.
        common_pipeline_component_configs (CommonPipelineComponentConfigs): The common pipeline component configurations.

    Returns:
        None
    """
    DEFAULT_CPU_REQUEST = "4"
    DEFAULT_MEMORY_REQUEST = "16G"
    # default to cpu image, overwrite later as needed
    task.container_spec.image = common_pipeline_component_configs.cpu_container_image
    # Forward runtime loader-selection env vars (set at compile time) onto the
    # component container so they reach the component pod and, in turn, its
    # worker pool.
    _propagate_loader_selection_env_vars(task)
    # We don't set the default requests here as VAI pipelines are broken and
    # we're only getting logs with a `e2-standard-4` box.
    # *AND* the only way to get that box is to use it as the default.
    # task.set_cpu_request(DEFAULT_CPU_REQUEST)
    # task.set_memory_limit(DEFAULT_MEMORY_REQUEST)
