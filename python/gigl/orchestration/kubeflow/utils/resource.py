from gigl.common.types.resource_config import CommonPipelineComponentConfigs
from kfp.dsl import PipelineTask


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
    # We don't set the default requests here as VAI pipelines are broken and
    # we're only getting logs with a `e2-standard-4` box.
    # *AND* the only way to get that box is to use it as the default.
    # task.set_cpu_request(DEFAULT_CPU_REQUEST)
    # task.set_memory_limit(DEFAULT_MEMORY_REQUEST)
