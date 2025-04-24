from kfp.dsl import PipelineTask

from gigl.common.types.resource_config import CommonPipelineComponentConfigs


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
    DEFAULT_MEMORY_REQUEST = "1Gi"
    # default to cpu image, overwrite later as needed
    task.container_spec.image = common_pipeline_component_configs.cpu_container_image
    task.set_cpu_request(DEFAULT_CPU_REQUEST)
    task.set_memory_request(DEFAULT_MEMORY_REQUEST)
