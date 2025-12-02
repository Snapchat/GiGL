from collections.abc import Mapping
from typing import Optional

from google.cloud.aiplatform_v1.types import Scheduling, env_var

from gigl.common import Uri
from gigl.common.services.vertex_ai import VertexAiJobConfig
from snapchat.research.gbml.gigl_resource_config_pb2 import VertexAiResourceConfig


def build_job_config(
    job_name: str,
    task_config_uri: Uri,
    resource_config_uri: Uri,
    command_str: str,
    args: Mapping[str, str],
    use_cuda: bool,
    container_uri: str,
    vertex_ai_resource_config: VertexAiResourceConfig,
    env_vars: list[env_var.EnvVar],
    labels: Optional[dict[str, str]] = None,
) -> VertexAiJobConfig:
    """Build a VertexAiJobConfig for training or inference jobs.

    This function constructs a configuration object for running GiGL training or inference
    jobs on Vertex AI. It assembles job arguments, sets appropriate job naming conventions,
    and configures resource specifications based on the provided parameters.

    Args:
        job_name (str): The base name for the job. Will be prefixed with "gigl_train_" or "gigl_infer_".
        is_inference (bool): Whether this is an inference job (True) or training job (False).
        task_config_uri (Uri): URI to the task configuration file.
        resource_config_uri (Uri): URI to the resource configuration file.
        command_str (str): The command to run in the container (will be split on spaces).
        args (Mapping[str, str]): Additional command-line arguments to pass to the job.
        use_cuda (bool): Whether to use CUDA. If True, adds --use_cuda flag.
        container_uri (str): The URI of the container image to use.
        vertex_ai_resource_config (VertexAiResourceConfig): Resource configuration including
            machine type, GPU type, replica count, timeout, and scheduling strategy.
        env_vars (list[env_var.EnvVar]): Environment variables to set in the container.
        labels (Optional[dict[str, str]]): Labels to associate with the job. Defaults to None.

    Returns:
        VertexAiJobConfig: A configuration object ready to be used with VertexAIService.launch_job().
    """
    job_args = (
        [
            f"--job_name={job_name}",
            f"--task_config_uri={task_config_uri}",
            f"--resource_config_uri={resource_config_uri}",
        ]
        + (["--use_cuda"] if use_cuda else [])
        + ([f"--{k}={v}" for k, v in args.items()])
    )

    command = command_str.strip().split(" ")

    job_config = VertexAiJobConfig(
        job_name=job_name,
        container_uri=container_uri,
        command=command,
        args=job_args,
        environment_variables=env_vars,
        machine_type=vertex_ai_resource_config.machine_type,
        accelerator_type=vertex_ai_resource_config.gpu_type.upper().replace("-", "_"),
        accelerator_count=vertex_ai_resource_config.gpu_limit,
        replica_count=vertex_ai_resource_config.num_replicas,
        labels=labels,
        timeout_s=vertex_ai_resource_config.timeout
        if vertex_ai_resource_config.timeout
        else None,
        # This should be `aiplatform.gapic.Scheduling.Strategy[inferencer_resource_config.scheduling_strategy]`
        # But mypy complains otherwise...
        # python/gigl/src/inference/v2/glt_inferencer.py:124: error: The type "type[Strategy]" is not generic and not indexable  [misc]
        # TODO(kmonte): Fix this
        scheduling_strategy=getattr(
            Scheduling.Strategy,
            vertex_ai_resource_config.scheduling_strategy,
        )
        if vertex_ai_resource_config.scheduling_strategy
        else None,
    )
    return job_config
