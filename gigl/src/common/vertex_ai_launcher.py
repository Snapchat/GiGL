"""Shared functionality for launching Vertex AI jobs for training and inference."""

from collections.abc import Mapping
from typing import Optional

from google.cloud.aiplatform_v1.types import Scheduling, accelerator_type, env_var

from gigl.common import Uri
from gigl.common.constants import (
    DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU,
    DEFAULT_GIGL_RELEASE_SRC_IMAGE_CUDA,
)
from gigl.common.logger import Logger
from gigl.common.services.vertex_ai import VertexAiJobConfig, VertexAIService
from gigl.env.distributed import COMPUTE_CLUSTER_LOCAL_WORLD_SIZE_ENV_KEY
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.types.pb_wrappers.gigl_resource_config import (
    GiglResourceConfigWrapper,
)
from snapchat.research.gbml.gigl_resource_config_pb2 import (
    VertexAiGraphStoreConfig,
    VertexAiResourceConfig,
)

logger = Logger()

_LAUNCHABLE_COMPONENTS: frozenset[GiGLComponents] = frozenset(
    [GiGLComponents.Trainer, GiGLComponents.Inferencer]
)


def launch_single_pool_job(
    vertex_ai_resource_config: VertexAiResourceConfig,
    job_name: str,
    task_config_uri: Uri,
    resource_config_uri: Uri,
    process_command: str,
    process_runtime_args: Mapping[str, str],
    resource_config_wrapper: GiglResourceConfigWrapper,
    cpu_docker_uri: Optional[str],
    cuda_docker_uri: Optional[str],
    component: GiGLComponents,
    vertex_ai_region: str,
) -> None:
    """Launch a single pool job on Vertex AI.

    Args:
        vertex_ai_resource_config: The Vertex AI resource configuration
        job_name: Full name for the Vertex AI job
        task_config_uri: URI to the task configuration
        resource_config_uri: URI to the resource configuration
        process_command: Command to run in the container
        process_runtime_args: Runtime arguments for the process
        resource_config_wrapper: Wrapper for the resource configuration
        cpu_docker_uri: Docker image URI for CPU execution
        cuda_docker_uri: Docker image URI for GPU execution
        component: The GiGL component (Trainer or Inferencer)
        vertex_ai_region: The Vertex AI region to launch the job in
    """
    if component not in _LAUNCHABLE_COMPONENTS:
        raise ValueError(
            f"Invalid component: {component}. Expected one of: {_LAUNCHABLE_COMPONENTS}"
        )
    is_cpu_execution = _determine_if_cpu_execution(
        vertex_ai_resource_config=vertex_ai_resource_config
    )
    cpu_docker_uri = cpu_docker_uri or DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU
    cuda_docker_uri = cuda_docker_uri or DEFAULT_GIGL_RELEASE_SRC_IMAGE_CUDA
    container_uri = cpu_docker_uri if is_cpu_execution else cuda_docker_uri

    job_config = _build_job_config(
        job_name=job_name,
        task_config_uri=task_config_uri,
        resource_config_uri=resource_config_uri,
        command_str=process_command,
        args=process_runtime_args,
        use_cuda=is_cpu_execution,
        container_uri=container_uri,
        vertex_ai_resource_config=vertex_ai_resource_config,
        env_vars=[env_var.EnvVar(name="TF_CPP_MIN_LOG_LEVEL", value="3")],
        labels=resource_config_wrapper.get_resource_labels(component=component),
    )
    logger.info(f"Launching {component.value} job with config: {job_config}")

    vertex_ai_service = VertexAIService(
        project=resource_config_wrapper.project,
        location=vertex_ai_region,
        service_account=resource_config_wrapper.service_account_email,
        staging_bucket=resource_config_wrapper.temp_assets_regional_bucket_path.uri,
    )
    vertex_ai_service.launch_job(job_config=job_config)


def launch_graph_store_enabled_job(
    vertex_ai_graph_store_config: VertexAiGraphStoreConfig,
    job_name: str,
    task_config_uri: Uri,
    resource_config_uri: Uri,
    process_command: str,
    process_runtime_args: Mapping[str, str],
    resource_config_wrapper: GiglResourceConfigWrapper,
    cpu_docker_uri: Optional[str],
    cuda_docker_uri: Optional[str],
    component: GiGLComponents,
) -> None:
    """Launch a graph store enabled job on Vertex AI with separate storage and compute pools.

    Args:
        vertex_ai_graph_store_config: The Vertex AI graph store configuration
        job_name: Full name for the Vertex AI job
        task_config_uri: URI to the task configuration
        resource_config_uri: URI to the resource configuration
        process_command: Command to run in the compute container
        process_runtime_args: Runtime arguments for the process
        resource_config_wrapper: Wrapper for the resource configuration
        cpu_docker_uri: Docker image URI for CPU execution
        cuda_docker_uri: Docker image URI for GPU execution
        component: The GiGL component (Trainer or Inferencer)
    """
    if component not in _LAUNCHABLE_COMPONENTS:
        raise ValueError(
            f"Invalid component: {component}. Expected one of: {_LAUNCHABLE_COMPONENTS}"
        )
    storage_pool_config = vertex_ai_graph_store_config.graph_store_pool
    compute_pool_config = vertex_ai_graph_store_config.compute_pool

    # Determine if CPU or GPU based on compute pool
    is_cpu_execution = _determine_if_cpu_execution(
        vertex_ai_resource_config=compute_pool_config
    )
    cpu_docker_uri = cpu_docker_uri or DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU
    cuda_docker_uri = cuda_docker_uri or DEFAULT_GIGL_RELEASE_SRC_IMAGE_CUDA
    container_uri = cpu_docker_uri if is_cpu_execution else cuda_docker_uri

    logger.info(f"Running {component.value} with command: {process_command}")

    num_compute_processes = (
        vertex_ai_graph_store_config.compute_cluster_local_world_size
    )
    if not num_compute_processes:
        if is_cpu_execution:
            num_compute_processes = 1
        else:
            num_compute_processes = vertex_ai_graph_store_config.compute_pool.gpu_limit

    # Add server/client environment variables
    environment_variables: list[env_var.EnvVar] = [
        env_var.EnvVar(name="TF_CPP_MIN_LOG_LEVEL", value="3"),
        env_var.EnvVar(
            name=COMPUTE_CLUSTER_LOCAL_WORLD_SIZE_ENV_KEY,
            value=str(num_compute_processes),
        ),
    ]

    labels = resource_config_wrapper.get_resource_labels(component=component)

    # Create compute pool job config
    compute_job_config = _build_job_config(
        job_name=job_name,
        task_config_uri=task_config_uri,
        resource_config_uri=resource_config_uri,
        command_str=process_command,
        args=process_runtime_args,
        use_cuda=is_cpu_execution,
        container_uri=container_uri,
        vertex_ai_resource_config=compute_pool_config,
        env_vars=environment_variables,
        labels=labels,
    )

    # Create storage pool job config
    storage_job_config = _build_job_config(
        job_name=job_name,
        task_config_uri=task_config_uri,
        resource_config_uri=resource_config_uri,
        command_str=f"python -m gigl.distributed.graph_store.storage_main",
        args={},  # No extra args for storage pool
        use_cuda=is_cpu_execution,
        container_uri=container_uri,
        vertex_ai_resource_config=storage_pool_config,
        env_vars=environment_variables,
        labels=labels,
    )

    # Determine region from compute pool or use default region
    region = (
        compute_pool_config.gcp_region_override
        if compute_pool_config.gcp_region_override
        else resource_config_wrapper.region
    )

    vertex_ai_service = VertexAIService(
        project=resource_config_wrapper.project,
        location=region,
        service_account=resource_config_wrapper.service_account_email,
        staging_bucket=resource_config_wrapper.temp_assets_regional_bucket_path.uri,
    )
    vertex_ai_service.launch_graph_store_job(
        compute_pool_job_config=compute_job_config,
        storage_pool_job_config=storage_job_config,
    )


def _build_job_config(
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
        # gigl/src/inference/v2/glt_inferencer.py:124: error: The type "type[Strategy]" is not generic and not indexable  [misc]
        # TODO(kmonte): Fix this
        scheduling_strategy=getattr(
            Scheduling.Strategy,
            vertex_ai_resource_config.scheduling_strategy,
        )
        if vertex_ai_resource_config.scheduling_strategy
        else None,
        boot_disk_size_gb=vertex_ai_resource_config.boot_disk_size_gb
        if vertex_ai_resource_config.boot_disk_size_gb
        else 100,  # Default to 100 GB for backward compatibility
    )
    return job_config


# TODO(svij): This function may need some work cc @zfan3, @xgao4
# i.e. dataloading may happen on gpu instead of inference. Curretly, there is no
# great support for gpu data loading, thus we assume inference is done on gpu and
# data loading is done on cpu. This will need to be revisited.
def _determine_if_cpu_execution(
    vertex_ai_resource_config: VertexAiResourceConfig,
) -> bool:
    """Determine whether CPU execution is required based on the resource configuration.

    Args:
        vertex_ai_resource_config: The Vertex AI resource configuration to check

    Returns:
        True if CPU execution is required, False if GPU execution is required
    """
    if (
        not vertex_ai_resource_config.gpu_type
        or vertex_ai_resource_config.gpu_type
        == accelerator_type.AcceleratorType.ACCELERATOR_TYPE_UNSPECIFIED.name  # type: ignore[attr-defined] # `name` is defined
    ):
        return True
    else:
        return False
