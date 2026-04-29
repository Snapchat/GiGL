"""Shared functionality for launching Vertex AI jobs for training and inference."""

from collections.abc import Mapping
from typing import Final, Optional

from google.cloud.aiplatform_v1.types import (
    ReservationAffinity,
    Scheduling,
    accelerator_type,
    env_var,
)

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
    VertexAiReservationAffinity,
    VertexAiResourceConfig,
)

logger = Logger()

_LAUNCHABLE_COMPONENTS: frozenset[GiGLComponents] = frozenset(
    [GiGLComponents.Trainer, GiGLComponents.Inferencer]
)

_RESERVATION_AFFINITY_KEY: Final[str] = "compute.googleapis.com/reservation-name"
_VALID_RESERVATION_AFFINITY_TYPES: Final[frozenset[str]] = frozenset(
    {"NO_RESERVATION", "ANY_RESERVATION", "SPECIFIC_RESERVATION"}
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
    tensorboard_logs_uri: Optional[Uri] = None,
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
        tensorboard_logs_uri: Optional TensorBoard log URI for trainer jobs
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
        use_cuda=not is_cpu_execution,
        container_uri=container_uri,
        vertex_ai_resource_config=vertex_ai_resource_config,
        env_vars=[env_var.EnvVar(name="TF_CPP_MIN_LOG_LEVEL", value="3")],
        labels=resource_config_wrapper.get_resource_labels(component=component),
        tensorboard_logs_uri=tensorboard_logs_uri,
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
    compute_commmand: str,
    compute_runtime_args: Mapping[str, str],
    storage_command: str,
    storage_args: Mapping[str, str],
    resource_config_wrapper: GiglResourceConfigWrapper,
    cpu_docker_uri: Optional[str],
    cuda_docker_uri: Optional[str],
    component: GiGLComponents,
    tensorboard_logs_uri: Optional[Uri] = None,
) -> None:
    """Launch a graph store enabled job on Vertex AI with separate storage and compute pools.

    Args:
        vertex_ai_graph_store_config: The Vertex AI graph store configuration
        job_name: Full name for the Vertex AI job
        task_config_uri: URI to the task configuration
        resource_config_uri: URI to the resource configuration
        compute_commmand: Command to run in the compute container
        compute_runtime_args: Runtime arguments for the compute process
        storage_command: Command to run in the storage container
        storage_args: Arguments to pass to the storage command
        resource_config_wrapper: Wrapper for the resource configuration
        cpu_docker_uri: Docker image URI for CPU execution
        cuda_docker_uri: Docker image URI for GPU execution
        component: The GiGL component (Trainer or Inferencer)
        tensorboard_logs_uri: Optional TensorBoard log URI for trainer jobs
    """
    if component not in _LAUNCHABLE_COMPONENTS:
        raise ValueError(
            f"Invalid component: {component}. Expected one of: {_LAUNCHABLE_COMPONENTS}"
        )
    storage_pool_config = vertex_ai_graph_store_config.graph_store_pool
    compute_pool_config = vertex_ai_graph_store_config.compute_pool

    # Compute workers may use GPUs, but storage workers always run the CPU
    # graph-store entrypoint.
    is_compute_cpu_execution = _determine_if_cpu_execution(
        vertex_ai_resource_config=compute_pool_config
    )
    cpu_docker_uri = cpu_docker_uri or DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU
    cuda_docker_uri = cuda_docker_uri or DEFAULT_GIGL_RELEASE_SRC_IMAGE_CUDA
    compute_container_uri = (
        cpu_docker_uri if is_compute_cpu_execution else cuda_docker_uri
    )

    logger.info(f"Running {component.value} with command: {compute_commmand}")

    num_compute_processes = (
        vertex_ai_graph_store_config.compute_cluster_local_world_size
    )
    if not num_compute_processes:
        if is_compute_cpu_execution:
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
        command_str=compute_commmand,
        args=compute_runtime_args,
        use_cuda=not is_compute_cpu_execution,
        container_uri=compute_container_uri,
        vertex_ai_resource_config=compute_pool_config,
        env_vars=environment_variables,
        labels=labels,
        tensorboard_logs_uri=tensorboard_logs_uri,
    )

    # Create storage pool job config
    storage_job_config = _build_job_config(
        job_name=job_name,
        task_config_uri=task_config_uri,
        resource_config_uri=resource_config_uri,
        command_str=storage_command,
        args=storage_args,
        use_cuda=False,
        container_uri=cpu_docker_uri,
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
    tensorboard_logs_uri: Optional[Uri] = None,
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
        tensorboard_logs_uri (Optional[Uri]): TensorBoard log URI for trainer jobs.

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
    base_output_dir = (
        _get_base_output_dir_from_tensorboard_logs_uri(
            tensorboard_logs_uri=tensorboard_logs_uri
        )
        if tensorboard_logs_uri is not None
        else None
    )

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
        boot_disk_size_gb=vertex_ai_resource_config.boot_disk_size_gb or 100,
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
        reservation_affinity=_build_reservation_affinity(
            vertex_ai_resource_config.reservation_affinity
        ),
        base_output_dir=base_output_dir,
        tensorboard_resource_name=(
            vertex_ai_resource_config.tensorboard_resource_name or None
            if base_output_dir is not None
            else None
        ),
    )
    return job_config


def _get_base_output_dir_from_tensorboard_logs_uri(
    tensorboard_logs_uri: Uri,
) -> str:
    """Return the CustomJob base output directory for a TensorBoard log URI.

    Args:
        tensorboard_logs_uri: GiGL TensorBoard log URI. This is expected to
            point at the ``logs/`` directory underneath the trainer asset dir.

    Returns:
        The parent directory to use as ``base_output_dir``.

    Raises:
        ValueError: If the URI does not contain a parent directory.
    """
    normalized_tensorboard_logs_uri = tensorboard_logs_uri.uri.rstrip("/")
    base_output_dir, separator, _ = normalized_tensorboard_logs_uri.rpartition("/")
    if not separator or not base_output_dir:
        raise ValueError(
            f"TensorBoard logs URI must include a parent directory, got {tensorboard_logs_uri.uri!r}."
        )
    return base_output_dir


def _build_reservation_affinity(
    affinity: VertexAiReservationAffinity,
) -> Optional[ReservationAffinity]:
    """Translate a proto VertexAiReservationAffinity to the SDK ReservationAffinity.

    Returns ``None`` when the proto is fully unset (empty ``type`` with no
    ``reservation_resource_names``) so the resulting ``MachineSpec`` is left
    at the Vertex AI default.

    Args:
        affinity: The proto reservation affinity (always non-None because
            proto3 singular message fields default to an empty instance).

    Returns:
        A ``ReservationAffinity`` message, or ``None`` when ``affinity`` is
        unset.

    Raises:
        ValueError: If ``affinity.type`` is empty but
            ``reservation_resource_names`` is set; if ``affinity.type`` is
            not one of the accepted values; if ``SPECIFIC_RESERVATION`` is
            requested without names; or if names are provided for
            ``NO_RESERVATION`` / ``ANY_RESERVATION``.
    """
    names = list(affinity.reservation_resource_names)
    if not affinity.type:
        if names:
            raise ValueError(
                "reservation_resource_names is set but reservation_affinity.type "
                "is empty. Set type to 'SPECIFIC_RESERVATION' or clear "
                "reservation_resource_names."
            )
        return None

    if affinity.type not in _VALID_RESERVATION_AFFINITY_TYPES:
        raise ValueError(
            f"Invalid reservation_affinity.type {affinity.type!r}. "
            f"Expected one of: {sorted(_VALID_RESERVATION_AFFINITY_TYPES)}."
        )
    # Mirrors the scheduling_strategy pattern above: mypy does not treat
    # ReservationAffinity.Type as subscriptable, so use getattr instead of
    # ReservationAffinity.Type[affinity.type].
    affinity_type = getattr(ReservationAffinity.Type, affinity.type)

    if affinity_type == ReservationAffinity.Type.SPECIFIC_RESERVATION:
        if not names:
            raise ValueError(
                "SPECIFIC_RESERVATION requires at least one entry in "
                "reservation_resource_names "
                "(format: projects/{p}/zones/{z}/reservations/{n})."
            )
        return ReservationAffinity(
            reservation_affinity_type=affinity_type,
            key=_RESERVATION_AFFINITY_KEY,
            values=names,
        )
    if names:
        raise ValueError(
            f"reservation_resource_names must be empty for type {affinity.type}."
        )
    return ReservationAffinity(reservation_affinity_type=affinity_type)


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
