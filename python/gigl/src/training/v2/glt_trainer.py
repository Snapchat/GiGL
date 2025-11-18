import argparse
from collections.abc import Mapping
from typing import Optional

from google.cloud.aiplatform_v1.types import accelerator_type, env_var

from gigl.common import Uri, UriFactory
from gigl.common.constants import (
    DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU,
    DEFAULT_GIGL_RELEASE_SRC_IMAGE_CUDA,
)
from gigl.common.logger import Logger
from gigl.common.services.vertex_ai import (
    VertexAIService,
    get_job_config_from_vertex_ai_resource_config,
)
from gigl.env.distributed import COMPUTE_CLUSTER_LOCAL_WORLD_SIZE_ENV_KEY
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.types.pb_wrappers.gigl_resource_config import (
    GiglResourceConfigWrapper,
)
from gigl.src.common.utils.metrics_service_provider import initialize_metrics
from snapchat.research.gbml.gigl_resource_config_pb2 import (
    LocalResourceConfig,
    VertexAiGraphStoreConfig,
    VertexAiResourceConfig,
)

logger = Logger()

# TODO: (svij) We should parameterize this in the future
DEFAULT_VERTEX_AI_TIMEOUT_S = 60 * 60 * 3  # 3 hours


# TODO: (svij) This function may need some work cc @zfan3, @xgao4
# i.e. dataloading may happen on gpu instead of inference. Curretly, there is no
# great support for gpu data loading, thus we assume inference is done on gpu and
# data loading is done on cpu. This will need to be revisited.
def _determine_if_cpu_training(
    trainer_resource_config: VertexAiResourceConfig,
) -> bool:
    """Determine whether CPU training is required based on the glt_training configuration."""
    if (
        not trainer_resource_config.gpu_type
        or trainer_resource_config.gpu_type
        == accelerator_type.AcceleratorType.ACCELERATOR_TYPE_UNSPECIFIED.name  # type: ignore[attr-defined] # `name` is defined
    ):
        return True
    else:
        return False


class GLTTrainer:
    """
    GiGL Component that runs a GLT Training using a provided class path
    """

    def _launch_single_pool_training(
        self,
        vertex_ai_resource_config: VertexAiResourceConfig,
        applied_task_identifier: AppliedTaskIdentifier,
        task_config_uri: Uri,
        resource_config_uri: Uri,
        training_process_command: str,
        training_process_runtime_args: Mapping[str, str],
        resource_config: GiglResourceConfigWrapper,
        cpu_docker_uri: Optional[str],
        cuda_docker_uri: Optional[str],
    ) -> None:
        is_cpu_training = _determine_if_cpu_training(
            trainer_resource_config=vertex_ai_resource_config
        )
        cpu_docker_uri = cpu_docker_uri or DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU
        cuda_docker_uri = cuda_docker_uri or DEFAULT_GIGL_RELEASE_SRC_IMAGE_CUDA
        container_uri = cpu_docker_uri if is_cpu_training else cuda_docker_uri

        job_config = get_job_config_from_vertex_ai_resource_config(
            applied_task_identifier=applied_task_identifier,
            is_inference=False,
            task_config_uri=task_config_uri,
            resource_config_uri=resource_config_uri,
            command_str=training_process_command,
            args=training_process_runtime_args,
            run_on_cpu=is_cpu_training,
            container_uri=container_uri,
            vertex_ai_resource_config=vertex_ai_resource_config,
            env_vars=[env_var.EnvVar(name="TF_CPP_MIN_LOG_LEVEL", value="3")],
        )
        logger.info(f"Launching training job with config: {job_config}")

        vertex_ai_service = VertexAIService(
            project=resource_config.project,
            location=resource_config.vertex_ai_trainer_region,
            service_account=resource_config.service_account_email,
            staging_bucket=resource_config.temp_assets_regional_bucket_path.uri,
        )
        vertex_ai_service.launch_job(job_config=job_config)

    def _launch_server_client_training(
        self,
        vertex_ai_graph_store_config: VertexAiGraphStoreConfig,
        applied_task_identifier: AppliedTaskIdentifier,
        task_config_uri: Uri,
        resource_config_uri: Uri,
        training_process_command: str,
        training_process_runtime_args: Mapping[str, str],
        resource_config_wrapper: GiglResourceConfigWrapper,
        cpu_docker_uri: Optional[str],
        cuda_docker_uri: Optional[str],
    ) -> None:
        """Launch a server/client training job on Vertex AI using graph store config."""
        storage_pool_config = vertex_ai_graph_store_config.graph_store_pool
        compute_pool_config = vertex_ai_graph_store_config.compute_pool

        # Determine if CPU or GPU based on compute pool
        is_cpu_training = _determine_if_cpu_training(
            trainer_resource_config=compute_pool_config
        )
        cpu_docker_uri = cpu_docker_uri or DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU
        cuda_docker_uri = cuda_docker_uri or DEFAULT_GIGL_RELEASE_SRC_IMAGE_CUDA
        container_uri = cpu_docker_uri if is_cpu_training else cuda_docker_uri

        logger.info(f"Running training with command: {training_process_command}")

        num_compute_processes = (
            vertex_ai_graph_store_config.compute_cluster_local_world_size
        )
        if not num_compute_processes:
            if is_cpu_training:
                num_compute_processes = 1
            else:
                num_compute_processes = (
                    vertex_ai_graph_store_config.compute_pool.gpu_limit
                )
        # Add server/client environment variables
        environment_variables: list[env_var.EnvVar] = [
            env_var.EnvVar(name="TF_CPP_MIN_LOG_LEVEL", value="3"),
            env_var.EnvVar(
                name=COMPUTE_CLUSTER_LOCAL_WORLD_SIZE_ENV_KEY,
                value=str(num_compute_processes),
            ),
        ]

        # Create compute pool job config
        compute_job_config = get_job_config_from_vertex_ai_resource_config(
            applied_task_identifier=applied_task_identifier,
            is_inference=False,
            task_config_uri=task_config_uri,
            resource_config_uri=resource_config_uri,
            command_str=training_process_command,
            args=training_process_runtime_args,
            run_on_cpu=is_cpu_training,
            container_uri=container_uri,
            vertex_ai_resource_config=compute_pool_config,
            env_vars=environment_variables,
        )

        # Create storage pool job config
        storage_job_config = get_job_config_from_vertex_ai_resource_config(
            applied_task_identifier=applied_task_identifier,
            is_inference=False,
            task_config_uri=task_config_uri,
            resource_config_uri=resource_config_uri,
            command_str="python -m gigl.distributed.server_client.server_main",
            args={},  # No extra args for storage pool
            run_on_cpu=is_cpu_training,
            container_uri=container_uri,
            vertex_ai_resource_config=storage_pool_config,
            env_vars=environment_variables,
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

    def __execute_VAI_training(
        self,
        applied_task_identifier: AppliedTaskIdentifier,
        task_config_uri: Uri,
        resource_config_uri: Uri,
        cpu_docker_uri: Optional[str] = None,
        cuda_docker_uri: Optional[str] = None,
    ) -> None:
        resource_config: GiglResourceConfigWrapper = get_resource_config(
            resource_config_uri=resource_config_uri
        )
        gbml_config_pb_wrapper = (
            GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
                gbml_config_uri=task_config_uri
            )
        )
        training_process_command = gbml_config_pb_wrapper.trainer_config.command
        if not training_process_command:
            raise ValueError(
                "Currently, GLT Trainer only supports training process command which"
                + f" was not provided in trainer config: {gbml_config_pb_wrapper.trainer_config}"
            )
        training_process_runtime_args = (
            gbml_config_pb_wrapper.trainer_config.trainer_args
        )

        if isinstance(resource_config.trainer_config, VertexAiResourceConfig):
            self._launch_single_pool_training(
                vertex_ai_resource_config=resource_config.trainer_config,
                applied_task_identifier=applied_task_identifier,
                task_config_uri=task_config_uri,
                resource_config_uri=resource_config_uri,
                training_process_command=training_process_command,
                training_process_runtime_args=training_process_runtime_args,
                resource_config=resource_config,
                cpu_docker_uri=cpu_docker_uri,
                cuda_docker_uri=cuda_docker_uri,
            )
        elif isinstance(resource_config.trainer_config, VertexAiGraphStoreConfig):
            self._launch_server_client_training(
                vertex_ai_graph_store_config=resource_config.trainer_config,
                applied_task_identifier=applied_task_identifier,
                task_config_uri=task_config_uri,
                resource_config_uri=resource_config_uri,
                training_process_command=training_process_command,
                training_process_runtime_args=training_process_runtime_args,
                resource_config_wrapper=resource_config,
                cpu_docker_uri=cpu_docker_uri,
                cuda_docker_uri=cuda_docker_uri,
            )
        else:
            raise NotImplementedError(
                f"Unsupported resource config for glt training: {type(resource_config.trainer_config).__name__}"
            )

    def run(
        self,
        applied_task_identifier: AppliedTaskIdentifier,
        task_config_uri: Uri,
        resource_config_uri: Uri,
        cpu_docker_uri: Optional[str] = None,
        cuda_docker_uri: Optional[str] = None,
    ) -> None:
        # TODO: Support local inference run i.e. non vertex AI
        resource_config_wrapper: GiglResourceConfigWrapper = get_resource_config(
            resource_config_uri=resource_config_uri
        )
        trainer_config = resource_config_wrapper.trainer_config

        if isinstance(trainer_config, LocalResourceConfig):
            # TODO: (svij) Implement local training
            raise NotImplementedError(
                f"Local GLT Trainer is not yet supported, please specify a {VertexAiResourceConfig.__name__} or {VertexAiGraphStoreConfig.__name__} resource config field."
            )
        elif isinstance(trainer_config, VertexAiResourceConfig) or isinstance(
            trainer_config, VertexAiGraphStoreConfig
        ):
            self.__execute_VAI_training(
                applied_task_identifier=applied_task_identifier,
                task_config_uri=task_config_uri,
                resource_config_uri=resource_config_uri,
                cpu_docker_uri=cpu_docker_uri,
                cuda_docker_uri=cuda_docker_uri,
            )
        else:
            raise NotImplementedError(
                f"Unsupported resource config for glt training: {type(trainer_config).__name__}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Program to generate embeddings from a GBML model"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        help="Unique identifier for the job name",
        required=True,
    )
    parser.add_argument(
        "--task_config_uri",
        type=str,
        help="A URI pointing to a GbmlConfig proto serialized as YAML",
        required=True,
    )
    parser.add_argument(
        "--resource_config_uri",
        type=str,
        help="A URI pointing to a GiGLResourceConfig proto serialized as YAML",
        required=True,
    )
    parser.add_argument(
        "--cpu_docker_uri",
        type=str,
        help="User Specified or KFP compiled Docker Image for CPU training",
        required=False,
    )
    parser.add_argument(
        "--cuda_docker_uri",
        type=str,
        help="User Specified or KFP compiled Docker Image for GPU training",
        required=False,
    )

    args = parser.parse_args()

    applied_task_identifier = AppliedTaskIdentifier(args.job_name)
    task_config_uri = UriFactory.create_uri(args.task_config_uri)
    resource_config_uri = UriFactory.create_uri(args.resource_config_uri)
    cpu_docker_uri, cuda_docker_uri = args.cpu_docker_uri, args.cuda_docker_uri

    initialize_metrics(task_config_uri=task_config_uri, service_name=args.job_name)

    glt_trainer = GLTTrainer()
    glt_trainer.run(
        applied_task_identifier=applied_task_identifier,
        task_config_uri=task_config_uri,
        resource_config_uri=resource_config_uri,
        cpu_docker_uri=cpu_docker_uri,
        cuda_docker_uri=cuda_docker_uri,
    )
