import argparse
from collections.abc import Mapping
from typing import Optional

from google.cloud.aiplatform_v1.types import Scheduling, accelerator_type, env_var

from gigl.common import Uri, UriFactory
from gigl.common.constants import (
    DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU,
    DEFAULT_GIGL_RELEASE_SRC_IMAGE_CUDA,
)
from gigl.common.logger import Logger
from gigl.common.services.vertex_ai import VertexAiJobConfig, VertexAIService
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.types.pb_wrappers.gigl_resource_config import (
    GiglResourceConfigWrapper,
)
from gigl.src.common.utils.metrics_service_provider import initialize_metrics
from snapchat.research.gbml.gigl_resource_config_pb2 import (
    LocalResourceConfig,
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

        job_args = (
            [
                f"--job_name={applied_task_identifier}",
                f"--task_config_uri={task_config_uri}",
                f"--resource_config_uri={resource_config_uri}",
            ]
            + ([] if is_cpu_training else ["--use_cuda"])
            + ([f"--{k}={v}" for k, v in training_process_runtime_args.items()])
        )

        command = training_process_command.strip().split(" ")
        logger.info(f"Running trainer with command: {command}")
        vai_job_name = f"gigl_train_{applied_task_identifier}"
        environment_variables: list[env_var.EnvVar] = [
            env_var.EnvVar(name="TF_CPP_MIN_LOG_LEVEL", value="3"),
        ]
        job_config = VertexAiJobConfig(
            job_name=vai_job_name,
            container_uri=container_uri,
            command=command,
            args=job_args,
            environment_variables=environment_variables,
            machine_type=vertex_ai_resource_config.machine_type,
            accelerator_type=vertex_ai_resource_config.gpu_type.upper().replace(
                "-", "_"
            ),
            accelerator_count=vertex_ai_resource_config.gpu_limit,
            replica_count=vertex_ai_resource_config.num_replicas,
            labels=resource_config.get_resource_labels(
                component=GiGLComponents.Trainer
            ),
            timeout_s=vertex_ai_resource_config.timeout
            if vertex_ai_resource_config.timeout
            else None,
            # This should be `aiplatform.gapic.Scheduling.Strategy[trainer_resource_config.scheduling_strategy]`
            # But mypy complains otherwise...
            # python/gigl/src/training/v2/glt_trainer.py:123: error: The type "type[Strategy]" is not generic and not indexable  [misc]
            # TODO(kmonte): Fix this
            scheduling_strategy=getattr(
                Scheduling.Strategy,
                trainer_resource_config.scheduling_strategy,
            )
            if trainer_resource_config.scheduling_strategy
            else None,
        )

        vertex_ai_service = VertexAIService(
            project=resource_config.project,
            location=resource_config.vertex_ai_trainer_region,
            service_account=resource_config.service_account_email,
            staging_bucket=resource_config.temp_assets_regional_bucket_path.uri,
        )
        vertex_ai_service.launch_job(job_config=job_config)

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
                f"Local GLT Inferencer is not yet supported, please specify a {VertexAiResourceConfig.__name__} resource config field."
            )
        elif isinstance(trainer_config, VertexAiResourceConfig):
            self.__execute_VAI_training(
                applied_task_identifier=applied_task_identifier,
                task_config_uri=task_config_uri,
                resource_config_uri=resource_config_uri,
                cpu_docker_uri=cpu_docker_uri,
                cuda_docker_uri=cuda_docker_uri,
            )
        else:
            raise NotImplementedError(
                f"Unsupported resource config for glt inference: {type(trainer_config).__name__}"
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
