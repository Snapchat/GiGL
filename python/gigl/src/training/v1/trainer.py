import argparse
from typing import Optional

import torch
from google.cloud.aiplatform_v1.types import accelerator_type

from gigl.common import Uri, UriFactory
from gigl.common.logger import Logger
from gigl.common.services.vertex_ai import VertexAiJobConfig, VertexAIService
from gigl.env.dep_constants import GIGL_SRC_IMAGE_CPU, GIGL_SRC_IMAGE_CUDA
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.utils.metrics_service_provider import initialize_metrics
from gigl.src.training.v1.lib.training_process import GnnTrainingProcess
from snapchat.research.gbml.gigl_resource_config_pb2 import (
    LocalResourceConfig,
    VertexAiResourceConfig,
)

logger = Logger()


class Trainer:
    """
    GiGL Component that trains a GNN model using the specified task and resource configurations.
    """

    def run(
        self,
        applied_task_identifier: AppliedTaskIdentifier,
        task_config_uri: Uri,
        resource_config_uri: Uri,
        cpu_docker_uri: Optional[str] = None,
        cuda_docker_uri: Optional[str] = None,
    ) -> None:
        resource_config = get_resource_config(resource_config_uri=resource_config_uri)
        trainer_config = resource_config.trainer_config

        is_cpu_training = self._determine_if_cpu_training(trainer_config)

        if isinstance(trainer_config, VertexAiResourceConfig):
            cpu_docker_uri = cpu_docker_uri or GIGL_SRC_IMAGE_CPU
            cuda_docker_uri = cuda_docker_uri or GIGL_SRC_IMAGE_CUDA
            container_uri = cpu_docker_uri if is_cpu_training else cuda_docker_uri

            job_args = [
                f"--job_name={applied_task_identifier}",
                f"--task_config_uri={task_config_uri}",
                f"--resource_config_uri={resource_config_uri}",
            ] + ([] if is_cpu_training else ["--use_cuda"])

            job_config = VertexAiJobConfig(
                job_name=applied_task_identifier,
                container_uri=container_uri,
                command=["python", "-m", "gigl.src.training.v1.lib.training_process"],
                args=job_args,
                environment_variables=[
                    {"name": "TF_CPP_MIN_LOG_LEVEL", "value": "3"},
                ],
                machine_type=trainer_config.machine_type,
                accelerator_type=trainer_config.gpu_type.upper().replace("-", "_"),
                accelerator_count=trainer_config.gpu_limit,
                replica_count=trainer_config.num_replicas,
                labels=resource_config.get_resource_labels(
                    component=GiGLComponents.Trainer
                ),
                timeout_s=trainer_config.timeout if trainer_config.timeout else None,
            )

            vertex_ai_service = VertexAIService(
                project=resource_config.project,
                location=resource_config.region,
                service_account=resource_config.service_account_email,
                staging_bucket=resource_config.temp_assets_regional_bucket_path.uri,
            )

            vertex_ai_service.launch_job(job_config=job_config)

        elif isinstance(trainer_config, LocalResourceConfig):
            training_process = GnnTrainingProcess()
            training_process.run(
                task_config_uri=task_config_uri,
                device=torch.device(
                    "cuda"
                    if not is_cpu_training and torch.cuda.is_available()
                    else "cpu"
                ),
            )
        else:
            raise ValueError(
                f"Unsupported trainer_config in resource_config: {type(trainer_config).__name__}"
            )

    def _determine_if_cpu_training(self, trainer_config) -> bool:
        """Determine whether CPU training is required based on the trainer configuration."""
        if isinstance(trainer_config, LocalResourceConfig):
            return True
        elif hasattr(trainer_config, "gpu_type") and (
            trainer_config.gpu_type
            == accelerator_type.AcceleratorType.ACCELERATOR_TYPE_UNSPECIFIED
            or trainer_config.gpu_type is None
        ):
            return True
        else:
            return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Program to generate embeddings from a GBML model"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        help="Unique identifier for the job name",
    )
    parser.add_argument(
        "--task_config_uri",
        type=str,
        help="Gbml config uri",
    )
    parser.add_argument(
        "--resource_config_uri",
        type=str,
        help="Runtime argument for resource and env specifications of each component",
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

    if not args.job_name or not args.task_config_uri or not args.resource_config_uri:
        raise RuntimeError("Missing command-line arguments")

    applied_task_identifier = AppliedTaskIdentifier(args.job_name)
    task_config_uri = UriFactory.create_uri(args.task_config_uri)
    resource_config_uri = UriFactory.create_uri(args.resource_config_uri)
    cpu_docker_uri, cuda_docker_uri = args.cpu_docker_uri, args.cuda_docker_uri

    initialize_metrics(task_config_uri=task_config_uri, service_name=args.job_name)

    trainer = Trainer()
    trainer.run(
        applied_task_identifier=applied_task_identifier,
        task_config_uri=task_config_uri,
        resource_config_uri=resource_config_uri,
        cpu_docker_uri=cpu_docker_uri,
        cuda_docker_uri=cuda_docker_uri,
    )
