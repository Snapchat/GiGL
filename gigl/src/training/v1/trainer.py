import argparse
from typing import Optional

import torch
from google.cloud.aiplatform_v1.types import accelerator_type

from gigl.common import Uri, UriFactory
from gigl.common.constants import (
    DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU,
    DEFAULT_GIGL_RELEASE_SRC_IMAGE_CUDA,
)
from gigl.common.logger import Logger
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.utils.metrics_service_provider import initialize_metrics
from gigl.src.common.vertex_ai_launcher import launch_single_pool_job
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
            gbml_config_pb_wrapper = (
                GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
                    gbml_config_uri=task_config_uri
                )
            )
            raw_tensorboard_logs_uri = (
                gbml_config_pb_wrapper.shared_config.trained_model_metadata.tensorboard_logs_uri
            )
            tensorboard_logs_uri = (
                UriFactory.create_uri(raw_tensorboard_logs_uri)
                if gbml_config_pb_wrapper.trainer_config.should_log_to_tensorboard
                and raw_tensorboard_logs_uri
                else None
            )
            launch_single_pool_job(
                vertex_ai_resource_config=trainer_config,
                job_name=str(applied_task_identifier),
                task_config_uri=task_config_uri,
                resource_config_uri=resource_config_uri,
                process_command="python -m gigl.src.training.v1.lib.training_process",
                process_runtime_args={},
                resource_config_wrapper=resource_config,
                cpu_docker_uri=cpu_docker_uri or DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU,
                cuda_docker_uri=cuda_docker_uri or DEFAULT_GIGL_RELEASE_SRC_IMAGE_CUDA,
                component=GiGLComponents.Trainer,
                vertex_ai_region=resource_config.vertex_ai_trainer_region,
                tensorboard_logs_uri=tensorboard_logs_uri,
            )

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
