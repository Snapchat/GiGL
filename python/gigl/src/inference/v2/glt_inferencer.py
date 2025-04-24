import argparse
from typing import Optional

from google.cloud.aiplatform_v1.types import accelerator_type

from gigl.common import Uri, UriFactory
from gigl.common.logger import Logger
from gigl.common.services.vertex_ai import VertexAiJobConfig, VertexAIService
from gigl.env.dep_constants import GIGL_SRC_IMAGE_CPU, GIGL_SRC_IMAGE_CUDA
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


# TODO: (svij) This function may need some work cc @zfan3, @xgao4
# i.e. dataloading may happen on gpu instead of inference. Curretly, there is no
# great support for gpu data loading, thus we assume inference is done on gpu and
# data loading is done on cpu. This will need to be revisited.
def _determine_if_cpu_inference(
    inferencer_resource_config: VertexAiResourceConfig,
) -> bool:
    """Determine whether CPU inference is required based on the glt_inferencer configuration."""
    if (
        not inferencer_resource_config.gpu_type
        or inferencer_resource_config.gpu_type
        == accelerator_type.AcceleratorType.ACCELERATOR_TYPE_UNSPECIFIED.name  # type: ignore[attr-defined] # `name` is defined
    ):
        return True
    else:
        return False


class GLTInferencer:
    """
    GiGL Component that runs a GLT Inference using a provided class path
    """

    def __execute_VAI_inference(
        self,
        applied_task_identifier: AppliedTaskIdentifier,
        task_config_uri: Uri,
        resource_config_uri: Uri,
        cpu_docker_uri: Optional[str] = None,
        cuda_docker_uri: Optional[str] = None,
    ) -> None:
        resource_config_wrapper: GiglResourceConfigWrapper = get_resource_config(
            resource_config_uri=resource_config_uri
        )
        gbml_config_pb_wrapper = (
            GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
                gbml_config_uri=task_config_uri
            )
        )
        inference_process_command = gbml_config_pb_wrapper.inferencer_config.command
        if not inference_process_command:
            raise ValueError(
                "Currently, GLT Inferencer only supports inferencer process command which"
                + f" was not provided in inferencer config: {gbml_config_pb_wrapper.inferencer_config}"
            )
        inference_process_runtime_args = (
            gbml_config_pb_wrapper.inferencer_config.inferencer_args
        )
        assert isinstance(
            resource_config_wrapper.inferencer_config, VertexAiResourceConfig
        )
        inferencer_resource_config: VertexAiResourceConfig = (
            resource_config_wrapper.inferencer_config
        )

        is_cpu_training = _determine_if_cpu_inference(
            inferencer_resource_config=inferencer_resource_config
        )
        cpu_docker_uri = cpu_docker_uri or GIGL_SRC_IMAGE_CPU
        cuda_docker_uri = cuda_docker_uri or GIGL_SRC_IMAGE_CUDA
        container_uri = cpu_docker_uri if is_cpu_training else cuda_docker_uri

        job_args = (
            [
                f"--job_name={applied_task_identifier}",
                f"--task_config_uri={task_config_uri}",
                f"--resource_config_uri={resource_config_uri}",
            ]
            + ([] if is_cpu_training else ["--use_cuda"])
            + ([f"--{k}={v}" for k, v in inference_process_runtime_args.items()])
        )

        command = inference_process_command.strip().split(" ")
        logger.info(f"Running inference with command: {command}")
        vai_job_name = f"gigl_infer_{applied_task_identifier}"
        job_config = VertexAiJobConfig(
            job_name=vai_job_name,
            container_uri=container_uri,
            command=command,
            args=job_args,
            environment_variables=[
                {"name": "TF_CPP_MIN_LOG_LEVEL", "value": "3"},
            ],
            machine_type=inferencer_resource_config.machine_type,
            accelerator_type=inferencer_resource_config.gpu_type.upper().replace(
                "-", "_"
            ),
            accelerator_count=inferencer_resource_config.gpu_limit,
            replica_count=inferencer_resource_config.num_replicas,
            labels=resource_config_wrapper.get_resource_labels(
                component=GiGLComponents.Inferencer
            ),
            timeout_s=inferencer_resource_config.timeout
            if inferencer_resource_config.timeout
            else None,
        )
        vertex_ai_service = VertexAIService(
            project=resource_config_wrapper.project,
            location=resource_config_wrapper.region,
            service_account=resource_config_wrapper.service_account_email,
            staging_bucket=resource_config_wrapper.temp_assets_regional_bucket_path.uri,
        )
        vertex_ai_service.launch_job(job_config=job_config)

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

        if isinstance(resource_config_wrapper.inferencer_config, LocalResourceConfig):
            raise NotImplementedError(
                f"Local GLT Inferencer is not yet supported, please specify a {VertexAiResourceConfig.__name__} resource config field."
            )
        elif isinstance(
            resource_config_wrapper.inferencer_config, VertexAiResourceConfig
        ):
            self.__execute_VAI_inference(
                applied_task_identifier=applied_task_identifier,
                task_config_uri=task_config_uri,
                resource_config_uri=resource_config_uri,
                cpu_docker_uri=cpu_docker_uri,
                cuda_docker_uri=cuda_docker_uri,
            )
        else:
            raise NotImplementedError(
                f"Unsupported resource config for glt inference: {type(resource_config_wrapper.inferencer_config).__name__}"
            )


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

    glt_inferencer = GLTInferencer()
    glt_inferencer.run(
        applied_task_identifier=applied_task_identifier,
        task_config_uri=task_config_uri,
        resource_config_uri=resource_config_uri,
        cpu_docker_uri=cpu_docker_uri,
        cuda_docker_uri=cuda_docker_uri,
    )
