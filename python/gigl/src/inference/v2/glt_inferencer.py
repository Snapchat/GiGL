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
from gigl.common.services.vertex_ai import VertexAIService
from gigl.env.distributed import (
    COMPUTE_CLUSTER_LOCAL_WORLD_SIZE_ENV_KEY,
    GRAPH_STORE_MAIN_FQDN,
)
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.translators.vertex_ai_job_translator import build_job_config
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

    def _launch_single_pool(
        self,
        vertex_ai_resource_config: VertexAiResourceConfig,
        applied_task_identifier: AppliedTaskIdentifier,
        task_config_uri: Uri,
        resource_config_uri: Uri,
        inference_process_command: str,
        inference_process_runtime_args: Mapping[str, str],
        resource_config_wrapper: GiglResourceConfigWrapper,
        cpu_docker_uri: Optional[str],
        cuda_docker_uri: Optional[str],
    ) -> None:
        """Launch a single pool inference job on Vertex AI."""
        is_cpu_inference = _determine_if_cpu_inference(
            inferencer_resource_config=vertex_ai_resource_config
        )
        cpu_docker_uri = cpu_docker_uri or DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU
        cuda_docker_uri = cuda_docker_uri or DEFAULT_GIGL_RELEASE_SRC_IMAGE_CUDA
        container_uri = cpu_docker_uri if is_cpu_inference else cuda_docker_uri

        job_config = build_job_config(
            job_name=str(applied_task_identifier),
            is_inference=True,
            task_config_uri=task_config_uri,
            resource_config_uri=resource_config_uri,
            command_str=inference_process_command,
            args=inference_process_runtime_args,
            use_cuda=is_cpu_inference,
            container_uri=container_uri,
            vertex_ai_resource_config=vertex_ai_resource_config,
            env_vars=[env_var.EnvVar(name="TF_CPP_MIN_LOG_LEVEL", value="3")],
            labels=resource_config_wrapper.get_resource_labels(
                component=GiGLComponents.Inferencer
            ),
        )
        logger.info(f"Launching inference job with config: {job_config}")

        vertex_ai_service = VertexAIService(
            project=resource_config_wrapper.project,
            location=resource_config_wrapper.vertex_ai_inferencer_region,
            service_account=resource_config_wrapper.service_account_email,
            staging_bucket=resource_config_wrapper.temp_assets_regional_bucket_path.uri,
        )
        vertex_ai_service.launch_job(job_config=job_config)

    def _launch_server_client(
        self,
        vertex_ai_graph_store_config: VertexAiGraphStoreConfig,
        applied_task_identifier: AppliedTaskIdentifier,
        task_config_uri: Uri,
        resource_config_uri: Uri,
        inference_process_command: str,
        inference_process_runtime_args: Mapping[str, str],
        resource_config_wrapper: GiglResourceConfigWrapper,
        cpu_docker_uri: Optional[str],
        cuda_docker_uri: Optional[str],
    ) -> None:
        """Launch a server/client inference job on Vertex AI using graph store config."""
        storage_pool_config = vertex_ai_graph_store_config.graph_store_pool
        compute_pool_config = vertex_ai_graph_store_config.compute_pool

        # Determine if CPU or GPU based on compute pool
        is_cpu_inference = _determine_if_cpu_inference(
            inferencer_resource_config=compute_pool_config
        )
        cpu_docker_uri = cpu_docker_uri or DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU
        cuda_docker_uri = cuda_docker_uri or DEFAULT_GIGL_RELEASE_SRC_IMAGE_CUDA
        container_uri = cpu_docker_uri if is_cpu_inference else cuda_docker_uri

        logger.info(f"Running inference with command: {inference_process_command}")

        num_compute_processes = (
            vertex_ai_graph_store_config.compute_cluster_local_world_size
        )
        if not num_compute_processes:
            if is_cpu_inference:
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

        labels = resource_config_wrapper.get_resource_labels(
            component=GiGLComponents.Inferencer
        )
        # Create compute pool job config
        compute_job_config = build_job_config(
            job_name=str(applied_task_identifier),
            is_inference=True,
            task_config_uri=task_config_uri,
            resource_config_uri=resource_config_uri,
            command_str=inference_process_command,
            args=inference_process_runtime_args,
            use_cuda=is_cpu_inference,
            container_uri=container_uri,
            vertex_ai_resource_config=compute_pool_config,
            env_vars=environment_variables,
            labels=labels,
        )

        # Create storage pool job config
        storage_job_config = build_job_config(
            job_name=str(applied_task_identifier),
            is_inference=True,
            task_config_uri=task_config_uri,
            resource_config_uri=resource_config_uri,
            command_str=f"python -m {GRAPH_STORE_MAIN_FQDN}",
            args={},  # No extra args for storage pool
            use_cuda=is_cpu_inference,
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

        if isinstance(
            resource_config_wrapper.inferencer_config, VertexAiResourceConfig
        ):
            self._launch_single_pool(
                vertex_ai_resource_config=resource_config_wrapper.inferencer_config,
                applied_task_identifier=applied_task_identifier,
                task_config_uri=task_config_uri,
                resource_config_uri=resource_config_uri,
                inference_process_command=inference_process_command,
                inference_process_runtime_args=inference_process_runtime_args,
                resource_config_wrapper=resource_config_wrapper,
                cpu_docker_uri=cpu_docker_uri,
                cuda_docker_uri=cuda_docker_uri,
            )
        elif isinstance(
            resource_config_wrapper.inferencer_config, VertexAiGraphStoreConfig
        ):
            self._launch_server_client(
                vertex_ai_graph_store_config=resource_config_wrapper.inferencer_config,
                applied_task_identifier=applied_task_identifier,
                task_config_uri=task_config_uri,
                resource_config_uri=resource_config_uri,
                inference_process_command=inference_process_command,
                inference_process_runtime_args=inference_process_runtime_args,
                resource_config_wrapper=resource_config_wrapper,
                cpu_docker_uri=cpu_docker_uri,
                cuda_docker_uri=cuda_docker_uri,
            )
        else:
            raise NotImplementedError(
                f"Unsupported resource config for glt inference: {type(resource_config_wrapper.inferencer_config).__name__}"
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

        if isinstance(resource_config_wrapper.inferencer_config, LocalResourceConfig):
            raise NotImplementedError(
                f"Local GLT Inferencer is not yet supported, please specify a {VertexAiResourceConfig.__name__} or {VertexAiGraphStoreConfig.__name__} resource config field."
            )
        elif isinstance(
            resource_config_wrapper.inferencer_config, VertexAiResourceConfig
        ) or isinstance(
            resource_config_wrapper.inferencer_config, VertexAiGraphStoreConfig
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
