import argparse
from typing import Optional

from gigl.common import Uri, UriFactory
from gigl.common.logger import Logger
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.types.pb_wrappers.gigl_resource_config import (
    GiglResourceConfigWrapper,
)
from gigl.src.common.utils.metrics_service_provider import initialize_metrics
from gigl.src.common.vertex_ai_launcher import (
    launch_graph_store_enabled_job,
    launch_single_pool_job,
)
from snapchat.research.gbml.gigl_resource_config_pb2 import (
    LocalResourceConfig,
    VertexAiGraphStoreConfig,
    VertexAiResourceConfig,
)

logger = Logger()


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

        job_name = f"gigl_infer_{applied_task_identifier}"

        if isinstance(
            resource_config_wrapper.inferencer_config, VertexAiResourceConfig
        ):
            launch_single_pool_job(
                vertex_ai_resource_config=resource_config_wrapper.inferencer_config,
                job_name=job_name,
                task_config_uri=task_config_uri,
                resource_config_uri=resource_config_uri,
                process_command=inference_process_command,
                process_runtime_args=inference_process_runtime_args,
                resource_config_wrapper=resource_config_wrapper,
                cpu_docker_uri=cpu_docker_uri,
                cuda_docker_uri=cuda_docker_uri,
                component=GiGLComponents.Inferencer,
                vertex_ai_region=resource_config_wrapper.vertex_ai_inferencer_region,
            )
        elif isinstance(
            resource_config_wrapper.inferencer_config, VertexAiGraphStoreConfig
        ):
            launch_graph_store_enabled_job(
                vertex_ai_graph_store_config=resource_config_wrapper.inferencer_config,
                job_name=job_name,
                task_config_uri=task_config_uri,
                resource_config_uri=resource_config_uri,
                compute_commmand=inference_process_command,
                compute_runtime_args=inference_process_runtime_args,
                storage_command=gbml_config_pb_wrapper.inferencer_config.graph_store_storage_config.command,
                storage_args=gbml_config_pb_wrapper.inferencer_config.graph_store_storage_config.storage_args,
                resource_config_wrapper=resource_config_wrapper,
                cpu_docker_uri=cpu_docker_uri,
                cuda_docker_uri=cuda_docker_uri,
                component=GiGLComponents.Inferencer,
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
