import argparse
from typing import Optional

from gigl.common import Uri, UriFactory
from gigl.common.logger import Logger
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.types.pb_wrappers.gigl_resource_config import (
    GiglResourceConfigWrapper,
)
from gigl.src.common.utils.metrics_service_provider import initialize_metrics
from gigl.src.inference.lib.assets import InferenceAssets
from gigl.src.inference.v1.gnn_inferencer import InferencerV1
from gigl.src.inference.v2.glt_inferencer import GLTInferencer

logger = Logger()


class Inferencer:
    """
    GiGL Component that runs static (GiGL) or dynamic (GLT) inference of a trained model on samples and outputs embedding and/or prediction assets.
    """

    def run(
        self,
        applied_task_identifier: AppliedTaskIdentifier,
        task_config_uri: Uri,
        resource_config_uri: Uri,
        custom_worker_image_uri: Optional[str] = None,
        cpu_docker_uri: Optional[str] = None,
        cuda_docker_uri: Optional[str] = None,
    ):
        gbml_config_wrapper = GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
            gbml_config_uri=task_config_uri
        )
        resource_config_wrapper: GiglResourceConfigWrapper = get_resource_config(
            resource_config_uri=resource_config_uri
        )

        # Prepare staging paths for inferencer assets by clearing the paths that inferencer
        # would be writing to, to avoid clobbering of data.
        InferenceAssets.prepare_staging_paths(
            applied_task_identifier=applied_task_identifier,
            gbml_config_pb_wrapper=gbml_config_wrapper,
            project=resource_config_wrapper.project,
        )

        if gbml_config_wrapper.should_use_experimental_glt_backend:
            inferencer_glt = GLTInferencer()
            inferencer_glt.run(
                applied_task_identifier=applied_task_identifier,
                task_config_uri=task_config_uri,
                resource_config_uri=resource_config_uri,
                cpu_docker_uri=cpu_docker_uri,
                cuda_docker_uri=cuda_docker_uri,
            )
        else:
            inferencer_v1 = InferencerV1(bq_gcp_project=resource_config_wrapper.project)
            inferencer_v1.run(
                applied_task_identifier=applied_task_identifier,
                task_config_uri=task_config_uri,
                custom_worker_image_uri=custom_worker_image_uri,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Program to run distributed inference")
    parser.add_argument(
        "--job_name",
        type=str,
        help="Unique identifier for the job name",
        required=True,
    )
    parser.add_argument(
        "--task_config_uri",
        type=str,
        help="Gbml config uri",
        required=True,
    )
    parser.add_argument(
        "--resource_config_uri",
        type=str,
        help="Runtime argument for resource and env specifications of each component",
        required=True,
    )
    parser.add_argument(
        "--custom_worker_image_uri",
        type=str,
        help="Docker image to use for the worker harness in dataflow",
        required=False,
    )

    parser.add_argument(
        "--cpu_docker_uri",
        type=str,
        help="User Specified or KFP compiled Docker Image for CPU inference",
        required=False,
    )
    parser.add_argument(
        "--cuda_docker_uri",
        type=str,
        help="User Specified or KFP compiled Docker Image for GPU inference",
        required=False,
    )
    args = parser.parse_args()

    task_config_uri = UriFactory.create_uri(args.task_config_uri)
    resource_config_uri = UriFactory.create_uri(args.resource_config_uri)
    custom_worker_image_uri = args.custom_worker_image_uri
    cpu_docker_uri = args.cpu_docker_uri
    cuda_docker_uri = args.cuda_docker_uri

    initialize_metrics(task_config_uri=task_config_uri, service_name=args.job_name)

    applied_task_identifier = AppliedTaskIdentifier(args.job_name)
    inferencer = Inferencer()
    inferencer.run(
        applied_task_identifier=applied_task_identifier,
        task_config_uri=task_config_uri,
        resource_config_uri=resource_config_uri,
        custom_worker_image_uri=custom_worker_image_uri,
        cpu_docker_uri=cpu_docker_uri,
        cuda_docker_uri=cuda_docker_uri,
    )
