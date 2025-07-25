"""
Util file for launching example training and inference jobs on Vertex AI.
This script:
    1.  Builds and pushes Docker images for CPU and CUDA environments
    2.  Launches training and inference jobs on Vertex AI, using specified task and resource configs.

Args: 
    --job_type: Type of job to run (training, inference, or training_and_inference).
    --task_config_uri: URI to the task configuration file.
    --resource_config_uri: URI to the resource configuration file.
Example usage:
    python -m examples.tutorial.KDD_2025.vertex_ai_launch --job_type training -- task_config_uri=gs://your-bucket/task_config.yaml --resource_config_uri=gs://your-bucket/resource_config.yaml
"""

import argparse
import datetime
import getpass
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from gigl.common import Uri
from gigl.common.types.uri.uri_factory import UriFactory
from gigl.env.pipelines_config import get_resource_config
from gigl.orchestration.local.runner import PipelineConfig, Runner
from gigl.src.common.types import AppliedTaskIdentifier
from scripts.build_and_push_docker_image import (
    build_and_push_cpu_image,
    build_and_push_cuda_image,
)


def main(
    job_type: Literal["training", "inference", "training_and_inference"],
    task_config_uri: Uri,
    resource_config_uri: Uri,
) -> None:
    resource_config = get_resource_config(resource_config_uri)
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    cuda_image = f"us-central1-docker.pkg.dev/{resource_config.project}/gigl-base-images/src-cuda:{date}"
    cpu_image = f"us-central1-docker.pkg.dev/{resource_config.project}/gigl-base-images/src-cpu:{date}"
    with ThreadPoolExecutor() as executor:
        executor.submit(build_and_push_cpu_image, image_name=cpu_image)
        executor.submit(build_and_push_cuda_image, image_name=cuda_image)

    pipeline_config = PipelineConfig(
        applied_task_identifier=AppliedTaskIdentifier(
            f"{getpass.getuser()}-gigl-tutorial-{job_type}-{date}"
        ),
        task_config_uri=task_config_uri,
        resource_config_uri=resource_config_uri,
        custom_cpu_docker_uri=cpu_image,
        custom_cuda_docker_uri=cuda_image,
    )

    if job_type == "training":
        Runner().run_trainer(pipeline_config)
    elif job_type == "inference":
        Runner().run_inferencer(pipeline_config)
    elif job_type == "training_and_inference":
        Runner().run_trainer(pipeline_config)
        Runner().run_inferencer(pipeline_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch a Vertex AI job.")
    parser.add_argument(
        "--job_type",
        type=str,
        choices=["training", "inference", "training_and_inference"],
        required=True,
    )
    parser.add_argument("--task_config_uri", type=str, required=True)
    parser.add_argument("--resource_config_uri", type=str, required=True)
    args = parser.parse_args()
    main(
        job_type=args.job_type,
        task_config_uri=UriFactory.create_uri(args.task_config_uri),
        resource_config_uri=UriFactory.create_uri(args.resource_config_uri),
    )
