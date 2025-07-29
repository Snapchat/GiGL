import datetime
import subprocess
from pathlib import Path
from typing import Optional

from gigl.common.constants import (
    DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG,
    DOCKER_LATEST_BASE_CUDA_IMAGE_NAME_WITH_TAG,
    DOCKER_LATEST_BASE_DATAFLOW_IMAGE_NAME_WITH_TAG,
)
from gigl.common.logger import Logger

logger = Logger()


CUSTOMER_SRC_DOCKERFILE_PATH = (
    Path(__file__).resolve() / "Dockerfile.customer_src"
).as_posix()


def build_and_push_customer_src_images(
    context_path: str,
    export_docker_artifact_registry: str,
    base_image_cuda: str = DOCKER_LATEST_BASE_CUDA_IMAGE_NAME_WITH_TAG,
    base_image_cpu: str = DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG,
    base_image_dataflow: str = DOCKER_LATEST_BASE_DATAFLOW_IMAGE_NAME_WITH_TAG,
) -> tuple[str, str, str]:
    """
    Package user provided code located at context_path into docker images based on the base images provided.
    The images are pushed to the export_docker_artifact_registry.

    Args:
        context_path (str): Root directory that will be copied into the docker images.
        export_docker_artifact_registry (str): Docker artifact registry to push the images to.
        base_image_cuda (str): Base image to use for the CUDA image.
        base_image_cpu (str): Base image to use for the CPU image.
        base_image_dataflow (str): Base image to use for the Dataflow image.
    Returns:
        tuple[str, str, str]: The names of cuda, cpu, and dataflow images.
    """
    logger.info(
        f"Building and pushing customer src images to {export_docker_artifact_registry}"
    )
    logger.info(
        f"Using base images: {base_image_cuda}, {base_image_cpu}, {base_image_dataflow}"
    )
    logger.info(f"Using context path: {context_path}")
    tag = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    export_cuda_image_name = f"{export_docker_artifact_registry}/src-cuda:{tag}"
    export_cpu_image_name = f"{export_docker_artifact_registry}/src-cpu:{tag}"
    export_dataflow_image_name = (
        f"{export_docker_artifact_registry}/src-cpu-dataflow:{tag}"
    )

    logger.info(f"Building and pushing cuda image to {export_cuda_image_name}")
    build_and_push_image(
        base_image=base_image_cuda,
        image_name=export_cuda_image_name,
        dockerfile_path=CUSTOMER_SRC_DOCKERFILE_PATH,
        context_path=context_path,
    )
    logger.info(f"Building and pushing cpu image to {export_cpu_image_name}")
    build_and_push_image(
        base_image=base_image_cpu,
        image_name=export_cpu_image_name,
        dockerfile_path=CUSTOMER_SRC_DOCKERFILE_PATH,
        context_path=context_path,
    )
    logger.info(f"Building and pushing dataflow image to {export_dataflow_image_name}")
    build_and_push_image(
        base_image=base_image_dataflow,
        image_name=export_dataflow_image_name,
        dockerfile_path=CUSTOMER_SRC_DOCKERFILE_PATH,
        context_path=context_path,
    )
    logger.info(f"Done building and pushing customer src images")
    return export_cuda_image_name, export_cpu_image_name, export_dataflow_image_name


def build_and_push_image(
    base_image: Optional[str],
    image_name: str,
    dockerfile_path: str,
    context_path: str,
    multi_arch: bool = False,
) -> None:
    """
    Builds and pushes a Docker image.

    Args:
        base_image (Optional[str]): The base image to use for the build.
        image_name (str): The name of the Docker image to build and push.
        dockerfile_path (str): The path to the Dockerfile to use for the build.
        context_path (str): The path to the context to use for the build.
        multi_arch (bool): Whether to build a multi-architecture Docker image. Defaults to False.
    """

    if multi_arch:
        build_command = [
            "docker",
            "buildx",
            "build",
            "--platform",
            "linux/amd64,linux/arm64",
            "-f",
            str(dockerfile_path),
            "-t",
            image_name,
            "--push",
        ]
    else:
        build_command = [
            "docker",
            "build",
            "-f",
            str(dockerfile_path),
            "-t",
            image_name,
        ]

    if base_image:
        build_command.extend(["--build-arg", f"BASE_IMAGE={base_image}"])

    build_command.append(context_path)

    logger.info(f"Running command: {' '.join(build_command)}")
    result = subprocess.run(
        build_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    if result.returncode != 0:
        logger.info(result.stdout.decode())
        logger.error(f"Command failed: {' '.join(build_command)}")
        raise RuntimeError(f"Docker build failed with exit code {result.returncode}")

    # Push image if it's not a multi-arch build (multi-arch images are pushed in the build step)
    if not multi_arch:
        push_command = ["docker", "push", image_name]
        result_push = subprocess.run(
            push_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        if result_push.returncode != 0:
            logger.info(result_push.stdout.decode())
            logger.error(f"Command failed: {' '.join(push_command)}")
            raise RuntimeError(
                f"Docker push failed with exit code {result_push.returncode}"
            )
