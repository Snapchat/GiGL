import argparse
import datetime
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

from gigl.common.constants import (
    DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG,
    DOCKER_LATEST_BASE_CUDA_IMAGE_NAME_WITH_TAG,
    DOCKER_LATEST_BASE_DATAFLOW_IMAGE_NAME_WITH_TAG,
)
from gigl.common.logger import Logger

logger = Logger()


class PredefinedImageType(Enum):
    """
    Enum representing predefined Docker image types.

    Attributes:
        CPU: Represents a CPU-based Docker image. This is the default image type, which is also used by orchestration vms.
        CUDA: Represents a CUDA-based Docker image. Will be used by distributed trainer/inferencer vms if cuda is available.
        DATAFLOW: Represents a Dataflow-based Docker image. Is build off of the CPU image.
    """

    CPU = "cpu"
    CUDA = "cuda"
    DATAFLOW = "dataflow"
    DEV_WORKBENCH = "dev_workbench"


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
    tag = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    export_cuda_image_name = f"{export_docker_artifact_registry}/src-cuda:{tag}"
    export_cpu_image_name = f"{export_docker_artifact_registry}/src-cpu:{tag}"
    export_dataflow_image_name = (
        f"{export_docker_artifact_registry}/src-cpu-dataflow:{tag}"
    )

    build_and_push_image(
        base_image=base_image_cuda,
        image_name=export_cuda_image_name,
        dockerfile_name="Dockerfile.customer_src",
        context_path=context_path,
    )
    build_and_push_image(
        base_image=base_image_cpu,
        image_name=export_cpu_image_name,
        dockerfile_name="Dockerfile.customer_src",
        context_path=context_path,
    )
    build_and_push_image(
        base_image=base_image_dataflow,
        image_name=export_dataflow_image_name,
        dockerfile_name="Dockerfile.customer_src",
        context_path=context_path,
    )
    return export_cuda_image_name, export_cpu_image_name, export_dataflow_image_name


def build_and_push_cpu_image(
    image_name: str,
) -> None:
    """
    Builds and pushes a CPU-based Docker image.

    Args:
        image_name (str): The name of the Docker image to build and push.
    """
    build_and_push_image(
        base_image=DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG,
        image_name=image_name,
        dockerfile_name="Dockerfile.src",
    )


def build_and_push_cuda_image(
    image_name: str,
) -> None:
    """
    Builds and pushes a CUDA-based Docker image.

    Args:
        image_name (str): The name of the Docker image to build and push.
    """
    build_and_push_image(
        base_image=DOCKER_LATEST_BASE_CUDA_IMAGE_NAME_WITH_TAG,
        image_name=image_name,
        dockerfile_name="Dockerfile.src",
    )


def build_and_push_dataflow_image(
    image_name: str,
) -> None:
    """
    Builds and pushes a Dataflow-based Docker image.

    Args:
        image_name (str): The name of the Docker image to build and push.
    """
    build_and_push_image(
        base_image=DOCKER_LATEST_BASE_DATAFLOW_IMAGE_NAME_WITH_TAG,
        image_name=image_name,
        dockerfile_name="Dockerfile.dataflow.src",
        multi_arch=True,
    )


def build_and_push_dev_workbench_image(
    image_name: str,
) -> None:
    """
    Builds and pushes a Dev Workbench Docker image.

    Args:
        image_name (str): The name of the Docker image to build and push.
    """
    build_and_push_image(
        base_image=None,
        image_name=image_name,
        dockerfile_name="Dockerfile.gigl_workbench_container",
        multi_arch=False,
    )


def build_and_push_image(
    base_image: Optional[str],
    image_name: str,
    dockerfile_name: str,
    multi_arch: bool = False,
    context_path: Optional[str] = None,
) -> None:
    """
    Builds and pushes a Docker image.

    Args:
        base_image (Optional[str]): The base image to use for the build.
        image_name (str): The name of the Docker image to build and push.
        dockerfile_name (str): The name of the Dockerfile to use for the build.
        multi_arch (bool): Whether to build a multi-architecture Docker image. Defaults to False.
    """
    root_dir = Path(__file__).resolve().parent.parent

    dockerfile_path = root_dir / "containers" / dockerfile_name

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

    if context_path is None:
        build_command.append(root_dir.as_posix())
    else:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and push Docker images.")
    parser.add_argument(
        "--image_name", required=True, help="Name for the built Docker image"
    )
    parser.add_argument(
        "--predefined_type",
        choices=[e.value for e in PredefinedImageType],
        required=False,
        help="Predefined image type to build. If specified, do not need to specify other arguments, except image_name.",
    )

    parser.add_argument("--base_image", help="Base image as an optional build argument")

    parser.add_argument(
        "--dockerfile_name", required=False, help="Dockerfile to use for the build"
    )
    parser.add_argument(
        "--multi_arch",
        action="store_true",
        help="Build a multi-architecture Docker image",
    )

    args = parser.parse_args()
    try:
        if args.predefined_type:
            if args.predefined_type == PredefinedImageType.CPU.value:
                build_and_push_cpu_image(image_name=args.image_name)
            elif args.predefined_type == PredefinedImageType.CUDA.value:
                build_and_push_cuda_image(image_name=args.image_name)
            elif args.predefined_type == PredefinedImageType.DATAFLOW.value:
                build_and_push_dataflow_image(image_name=args.image_name)
            elif args.predefined_type == PredefinedImageType.DEV_WORKBENCH.value:
                build_and_push_dev_workbench_image(image_name=args.image_name)
            else:
                raise ValueError(f"Invalid predefined_type: {args.predefined_type}")
        else:
            assert (
                args.base_image
            ), "base_image is required if predefined_type is not specified"
            assert (
                args.dockerfile_name
            ), "dockerfile_name is required if predefined_type is not specified"
            build_and_push_image(
                base_image=args.base_image,
                image_name=args.image_name,
                dockerfile_name=args.dockerfile_name,
                multi_arch=args.multi_arch,
            )
    except subprocess.CalledProcessError as e:
        logger.error(f"Command '{e.cmd}' returned non-zero exit status {e.returncode}.")
        sys.exit(e.returncode)
