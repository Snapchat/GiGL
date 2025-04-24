import argparse
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
    CPU = "cpu"
    CUDA = "cuda"
    DATAFLOW = "dataflow"


def build_and_push_cpu_image(
    image_name: str,
) -> None:
    build_and_push_image(
        base_image=DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG,
        image_name=image_name,
        dockerfile_name="Dockerfile.src",
    )


def build_and_push_cuda_image(
    image_name: str,
) -> None:
    build_and_push_image(
        base_image=DOCKER_LATEST_BASE_CUDA_IMAGE_NAME_WITH_TAG,
        image_name=image_name,
        dockerfile_name="Dockerfile.src",
    )


def build_and_push_dataflow_image(
    image_name: str,
) -> None:
    build_and_push_image(
        base_image=DOCKER_LATEST_BASE_DATAFLOW_IMAGE_NAME_WITH_TAG,
        image_name=image_name,
        dockerfile_name="Dockerfile.dataflow.src",
        multi_arch=True,
    )


def build_and_push_image(
    base_image: Optional[str],
    image_name: str,
    dockerfile_name: str,
    multi_arch: bool = False,
) -> None:
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

    build_command.append(".")

    logger.info(f"Running command: {' '.join(build_command)}")
    subprocess.run(build_command, check=True)

    # Push image if it's not a multi-arch build (multi-arch images are pushed in the build step)
    if not multi_arch:
        push_command = ["docker", "push", image_name]
        subprocess.run(push_command, check=True)


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
