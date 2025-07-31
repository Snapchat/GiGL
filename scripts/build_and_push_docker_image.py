import argparse
import subprocess
import sys
from enum import Enum
from pathlib import Path

from gigl.common.constants import (
    DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG,
    DOCKER_LATEST_BASE_CUDA_IMAGE_NAME_WITH_TAG,
    DOCKER_LATEST_BASE_DATAFLOW_IMAGE_NAME_WITH_TAG,
)
from gigl.common.logger import Logger
from gigl.orchestration.img_builder import build_and_push_image

logger = Logger()

ROOT_DIR = Path(__file__).resolve().parent.parent
CONTEXT_PATH: str = ROOT_DIR.as_posix()


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


def build_and_push_cpu_image(
    image_name: str,
) -> None:
    """
    Builds and pushes a CPU-based Docker image.

    Args:
        image_name (str): The name of the Docker image to build and push.
    """
    dockerfile_path: str = (ROOT_DIR / "containers" / "Dockerfile.src").as_posix()

    build_and_push_image(
        base_image=DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG,
        image_name=image_name,
        dockerfile_path=dockerfile_path,
        context_path=CONTEXT_PATH,
        multi_arch=False,
    )


def build_and_push_cuda_image(
    image_name: str,
) -> None:
    """
    Builds and pushes a CUDA-based Docker image.

    Args:
        image_name (str): The name of the Docker image to build and push.
    """
    dockerfile_path: str = (ROOT_DIR / "containers" / "Dockerfile.src").as_posix()

    build_and_push_image(
        base_image=DOCKER_LATEST_BASE_CUDA_IMAGE_NAME_WITH_TAG,
        image_name=image_name,
        dockerfile_path=dockerfile_path,
        context_path=CONTEXT_PATH,
        multi_arch=False,
    )


def build_and_push_dataflow_image(
    image_name: str,
) -> None:
    """
    Builds and pushes a Dataflow-based Docker image.

    Args:
        image_name (str): The name of the Docker image to build and push.
    """
    dockerfile_path: str = (
        ROOT_DIR / "containers" / "Dockerfile.dataflow.src"
    ).as_posix()

    build_and_push_image(
        base_image=DOCKER_LATEST_BASE_DATAFLOW_IMAGE_NAME_WITH_TAG,
        image_name=image_name,
        dockerfile_path=dockerfile_path,
        context_path=CONTEXT_PATH,
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
    dockerfile_path: str = (
        ROOT_DIR / "containers" / "Dockerfile.gigl_workbench_container"
    ).as_posix()

    build_and_push_image(
        base_image=None,
        image_name=image_name,
        dockerfile_path=dockerfile_path,
        context_path=CONTEXT_PATH,
        multi_arch=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and push Docker images.")
    parser.add_argument(
        "--image_name", required=True, help="Name for the built Docker image"
    )
    parser.add_argument(
        "--predefined_type",
        choices=[e.value for e in PredefinedImageType],
        required=True,
        help="Predefined image type to build. If specified, do not need to specify other arguments, except image_name.",
    )

    args = parser.parse_args()
    try:
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

    except subprocess.CalledProcessError as e:
        logger.error(f"Command '{e.cmd}' returned non-zero exit status {e.returncode}.")
        sys.exit(e.returncode)
