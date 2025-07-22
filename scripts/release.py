"""
This script can be used to release GiGL assets publicly.

Usage:
Release CPU src images:
python scripts/release.py --release_type="cpu_src_images"

Release CUDA src images:
python scripts/release.py --release_type="cuda_src_images"

Release Dev images:
python scripts/release.py --release_type="dev_images"

Release KFP pipeline:
python scripts/release.py --release_type="kfp_pipeline"
"""

import argparse

from gigl.common import GcsUri
from gigl.common.constants import (
    DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU,
    DEFAULT_GIGL_RELEASE_SRC_IMAGE_CUDA,
    DEFAULT_GIGL_RELEASE_SRC_IMAGE_DATAFLOW_CPU,
    DEFAULT_GIGL_RELEASE_DEV_WORKBENCH_IMAGE,
    DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG,
    DOCKER_LATEST_BASE_CUDA_IMAGE_NAME_WITH_TAG,
    DOCKER_LATEST_BASE_DATAFLOW_IMAGE_NAME_WITH_TAG,
)
from gigl.env.dep_constants import GIGL_PUBLIC_BUCKET_NAME
from gigl.orchestration.kubeflow.runner import KfpOrchestrator

from .build_and_push_docker_image import build_and_push_image
from .bump_version import get_current_version


def release_cpu_src_images():
    build_and_push_image(
        base_image=DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG,
        image_name=DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU,
        dockerfile_name="Dockerfile.src",
    )
    print(f"Pushed CPU src image to {DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU}")
    build_and_push_image(
        base_image=DOCKER_LATEST_BASE_DATAFLOW_IMAGE_NAME_WITH_TAG,
        image_name=DEFAULT_GIGL_RELEASE_SRC_IMAGE_DATAFLOW_CPU,
        dockerfile_name="Dockerfile.dataflow.src",
        multi_arch=True,
    )
    print(f"Pushed Dataflow src image to {DEFAULT_GIGL_RELEASE_SRC_IMAGE_DATAFLOW_CPU}")


def release_cuda_src_images():
    build_and_push_image(
        base_image=DOCKER_LATEST_BASE_CUDA_IMAGE_NAME_WITH_TAG,
        image_name=DEFAULT_GIGL_RELEASE_SRC_IMAGE_CUDA,
        dockerfile_name="Dockerfile.src",
    )
    print(f"Pushed CUDA src image to {DEFAULT_GIGL_RELEASE_SRC_IMAGE_CUDA}")


def release_dev_images():
    build_and_push_image(
        base_image=None,
        image_name=DEFAULT_GIGL_RELEASE_DEV_WORKBENCH_IMAGE,
        dockerfile_name="Dockerfile.dev_workbench_container",
    )
    print(f"Pushed Dev image to {DEFAULT_GIGL_RELEASE_DEV_WORKBENCH_IMAGE}")


def release_kfp_pipeline():
    version = get_current_version()
    if version is None:
        raise ValueError("Current version not found")

    kfp_orchestrator = KfpOrchestrator()
    kfp_orchestrator.compile(
        cuda_container_image=DEFAULT_GIGL_RELEASE_SRC_IMAGE_CUDA,
        cpu_container_image=DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU,
        dataflow_container_image=DEFAULT_GIGL_RELEASE_SRC_IMAGE_DATAFLOW_CPU,
        dst_compiled_pipeline_path=GcsUri(
            uri=f"gs://{GIGL_PUBLIC_BUCKET_NAME}/releases/gigl-pipeline-{version}.yaml",
        ),
        tag=version,
    )
    print(
        f"Pushed KFP pipeline to {GIGL_PUBLIC_BUCKET_NAME}/releases/gigl-pipeline-{version}.yaml"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Release the GiGL version")
    parser.add_argument(
        "--release_type",
        help="Specify the type of release to be done",
        choices=["cpu_src_images", "cuda_src_images", "dev_images", "kfp_pipeline"],
        required=True,
    )

    args = parser.parse_args()

    if args.release_type == "cpu_src_images":
        release_cpu_src_images()
    elif args.release_type == "cuda_src_images":
        release_cuda_src_images()
    elif args.release_type == "dev_images":
        release_dev_images()
    elif args.release_type == "kfp_pipeline":
        release_kfp_pipeline()
