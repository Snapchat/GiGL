"""
This script can be used to:
- Get the current version of GiGL
- Bump the version of GiGL, and subsequently release the src images and KFP pipeline

Example Usage:
Bump patch version and release resources:
python scripts/versioning_and_release.py --bump_type patch --project gigl-public-ci

Bump the nightly version and release resources:
python scripts/versioning_and_release.py --bump_type nightly --project gigl-public-ci

Get current version:
python scripts/versioning_and_release.py --get_current_version
"""

import argparse
import datetime
import re
from typing import Literal, Optional

from gigl.common import GcsUri
from gigl.common.constants import (
    DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG,
    DOCKER_LATEST_BASE_CUDA_IMAGE_NAME_WITH_TAG,
    GIGL_ROOT_DIR,
    PATH_GIGL_PKG_INIT_FILE,
)
from gigl.env.dep_constants import GIGL_PUBLIC_BUCKET_NAME
from gigl.orchestration.kubeflow.runner import KfpOrchestrator

from .build_and_push_docker_image import build_and_push_image


def get_current_version() -> Optional[str]:
    with open(PATH_GIGL_PKG_INIT_FILE, "r") as f:
        content = f.read()
        match = re.search(r'__version__ = "(.*?)"', content)
        if match:
            return match.group(1)
    return None


def update_version(version: str) -> None:
    with open(PATH_GIGL_PKG_INIT_FILE, "r") as f:
        content = f.read()
    updated_content = re.sub(
        r'__version__ = "(.*?)"', f'__version__ = "{version}"', content
    )
    with open(PATH_GIGL_PKG_INIT_FILE, "w") as f:
        f.write(updated_content)


def update_dep_vars_env(
    cuda_image_name: str,
    cpu_image_name: str,
    dataflow_image_name: str,
) -> None:
    print(
        f"Updating dep_vars.env with {cuda_image_name}, {cpu_image_name}, {dataflow_image_name}"
    )

    dep_vars_env_path = f"{GIGL_ROOT_DIR}/dep_vars.env"
    with open(dep_vars_env_path, "r") as f:
        content = f.read()

    # Update the Docker image names with the new version
    content = re.sub(
        r"DEFAULT_GIGL_RELEASE_SRC_IMAGE_CUDA=([^:]+):.*",
        f"DEFAULT_GIGL_RELEASE_SRC_IMAGE_CUDA=\\1:{cuda_image_name}",
        content,
    )
    content = re.sub(
        r"DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU=([^:]+):.*",
        f"DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU=\\1:{cpu_image_name}",
        content,
    )
    content = re.sub(
        r"DEFAULT_GIGL_RELEASE_SRC_IMAGE_DATAFLOW_CPU=([^:]+):.*",
        f"DEFAULT_GIGL_RELEASE_SRC_IMAGE_DATAFLOW_CPU=\\1:{dataflow_image_name}",
        content,
    )

    with open(dep_vars_env_path, "w") as f:
        f.write(content)


def update_pyproject(version: str) -> None:
    path = f"{GIGL_ROOT_DIR}/python/pyproject.toml"
    with open(path, "r") as f:
        content = f.read()
    content = re.sub(r'(version\s*)=\s*"[\d\.]+"', f'\\1= "{version}"', content)
    with open(path, "w") as f:
        f.write(content)


def push_src_images(
    cuda_image_name: str,
    cpu_image_name: str,
    dataflow_image_name: str,
) -> None:
    print(f"Building and pushing CUDA image to {cuda_image_name}")
    build_and_push_image(
        base_image=DOCKER_LATEST_BASE_CUDA_IMAGE_NAME_WITH_TAG,
        image_name=cuda_image_name,
        dockerfile_name="Dockerfile.src",
    )
    print(f"Building and pushing CPU image to {cpu_image_name}")
    build_and_push_image(
        base_image=DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG,
        image_name=cpu_image_name,
        dockerfile_name="Dockerfile.src",
    )
    print(f"Building and pushing Dataflow image to {dataflow_image_name}")
    build_and_push_image(
        base_image=DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG,
        image_name=dataflow_image_name,
        dockerfile_name="Dockerfile.dataflow.src",
        multi_arch=True,
    )


def get_new_version(
    bump_type: Literal["major", "minor", "patch", "nightly"], curr_version: str
) -> str:
    major, minor, patch = map(int, curr_version.split("."))
    if bump_type == "major":
        major += 1
        minor, patch = 0, 0
    elif bump_type == "minor":
        minor += 1
        patch = 0
    elif bump_type == "patch":
        patch += 1
    new_version = f"{major}.{minor}.{patch}"
    if bump_type == "nightly":
        new_version += f"-nightly.{datetime.datetime.now().strftime('%Y%m%d')}"
    return new_version


def bump_version_and_release_resources(
    bump_type: Literal["major", "minor", "patch", "nightly"],
    project: str,
    version_override: Optional[str] = None,
) -> None:
    version: Optional[str] = get_current_version()
    if version is None:
        raise ValueError("Current version not found")

    new_version: str
    if version_override:
        new_version = version_override
    else:
        new_version = get_new_version(bump_type=bump_type, curr_version=version)

    print(f"Bumping GiGL to version {new_version}")
    base_image_registry = f"us-central1-docker.pkg.dev/{project}/public-gigl"
    cuda_image_name = f"{base_image_registry}/src-cuda:{new_version}"
    cpu_image_name = f"{base_image_registry}/src-cpu:{new_version}"
    dataflow_image_name = f"{base_image_registry}/src-cpu-dataflow:{new_version}"

    push_src_images(
        cuda_image_name=cuda_image_name,
        cpu_image_name=cpu_image_name,
        dataflow_image_name=dataflow_image_name,
    )
    update_dep_vars_env(
        cuda_image_name=cuda_image_name,
        cpu_image_name=cpu_image_name,
        dataflow_image_name=dataflow_image_name,
    )
    update_version(version=new_version)
    update_pyproject(version=new_version)
    kfp_orchestrator = KfpOrchestrator()
    kfp_orchestrator.compile(
        cuda_container_image=cuda_image_name,
        cpu_container_image=cpu_image_name,
        dataflow_container_image=dataflow_image_name,
        dst_compiled_pipeline_path=GcsUri(
            uri=f"gs://{GIGL_PUBLIC_BUCKET_NAME}/releases/gigl-pipeline-{new_version}.yaml",
        ),
        tag=new_version,
    )

    print(
        f"Bumped to GiGL Version: {new_version}! To release, raise a PR with these changes and after it is merged, tag main with the version and run make release_gigl."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom arguments for version bump")
    parser.add_argument(
        "--bump_type",
        help="Specify major, minor, or patch release",
        choices=["major", "minor", "patch", "nightly"],
        default="nightly",
    )
    parser.add_argument(
        "--project",
        type=str,
        help="GCP project id",
    )
    parser.add_argument(
        "--get_current_version",
        action="store_true",
        help="Instead of bumping the version, get the current version",
    )
    parser.add_argument(
        "--version_override",
        type=str,
        help="Override the version to be bumped",
    )

    args = parser.parse_args()

    if args.get_current_version:
        print(get_current_version())
        exit(0)

    bump_version_and_release_resources(
        bump_type=args.bump_type,
        project=args.project,
        version_override=args.version_override,
    )
