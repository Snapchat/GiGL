import argparse
import re
from typing import Optional

from gigl.common.constants import (
    DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG,
    DOCKER_LATEST_BASE_CUDA_IMAGE_NAME_WITH_TAG,
    GIGL_ROOT_DIR,
    PATH_GIGL_PKG_INIT_FILE,
)
from gigl.env.pipelines_config import get_resource_config

from .build_and_push_docker_image import build_and_push_image


def get_current_version(filename: str) -> Optional[str]:
    with open(filename, "r") as f:
        content = f.read()
        match = re.search(r'__version__ = "([\d\.]+)"', content)
        if match:
            return match.group(1)
    return None


def update_version(filename: str, version: str) -> None:
    with open(filename, "r") as f:
        content = f.read()
    updated_content = re.sub(
        r'__version__ = "([\d\.]+)"', f'__version__ = "{version}"', content
    )
    with open(filename, "w") as f:
        f.write(updated_content)


def update_dep_constants(version: str) -> None:
    path = f"{GIGL_ROOT_DIR}/python/gigl/env/dep_constants.py"
    with open(path, "r") as f:
        content = f.read()
    content = re.sub(r"gigl_src_cuda:[\d\.]+", f"gigl_src_cuda:{version}", content)
    content = re.sub(r"gigl_src_cpu:[\d\.]+", f"gigl_src_cpu:{version}", content)
    content = re.sub(
        r"gigl_src_dataflow:[\d\.]+", f"gigl_src_dataflow:{version}", content
    )
    with open(path, "w") as f:
        f.write(content)


def update_pyproject(version: str) -> None:
    path = f"{GIGL_ROOT_DIR}/python/pyproject.toml"
    with open(path, "r") as f:
        content = f.read()
    content = re.sub(r'(version\s*)=\s*"[\d\.]+"', f'\\1= "{version}"', content)
    with open(path, "w") as f:
        f.write(content)


def bump_version(
    bump_type: str, cuda_image_name: str, cpu_image_name: str, dataflow_image_name: str
) -> None:
    version: Optional[str] = get_current_version(filename=str(PATH_GIGL_PKG_INIT_FILE))
    if version is None:
        raise ValueError("Current version not found")

    major, minor, patch = map(int, version.split("."))
    if bump_type == "major":
        major += 1
        minor, patch = 0, 0
    elif bump_type == "minor":
        minor += 1
        patch = 0
    else:
        patch += 1

    new_version = f"{major}.{minor}.{patch}"

    print(f"Bumping GiGL to version {new_version}")
    cuda_image = f"{cuda_image_name}:{new_version}"
    cpu_image = f"{cpu_image_name}:{new_version}"
    dataflow_image = f"{dataflow_image_name}:{new_version}"

    build_and_push_image(
        base_image=DOCKER_LATEST_BASE_CUDA_IMAGE_NAME_WITH_TAG,
        image_name=cuda_image,
        dockerfile_name="Dockerfile.src",
    )
    build_and_push_image(
        base_image=DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG,
        image_name=cpu_image,
        dockerfile_name="Dockerfile.src",
    )
    build_and_push_image(
        base_image=DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG,
        image_name=dataflow_image,
        dockerfile_name="Dockerfile.dataflow.src",
        multi_arch=True,
    )

    update_version(filename=str(PATH_GIGL_PKG_INIT_FILE), version=new_version)
    update_dep_constants(version=new_version)
    update_pyproject(version=new_version)

    print(
        f"Bumped to GiGL Version: {new_version}! To release, raise a PR with these changes and after it is merged, tag main with the version and run make release_gigl."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom arguments for version bump")
    parser.add_argument("--bump_type", help="Specify major, minor, or patch release")
    parser.add_argument(
        "--resource_config_uri",
        type=str,
        help="Runtime argument for resource and env specifications of each component",
        required=True,
    )
    parser.add_argument(
        "--cuda_image_name",
        help="Specify custom name for cuda gigl base image",
    )
    parser.add_argument(
        "--cpu_image_name",
        help="Specify custom name for cpu gigl base image",
    )
    parser.add_argument(
        "--dataflow_image_name",
        help="Specify custom name for dataflow gigl base image",
    )
    args = parser.parse_args()

    resource_config = get_resource_config(args.resource_config_uri)
    project = resource_config.project

    cuda_image_name = (
        args.cuda_image_name or f"gcr.io/{project}/gigl_src_images/gigl_src_cuda"
    )
    cpu_image_name = (
        args.cpu_image_name or f"gcr.io/{project}/gigl_src_images/gigl_src_cpu"
    )
    dataflow_image_name = (
        args.dataflow_image_name
        or f"gcr.io/{project}/gigl_src_images/gigl_src_dataflow"
    )

    try:
        bump_version(
            bump_type=args.bump_type,
            cuda_image_name=args.cuda_image_name,
            cpu_image_name=args.cpu_image_name,
            dataflow_image_name=args.dataflow_image_name,
        )
    except RuntimeError as e:
        print(f"Error: {e}")
