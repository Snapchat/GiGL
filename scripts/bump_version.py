"""
This script can be used to:
- Get the current version of GiGL
- Bump the version labels of GiGL
Note: This does not release the src images and KFP pipeline, it only updates the version labels.
The usage of bump_version.py, and the overall release process of the src images and KFP pipeline
is handled by the .github/workflows/create_release.yml workflow.

Example Usage:
Bump patch version:
python scripts/bump_version.py --bump_type patch --project gigl-public-ci

Bump the nightly version:
python scripts/bump_version.py --bump_type nightly --project gigl-public-ci

Get current version:
python scripts/bump_version.py --get_current_version
"""

import argparse
import datetime
import re
from typing import Literal, Optional

from gigl.common.constants import GIGL_ROOT_DIR, PATH_GIGL_PKG_INIT_FILE
from gigl.env.dep_constants import GIGL_PUBLIC_BUCKET_NAME


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
    dev_workbench_image_name: str,
    kfp_pipeline_path: str,
) -> None:
    print(
        f"Updating dep_vars.env with: "
        + f"cuda_image: {cuda_image_name}, "
        + f"cpu_image: {cpu_image_name}, "
        + f"dataflow_image: {dataflow_image_name}, "
        + f"dev_workbench_image: {dev_workbench_image_name}, "
        + f"kfp_pipeline: {kfp_pipeline_path}"
    )

    dep_vars_env_path = f"{GIGL_ROOT_DIR}/dep_vars.env"
    with open(dep_vars_env_path, "r") as f:
        content = f.read()

    content = re.sub(
        r"DEFAULT_GIGL_RELEASE_SRC_IMAGE_CUDA=.*",
        r"DEFAULT_GIGL_RELEASE_SRC_IMAGE_CUDA=" + cuda_image_name,
        content,
    )
    content = re.sub(
        r"DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU=.*",
        r"DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU=" + cpu_image_name,
        content,
    )
    content = re.sub(
        r"DEFAULT_GIGL_RELEASE_SRC_IMAGE_DATAFLOW_CPU=.*",
        r"DEFAULT_GIGL_RELEASE_SRC_IMAGE_DATAFLOW_CPU=" + dataflow_image_name,
        content,
    )
    content = re.sub(
        r"DEFAULT_GIGL_RELEASE_DEV_WORKBENCH_IMAGE=.*",
        r"DEFAULT_GIGL_RELEASE_DEV_WORKBENCH_IMAGE=" + dev_workbench_image_name,
        content,
    )
    content = re.sub(
        r"DEFAULT_GIGL_RELEASE_KFP_PIPELINE_PATH=.*",
        r"DEFAULT_GIGL_RELEASE_KFP_PIPELINE_PATH=" + kfp_pipeline_path,
        content,
    )

    with open(dep_vars_env_path, "w") as f:
        f.write(content)


def update_pyproject(version: str) -> None:
    path = f"{GIGL_ROOT_DIR}/pyproject.toml"
    with open(path, "r") as f:
        content = f.read()
    content = re.sub(r'(version\s*)=\s*"[\d\.]+"', f'\\1= "{version}"', content)
    with open(path, "w") as f:
        f.write(content)


def get_new_version(
    bump_type: Literal["major", "minor", "patch", "nightly"], curr_version: str
) -> str:
    version_parts = curr_version.split(".")
    # We may have dev suffixes, so we only unpack first three parts
    major, minor, patch = (
        int(version_parts[0]),
        int(version_parts[1]),
        int(version_parts[2]),
    )
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
        # PEP 440 compliant
        new_version += f".dev{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    return new_version


def bump_version(
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
    dev_workbench_image_name = f"{base_image_registry}/gigl-dev-workbench:{new_version}"
    kfp_pipeline_path = f"gs://{GIGL_PUBLIC_BUCKET_NAME}/releases/pipelines/gigl-pipeline-{new_version}.yaml"

    update_dep_vars_env(
        cuda_image_name=cuda_image_name,
        cpu_image_name=cpu_image_name,
        dataflow_image_name=dataflow_image_name,
        dev_workbench_image_name=dev_workbench_image_name,
        kfp_pipeline_path=kfp_pipeline_path,
    )
    update_version(version=new_version)
    update_pyproject(version=new_version)

    print(
        f"Bumped to GiGL Version: {new_version}! To release, raise a PR with these changes and after it is merged, tag main with the version and run make release_gigl."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bump the version of GiGL")
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

    assert args.project, "Project is required to bump the version"

    bump_version(
        bump_type=args.bump_type,
        project=args.project,
        version_override=args.version_override,
    )
