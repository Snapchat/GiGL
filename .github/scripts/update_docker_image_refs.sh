#!/bin/bash
# Script to update gigl/dep_vars.env and cloud builder config with new Docker image references.
# Each GIGL_*_IMAGE input is the tag one build-base-docker-images run shares; the run pushes every image as
# <input>-py311, -py312 and -py313. This script writes each of those refs to its _PY<major><minor> key in
# gigl/dep_vars.env, and the builder for the minor in .python-version to the Cloud Build _BUILDER_IMAGE default.

set -e

DEFAULT_BUILDER_IMAGE="${GIGL_BUILDER_IMAGE}-py$(cut -d. -f1,2 .python-version | tr -d .)"

for PY in 311 312 313; do
    echo "Writing the py${PY} image refs to gigl/dep_vars.env"
    sed -i "s|^DOCKER_LATEST_BASE_CUDA_IMAGE_NAME_WITH_TAG_PY${PY}=.*|DOCKER_LATEST_BASE_CUDA_IMAGE_NAME_WITH_TAG_PY${PY}=${GIGL_BASE_CUDA_IMAGE}-py${PY}|" gigl/dep_vars.env
    sed -i "s|^DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG_PY${PY}=.*|DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG_PY${PY}=${GIGL_BASE_CPU_IMAGE}-py${PY}|" gigl/dep_vars.env
    sed -i "s|^DOCKER_LATEST_BASE_DATAFLOW_IMAGE_NAME_WITH_TAG_PY${PY}=.*|DOCKER_LATEST_BASE_DATAFLOW_IMAGE_NAME_WITH_TAG_PY${PY}=${GIGL_BASE_DATAFLOW_IMAGE}-py${PY}|" gigl/dep_vars.env
    sed -i "s|^DOCKER_LATEST_BUILDER_IMAGE_NAME_WITH_TAG_PY${PY}=.*|DOCKER_LATEST_BUILDER_IMAGE_NAME_WITH_TAG_PY${PY}=${GIGL_BUILDER_IMAGE}-py${PY}|" gigl/dep_vars.env
done

echo "Writing the default builder to the Cloud Build config: ${DEFAULT_BUILDER_IMAGE}"
sed -i "s|^  _BUILDER_IMAGE: .*|  _BUILDER_IMAGE: \"${DEFAULT_BUILDER_IMAGE}\"|" .github/cloud_builder/run_command_on_active_checkout.yaml
