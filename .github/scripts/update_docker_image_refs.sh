#!/bin/bash
# Script to update gigl/dep_vars.env and cloud builder config with new Docker image references.
# Each GIGL_*_IMAGE input is the tag one build-base-docker-images run shares; the run pushes every image as
# <input>-py<major><minor> for each minor in SUPPORTED_PYTHON_VERSIONS. This script writes each of those refs to its
# _py<major><minor> key in gigl/dep_vars.env, and the builder for the minor in .python-version to the Cloud Build
# _BUILDER_IMAGE default.

set -e

DEFAULT_BUILDER_IMAGE="${GIGL_BUILDER_IMAGE}-py$(cut -d. -f1,2 .python-version | tr -d .)"

VERSIONS="$(grep '^SUPPORTED_PYTHON_VERSIONS=' gigl/dep_vars.env | cut -d= -f2)"
for VERSION in ${VERSIONS//,/ }; do
    TAG="py$(cut -d. -f1,2 <<< "${VERSION}" | tr -d .)"
    echo "Writing the ${TAG} image refs to gigl/dep_vars.env"
    sed -i "s|^DOCKER_LATEST_BASE_CUDA_IMAGE_NAME_WITH_TAG_${TAG}=.*|DOCKER_LATEST_BASE_CUDA_IMAGE_NAME_WITH_TAG_${TAG}=${GIGL_BASE_CUDA_IMAGE}-${TAG}|" gigl/dep_vars.env
    sed -i "s|^DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG_${TAG}=.*|DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG_${TAG}=${GIGL_BASE_CPU_IMAGE}-${TAG}|" gigl/dep_vars.env
    sed -i "s|^DOCKER_LATEST_BASE_DATAFLOW_IMAGE_NAME_WITH_TAG_${TAG}=.*|DOCKER_LATEST_BASE_DATAFLOW_IMAGE_NAME_WITH_TAG_${TAG}=${GIGL_BASE_DATAFLOW_IMAGE}-${TAG}|" gigl/dep_vars.env
    sed -i "s|^DOCKER_LATEST_BUILDER_IMAGE_NAME_WITH_TAG_${TAG}=.*|DOCKER_LATEST_BUILDER_IMAGE_NAME_WITH_TAG_${TAG}=${GIGL_BUILDER_IMAGE}-${TAG}|" gigl/dep_vars.env
done

echo "Writing the default builder to the Cloud Build config: ${DEFAULT_BUILDER_IMAGE}"
sed -i "s|^  _BUILDER_IMAGE: .*|  _BUILDER_IMAGE: \"${DEFAULT_BUILDER_IMAGE}\"|" .github/cloud_builder/run_command_on_active_checkout.yaml
