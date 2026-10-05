#!/bin/bash
# Script to update gigl/dep_vars.env and cloud builder config with new Docker image references.
# The GIGL_*_IMAGE inputs are stems; each image is published as <stem>-py3XX for every supported
# Python minor. dep_vars.env and the Cloud Build builder name store the stems; the Cloud Build
# _BUILDER_SUFFIX default is the suffix for the minor in .python-version.

set -e

DEFAULT_PYTHON_SUFFIX="-py$(cut -d. -f1,2 .python-version | tr -d .)"

echo "Writing new image names to gigl/dep_vars.env:"
echo "  DOCKER_LATEST_BASE_CUDA_IMAGE_NAME_WITH_TAG=${GIGL_BASE_CUDA_IMAGE}"
echo "  DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG=${GIGL_BASE_CPU_IMAGE}"
echo "  DOCKER_LATEST_BASE_DATAFLOW_IMAGE_NAME_WITH_TAG=${GIGL_BASE_DATAFLOW_IMAGE}"
echo "Writing the builder image to the Cloud Build config: ${GIGL_BUILDER_IMAGE}, default suffix ${DEFAULT_PYTHON_SUFFIX}"

sed -i "s|^DOCKER_LATEST_BASE_CUDA_IMAGE_NAME_WITH_TAG=.*|DOCKER_LATEST_BASE_CUDA_IMAGE_NAME_WITH_TAG=${GIGL_BASE_CUDA_IMAGE}|" gigl/dep_vars.env
sed -i "s|^DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG=.*|DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG=${GIGL_BASE_CPU_IMAGE}|" gigl/dep_vars.env
sed -i "s|^DOCKER_LATEST_BASE_DATAFLOW_IMAGE_NAME_WITH_TAG=.*|DOCKER_LATEST_BASE_DATAFLOW_IMAGE_NAME_WITH_TAG=${GIGL_BASE_DATAFLOW_IMAGE}|" gigl/dep_vars.env
sed -i "s|^  _BUILDER_IMAGE: .*|  _BUILDER_IMAGE: \"${GIGL_BUILDER_IMAGE}${DEFAULT_PYTHON_SUFFIX}\"|" .github/cloud_builder/run_command_on_active_checkout.yaml
