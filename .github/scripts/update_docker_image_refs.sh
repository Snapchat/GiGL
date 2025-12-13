#!/bin/bash
# Script to update dep_vars.env and cloud builder config with new Docker image references

set -e

echo "Writing new image names to dep_vars.env:"
echo "  DOCKER_LATEST_BASE_CUDA_IMAGE_NAME_WITH_TAG=${GIGL_BASE_CUDA_IMAGE}"
echo "  DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG=${GIGL_BASE_CPU_IMAGE}"
echo "  DOCKER_LATEST_BASE_DATAFLOW_IMAGE_NAME_WITH_TAG=${GIGL_BASE_DATAFLOW_IMAGE}"
echo "  DOCKER_LATEST_BUILDER_IMAGE_NAME_WITH_TAG=${GIGL_BUILDER_IMAGE}"

sed -i "s|^DOCKER_LATEST_BASE_CUDA_IMAGE_NAME_WITH_TAG=.*|DOCKER_LATEST_BASE_CUDA_IMAGE_NAME_WITH_TAG=${GIGL_BASE_CUDA_IMAGE}|" dep_vars.env
sed -i "s|^DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG=.*|DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG=${GIGL_BASE_CPU_IMAGE}|" dep_vars.env
sed -i "s|^DOCKER_LATEST_BASE_DATAFLOW_IMAGE_NAME_WITH_TAG=.*|DOCKER_LATEST_BASE_DATAFLOW_IMAGE_NAME_WITH_TAG=${GIGL_BASE_DATAFLOW_IMAGE}|" dep_vars.env
sed -i "s|name: us-central1-docker\.pkg\.dev.*|name: ${GIGL_BUILDER_IMAGE}|" .github/cloud_builder/run_command_on_active_checkout.yaml
