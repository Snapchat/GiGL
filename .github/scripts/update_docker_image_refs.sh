#!/bin/bash
# Script to update gigl/dep_vars.env with new Docker image references.
# Each GIGL_*_IMAGE input is the tag one build-base-docker-images run shares; the run pushes every image as
# <input>-py<major><minor> for each minor in SUPPORTED_PYTHON_VERSIONS. This script writes each of those refs to its
# _py<major><minor> key in gigl/dep_vars.env.

set -e

# Fails when the key is missing: sed would silently leave a minor without keys unchanged.
set_key() {
    grep -q "^$1=" gigl/dep_vars.env || { echo "gigl/dep_vars.env has no $1" >&2; exit 1; }
    sed -i "s|^$1=.*|$1=$2|" gigl/dep_vars.env
}

VERSIONS="$(grep '^SUPPORTED_PYTHON_VERSIONS=' gigl/dep_vars.env | cut -d= -f2)"
for VERSION in ${VERSIONS//,/ }; do
    TAG="py$(cut -d. -f1,2 <<< "${VERSION}" | tr -d .)"
    echo "Writing the ${TAG} image refs to gigl/dep_vars.env"
    set_key "DOCKER_LATEST_BASE_CUDA_IMAGE_NAME_WITH_TAG_${TAG}" "${GIGL_BASE_CUDA_IMAGE}-${TAG}"
    set_key "DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG_${TAG}" "${GIGL_BASE_CPU_IMAGE}-${TAG}"
    set_key "DOCKER_LATEST_BASE_DATAFLOW_IMAGE_NAME_WITH_TAG_${TAG}" "${GIGL_BASE_DATAFLOW_IMAGE}-${TAG}"
    set_key "DOCKER_LATEST_BUILDER_IMAGE_NAME_WITH_TAG_${TAG}" "${GIGL_BUILDER_IMAGE}-${TAG}"
done
