#!/bin/bash
# Prints the builder image ref for a Python minor, read from gigl/dep_vars.env in the current directory.
# Fails if the key is missing or empty.
#
# Usage: builder_image.sh [<major.minor>]
# With no argument the minor comes from .python-version in the current directory.
#   builder_image.sh      with .python-version 3.11.14 reads DOCKER_LATEST_BUILDER_IMAGE_NAME_WITH_TAG_py311
#   builder_image.sh 3.13 reads DOCKER_LATEST_BUILDER_IMAGE_NAME_WITH_TAG_py313

set -euo pipefail

MINOR="${1:-$(cut -d. -f1,2 .python-version)}"
KEY="DOCKER_LATEST_BUILDER_IMAGE_NAME_WITH_TAG_py${MINOR//./}"

REF="$(grep "^${KEY}=" gigl/dep_vars.env | cut -d= -f2- || true)"
if [[ -z "${REF}" ]]; then
  echo "Error: ${KEY} is missing or empty in gigl/dep_vars.env" >&2
  exit 1
fi
echo "${REF}"
