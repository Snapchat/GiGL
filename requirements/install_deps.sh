#!/bin/bash
set -e
set -x

# We use the uv package manager for everything
curl -LsSf https://astral.sh/uv/install.sh | sh

# TO DO: Figure out which machine you are on and based off that install the right deps.
# https://docs.astral.sh/uv/reference/cli/#uv-sync
uv sync --extra pyg27-torch28-cpu --group dev
