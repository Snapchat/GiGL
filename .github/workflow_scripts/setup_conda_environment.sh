#!/bin/bash

# Ensure the script fails on error
set -e

# Variables (Consider passing these as arguments to the script instead of hardcoding)
CONDA_ENV_NAME=gnn
PYTHON_VERSION=3.8

echo "Initializing environment and installing dependencies"

if [ ! -d "$HOME/miniconda" ]; then
  echo "Miniconda not found, installing..."
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $HOME/miniconda.sh
  bash $HOME/miniconda.sh -b -p $HOME/miniconda
else
  echo "Miniconda already installed"
fi

# Initialize conda in script
source $HOME/miniconda/etc/profile.d/conda.sh
conda init bash

# Create the conda environment
conda create -y -c conda-forge --name ${CONDA_ENV_NAME} python=${PYTHON_VERSION} pip-tools

# Reactivate conda.sh to ensure the conda activate command works
source $HOME/miniconda/etc/profile.d/conda.sh
conda activate ${CONDA_ENV_NAME}

echo "${CONDA_ENV_NAME} environment is ready and activated."