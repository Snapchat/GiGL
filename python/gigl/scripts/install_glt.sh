#!/bin/bash

set -e
set -x

is_running_on_mac() {
    [ "$(uname)" == "Darwin" ]
    return $?
}

# Only install GLT if not running on Mac.
if ! is_running_on_mac;
then
    # Without Ninja, we build sequentially which is very slow.
    echo "Installing Ninja as a build backend..."
    # Environments with sudo may require sudo to install ninja-build i.e. certain CI/CD environments.
    # Whereas our docker images do not require sudo neither have it; thus this needs to be conditional.
    if command -v sudo &> /dev/null; then
        sudo apt-get update -y
        sudo apt install ninja-build
    else
        apt-get update -y
        apt install ninja-build
    fi
    echo "Installing GraphLearn-Torch"
    # Occasionally, there is an existing GLT folder, delete it so we can clone.
    rm -rf graphlearn-for-pytorch
    # We upstreamed some bug fixes recently to GLT which have not been released yet.
    # * https://github.com/alibaba/graphlearn-for-pytorch/pull/154
    # * https://github.com/alibaba/graphlearn-for-pytorch/pull/153
    # * https://github.com/alibaba/graphlearn-for-pytorch/pull/151
    # Thus, checking out a specific commit instead of a tagged version.
    git clone https://github.com/alibaba/graphlearn-for-pytorch.git \
        && cd graphlearn-for-pytorch \
        && git checkout 26fe3d4e050b081bc51a79dc9547f244f5d314da \
        && git submodule update --init \
        && bash install_dependencies.sh
    if has_cuda_driver;
    then
        echo "Will use CUDA for GLT..."
        TORCH_CUDA_ARCH_LIST="7.5" WITH_CUDA="ON" python setup.py bdist_wheel
    else
        echo "Will use CPU for GLT..."
        WITH_CUDA="OFF" python setup.py bdist_wheel
    fi
    pip install dist/*.whl \
        && cd .. \
        && rm -rf graphlearn-for-pytorch
else
    echo "Skipping install of GraphLearn-Torch on Mac"
fi
