#!/bin/bash

set -e
set -x

is_running_on_mac() {
    [ "$(uname)" == "Darwin" ]
    return $?
}

has_cuda_driver() {
    # Use the whereis command to locate the CUDA driver
    cuda_location=$(whereis cuda)

    # Check if the cuda_location variable is empty
    if [[ -z "$cuda_location" || "$cuda_location" == "cuda:" ]]; then
        echo "CUDA driver not found."
        return 1
    else
        echo "CUDA driver found at: $cuda_location"
        return 0
    fi
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
    # Resolve this script's directory BEFORE cd'ing, so the patch path survives the cd below.
    GIGL_SCRIPTS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    git clone https://github.com/alibaba/graphlearn-for-pytorch.git \
        && cd graphlearn-for-pytorch \
        && git checkout 88ff111ac0d9e45c6c9d2d18cfc5883dca07e9f9 \
        && git submodule update --init \
        && bash install_dependencies.sh
    # Local patches applied on top of the pinned commit, in order. 0001 replaces the
    # at::_unique distinct-count in InitCPUGraphFromCSR with an exact bitmap count: _unique's
    # sort allocates ~3x the size of `indices` transiently (measured 3.00x at 200M edges), which
    # is ~94 GiB at a 4.2B-edge partition and OOM-kills the graph init on hosts whose budget
    # assumes init holds no copy of the topology. `patch` rather than `git apply` so failure
    # modes are the familiar reject files; `--forward` + explicit failure so a re-run or an
    # upstream change that absorbs the fix stops the build LOUDLY instead of shipping an
    # unpatched wheel that dies 3 hours into a 16-GPU job.
    for glt_patch in "${GIGL_SCRIPTS_DIR}"/patches/*.patch; do
        echo "Applying ${glt_patch}"
        patch -p1 --forward < "${glt_patch}" || { echo "FATAL: ${glt_patch} did not apply"; exit 1; }
    done
    if has_cuda_driver;
    then
        echo "Will use CUDA for GLT..."

        # Potential values for TORCH_CUDA_ARCH_LIST: (not all are tested)
        # 6.0 = Pascal support i.e. Tesla P100 - CUDA 8 or later
        # 6.1 = Pascal support i.e. Tesla P4 - CUDA 8 or later
        # 7.0 = Volta support i.e. Tesla V100 - CUDA 9 or later
        # 7.5 = Turing support i.e. Tesla T4 - CUDA 10 or later
        # 8.0 = Ampere support i.e. A100 - CUDA 11 or later
        # 8.9 = Ada Lovelace support i.e. L4 - CUDA 11.8 or later
        # 9.0 = Hopper support i.e. H100 , H200 - CUDA 12.0 or later
        # 10.0 = Blackwell support i.e. B200 - CUDA 12.6 or later
        # 12.0 = Blackwell support i.e. RTX6000 - CUDA 12.8 or later
        # List of Nvidia GPUS: https://developer.nvidia.com/cuda-gpus
        TORCH_CUDA_ARCH_LIST="7.5" WITH_CUDA="ON" python setup.py bdist_wheel
    else
        echo "Will use CPU for GLT..."
        WITH_CUDA="OFF" python setup.py bdist_wheel
    fi
    uv pip install dist/*.whl \
        && cd .. \
        && rm -rf graphlearn-for-pytorch
    # Applying a patch and SHIPPING it are different guarantees: the loop above proves the source
    # tree changed, this proves the installed .so behaves. A stale build dir or a second wheel on
    # the path would otherwise produce an image that silently OOMs (0001) or rejects the int32
    # topology the trainer builds (0002), hours into a multi-GPU job.
    echo "Verifying the GLT patches took effect in the installed wheel"
    python "${GIGL_SCRIPTS_DIR}/verify_glt_patches.py" \
        || { echo "FATAL: GLT patches did not take effect in the installed wheel"; exit 1; }
else
    echo "Skipping install of GraphLearn-Torch on Mac"
fi
