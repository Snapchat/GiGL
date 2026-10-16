#!/bin/bash

set -e
set -x

# Upstream GLT commit we build. We upstreamed some bug fixes recently to GLT which have not been released yet.
# * https://github.com/alibaba/graphlearn-for-pytorch/pull/154
# * https://github.com/alibaba/graphlearn-for-pytorch/pull/153
# * https://github.com/alibaba/graphlearn-for-pytorch/pull/151
# Thus, checking out a specific commit instead of a tagged version.
GLT_COMMIT="88ff111ac0d9e45c6c9d2d18cfc5883dca07e9f9"
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
GLT_CUDA_ARCH_LIST="7.5"

# Resolve this script's directory BEFORE cd'ing, so the patch path survives the cd below.
GIGL_SCRIPTS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Built wheels are cached per machine so only the first environment compiles GLT.
GLT_WHEEL_CACHE_ROOT="${GIGL_GLT_WHEEL_CACHE_DIR:-$HOME/.cache/gigl/glt-wheels}"

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

# Name of the cache entry for the wheel this script would build. Every input that changes the wheel
# is part of the name: the pinned commit, the contents of the local patches, the CPU/CUDA variant
# (and CUDA arch list), the torch version it links against, and the python ABI tag.
compute_cache_key() {
    local patches_hash variant torch_version python_tag
    patches_hash=$(cat $(LC_ALL=C ls "${GIGL_SCRIPTS_DIR}"/patches/*.patch) | sha256sum | cut -c1-12)
    if has_cuda_driver > /dev/null; then
        variant="cuda${GLT_CUDA_ARCH_LIST}"
    else
        variant="cpu"
    fi
    torch_version=$(python -c 'import torch; print(torch.__version__)')
    python_tag=$(python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
    echo "glt-${GLT_COMMIT:0:8}-p${patches_hash}-${variant}-torch${torch_version}-${python_tag}"
}

# Applying a patch and SHIPPING it are different guarantees: the patch loop proves the source
# tree changed, this proves the installed .so behaves. A stale build dir or a second wheel on
# the path would otherwise produce an image that silently OOMs (bitmap count) or rejects the int32
# topology the trainer builds, hours into a multi-GPU job.
verify_patches() {
    echo "Verifying the GLT patches took effect in the installed wheel"
    python "${GIGL_SCRIPTS_DIR}/verify_glt_patches.py"
}

# Clones GLT, applies the local patches, builds a wheel, and publishes it into the cache entry
# directory $1. The entry is written to a temp dir and renamed into place so a concurrent run
# never sees a half-written entry.
build_wheel_into_cache() {
    local cache_entry="$1"
    # Without Ninja, torch's cpp_extension builds sequentially which is very slow. Ninja comes from the
    # `glt-build-backend` dependency group, whose binary lives in the venv's bin dir.
    export PATH="$(dirname "$(command -v python)"):${PATH}"
    echo "Installing GraphLearn-Torch"
    # Occasionally, there is an existing GLT folder, delete it so we can clone.
    rm -rf graphlearn-for-pytorch
    git clone https://github.com/alibaba/graphlearn-for-pytorch.git \
        && cd graphlearn-for-pytorch \
        && git checkout "${GLT_COMMIT}" \
        && git submodule update --init \
        && bash install_dependencies.sh
    # TODO (dsaini2-sc): once a Snapchat-maintained GLT fork exists, land these patches as
    # commits on the fork, pin to it, and retire this apply loop.
    # Local patches applied on top of the pinned commit, in order. The CSR patch replaces the
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
        TORCH_CUDA_ARCH_LIST="${GLT_CUDA_ARCH_LIST}" WITH_CUDA="ON" python setup.py bdist_wheel
    else
        echo "Will use CPU for GLT..."
        WITH_CUDA="OFF" python setup.py bdist_wheel
    fi
    mkdir -p "${GLT_WHEEL_CACHE_ROOT}"
    local tmp_entry
    tmp_entry=$(mktemp -d "${GLT_WHEEL_CACHE_ROOT}/.tmp.XXXXXX")
    cp dist/*.whl "${tmp_entry}/"
    cd ..
    rm -rf graphlearn-for-pytorch
    rm -rf "${cache_entry}"
    mv "${tmp_entry}" "${cache_entry}" || rm -rf "${tmp_entry}"
}

# Installs the cached wheel in $1 and verifies it. Returns non-zero if either step fails.
install_and_verify_cached_wheel() {
    local cache_entry="$1"
    uv pip install "${cache_entry}"/*.whl || return 1
    verify_patches || return 1
}

main() {
    if [[ "$1" == "--print-cache-key" ]]; then
        compute_cache_key
        return 0
    fi

    # Only install GLT if not running on Mac.
    if is_running_on_mac;
    then
        echo "Skipping install of GraphLearn-Torch on Mac"
        return 0
    fi

    local cache_key cache_entry
    cache_key=$(compute_cache_key)
    cache_entry="${GLT_WHEEL_CACHE_ROOT}/${cache_key}"

    if compgen -G "${cache_entry}/*.whl" > /dev/null; then
        echo "GLT wheel cache hit: ${cache_entry}"
        if install_and_verify_cached_wheel "${cache_entry}"; then
            return 0
        fi
        echo "WARNING: cached GLT wheel ${cache_entry} failed to install or verify; evicting it and rebuilding"
        rm -rf "${cache_entry}"
    else
        echo "GLT wheel cache miss: ${cache_entry}"
    fi

    build_wheel_into_cache "${cache_entry}"
    uv pip install "${cache_entry}"/*.whl
    verify_patches || { echo "FATAL: GLT patches did not take effect in the installed wheel"; exit 1; }
}

# Run only when executed, so the functions above can be sourced.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
