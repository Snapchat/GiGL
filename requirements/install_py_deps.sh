#!/bin/bash
set -e
set -x

DEV=0  # Flag to install dev dependencies.
# Flag to skip installing GiGL lib dependencies, i.e. only dev tools will be installed if DEV=1.
SKIP_GIGL_LIB_DEPS_INSTALL=0
# Flag to skip installing GraphLearn-for-PyTorch (GLT). Note: if SKIP_GIGL_LIB_DEPS_INSTALL=1,
# GLT will never be installed (i.e. SKIP_GLT_INSTALL flag will be treated as set to 1)
SKIP_GLT_INSTALL=0


for arg in "$@"
do
    case $arg in
    --dev)
        DEV=1
        shift
        ;;
    --skip-gigl-lib-deps-install)
        SKIP_GIGL_LIB_DEPS_INSTALL=1
        shift
        ;;
    --skip-glt-install)
        SKIP_GLT_INSTALL=1
        shift
        ;;
    esac
done

### Helper functions ###
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

is_running_on_mac() {
    [ "$(uname)" == "Darwin" ]
    return $?
}

is_running_on_m1_mac() {
    [ "$(uname)" == "Darwin" ] && [ $(uname -m) == 'arm64' ]
    return $?
}

### Installation Functions ###
install_uv_if_needed() {
    # We use the uv package manager
    # Check if uv is already installed
    if ! command -v uv &> /dev/null
    then
        echo "uv could not be found. Installing uv..."
        EXPECTED_SHA256="8402ab80d2ef54d7044a71ea4e4e1e8db3b20c87c7bffbc30bff59f1e80ebbd5"
        curl -LsSf -o uv_installer.sh https://astral.sh/uv/0.9.5/install.sh # Matches the version in .github/actions/setup-python-tools/action.yml

        # Verify SHA256 checksum - script will exit if this fails due to set -e
        if ! echo "$EXPECTED_SHA256  uv_installer.sh" | sha256sum -c -; then
            echo "ERROR: SHA256 checksum verification failed for uv installer!" >&2
            rm -f uv_installer.sh
            exit 1
        fi

        sh uv_installer.sh
        rm -f uv_installer.sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
}

install_dev_tools() {
    echo "Setting up required dev tooling"
    # Install tools needed to run spark/scala code
    mkdir -p tools/python_protoc

    echo "Installing tooling for python protobuf"
    # https://github.com/protocolbuffers/protobuf/releases/tag/v3.19.6
    # This version should be smaller than the one used by client i.e. the protobuf client version specified in
    # pyproject.toml file should be >= v3.19.6
    # TODO (svij-sc): update protoc + protobuff
    if is_running_on_mac;
    then
        curl -L -o tools/python_protoc/python_protoc_3_19_6.zip "https://github.com/protocolbuffers/protobuf/releases/download/v3.19.6/protoc-3.19.6-osx-x86_64.zip"
    else
        curl -L -o tools/python_protoc/python_protoc_3_19_6.zip "https://github.com/protocolbuffers/protobuf/releases/download/v3.19.6/protoc-3.19.6-linux-x86_64.zip"
    fi
    unzip -o tools/python_protoc/python_protoc_3_19_6.zip -d tools/python_protoc
    rm tools/python_protoc/python_protoc_3_19_6.zip

    echo "Finished setting up required dev tooling"
}

install_gigl_lib_deps() {
    echo "Installing GiGL lib"
    extra_deps=("experimental" "transform")
    if is_running_on_mac;
    then
        echo "Setting up Mac CPU environment"
        extra_deps+=("pyg27-torch28-cpu")
    else
        if has_cuda_driver;
        then
            echo "Setting up Linux CUDA environment"
            extra_deps+=("pyg27-torch28-cu128")
        else
            echo "Setting up Linux CPU environment"
            extra_deps+=("pyg27-torch28-cpu")
        fi
    fi

    extra_deps_clause=()
    for dep in "${extra_deps[@]}"; do
        extra_deps_clause+=(--extra "$dep")
    done

    flag_use_inexact_match=""
    # If we are using system python, we want to use inexact match for dependencies so we don't override system packages.
    # We currently, only do this in our Dockerfile.cuda.base, and Dockerfile.dataflow.base images, as they have python
    # pre-installed with relevant packages - and currently reinstalling these in the trivial manner in a new virtual
    # environment is not able to correctly symlink existing optimizations / bootstrap scripts available in the parent
    # docker images of our base images.
    if [[ "${UV_SYSTEM_PYTHON}" == "true" ]]
    then
        echo "Recognized using system python due to UV_SYSTEM_PYTHON = true."
        echo "Will use inexact match for dependencies so we don't override system packages."
        # Syncing is "exact" by default, which means it will remove any packages that are not present in the lockfile.
        # To retain extraneous packages, use the --inexact option:
        # https://docs.astral.sh/uv/concepts/projects/sync/#retaining-extraneous-packages
        # This is useful for example when we might have packages pre-installed i.e. torch, pyg, etc.
        flag_use_inexact_match="--inexact"
    fi

    # https://docs.astral.sh/uv/reference/cli/#uv-sync
    sync_args=(${extra_deps_clause[@]} --locked ${flag_use_inexact_match} --no-install-package gigl-core)
    if [[ $DEV -eq 1 ]]
    then
        sync_args+=(--group dev)
    else
        sync_args+=(--group gigl-core-build-backend --group glt-build-backend)
    fi

    # gigl-core and graphlearn-torch (GLT) both compile against torch, which is a runtime
    # dep of gigl rather than a declared build dep, so neither can build in an isolated
    # build environment (see no-build-isolation-package in pyproject.toml). They also
    # cannot build before torch is in the venv, so the install runs in phases:
    #   1. everything except the two source builds, so torch lands in the venv;
    #   2. gigl-core via `uv pip install --no-build-isolation`, because uv sync may still
    #      isolate workspace members;
    #   3. GLT via a second `uv sync`, which now finds torch in the ambient venv and builds
    #      the commit pinned in pyproject.toml / uv.lock. uv caches the resulting wheel per
    #      GLT commit, so only the first environment on a machine pays for the compile.
    uv sync "${sync_args[@]}" --no-install-package graphlearn-torch
    uv pip install --no-build-isolation ./gigl-core/

    if [[ "${SKIP_GLT_INSTALL}" -eq 0 ]]
    then
        glt_build_env=(WITH_CUDA=OFF)
        if ! is_running_on_mac && has_cuda_driver;
        then
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
            glt_build_env=(WITH_CUDA=ON TORCH_CUDA_ARCH_LIST="7.5")
        fi
        env "${glt_build_env[@]}" uv sync "${sync_args[@]}"
    fi
}

### Main Script ###
main() {
    install_uv_if_needed

    if [[ $DEV -eq 1 ]]
    then
        install_dev_tools
    fi

    if [[ $SKIP_GIGL_LIB_DEPS_INSTALL -eq 0 ]]
    then
        install_gigl_lib_deps
    fi
}

main
echo "Finished installation"
