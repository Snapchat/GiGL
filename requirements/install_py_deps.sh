#!/bin/bash
set -e
set -x

DEV=0  # Flag to install dev dependencies.
# Flag to skip installing GiGL lib dependencies, i.e. only dev tools will be installed if DEV=1.
SKIP_GIGL_LIB_DEPS_INSTALL=0
# Flag to skip GLT post install. Note: if SKIP_GIGL_LIB_DEPS_INSTALL=1,
# GLT will never be installed (i.e. SKIP_GLT_POST_INSTALL flag will be treated as set to 1)
SKIP_GLT_POST_INSTALL=0


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
    --skip-glt-post-install)
        SKIP_GLT_POST_INSTALL=1
        shift
        ;;
    esac
done

# Directory the pinned uv is installed into; every uv call below goes through ${UV}.
export UV_INSTALL_DIR="${HOME}/.local/bin"
UV="${UV_INSTALL_DIR}/uv"

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
install_uv() {
    # We use the uv package manager. The pinned version is installed at UV_INSTALL_DIR and called from
    # there, so a uv that a base image or the host puts on PATH never builds the venv. The download is
    # skipped only when that path already holds the pinned version (e.g. a second run on a dev host).
    UV_VERSION="0.12.9" # Matches the version in .github/actions/setup-python-tools/action.yml
    if [[ -x "${UV}" ]] && [[ "$("${UV}" --version)" == "uv ${UV_VERSION} "* ]]; then
        echo "uv ${UV_VERSION} is already installed at ${UV}"
        export PATH="$UV_INSTALL_DIR:$PATH"
        return 0
    fi
    # The installer places uv and uvx directly in UV_INSTALL_DIR.
    EXPECTED_SHA256="222e006c0fe4a0d793031833e469b21df72311f4e3526ffecca0e19e6dfabc32"
    curl -LsSf -o uv_installer.sh "https://astral.sh/uv/${UV_VERSION}/install.sh"

    # Verify SHA256 checksum - script will exit if this fails due to set -e
    if ! echo "$EXPECTED_SHA256  uv_installer.sh" | sha256sum -c -; then
        echo "ERROR: SHA256 checksum verification failed for uv installer!" >&2
        rm -f uv_installer.sh
        exit 1
    fi

    sh uv_installer.sh
    rm -f uv_installer.sh
    export PATH="$UV_INSTALL_DIR:$PATH"
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

    # gigl-core's CMake build requires torch to be present during the build, but
    # torch is a runtime dep of gigl, not a declared build dep of gigl-core.
    # uv sync may ignore no-build-isolation-package for workspace members and always
    # builds them in an isolated environment where torch is absent.
    # Workaround: phase 1 installs everything except gigl-core (so torch lands in
    # the venv); phase 2 uses `uv pip install --no-build-isolation` which explicitly
    # skips isolation and lets CMake find torch from the ambient venv.
    if [[ $DEV -eq 1 ]]
    then
        # https://docs.astral.sh/uv/reference/cli/#uv-sync
        "${UV}" sync ${extra_deps_clause[@]} --group dev --locked --no-install-package gigl-core
    else
        "${UV}" sync ${extra_deps_clause[@]} --group gigl-core-build-backend --group glt-build-backend --locked --no-install-package gigl-core
    fi
    "${UV}" pip install --no-build-isolation ./gigl-core/

    # Taken from https://stackoverflow.com/questions/59895/how-do-i-get-the-directory-where-a-bash-script-is-located-from-within-the-script
    # We do this so if `install_py_deps.sh` is run from a different directory, the script can still find the post_install.py file.
    if [[ "${SKIP_GLT_POST_INSTALL}" -eq 0 ]]
    then
        SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
        "${UV}" run python $SCRIPT_DIR/../gigl/scripts/post_install.py
    fi
}

### Main Script ###
main() {
    install_uv

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
