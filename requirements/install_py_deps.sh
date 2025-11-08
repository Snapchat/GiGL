#!/bin/bash
set -e
set -x

DEV=0  # Flag to install dev dependencies.
PIP_ARGS="--no-deps"  # We don't want to install dependencies when installing packages from hashed requirements files.
SKIP_GLT_POST_INSTALL=0 # Flag to skip GLT post install.
PIP_CREDENTIALS_MOUNTED=0  # When running this script in Docker environments, we may wish to mount pip credentials to install packages from a private repository.


for arg in "$@"
do
    case $arg in
    --dev)
        DEV=1
        shift
        ;;
    --no-pip-cache)
        PIP_ARGS+=" --no-cache-dir"
        shift
        ;;
    --skip-glt-post-install)
        SKIP_GLT_POST_INSTALL=1
        shift
        ;;
    --mount-pip-credentials)
        PIP_CREDENTIALS_MOUNTED=1
        shift
        ;;
    esac
done

# We use the uv package manager
# Check if uv is already installed
if ! command -v uv &> /dev/null
then
    echo "uv could not be found. Installing uv..."
    EXPECTED_SHA256="8402ab80d2ef54d7044a71ea4e4e1e8db3b20c87c7bffbc30bff59f1e80ebbd5"
    curl -LsSf -o install.sh https://astral.sh/uv/0.9.5/install.sh # Matches the version in .github/actions/setup-python-tools/action.yml

    # Verify SHA256 checksum - script will exit if this fails due to set -e
    if ! echo "$EXPECTED_SHA256  install.sh" | sha256sum -c -; then
        echo "ERROR: SHA256 checksum verification failed for uv installer!" >&2
        rm -f install.sh
        exit 1
    fi

    sh install.sh
    source $HOME/.local/bin/env
fi



REQ_FILE_PREFIX=""
if [[ $DEV -eq 1 ]]
then
  echo "Recognized '--dev' flag is set. Will also install dev dependencies."
  REQ_FILE_PREFIX="dev_"
fi

if [[ $PIP_CREDENTIALS_MOUNTED -eq 1 ]]
then
    echo "Recognized '--mount-pip-credentials' flag is set. Will use the mounted pip credentials (expected at /root/.pip/pip.conf)."
    cp /root/.pip/pip.conf /etc/pip.conf
    echo "Contents of /etc/pip.conf:"
    cat /etc/pip.conf
fi

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

extra_deps=("experimental" "transform")
if is_running_on_mac;
then
    echo "Setting up Mac CPU environment"
    req_file="requirements/${REQ_FILE_PREFIX}darwin_arm64_requirements_unified.txt"
    extra_deps+=("pyg27-torch28-cpu")
else
    if has_cuda_driver;
    then
        echo "Setting up Linux CUDA environment"
        # req_file="requirements/${REQ_FILE_PREFIX}linux_cuda_requirements_unified.txt"
        extra_deps+=("pyg27-torch28-cu128")
    else
        echo "Setting up Linux CPU environment"
        # req_file="requirements/${REQ_FILE_PREFIX}linux_cpu_requirements_unified.txt"
        extra_deps+=("pyg27-torch28-cpu")
    fi
fi

extra_deps_clause=()
for dep in "${extra_deps[@]}"; do
    extra_deps_clause+=(--extra "$dep")
done

flag_use_inexact_match=""
if [[ "${UV_SYSTEM_PYTHON}" == "true" ]]
then
    echo "Recognized using system python."
    echo "Will use inexact match for dependencies so we don't override system packages."
    flag_use_inexact_match="--inexact"
fi

if [[ $DEV -eq 1 ]]
then
    # https://docs.astral.sh/uv/reference/cli/#uv-sync
    uv sync ${extra_deps_clause[@]} --group dev --locked ${flag_use_inexact_match}
else
    uv sync ${extra_deps_clause[@]} --locked ${flag_use_inexact_match}
fi

# echo "Installing from ${req_file}"
# pip install -r $req_file $PIP_ARGS
# source .venv/bin/activate

# Taken from https://stackoverflow.com/questions/59895/how-do-i-get-the-directory-where-a-bash-script-is-located-from-within-the-script
# We do this so if `install_py_deps.sh` is run from a different directory, the script can still find the post_install.py file.
if [[ "${SKIP_GLT_POST_INSTALL}" -eq 0 ]]
then
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
    uv run python $SCRIPT_DIR/../python/gigl/scripts/post_install.py
fi


if [[ $DEV -eq 1 ]]
then
    echo "Setting up required dev tooling"
    # Install tools needed to run spark/scala code
    mkdir -p tools/python_protoc

    echo "Installing tooling for python protobuf"
    # https://github.com/protocolbuffers/protobuf/releases/tag/v3.19.6
    # This version should be smaller than the one used by client i.e. the protobuf client version specified in
    # common-requirements.txt file should be >= v3.19.6
    # TODO (svij-sc): update protoc + protobuff
    if is_running_on_mac;
    then
        curl -L -o tools/python_protoc/python_protoc_3_19_6.zip "https://github.com/protocolbuffers/protobuf/releases/download/v3.19.6/protoc-3.19.6-osx-x86_64.zip"
    else
        curl -L -o tools/python_protoc/python_protoc_3_19_6.zip "https://github.com/protocolbuffers/protobuf/releases/download/v3.19.6/protoc-3.19.6-linux-x86_64.zip"
    fi
    unzip -o tools/python_protoc/python_protoc_3_19_6.zip -d tools/python_protoc
    rm tools/python_protoc/python_protoc_3_19_6.zip

fi

if [[ $PIP_CREDENTIALS_MOUNTED -eq 1 ]]
then
    echo "Removing mounted pip credentials."
    rm /etc/pip.conf
fi

echo "Finished installation"
