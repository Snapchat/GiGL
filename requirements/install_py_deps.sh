#!/bin/bash
set -e
set -x

DEV=0  # Flag to install dev dependencies.
PIP_ARGS="--no-deps"  # We don't want to install dependencies when installing packages from hashed requirements files.
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
    --mount-pip-credentials)
        PIP_CREDENTIALS_MOUNTED=1
        shift
        ;;
    esac
done

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

pip install --upgrade pip

if is_running_on_mac;
then
    echo "Setting up Mac CPU environment"
    req_file="requirements/${REQ_FILE_PREFIX}darwin_arm64_requirements_unified.txt"
else
    if has_cuda_driver;
    then
        echo "Setting up Linux CUDA environment"
        req_file="requirements/${REQ_FILE_PREFIX}linux_cuda_requirements_unified.txt"
    else
        echo "Setting up Linux CPU environment"
        req_file="requirements/${REQ_FILE_PREFIX}linux_cpu_requirements_unified.txt"
    fi
fi

echo "Installing from ${req_file}"
pip install -r $req_file $PIP_ARGS

python python/gigl/scripts/post_install.py

conda install --override-channels --channel conda-forge gperftools # tcmalloc, ref: https://google.github.io/tcmalloc/overview.html

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

conda clean -afy
echo "Finished installation"
