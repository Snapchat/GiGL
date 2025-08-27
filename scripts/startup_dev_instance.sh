#!/usr/bin/bash

# This script is to be used to setup pre-reqs for a new dev instance created w/ `scripts/create_dev_instance.py`

set -e
set -x

username=$(whoami)
sa_account=$(gcloud config list account --format "value(core.account)")

MAMBA_RELEASE="25.3.1-0" # https://github.com/conda-forge/miniforge/releases/tag/25.3.1-0

has_mamba_installed() {
    if ! mamba --version > /dev/null 2>&1; then
        echo "Mamba not found, will try again, trying to see if it is installed but not initialized"
        if ! source "/opt/conda/etc/profile.d/conda.sh" || ! source "/opt/conda/etc/profile.d/mamba.sh"; then
            echo "Failed to find mamba"
            return 1
        fi
        if ! mamba --version > /dev/null 2>&1; then
            echo "WARNING: Failed to initialize mamba correctly. Script may fail."
            return 1
        fi
    fi
    echo "Mamba found in the current shell"
    return 0
}

# [SETUP GCOMMON DEV TOOLS]
sudo apt-get update

# Install essential development tools and utilities:
# build-essential: GNU C/C++ compiler (gcc/g++)
# make: Powerful automation tool that reads 'Makefile' files to manage and compile software projects.
# unzip: Simple but necessary utility for decompressing files from the common .zip archive format.
# podman: Alternate to docker, daemonless container engine that allows you to build, run, and manage OCI-compatible containers.
sudo apt-get install -y build-essential make unzip podman

# We still install docker for backwards compatibility of older projects
if ! command -v docker >/dev/null 2>&1; then
    if ! grep -qi 'ubuntu' /etc/os-release; then
        echo "ERROR: Docker not installed, and running on non-ubuntu OS. Please install docker manually."
        echo "Will exit now."
        exit 1
    fi

    # Docker install instructions: https://docs.docker.com/engine/install/ubuntu/
    # Uninstall all conflicting packages, if they exist
    for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do
        sudo apt-get remove $pkg;
    done

    sudo apt-get install ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc

    # Add the repository to Apt sources:
    echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update

    # Install the Docker packages.
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

fi

# Permission for docker.sock
sudo chmod 777 /var/run/docker.sock
# Ensure we can run multi-arch docker builds
docker buildx create --driver=docker-container --use
sudo apt-get install -y qemu-user-static
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes

# Opinionated decision to install zsh as well - its better than bash
if ! command -v zsh >/dev/null 2>&1; then
    echo "Zsh is not installed. Installing zsh..."
    sudo apt install -y zsh
    echo "Installing oh-my-zsh: https://ohmyz.sh/#install"
    # RUNZSH=no → prevents the script from launching zsh right after install
    # CHSH=yes → changes default shell to zsh
    RUNZSH=no CHSH=yes sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

    # Ensure .zprofile is sourced in interactive shells
    # This is for convenience, as if you ever install tools like coursier, they will write to your non-interactive shells
    # files like .zprofile, so you will have to restart your ssh session to ensure the environment variables are loaded
    # vs. now you would just need to create a new shell.
    grep -q 'source .*\.zprofile' ~/.zshrc || \
    echo '[[ -e ~/.zprofile ]] && source ~/.zprofile' | cat - ~/.zshrc > /tmp/zshrc && mv /tmp/zshrc ~/.zshrc
fi
# ========== [SETUP COMMON DEV TOOLS]


# [SETUP PYTHON ENV]
if ! has_mamba_installed; then
    echo "Mamba is not installed. Installing miniforge..."
    wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/download/${MAMBA_RELEASE}/Miniforge3-$(uname)-$(uname -m).sh"
    sudo bash Miniforge3.sh -b -p "/opt/conda" # This is the default place where conda-forge enabled GCP images install conda/mamba, etc

    # Source mamba and conda so we can use it in the current shell
    source "/opt/conda/etc/profile.d/conda.sh"
    source "/opt/conda/etc/profile.d/mamba.sh" # Mamba support

    # Initialize conda and mamba for zsh, and bash
    # We initialize conda for backwards compatibility
    conda init bash
    echo 'eval "$(mamba shell hook --shell bash)"' >> ~/.bashrc
    conda init zsh
    echo 'eval "$(mamba shell hook --shell zsh)"' >> ~/.zshrc
fi

# Allow all users to read/write to conda dir to manage environments
sudo chmod -R 755 /opt/conda/
sudo chmod 777 /opt/conda/pkgs/
sudo chmod 777 /opt/conda/envs/

echo "\n=========================\n"
echo "Done! Please restart your machine to ensure setup is complete"
