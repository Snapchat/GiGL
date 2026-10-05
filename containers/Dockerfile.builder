# syntax=docker/dockerfile:1

# This dockerfile is contains all Dev dependencies, and is used by gcloud
# builders for running tests, et al.

FROM ubuntu:noble-20251001

SHELL ["/bin/bash", "-c"]

# Non-interactive install
ENV DEBIAN_FRONTEND=noninteractive

# Install base dependencies
RUN apt-get update && apt-get install && apt-get install -y \
    curl \
    tar \
    unzip \
    bash \
    openjdk-11-jdk \
    git \
    cmake \
    sudo \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Dec 1, 2025 (svij-sc):
# GCP Cloud build agents run an older version of docker deamon
# with max Docker API version support of 1.41. https://docs.cloud.google.com/build/docs/overview#docker
# At the time of writing Docker Client > v28 has deprecated support for < v1.44.
# https://docs.docker.com/engine/release-notes/29/#breaking-changes
# Thus we use v28.5.2, and also manually set the API version to 1.41 to ensure compatibility.
ENV DOCKER_CLIENT_VERSION=28.5.2
ENV DOCKER_API_VERSION=1.41
RUN curl -fsSL https://get.docker.com -o get-docker.sh && \
    sh get-docker.sh --version ${DOCKER_CLIENT_VERSION} && \
    rm get-docker.sh

# Install Google Cloud CLI
RUN mkdir -p /tools && \
    curl -o /tools/google-cloud-cli-linux-x86_64.tar.gz https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz && \
    tar -xzf /tools/google-cloud-cli-linux-x86_64.tar.gz -C /tools/ && \
    bash /tools/google-cloud-sdk/install.sh --quiet --path-update=true --usage-reporting=false && \
    rm -rf /tools/google-cloud-cli-linux-x86_64.tar.gz

ENV PATH="/tools/google-cloud-sdk/bin:/usr/lib/jvm/java-1.11.0-openjdk-amd64/bin:$PATH"
ENV JAVA_HOME="/usr/lib/jvm/java-1.11.0-openjdk-amd64"

WORKDIR /gigl_deps
# We copy the tools directory from the host machine to the container
# to avoid re-downloading the dependencies as some of them require GCP credentials.
# and, mounting GCP credentials to build time can be a pain and more prone to
# accidental leaking of credentials.
COPY tools tools
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY gigl/dep_vars.env gigl/dep_vars.env
COPY requirements requirements
# Needed to install GLT
COPY gigl/scripts gigl/scripts


# gigl-core is a path dependency in pyproject.toml. uv sync needs its metadata to
# resolve the lockfile. Copying only the build manifest (no C++ sources) so cmake
# configures but compiles nothing — the src Dockerfile installs the real wheel later.
COPY gigl-core/pyproject.toml gigl-core/pyproject.toml
COPY gigl-core/CMakeLists.txt gigl-core/CMakeLists.txt
COPY gigl-core/README.md gigl-core/README.md
# The Python version this image's venv is built on, e.g. 3.13.15. Required.
ARG PYTHON_VERSION
# Set before the install so uv builds the venv on PYTHON_VERSION, and kept in the image so that every later
# uv command stays on that venv instead of resolving .python-version. The Makefile reads it as well.
ENV UV_PYTHON=${PYTHON_VERSION}
# only-managed keeps uv from settling for a system Python that matches the request.
RUN : "${UV_PYTHON:?Pass --build-arg PYTHON_VERSION=<X.Y.Z>}" \
    && UV_PYTHON_PREFERENCE=only-managed bash ./requirements/install_py_deps.sh --dev

# The UV_PROJECT_ENVIRONMENT environment variable can be used to configure the project virtual environment path
# Since the above command should have created the .venv, we activate by default for any future uv commands.
# We also need to set VIRTUAL_ENV so pip envocations can find the virtual environment.
ENV UV_PROJECT_ENVIRONMENT=/gigl_deps/.venv
ENV VIRTUAL_ENV="${UV_PROJECT_ENVIRONMENT}"
# We just created a virtual environment, lets add the bin to the path
ENV PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"
# We also need to make UV detectable by the system
ENV PATH="/root/.local/bin:${PATH}"
RUN bash ./requirements/install_scala_deps.sh

WORKDIR /

CMD [ "/bin/bash" ]
