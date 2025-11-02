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

RUN curl -fsSL https://get.docker.com -o get-docker.sh && \
    sh get-docker.sh && \
    rm get-docker.sh

# Install Google Cloud CLI
RUN mkdir -p /tools && \
    curl -o /tools/google-cloud-cli-linux-x86_64.tar.gz https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz && \
    tar -xzf /tools/google-cloud-cli-linux-x86_64.tar.gz -C /tools/ && \
    bash /tools/google-cloud-sdk/install.sh --quiet --path-update=true --usage-reporting=false && \
    rm -rf /tools/google-cloud-cli-linux-x86_64.tar.gz

ENV PATH="/tools/google-cloud-sdk/bin:/usr/lib/jvm/java-1.11.0-openjdk-amd64/bin:$PATH"
ENV JAVA_HOME="/usr/lib/jvm/java-1.11.0-openjdk-amd64"

# We copy the tools directory from the host machine to the container
# to avoid re-downloading the dependencies as some of them require GCP credentials.
# and, mounting GCP credentials to build time can be a pain and more prone to
# accidental leaking of credentials.
COPY tools gigl_deps/tools
COPY pyproject.toml gigl_deps/pyproject.toml
COPY uv.lock gigl_deps/uv.lock
COPY dep_vars.env gigl_deps/dep_vars.env
COPY requirements gigl_deps/requirements
# Needed to install glt dependencies - which is done.
COPY python/gigl/scripts gigl_deps/python/gigl/scripts


COPY .python-version tmp/.python-version
# ENV UV_SYSTEM_PYTHON=1
RUN cd gigl_deps && bash ./requirements/install_py_deps.sh --no-pip-cache --dev

# The UV_PROJECT_ENVIRONMENT environment variable can be used to configure the project virtual environment path
# Since the above command should have created the .venv, we activate by default for any future uv commands.
# We also need to set VIRTUAL_ENV so pip envocations can find the virtual environment.
ENV UV_PROJECT_ENVIRONMENT=/gigl_deps/.venv
ENV VIRTUAL_ENV="${UV_PROJECT_ENVIRONMENT}"
# We just created a virtual environment, lets add the bin to the path
ENV PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"
# We also need to make UV detectable by the system
ENV PATH="/root/.local/bin:${PATH}"
RUN uv tools install pip==25.3

RUN cd gigl_deps && bash ./requirements/install_scala_deps.sh

CMD [ "/bin/bash" ]
