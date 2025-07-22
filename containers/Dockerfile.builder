# syntax=docker/dockerfile:1

# This dockerfile is contains all Dev dependencies, and is used by gcloud
# builders for running tests, et al.

FROM condaforge/miniforge3:25.3.0-1

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

# Create the environment:
# TODO: (svij) Build env using single entrypoint `make initialize_environment` for better maintainability
RUN conda create -y -c conda-forge --name gigl python=3.9 pip

# Update path so any call for python executables in the built image defaults to using the gnn conda environment
ENV PATH=/opt/conda/envs/gigl/bin:$PATH
# For debugging purposes, we also initialize respective conda env in bashrc
RUN conda init bash
RUN echo "conda activate gigl" >> ~/.bashrc

# We copy the tools directory from the host machine to the container
# to avoid re-downloading the dependencies as some of them require GCP credentials.
# and, mounting GCP credentials to build time can be a pain and more prone to
# accidental leaking of credentials.
COPY tools tools/gigl/tools
COPY requirements tools/gigl/requirements

RUN pip install --upgrade pip
RUN cd tools/gigl && bash ./requirements/install_py_deps.sh --no-pip-cache --dev
RUN cd tools/gigl && bash ./requirements/install_scala_deps.sh

CMD [ "/bin/bash" ]
