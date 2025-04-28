# syntax=docker/dockerfile:1

# This dockerfile is contains all Dev dependencies, and is used by gcloud
# builders for running tests, et al.

FROM continuumio/miniconda3:4.12.0

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

RUN echo 'export PATH="/tools/google-cloud-sdk/bin:/usr/lib/jvm/java-1.11.0-openjdk-amd64/bin:$PATH"' >> /root/.bashrc
RUN echo 'export JAVA_HOME="/usr/lib/jvm/java-1.11.0-openjdk-amd64"' >> /root/.bashrc

# Create the environment:
# TODO: (svij) Build env using single entrypoint `make initialize_environment` for better maintainability
RUN conda create -y --name gigl python=3.9 pip

# Update path so any call for python executables in the built image defaults to using the gnn conda environment
ENV PATH=/opt/conda/envs/gigl/bin:$PATH

RUN conda init bash
RUN echo "conda activate gigl" >> ~/.bashrc

COPY requirements tools/gigl/requirements
RUN source ~/.bashrc && pip install --upgrade pip
RUN source ~/.bashrc && cd tools/gigl && bash ./requirements/install_py_deps.sh --no-pip-cache --dev
# TODO: (svij) Enable install_scala_deps.sh to inside Docker image build
# RUN source ~/.bashrc && cd tools/gigl && bash ./requirements/install_scala_deps.sh

CMD [ "/bin/bash" ]
