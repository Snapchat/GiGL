# Installation

## Supported Environments

These are the current environments supported by GiGL

| Python | Mac (Arm64) CPU | Linux CPU | Linux CUDA | PyTorch | PyG |
| ------ | --------------- | --------- | ---------- | ------- | --- |
| 3.9    | Partial Support | Supported | 12.1       | 2.5     | 2.5 |

## Install Prerequisites - setting up your dev machine

Below we provide two ways to bootstrap an environment for using and/or developing GiGL

````{dropdown} (Recommended) Developing/experimenting on a GCP cloud instance.
:color: primary

  We will need to create a GCP instance and setup needed pre-requisites to install and use GiGL.

  You can use our `create_dev_instance.py` script to automatically create an instance for you:
  ```bash
    python <(curl -s https://raw.githubusercontent.com/Snapchat/GiGL/refs/heads/main/scripts/create_dev_instance.py)
  ```
  Next, ssh into your instance. It will most likely ask you to install gpu drivers, follow instructions and do so.
  Once you install the drivers, make sure to restart the instance once you do so to ensure the the ops agent for monitoring is also working. You may also need to navigate to the GCP compute instance UI, and under the `Observability` tab of your instance click
  the "Install OPS Agent" button under the GPU metrics to ensure the GPU metrics are also being reported.

  Once done, ensure you can run multiarch docker builds by running following command:
  ```
  docker buildx create --driver=docker-container --use
  sudo apt-get install qemu-user-static
  docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
  ```

````

````{dropdown} Manual Setup
:color: primary

  1. If on MAC, Install [Homebrew](https://brew.sh/).

  2. Install [Conda](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install):

  3. Install [Docker](https://docs.docker.com/desktop/) and the relevant `buildx` drivers (if using old versions of docker):

      Once installed, ensure you can run multiarch docker builds by running following command:

      Linux:
      ```bash
      docker buildx create --driver=docker-container --use
      sudo apt-get install qemu-user-static
      docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
      ```

      Mac:
      ```bash
      docker buildx create --driver=docker-container --use
      brew install qemu
      docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
      ```

  4. Ensure you have make installed.

      ```bash
      make --version
      ```

      **Install make on Linux:**
      ```
      apt-get update && apt-get upgrade -y && apt-get install -y cmake
      ```

      **Install make on MAC:**
      ```bash
      brew install make
      ```

      Subsequently, you should be able to use `gmake` in all places where we use `make` since brew formula has installed GNU "make" as "gmake".
      See: https://formulae.brew.sh/formula/make


  5. Follow the glcoud cli install [instructions](https://cloud.google.com/sdk/docs/install)

  6. Then, setup your gcloud environment:

    ```bash
    gcloud init # setup glcoud CLI
    gcloud auth application-default login # Auth gcloud cli
    gcloud auth configure-docker us-central1-docker.pkg.dev # Setup docker auth for GiGL images.
    ```
````

## Install GiGL

### Install from source

There are various ways to use GiGL. Firstly, clone the repo locally.

```bash
git clone https://github.com/Snapchat/GiGL.git
```

From the root directory:

```bash
make initialize_environment
conda activate gnn
```

This creates a Python 3.9 environment with some basic utilities. Next, to install all user dependencies. Note: The
command below will try its best ot infer your environment and install necessary reqs i.e. if CUDA is available it will
try to install the necessary gpu deps, otherwise it will install cpu deps.

```bash
make install_deps
```

If you *instead* want to contribute and/or extend GiGL. You can install the developer deps which includes some extra
tooling useful for contributions:

```bash
make install_dev_deps
```

### Install Wheel (In Development)

```{caution}
The instructions below are in development and are not recommended. We are working on making our whls publicly accessible, for the time being you will need to [install from source](#install-from-source).
```

1. Create a python environment

```bash
conda create -y -c conda-forge --name gigl python=3.9
conda activate gigl
```

2. Install GiGL

**Placeholder - `external-snap-ci-github-gigl/gigl/` registry is not made public yet**

We host GiGL on GCP Artifact Registry, thus we need to install Google Artifact Registry Keyring, which will allow you
interact with Python repositories stored in Artifact Registry.

```bash
pip install keyrings.google-artifactregistry-auth
```

Install GiGL + necessary tooling for Torch 2.5 + Cuda12.1

```bash
pip install "gigl[torch25-cuda-121,transform]==0.0.2" \
  --index-url=https://us-central1-python.pkg.dev/external-snap-ci-github-gigl/gigl/simple/ \
  --extra-index-url=https://pypi.org/simple
```

Install GiGL + necessary tooling for Torch 2.5 + CPU

```bash
pip install "gigl[torch25-cpu,transform]==0.0.2" \
  --index-url=https://us-central1-python.pkg.dev/external-snap-ci-github-gigl/gigl/simple/ \
  --extra-index-url=https://pypi.org/simple
```
