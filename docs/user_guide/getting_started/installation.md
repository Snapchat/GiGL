# Installation

## Supported Environments

These are the current environments supported by GiGL

| Python      | Mac (Arm64) CPU | Linux CPU | Linux CUDA | PyTorch | PyG |
| ----------- | --------------- | --------- | ---------- | ------- | --- |
| 3.11 – 3.13 | Supported       | Supported | 12.8       | 2.8     | 2.7 |

Python 3.11 is the default, and the full CI suite runs on it. GiGL's Docker images are published for each supported
minor; see [Docker images and Python versions](#docker-images-and-python-versions).

## Available Versions

GiGL is distributed as two wheels that are installed together:

- **`gigl`** — pure Python package (same wheel for CPU and CUDA users)
- **`gigl-core`** — compiled C++/CUDA extensions, ABI-bound to the torch variant

You do not need to install `gigl-core` directly; it is a dependency of `gigl` and is resolved automatically from the
same registry.

Each registry is self-contained — you only need one GCP extra-index URL:

| Variant   | Registry                                                                                                                                                                      |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CPU       | [gigl (CPU registry)](https://console.cloud.google.com/artifacts/python/external-snap-ci-github-gigl/us-central1/gigl/gigl?project=external-snap-ci-github-gigl)              |
| CUDA 12.8 | [gigl-cu128 (CUDA registry)](https://console.cloud.google.com/artifacts/python/external-snap-ci-github-gigl/us-central1/gigl/gigl-cu128?project=external-snap-ci-github-gigl) |

## Install Prerequisites - setting up your dev machine

Below we provide two ways to bootstrap an environment for using and/or developing GiGL

````{dropdown} (Recommended) Developing/experimenting on a GCP cloud instance.
:color: primary

  1. Create dev instance
  We will need to create a GCP instance and setup needed pre-requisites to install and use GiGL.

  You can use our `create_dev_instance.py` script to automatically create an instance for you:
  ```bash
    python <(curl -s https://raw.githubusercontent.com/Snapchat/GiGL/refs/heads/main/scripts/create_dev_instance.py)
  ```
  Next, ssh into your instance. It might ask you to install gpu drivers, follow instructions and do so.

  2. Install some pre-reqs on your instance. The script below tries to automate installation of the following pre-reqs:
  `make, unzip, qemu-user-static, docker, docker-buildx, mamba/conda`
  ```bash
    bash -c "$(curl -s https://raw.githubusercontent.com/Snapchat/GiGL/refs/heads/main/scripts/scripts/startup_dev_instance.sh)"
  ```

  3. Once you are done, make sure to restart the instance. You may also need to navigate to the GCP compute instance UI, and under the `Observability` tab of your instance click the "Install OPS Agent" button under the GPU metrics to ensure the GPU metrics are also being reported.

  Next, Follow instructions to [install GiGL](#install-gigl)
````

````{dropdown} Manual Setup
:color: primary

  1. If on MAC, Install [Homebrew](https://brew.sh/).

  2. Install [Docker](https://docs.docker.com/desktop/) and the relevant `buildx` drivers (if using old versions of docker):

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

### Install Wheel

1. Create a python virtual environment w/ `python>=3.11,<3.14`

2. Install GiGL

#### Install GiGL + necessary tooling for PyG 2.7 + Torch 2.8 on CUDA 12.8

The `gigl-cu128` registry needs Google Cloud credentials; anonymous requests get HTTP 401. Install the Artifact Registry
keyring backend into the same environment, and pip authenticates with your Application Default Credentials
(`gcloud auth application-default login`). See
[Artifact Registry authentication](https://cloud.google.com/artifact-registry/docs/python/authentication).

```bash
pip install keyring keyrings.google-artifactregistry-auth
pip install "gigl[pyg27-torch28-cu128, transform]" \
--extra-index-url=https://us-central1-python.pkg.dev/external-snap-ci-github-gigl/gigl-cu128/simple/ \
--extra-index-url=https://download.pytorch.org/whl/cu128 \
--extra-index-url=https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

#### Install GiGL + necessary tooling for PyG 2.7 + Torch 2.8 on CPU

```bash
pip install "gigl[pyg27-torch28-cpu, transform]" \
--extra-index-url=https://us-central1-python.pkg.dev/external-snap-ci-github-gigl/gigl/simple/ \
--extra-index-url=https://download.pytorch.org/whl/cpu \
--extra-index-url=https://data.pyg.org/whl/torch-2.8.0+cpu.html
```

pip resolves and installs `gigl-core` automatically from the same GCP registry. No separate install step is needed.

Currently, building/using wheels for GLT is error prone, thus we opt to install from source every time. Run post-install
script to setup GLT dependency:

```bash
gigl-post-install
```

### Install from source

```bash
git clone https://github.com/Snapchat/GiGL.git
```

If you are just using (not developing) GiGL, from the root directory:

```bash
make install_deps
```

If you *instead* want to contribute and/or extend GiGL. You can install the developer deps which includes some extra
tooling:

```bash
make install_dev_deps
```

### Docker images and Python versions

GiGL publishes every runtime Docker image and the KFP pipeline once per supported Python minor, with a `-py311`,
`-py312` or `-py313` suffix: for example `src-cpu:<version>-py312` and `gigl-pipeline-<version>-py312.yaml`. There is no
unsuffixed ref. `gigl/dep_vars.env` lists each image and the pipeline once per minor, under keys ending in `_PY311`,
`_PY312` and `_PY313`, and `gigl.common.constants` picks the running interpreter's, so the default images and pipeline
always match the launching interpreter. That matters because Ray requires every node in a cluster to run the same Python
version, and Dataflow requires the worker container's Python minor to match the launching environment's.

Each image runs the exact Python version listed for its minor in the matrix of
`.github/workflows/build-base-docker-images.yml` (currently 3.11.14, 3.12.14 and 3.13.15); the 3.11 entry equals
`.python-version`.

To build your own base images, pass the full Python version as `PYTHON_VERSION`; it is required. The images are built
for `linux/amd64` only, because tensorflow-data-validation publishes no aarch64 Linux wheel.

```bash
for image in cpu cuda dataflow; do
  docker build --platform linux/amd64 -f "containers/Dockerfile.${image}.base" \
    --build-arg PYTHON_VERSION=3.12.14 -t "gigl-${image}-base:dev-py312" .
done
```

Build src images on those bases with `--build-arg BASE_IMAGE=...` to `containers/Dockerfile.src` (CPU and CUDA) or
`containers/Dockerfile.dataflow.src`. `scripts/build_and_push_docker_image.py` instead builds on the published base for
the running interpreter's minor, the `DOCKER_LATEST_BASE_*` key for it in `gigl/dep_vars.env`.
