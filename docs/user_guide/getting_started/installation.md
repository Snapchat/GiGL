# Installation

## Supported Environments

These are the current environments supported by GiGL

| Python | Mac (Arm64) CPU | Linux CPU | Linux CUDA | PyTorch | PyG |
| ------ | --------------- | --------- | ---------- | ------- | --- |
| 3.9    | Partial Support | Supported | 12.1       | 2.5     | 2.5 |

## Available Versions

You can see the available wheels for GiGL [here](https://console.cloud.google.com/artifacts/python/external-snap-ci-github-gigl/us-central1/gigl/gigl?project=external-snap-ci-github-gigl)

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

1. Create a python virtual environment w/ `python==3.11.*`

2. Install GiGL

#### Install GiGL + necessary tooling for PyG 2.7 + Torch 2.8 on Cuda12.8

```bash
pip install "gigl[pyg27-torch28-cu128, transform]==0.1.0" \
--extra-index-url=https://us-central1-python.pkg.dev/external-snap-ci-github-gigl/gigl/simple/ \
--extra-index-url=https://download.pytorch.org/whl/cu128 \
--extra-index-url=https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

Currently, building/using wheels for GLT is error prone, thus we opt to install from source every time. Run post-install
script to setup GLT dependency:

```bash
gigl-post-install
```

#### Install GiGL + necessary tooling for PyG 2.7 + Torch 2.8 on CPU

```bash
pip install "gigl[pyg27-torch28-cpu, transform]==0.1.0" \
--extra-index-url=https://us-central1-python.pkg.dev/external-snap-ci-github-gigl/gigl/simple/ \
--extra-index-url=https://download.pytorch.org/whl/cpu \
--extra-index-url=https://data.pyg.org/whl/torch-2.8.0+cpu.html
```

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
