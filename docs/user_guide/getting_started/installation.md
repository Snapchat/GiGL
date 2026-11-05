# Installation

## Supported Environments

These are the current environments supported by GiGL

| Python | Mac (Arm64) CPU | Linux CPU | Linux CUDA | PyTorch | PyG |
| ------ | --------------- | --------- | ---------- | ------- | --- |
| 3.11   | Supported       | Supported | 12.8       | 2.8     | 2.7 |

## Available Versions

GiGL is distributed as two wheels that are installed together:

- **`gigl`** — pure Python package (same wheel for CPU and CUDA users)
- **`gigl-core`** — compiled C++/CUDA extensions, ABI-bound to the torch variant

`gigl` pins the matching `gigl-core` version. Both wheels are published to one registry per torch variant, and each
registry holds everything GiGL itself needs:

| Variant   | Package index URL                                                                    | Registry                                                                                                                                                                      |
| --------- | ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CPU       | `https://us-central1-python.pkg.dev/external-snap-ci-github-gigl/gigl/simple/`       | [gigl (CPU registry)](https://console.cloud.google.com/artifacts/python/external-snap-ci-github-gigl/us-central1/gigl/gigl?project=external-snap-ci-github-gigl)              |
| CUDA 12.8 | `https://us-central1-python.pkg.dev/external-snap-ci-github-gigl/gigl-cu128/simple/` | [gigl-cu128 (CUDA registry)](https://console.cloud.google.com/artifacts/python/external-snap-ci-github-gigl/us-central1/gigl-cu128/gigl?project=external-snap-ci-github-gigl) |

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

GiGL and its torch and PyG dependencies come from four package indexes:

- the GiGL registry for your variant (see [Available Versions](#available-versions))
- the PyTorch index
- the PyG wheel page
- PyPI, for everything else

We recommend [uv](https://docs.astral.sh/uv/getting-started/installation/), which can pin each package to the index it
must come from.

```{note}
Avoid `pip install --extra-index-url` for GiGL. pip picks the highest version of each package across every index it is
given, so a package published to PyPI under the name `gigl-core` would win over the one in the GiGL registry.
```

1. Create a project directory with the `pyproject.toml` below.

   It declares one extra per variant. Change `name` and add your own dependencies to `dependencies`.

   `gigl-core` is listed next to `gigl` because uv applies `[tool.uv.sources]` only to direct dependencies. Without it,
   uv would look for `gigl-core` on PyPI.

   ```toml
   [project]
   name = "my-gigl-project"
   version = "0.0.1"
   requires-python = "==3.11.*"
   dependencies = [
     # ninja is needed to build GraphLearn-Torch in `gigl-post-install`.
     "ninja",
   ]

   [project.optional-dependencies]
   pyg27-torch28-cpu = [
     "gigl[pyg27-torch28-cpu, transform, experimental]==0.3.1",
     "gigl-core", # gigl pins the matching version.
     "torch",
     "torchrec",
     "fbgemm-gpu",
     "pyg_lib",
     "torch_scatter",
     "torch_sparse",
     "torch_cluster",
     "torch_spline_conv",
   ]
   pyg27-torch28-cu128 = [
     "gigl[pyg27-torch28-cu128, transform, experimental]==0.3.1",
     "gigl-core", # gigl pins the matching version.
     "torch",
     "torchrec",
     "fbgemm-gpu",
     "pyg_lib",
     "torch_scatter",
     "torch_sparse",
     "torch_cluster",
     "torch_spline_conv",
   ]

   [tool.uv]
   # The CPU and CUDA extras pull different torch builds, so they are resolved separately.
   conflicts = [
     [{ extra = "pyg27-torch28-cpu" }, { extra = "pyg27-torch28-cu128" }],
   ]
   required-environments = [
     "sys_platform == 'linux' and platform_machine == 'x86_64'",
   ]

   # PyPI stays the default index. Every index below is `explicit`, so uv only uses it for
   # the packages mapped to it in [tool.uv.sources].
   [[tool.uv.index]]
   name = "torch-cpu"
   url = "https://download.pytorch.org/whl/cpu"
   explicit = true

   [[tool.uv.index]]
   name = "torch-cu128"
   url = "https://download.pytorch.org/whl/cu128"
   explicit = true

   # PyG publishes a flat HTML page of wheels, not a PEP 503 index.
   [[tool.uv.index]]
   name = "pyg-torch28-cpu"
   url = "https://data.pyg.org/whl/torch-2.8.0+cpu.html"
   explicit = true
   format = "flat"

   [[tool.uv.index]]
   name = "pyg-torch28-cu128"
   url = "https://data.pyg.org/whl/torch-2.8.0+cu128.html"
   explicit = true
   format = "flat"

   [[tool.uv.index]]
   name = "gigl-cpu"
   url = "https://us-central1-python.pkg.dev/external-snap-ci-github-gigl/gigl/simple/"
   explicit = true

   [[tool.uv.index]]
   name = "gigl-cu128"
   url = "https://us-central1-python.pkg.dev/external-snap-ci-github-gigl/gigl-cu128/simple/"
   explicit = true

   [tool.uv.sources]
   torch = [
     { extra = "pyg27-torch28-cpu", index = "torch-cpu" },
     { extra = "pyg27-torch28-cu128", index = "torch-cu128" },
   ]
   torchrec = [
     { extra = "pyg27-torch28-cpu", index = "torch-cpu" },
     { extra = "pyg27-torch28-cu128", index = "torch-cu128" },
   ]
   fbgemm-gpu = [
     { extra = "pyg27-torch28-cpu", index = "torch-cpu" },
     { extra = "pyg27-torch28-cu128", index = "torch-cu128" },
   ]
   pyg_lib = [
     { extra = "pyg27-torch28-cpu", index = "pyg-torch28-cpu" },
     { extra = "pyg27-torch28-cu128", index = "pyg-torch28-cu128" },
   ]
   torch_scatter = [
     { extra = "pyg27-torch28-cpu", index = "pyg-torch28-cpu" },
     { extra = "pyg27-torch28-cu128", index = "pyg-torch28-cu128" },
   ]
   torch_sparse = [
     { extra = "pyg27-torch28-cpu", index = "pyg-torch28-cpu" },
     { extra = "pyg27-torch28-cu128", index = "pyg-torch28-cu128" },
   ]
   torch_cluster = [
     { extra = "pyg27-torch28-cpu", index = "pyg-torch28-cpu" },
     { extra = "pyg27-torch28-cu128", index = "pyg-torch28-cu128" },
   ]
   torch_spline_conv = [
     { extra = "pyg27-torch28-cpu", index = "pyg-torch28-cpu" },
     { extra = "pyg27-torch28-cu128", index = "pyg-torch28-cu128" },
   ]
   gigl = [
     { extra = "pyg27-torch28-cpu", index = "gigl-cpu" },
     { extra = "pyg27-torch28-cu128", index = "gigl-cu128" },
   ]
   gigl-core = [
     { extra = "pyg27-torch28-cpu", index = "gigl-cpu" },
     { extra = "pyg27-torch28-cu128", index = "gigl-cu128" },
   ]
   ```

2. Install the extra for your variant.

   CUDA 12.8:

   ```bash
   uv sync --extra pyg27-torch28-cu128
   ```

   CPU:

   ```bash
   uv sync --extra pyg27-torch28-cpu
   ```

3. Build GraphLearn-Torch (GLT).

   GLT is not on PyPI; the post-install script builds a patched wheel from source and caches it per machine under
   `~/.cache/gigl/glt-wheels/`, so only the first environment on a machine compiles it. The build uses the `ninja`
   declared in `dependencies` above.

   ```bash
   uv run gigl-post-install
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
