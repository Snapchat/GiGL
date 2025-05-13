# Installation

We are working on making our whls publicly accessible, for the time being you will need to install from source.

There are various ways to use GiGL. The recommended solution is to set up a conda environment and use some handy
commands:

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

## GCP Installation

If you are using a GCP cloud instance, you can also leverage our handy script `scripts/setup_environment.sh` to help you
bootstrap your instance. i.e. first you can clone the repo to your instance then run:

```
bash scripts/setup_environment.sh
```

## Supported Environments

These are the current environments supported by GiGL

| Python | Mac (Arm64) CPU | Linux CPU | Linux CUDA | PyTorch | PyG |
| ------ | --------------- | --------- | ---------- | ------- | --- |
| 3.9    | Partial Support | Supported | 12.1       | 2.5     | 2.5 |
