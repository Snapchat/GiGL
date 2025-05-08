# Installation

There are various ways to use GiGL, depending on your preferred environment. These are the current environments
supported by GiGL

| | Mac (Arm64) | Linux CPU | CUDA 11.8 | CUDA 12.1 | | ------ | ----------------- | ----------------- |
----------------- | ----------------- | | Python | | | | | | 3.9 | Supported | Supported | Supported | Not Yet Supported
| | 3.10 | Not Yet Supported | Not Yet Supported | Not Yet Supported | Not Yet Supported |

The easiest way to set up [gigl](https://pypi.org/project/gigl/) is to install it using pip. However, before installing
the package, make sure you have the following prerequisites:

- PyTorch Version: 2.1.2 (see [PyTorch Installation Docs](https://pytorch.org/get-started/locally/))
- Torchvision Version: 0.16.2
- Torchaudio Version: 2.1.2

To simplify this process, the steps to create a new conda enviornment and install gigl (and its dependencies) are shown
below (seperated by platform/OS).

## Installation Steps

::::{tab-set}

:::{tab-item} ARM Mac

Create the conda environment (python 3.9)

```bash
conda create -y -c conda-force --name ANY_NAME python=3.9 pip-tools
```

Activate the newly created environment:

```bash
conda activate ANY_NAME
```

Install prerequisites

```bash
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 -c pytorch -y
```

Install GiGL

```bash
pip install gigl[torch21-cpu, transform]
```

:::

:::{tab-item} Linux CPU

Create the conda environment (python 3.9)

```bash
conda create -y -c conda-force --name ANY_NAME python=3.9 pip-tools
```

Activate the newly created environment:

```bash
conda activate ANY_NAME
```

Install prerequisites

```bash
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 cpuonly -c pytorch -y
```

Install GiGL

```bash
pip install gigl[torch21-cpu, transform]
```

:::

:::{tab-item} Linux CUDA 11.8

Create the conda environment (python 3.9)

```bash
conda create -y -c conda-forge --name ANY_NAME python=3.9 pip-tools
```

Activate the newly created environment:

```bash
conda activate ANY_NAME
```

Install prerequisites

```bash
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

Install GiGL

```bash
pip install gigl[torch21-cuda-118, transform]
```

:::

:::: :::::
