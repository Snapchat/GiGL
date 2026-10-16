#!/usr/bin/env python3
"""
Once GiGL is installed w/ `pip install gigl`, this script can be executed by running:
`gigl-post-install`

This script is used to install the dependencies for GIGL.
- Currently, it installs GLT: it reuses a cached wheel if one exists, otherwise builds it with install_glt.sh.
"""

import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import torch

from gigl.scripts.glt_wheel_cache import (
    compute_cache_key,
    find_cached_wheel,
    publish_wheel,
    read_patches,
)

_SCRIPTS_DIR = Path(__file__).parent

# Upstream GLT commit we build. We upstreamed some bug fixes recently to GLT which have not been released yet.
# * https://github.com/alibaba/graphlearn-for-pytorch/pull/154
# * https://github.com/alibaba/graphlearn-for-pytorch/pull/153
# * https://github.com/alibaba/graphlearn-for-pytorch/pull/151
# Thus, checking out a specific commit instead of a tagged version.
GLT_COMMIT = "88ff111ac0d9e45c6c9d2d18cfc5883dca07e9f9"
# Potential values for TORCH_CUDA_ARCH_LIST: (not all are tested)
# 6.0 = Pascal support i.e. Tesla P100 - CUDA 8 or later
# 6.1 = Pascal support i.e. Tesla P4 - CUDA 8 or later
# 7.0 = Volta support i.e. Tesla V100 - CUDA 9 or later
# 7.5 = Turing support i.e. Tesla T4 - CUDA 10 or later
# 8.0 = Ampere support i.e. A100 - CUDA 11 or later
# 8.9 = Ada Lovelace support i.e. L4 - CUDA 11.8 or later
# 9.0 = Hopper support i.e. H100 , H200 - CUDA 12.0 or later
# 10.0 = Blackwell support i.e. B200 - CUDA 12.6 or later
# 12.0 = Blackwell support i.e. RTX6000 - CUDA 12.8 or later
# List of Nvidia GPUS: https://developer.nvidia.com/cuda-gpus
GLT_CUDA_ARCH_LIST = "7.5"


def has_cuda() -> bool:
    """Whether the host has a CUDA toolkit; the Python equivalent of the `whereis cuda` check."""
    return shutil.which("nvcc") is not None or Path("/usr/local/cuda").exists()


def install_glt_wheel(
    cache_entry: Path,
    build_wheel: Callable[[Path], None],
    install_wheel: Callable[[Path], None],
    verify: Callable[[], None],
) -> None:
    """Installs a verified GLT wheel, building it only if the cache has no good one.

    On a cache hit the cached wheel is installed and verified; if either step fails, the entry is
    deleted and the wheel is built. After a fresh build, a failure to install or verify is fatal.
    The callables signal failure by raising (e.g. ``subprocess.CalledProcessError``).

    Args:
        cache_entry: Cache entry directory for this wheel's cache key.
        build_wheel: Builds a wheel into the given directory.
        install_wheel: Installs the given wheel.
        verify: Checks that the patches are live in the installed wheel.
    """
    cached_wheel = find_cached_wheel(cache_entry)
    if cached_wheel is not None:
        print(f"GLT wheel cache hit: {cached_wheel}")
        try:
            install_wheel(cached_wheel)
            verify()
            return
        except (subprocess.CalledProcessError, OSError) as e:
            print(
                f"WARNING: cached GLT wheel {cached_wheel} failed to install or verify ({e}); evicting it and rebuilding"
            )
            shutil.rmtree(cache_entry, ignore_errors=True)
    else:
        print(f"GLT wheel cache miss: {cache_entry}")

    with tempfile.TemporaryDirectory() as build_dir:
        build_wheel(Path(build_dir))
        (built_wheel,) = Path(build_dir).glob("*.whl")
        published_wheel = publish_wheel(built_wheel, cache_entry)
    install_wheel(published_wheel)
    verify()


def main():
    """Main entry point for the post-install script."""
    print("Running GIGL post-install script...")
    if platform.system() == "Darwin":
        print("Skipping install of GraphLearn-Torch on Mac")
        return 0

    variant = f"cuda{GLT_CUDA_ARCH_LIST}" if has_cuda() else "cpu"
    cache_key = compute_cache_key(
        commit=GLT_COMMIT,
        patch_contents=read_patches(_SCRIPTS_DIR / "patches"),
        variant=variant,
        torch_version=torch.__version__,
        python_tag=f"cp{sys.version_info.major}{sys.version_info.minor}",
    )
    cache_root = Path(
        os.environ.get("GIGL_GLT_WHEEL_CACHE_DIR")
        or Path.home() / ".cache" / "gigl" / "glt-wheels"
    )

    def build_wheel(output_dir: Path) -> None:
        subprocess.run(
            [
                "bash",
                str(_SCRIPTS_DIR / "install_glt.sh"),
                "--commit", GLT_COMMIT,
                "--cuda-arch-list", GLT_CUDA_ARCH_LIST if variant != "cpu" else "",
                "--output-dir", str(output_dir),
            ],
            check=True,
        )  # fmt: skip

    def install_wheel(wheel: Path) -> None:
        subprocess.run(["uv", "pip", "install", str(wheel)], check=True)

    def verify() -> None:
        # Applying a patch and SHIPPING it are different guarantees: this proves the installed .so
        # behaves, in a fresh process so it imports the wheel just installed.
        print("Verifying the GLT patches took effect in the installed wheel")
        subprocess.run(
            [sys.executable, str(_SCRIPTS_DIR / "verify_glt_patches.py")], check=True
        )

    try:
        install_glt_wheel(cache_root / cache_key, build_wheel, install_wheel, verify)
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"FATAL: GLT install failed: {e}")
        return 1
    print("Post-install script finished running")
    return 0


if __name__ == "__main__":
    sys.exit(main())
