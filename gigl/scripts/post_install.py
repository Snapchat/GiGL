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
import sysconfig
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Final

import torch
import torch.version

from gigl.scripts.glt_wheel_cache import (
    GLT_COMMIT,
    cache_entry_lock,
    compute_cache_key,
    describe_wheel,
    find_cached_wheel,
    publish_wheel,
    read_patches,
)

_SCRIPTS_DIR: Final = Path(__file__).parent

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
GLT_CUDA_ARCH_LIST: Final = "7.5"


def cache_root_from_environment() -> Path | None:
    """Returns the wheel cache root, or None when the cache is disabled.

    The cache only pays off on a developer machine that creates several environments. It is off
    when ``GIGL_GLT_WHEEL_CACHE`` is ``0``, when ``GIGL_GLT_WHEEL_CACHE_DIR`` is set but empty, when
    ``CI`` is set, or when running as root (image builds, CI runners), so no wheel is ever written
    into an image layer or a shared runner's home directory.

    Example:
        >>> os.environ["GIGL_GLT_WHEEL_CACHE"] = "0"
        >>> cache_root_from_environment() is None
        True

    Returns:
        The cache root directory, ``GIGL_GLT_WHEEL_CACHE_DIR`` or ``~/.cache/gigl/glt-wheels``, or
        None when the cache is disabled.
    """
    if os.environ.get("GIGL_GLT_WHEEL_CACHE") == "0":
        print("GLT wheel cache disabled: GIGL_GLT_WHEEL_CACHE=0")
        return None
    cache_dir = os.environ.get("GIGL_GLT_WHEEL_CACHE_DIR")
    if cache_dir == "":
        print("GLT wheel cache disabled: GIGL_GLT_WHEEL_CACHE_DIR is empty")
        return None
    if "CI" in os.environ:
        print("GLT wheel cache disabled: CI is set")
        return None
    if os.geteuid() == 0:
        print("GLT wheel cache disabled: running as root")
        return None
    return (
        Path(cache_dir) if cache_dir else Path.home() / ".cache" / "gigl" / "glt-wheels"
    )


def install_glt_wheel(
    cache_entry: Path | None,
    build_wheel: Callable[[Path], None],
    install_wheel: Callable[[Path], None],
    verify: Callable[[], None],
) -> None:
    """Installs a verified GLT wheel, building it only if the cache has no good one.

    On a cache hit the cached wheel is installed and verified; if either step fails, the entry is
    deleted and the wheel is built. After a fresh build, a failure to install or verify is fatal.
    The callables signal a bad wheel by raising ``subprocess.CalledProcessError``; any other
    exception propagates and leaves the entry in place.

    With ``cache_entry`` None the cache is disabled: the wheel is built, installed and verified
    from a temp dir and nothing is published.

    The caller must hold ``cache_entry_lock(cache_entry)`` so the check, eviction, build and publish
    of one key are serialized across processes.

    Example:
        >>> entry = Path("/tmp/cache/glt-key")
        >>> with cache_entry_lock(entry):
        ...     install_glt_wheel(entry, build_wheel, install_wheel, verify)
        GLT wheel cache miss: /tmp/cache/glt-key

    Args:
        cache_entry: Cache entry directory for this wheel's cache key, or None to bypass the cache.
        build_wheel: Builds a wheel into the given directory.
        install_wheel: Installs the given wheel.
        verify: Checks that the patches are live in the installed wheel.
    """
    if cache_entry is None:
        with tempfile.TemporaryDirectory() as build_dir:
            build_wheel(Path(build_dir))
            (built_wheel,) = Path(build_dir).glob("*.whl")
            print(f"Built GLT wheel: {describe_wheel(built_wheel)}")
            install_wheel(built_wheel)
        verify()
        return

    cached_wheel = find_cached_wheel(cache_entry)
    if cached_wheel is not None:
        print(f"GLT wheel cache hit: {describe_wheel(cached_wheel)}")
        try:
            install_wheel(cached_wheel)
            verify()
            return
        except subprocess.CalledProcessError as e:
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
    print(f"Published GLT wheel to cache: {describe_wheel(published_wheel)}")
    install_wheel(published_wheel)
    verify()


def main():
    """Main entry point for the post-install script."""
    print("Running GIGL post-install script...")
    if platform.system() == "Darwin":
        print("Skipping install of GraphLearn-Torch on Mac")
        return 0

    # install_py_deps.sh's CUDA check picks the torch extra, and GLT must match that torch build,
    # so the installed torch is the single source of truth for the variant.
    variant = f"cuda{GLT_CUDA_ARCH_LIST}" if torch.version.cuda is not None else "cpu"
    cache_key = compute_cache_key(
        commit=GLT_COMMIT,
        patch_contents=read_patches(_SCRIPTS_DIR / "patches"),
        build_script=(_SCRIPTS_DIR / "install_glt.sh").read_bytes(),
        variant=variant,
        torch_version=torch.__version__,
        python_tag=f"cp{sys.version_info.major}{sys.version_info.minor}",
        platform_tag=sysconfig.get_platform(),
    )
    cache_root = cache_root_from_environment()

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
            # ninja lives next to the interpreter; install_glt.sh must find it even if the venv is not activated.
            env={**os.environ, "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"},
        )  # fmt: skip

    def install_wheel(wheel: Path) -> None:
        # pip is a declared runtime dependency of gigl, so it is present wherever gigl is, which uv is
        # not (the documented path is `pip install gigl` then `gigl-post-install`). Installing through
        # sys.executable targets the interpreter running this script. --no-deps keeps pip from resolving
        # the wheel's torch requirement against PyPI; --force-reinstall makes a cache hit reinstall even
        # when the same GLT version is already present.
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "--force-reinstall", "--no-deps", "--disable-pip-version-check", str(wheel),
            ],
            check=True,
        )  # fmt: skip

    def verify() -> None:
        # Applying a patch and SHIPPING it are different guarantees: this proves the installed .so
        # behaves, in a fresh process so it imports the wheel just installed.
        print("Verifying the GLT patches took effect in the installed wheel")
        subprocess.run(
            [sys.executable, str(_SCRIPTS_DIR / "verify_glt_patches.py")], check=True
        )

    try:
        if cache_root is None:
            install_glt_wheel(None, build_wheel, install_wheel, verify)
        else:
            cache_entry = cache_root / cache_key
            with cache_entry_lock(cache_entry):
                install_glt_wheel(cache_entry, build_wheel, install_wheel, verify)
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"FATAL: GLT install failed: {e}")
        return 1
    print("Post-install script finished running")
    return 0


if __name__ == "__main__":
    sys.exit(main())
