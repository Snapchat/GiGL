"""Pure helpers for the per-machine cache of built GraphLearn-Torch wheels.

Standard library only: this runs from ``post_install.py`` during environment setup, before the
rest of the repo's dependencies are necessarily importable.
"""

import hashlib
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path


def compute_cache_key(
    commit: str,
    patch_contents: Sequence[bytes],
    variant: str,
    torch_version: str,
    python_tag: str,
) -> str:
    """Names the cache entry for the wheel built from the given inputs.

    Every input that changes the built wheel is part of the name.

    Args:
        commit: Pinned upstream GLT commit sha; its first 8 characters are used.
        patch_contents: Contents of the local patches, hashed in the order given.
        variant: Build variant, e.g. ``cpu`` or ``cuda7.5``.
        torch_version: Torch version the wheel links against, e.g. ``2.8.0+cu128``.
        python_tag: Python ABI tag, e.g. ``cp311``.

    Returns:
        The cache entry name, ``glt-<commit8>-p<patches hash>-<variant>-torch<version>-<tag>``.
    """
    patches_hash = hashlib.sha256(b"".join(patch_contents)).hexdigest()[:12]
    return (
        f"glt-{commit[:8]}-p{patches_hash}-{variant}-torch{torch_version}-{python_tag}"
    )


def read_patches(patches_dir: Path) -> list[bytes]:
    """Reads every ``*.patch`` file in a directory.

    Args:
        patches_dir: Directory containing the patch files.

    Returns:
        The file contents, sorted by filename (the order the patches are applied in).
    """
    return [
        path.read_bytes()
        for path in sorted(patches_dir.glob("*.patch"), key=lambda p: p.name)
    ]


def find_cached_wheel(cache_entry: Path) -> Path | None:
    """Finds the wheel in a cache entry.

    Args:
        cache_entry: The cache entry directory; it may not exist.

    Returns:
        The wheel path, or None if there is none. An entry holds exactly one wheel; if it
        somehow holds several, the first by filename is returned.
    """
    wheels = sorted(cache_entry.glob("*.whl"), key=lambda p: p.name)
    return wheels[0] if wheels else None


def publish_wheel(wheel: Path, cache_entry: Path) -> Path:
    """Publishes a wheel as a cache entry, replacing any existing entry.

    The entry is assembled in a temp dir inside the cache root and renamed into place, so a
    concurrent run never sees a half-written entry.

    Args:
        wheel: The built wheel to publish.
        cache_entry: Destination entry directory, ``<cache root>/<cache key>``.

    Returns:
        The path of the wheel inside the cache entry.
    """
    cache_entry.parent.mkdir(parents=True, exist_ok=True)
    tmp_entry = Path(tempfile.mkdtemp(prefix=".tmp.", dir=cache_entry.parent))
    try:
        shutil.copy2(wheel, tmp_entry / wheel.name)
        shutil.rmtree(cache_entry, ignore_errors=True)
        tmp_entry.rename(cache_entry)
    except BaseException:
        shutil.rmtree(tmp_entry, ignore_errors=True)
        raise
    return cache_entry / wheel.name
