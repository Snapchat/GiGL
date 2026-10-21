"""Pure helpers for the per-machine cache of built GraphLearn-Torch wheels.

Standard library only: this runs from ``post_install.py`` during environment setup, before the
rest of the repo's dependencies are necessarily importable.
"""

import fcntl
import hashlib
import shutil
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

# Upstream GLT commit we build. We upstreamed some bug fixes recently to GLT which have not been released yet.
# * https://github.com/alibaba/graphlearn-for-pytorch/pull/154
# * https://github.com/alibaba/graphlearn-for-pytorch/pull/153
# * https://github.com/alibaba/graphlearn-for-pytorch/pull/151
# Thus, checking out a specific commit instead of a tagged version.
GLT_COMMIT = "88ff111ac0d9e45c6c9d2d18cfc5883dca07e9f9"


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

    Raises:
        FileNotFoundError: If the directory has no ``*.patch`` files, e.g. they were left out of the
            installed package; hashing an empty set would give a cache key that ignores the patches.
    """
    paths = sorted(patches_dir.glob("*.patch"), key=lambda p: p.name)
    if not paths:
        raise FileNotFoundError(f"No *.patch files found in {patches_dir}")
    return [path.read_bytes() for path in paths]


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


@contextmanager
def cache_entry_lock(cache_entry: Path) -> Iterator[None]:
    """Holds an exclusive cross-process lock for one cache entry.

    Blocks until any other holder releases it. The lock file sits next to the entry, not inside it,
    so evicting or replacing the entry does not delete the lock.

    Args:
        cache_entry: The cache entry directory the lock guards; it may not exist.

    Yields:
        None, while the lock is held.
    """
    cache_entry.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_entry.parent / f"{cache_entry.name}.lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def publish_wheel(wheel: Path, cache_entry: Path) -> Path:
    """Publishes a wheel as a cache entry, replacing any existing entry.

    The entry is assembled in a temp dir inside the cache root and renamed into place, so a
    concurrent run never sees a half-written entry.

    The caller must hold ``cache_entry_lock(cache_entry)``; otherwise replacing the entry can delete
    a wheel another process just published.

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
