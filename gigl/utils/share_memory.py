import ctypes
import errno
import math
import mmap
import os
import tempfile
from collections import abc
from multiprocessing.reduction import ForkingPickler
from typing import Optional, TypeVar, Union, cast

import torch

# Imported for its side effect as much as its symbols: importing torch.multiprocessing runs
# init_reductions(), which registers torch's reducer for torch.Tensor. The spill-aware reducer near
# the bottom of this module must register AFTER that so it replaces torch's entry; a later import
# of torch.multiprocessing elsewhere is a cached no-op and cannot clobber it
import torch.multiprocessing
from graphlearn_torch.partition import PartitionBook, RangePartitionBook
from torch.multiprocessing.reductions import reduce_tensor as _torch_reduce_tensor

from gigl.common.logger import Logger
from gigl.utils.host_memory import available_memory_bytes

logger = Logger()

_KeyType = TypeVar("_KeyType")  # Generic Key Type

# Directory for spilling large tensors to disk instead of POSIX shared memory. When unset,
# behaviour is exactly as before (everything goes to /dev/shm).
#
# Read from the ENVIRONMENT rather than passed as an argument on purpose: the dataset is
# built inside `mp.spawn` (dataset_factory.py), which starts a fresh interpreter, so
# module-level state set by a parent process does NOT survive but the environment does --
# it is inherited by the spawned child
_SPILL_DIR_ENV = "GIGL_TENSOR_SPILL_DIR"
_SPILL_MIN_BYTES_ENV = "GIGL_TENSOR_SPILL_MIN_BYTES"
# Set by whichever process cleans the spill directory, so its descendants know not to.
_SPILL_PREPARED_ENV = "GIGL_TENSOR_SPILL_DIR_PREPARED"
_DEFAULT_SPILL_MIN_BYTES = 2 * 2**30  # 2 GiB
# Headroom left on the shared-memory mount when preallocating there. Other tensors, torch's own
# queues and the sampling workers' channels all share it, and a shared storage that overruns the
# mount takes SIGBUS on write rather than failing at allocation.
_SHARED_MEMORY_RESERVE_BYTES = 2 * 2**30  # 2 GiB
# Cgroup headroom to leave free when a scattered destination is placed in memory. Sized to what
# is allocated AFTER this check runs (a CSR build's cursor clone and chunk/sort transients, which
# together reach several GiB at large-graph scale) plus allocator and race margin. Already-charged
# dirty pages are not itemised: the headroom read here reflects them. A later allocation re-runs
# this check against whatever headroom is left and falls back to disk if the answer is no
_DEFAULT_RANDOM_ACCESS_RESERVE_BYTES = 16 * 2**30  # 16 GiB

# Quantized tensors carry a quantizer beside their storage; rebuilding one from raw bytes produces
# a tensor that trips internal torch assertions on first use (qscheme, copy_). Refused rather than
# mapped wrong
_UNSPILLABLE_DTYPES = frozenset(
    dtype
    for dtype in (
        getattr(torch, name, None)
        for name in ("qint8", "quint8", "qint32", "quint4x2", "quint2x4")
    )
    if dtype is not None
)


def prepare_spill_dir() -> None:
    """Clear leftover spill files, ONCE per run, before anything in the run can spill.

    Cleanup is done at the START of a run, not with ``atexit``. The tensors are spilled inside the
    ``mp.spawn`` dataset-building child (``dataset_factory._build_dataset_process``), and that
    child **exits before the trainer uses the dataset** -- so unlinking on its exit would delete
    files the trainer and its sampling workers still have mapped by path. Removing stale files up
    front bounds disk use without that hazard.

    Exactly one process may do this, and it must be a process that starts before any spilling
    one -- a run has several spilling children in sequence, and a later one cannot tell a
    sibling's live files from a stale run's (the first spiller does clean up itself when no
    ancestor did, which covers standalone use and tests). Per-process age-based cleanup is NOT a
    substitute and is actively wrong: with sequential loading, the edge child spills and exits,
    then the node child starts, sees the edge files as older than itself, and deletes files the
    parent has not mapped yet. The marker below is an environment variable precisely because
    ``spawn`` children inherit the environment, so a child can tell that its parent already did
    this.

    Idempotent, and safe to call when spilling is disabled.
    """
    spill_dir = _spill_dir()
    if spill_dir is None:
        return
    if os.environ.get(_SPILL_PREPARED_ENV) == "1":
        logger.info(
            f"share_memory: {spill_dir} was already prepared by an ancestor process; "
            f"leaving its contents alone"
        )
        _cleared_spill_dirs.add(spill_dir)
        return
    _clear_spill_dir(spill_dir)
    # Inherited by every process spawned from here on, which is how they know to skip cleanup.
    os.environ[_SPILL_PREPARED_ENV] = "1"


def _clear_spill_dir(spill_dir: str) -> None:
    """Unlink every spill file this module could have written under ``spill_dir``."""
    if spill_dir in _cleared_spill_dirs:
        return
    _cleared_spill_dirs.add(spill_dir)
    removed = 0
    freed = 0
    try:
        for entry in os.scandir(spill_dir):
            if not (
                entry.name.startswith(_SPILL_PREFIX) and entry.name.endswith(".bin")
            ):
                continue
            try:
                freed += entry.stat().st_size
                os.unlink(entry.path)
                removed += 1
            except OSError:
                pass
    except FileNotFoundError:
        return
    if removed:
        logger.info(
            f"share_memory: removed {removed} stale spill file(s) from {spill_dir} "
            f"({freed / 2**30:.1f} GiB)"
        )


def _ensure_spill_dir_prepared(spill_dir: str) -> None:
    """Called before writing a spill file.

    If some ancestor called :func:`prepare_spill_dir`, do nothing -- deleting anything here risks
    removing a sibling's live file. Otherwise this process is the first in its run to spill, so it
    takes on the cleanup itself; that covers standalone use and tests, where no orchestrator ran.
    """
    if os.environ.get(_SPILL_PREPARED_ENV) == "1":
        return
    _clear_spill_dir(spill_dir)
    os.environ[_SPILL_PREPARED_ENV] = "1"


_cleared_spill_dirs: set[str] = set()
_SPILL_PREFIX = "spill_"


def _spill_dir() -> Optional[str]:
    d = os.environ.get(_SPILL_DIR_ENV)
    return d or None


def _env_bytes(name: str, default: int) -> int:
    """A byte count from the environment, falling back to ``default`` when unset or unparseable."""
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _spill_min_bytes() -> int:
    return _env_bytes(_SPILL_MIN_BYTES_ENV, _DEFAULT_SPILL_MIN_BYTES)


def is_tensor_spilling_enabled() -> bool:
    """Whether ``GIGL_TENSOR_SPILL_DIR`` is set, i.e. large tensors should go to disk."""
    return _spill_dir() is not None


def is_disk_backed(tensor: torch.Tensor) -> bool:
    """Whether ``tensor`` (or a view of one) is an mmap over a real file rather than RAM.

    Read off the tensor itself: a storage created by ``from_file(shared=True)`` records the path
    in ``untyped_storage().filename``, and a view shares its base's storage, so the label travels
    with it. ``filename`` is None for everything else -- including tmpfs-shared storages
    (``share_memory_()``, ``_new_shared``), which are RAM and must not be treated as spilled.
    """
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
        return False
    if tensor.layout is not torch.strided:
        return False
    return tensor.untyped_storage().filename is not None


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Row-major strides for ``shape``, computed because ``set_`` with a size requires a stride."""
    stride = [1] * len(shape)
    for axis in range(len(shape) - 2, -1, -1):
        stride[axis] = stride[axis + 1] * int(shape[axis + 1])
    return tuple(stride)


def _tensor_over_file(
    path: str, dtype: torch.dtype, shape: tuple[int, ...]
) -> torch.Tensor:
    """A writable tensor mapped over the whole of ``path``, or raise ``OSError``.

    ``shared=True`` is MAP_SHARED: always writable, writes reach the file, and the storage records
    ``path`` in ``filename`` -- which is what marks the tensor disk-backed everywhere else in this
    module and makes it pickle by path. A read-only mapping is never handed back: downstream code
    is not audited for in-place mutation, and a write through PROT_READ storage is a SIGSEGV, not
    an exception.
    """
    n_bytes = math.prod(shape) * torch.empty(0, dtype=dtype).element_size()
    # cast because the stub types from_file's return as the storage base class, which set_ rejects
    storage = cast(
        torch.UntypedStorage,
        torch.UntypedStorage.from_file(path, shared=True, nbytes=n_bytes),
    )
    tensor = torch.empty(0, dtype=dtype)
    tensor.set_(storage, 0, tuple(shape), _contiguous_strides(tuple(shape)))
    return tensor


def load_spilled_tensor(
    path: str, dtype: torch.dtype, shape: tuple[int, ...]
) -> torch.Tensor:
    """Re-map a spilled tensor in the current process, without copying its bytes anywhere.

    The counterpart to :func:`spill_tensor_to_disk` for a path that arrived through some channel
    other than pickling a tensor (pickling already re-maps by itself).

    Raises:
        ValueError: If the file is the wrong size for ``shape`` -- which would otherwise be read
            as silently wrong data.
        OSError: If the file cannot be mapped writable.
    """
    expected_bytes = math.prod(shape) * torch.empty(0, dtype=dtype).element_size()
    actual_bytes = os.path.getsize(path)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Spill file {path} is {actual_bytes} bytes but {shape} of {dtype} needs "
            f"{expected_bytes}. Refusing to map it."
        )
    return _tensor_over_file(path, dtype, tuple(shape))


def spill_tensor_to_disk(tensor: torch.Tensor) -> Optional[torch.Tensor]:
    """Write ``tensor`` to disk and return a tensor mapped over the file, or ``None`` to keep it.

    ``None`` covers every reason not to spill -- spilling disabled, tensor below the threshold, IO
    failure -- so callers can treat it as "keep what you had". The returned tensor's storage knows
    its file (:func:`is_disk_backed`), so :func:`share_memory` leaves it alone and pickling ships
    the path rather than the bytes.

    Exposed for callers that hold a tensor directly rather than inside a Mapping, which
    :func:`share_memory` cannot substitute into. The partitioned node feature matrix is the case
    that matters: it is created after loading, so the spill in ``load_torch_tensors`` never sees
    it, and it is resident during the graph build.
    """
    spill_dir = _spill_dir()
    if spill_dir is None:
        return None
    if 0 in tensor.shape:
        return None
    if tensor.is_quantized:
        logger.warning(
            f"share_memory: dtype {tensor.dtype} not spillable, using shared memory"
        )
        return None
    # Already living in a file -- most likely allocated by allocate_disk_backed rather than filled
    # in memory and spilled. Copying it to a second file would double the disk use and the time
    # for no benefit
    if is_disk_backed(tensor):
        logger.info(
            f"share_memory: {tuple(tensor.shape)} {tensor.dtype} is already on disk at "
            f"{tensor.untyped_storage().filename}; not writing a second copy"
        )
        return tensor
    if tensor.numel() * tensor.element_size() < _spill_min_bytes():
        return None
    _ensure_spill_dir_prepared(spill_dir)
    return _spill_to_file(tensor, spill_dir)


def _fadvise_dontneed(fd: int) -> None:
    """``POSIX_FADV_DONTNEED`` over the whole file (length 0 means to end of file).

    Raises ``OSError`` (ENOSYS) on platforms without ``posix_fadvise``, resolved with getattr
    because the symbol does not exist off Linux, and macOS dev machines still import and
    type-check this module (its tests skip there).
    """
    fadvise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if fadvise is None or dontneed is None:
        raise OSError(errno.ENOSYS, "posix_fadvise is unavailable on this platform")
    fadvise(fd, 0, 0, dontneed)


def _reserve_file_blocks(fd: int, n_bytes: int, spill_dir: str) -> None:
    """Reserve every block of a spill file now, or raise ``OSError``.

    Running out of space must be an error HERE, not a SIGBUS on some later page write -- a signal,
    not an exception, which no ``try`` can catch. A filesystem that cannot reserve at all
    (``posix_fallocate`` unsupported) raises too: proceeding with an unreserved mapping would keep
    the SIGBUS window open, so the caller falls back to memory instead.
    """
    # getattr rather than a direct reference: the symbol does not exist on non-Linux platforms,
    # where the module must still import and merely refuse to spill
    fallocate = getattr(os, "posix_fallocate", None)
    if fallocate is None:
        raise OSError(
            errno.ENOSYS,
            f"posix_fallocate is unavailable on this platform, so a "
            f"{n_bytes / 2**30:.1f} GiB mapping under {spill_dir} cannot be made SIGBUS-safe",
        )
    while True:
        try:
            fallocate(fd, 0, n_bytes)
            return
        except InterruptedError:
            continue  # EINTR: the syscall was interrupted, not refused
        except OSError as reserve_error:
            if reserve_error.errno in (errno.EOPNOTSUPP, errno.ENOSYS, errno.EINVAL):
                # The filesystem cannot reserve. An unreserved mapping faults with SIGBUS if the
                # filesystem fills, so refuse rather than proceed without the guarantee
                raise OSError(
                    reserve_error.errno,
                    f"{spill_dir} does not support reserving space ({reserve_error}); "
                    f"refusing an unreserved {n_bytes / 2**30:.1f} GiB mapping",
                ) from reserve_error
            # ENOSPC, EDQUOT, EFBIG, EIO and anything else unexpected: the caller cleans up and
            # uses memory instead
            raise


def allocate_disk_backed(
    shape: tuple[int, ...], dtype: torch.dtype
) -> Optional[torch.Tensor]:
    """An EMPTY writable tensor whose bytes live in a file, or None to fall back to RAM.

    The counterpart to :func:`spill_tensor_to_disk` for a buffer that does not exist yet. Spilling
    fills RAM and then copies it out, so it needs the full size in anonymous memory at least once;
    allocating the destination as the file from the start means those bytes never occupy anonymous
    memory at all.

    Blocks are RESERVED up front with ``posix_fallocate`` rather than left sparse, and reservation
    is mandatory. ``from_file`` gives the file its full apparent size without allocating it, so a
    filesystem that fills later fails at the moment a page is written -- and a write to a mapping
    that cannot be backed raises **SIGBUS**, which is a signal, not an exception: it bypasses
    every ``try`` here and kills the process with no traceback. Reserving turns that into an
    ``OSError`` at allocation time, where it can be reported and fallen back on; a filesystem that
    cannot reserve at all gets None rather than an unreserved mapping.

    Sequential writes through the mapping cost ~3.1x their in-memory equivalent (measured on a wide
    fp32 matrix written row-by-row). What that buys: the destination's bytes stop being
    unreclaimable anonymous memory -- still charged to the cgroup while resident, but reclaimable
    page cache the kernel can evict under pressure instead of OOM-killing.

    Returns None whenever a file-backed buffer is not available or not worth it -- spilling
    disabled, below the size threshold, quantized dtype, no room to reserve, reservation
    unsupported by the filesystem, IO failure -- so callers can simply fall back to
    ``torch.empty``.
    """
    spill_dir = _spill_dir()
    if spill_dir is None:
        return None
    if 0 in shape:
        return None
    if dtype in _UNSPILLABLE_DTYPES:
        return None
    n_bytes = math.prod(shape) * torch.empty(0, dtype=dtype).element_size()
    if n_bytes < _spill_min_bytes():
        return None
    _ensure_spill_dir_prepared(spill_dir)

    path: Optional[str] = None
    try:
        os.makedirs(spill_dir, exist_ok=True)
        fd, path = tempfile.mkstemp(dir=spill_dir, prefix=_SPILL_PREFIX, suffix=".bin")
        try:
            _reserve_file_blocks(fd, n_bytes, spill_dir)
        finally:
            os.close(fd)
        # Mapped only AFTER the reservation. from_file leaves an existing file's allocated blocks
        # intact (the tests hold this), so the mapping below is fully backed and SIGBUS-safe
        tensor = _tensor_over_file(path, dtype, tuple(shape))
        logger.info(
            f"share_memory: allocated {n_bytes / 2**30:.1f} GiB buffer {tuple(shape)} {dtype} "
            f"on disk at {path} instead of in anonymous memory"
        )
        return tensor
    except Exception as e:  # noqa: BLE001
        if path is not None:
            try:
                os.unlink(path)
            except OSError:
                pass
        logger.error(
            f"share_memory: could not allocate {n_bytes / 2**30:.1f} GiB on disk "
            f"({type(e).__name__}: {e}); using memory"
        )
        return None


def _spill_to_file(tensor: torch.Tensor, spill_dir: str) -> Optional[torch.Tensor]:
    """Write ``tensor`` to a file under ``spill_dir`` and return a tensor mapped over it.

    Returns None if the tensor cannot be spilled (reservation refused by the filesystem, IO
    failure), in which case the caller should keep the tensor it already had.

    Why this helps: ``/dev/shm`` is tmpfs, i.e. RAM. Node features for a ~1B-node graph approach
    a terabyte at fp32; even split across replicas that is the largest resident tensor, and
    ``share_memory_()`` costs 1x in tmpfs plus a transient 2x in RSS. Backing the bytes with
    a real file turns them into evictable page cache instead.
    """
    path: Optional[str] = None
    try:
        os.makedirs(spill_dir, exist_ok=True)
        n_bytes = tensor.numel() * tensor.element_size()
        # mkstemp, not a name derived from shape and byte count. A derived name collides for two
        # tensors with the same shape and element width -- including different dtypes of the same
        # width -- and reopening and resizing that path would silently replace the contents still
        # backing the first tensor's live mapping. Nothing would raise; the features would simply
        # be wrong
        fd, path = tempfile.mkstemp(dir=spill_dir, prefix=_SPILL_PREFIX, suffix=".bin")
        try:
            # Same SIGBUS-safety rule as allocate_disk_backed: every block is reserved before the
            # copy starts, so a filesystem that fills mid-copy fails as OSError here, not as a
            # signal on some later page write
            _reserve_file_blocks(fd, n_bytes, spill_dir)
        finally:
            os.close(fd)
        writable = _tensor_over_file(path, tensor.dtype, tuple(tensor.shape))
        # Copy in ~1 GiB slices so we never hold a second full-size buffer.
        row_bytes = (
            max(1, int(tensor[0].numel()) * tensor.element_size())
            if tensor.dim() > 1
            else tensor.element_size()
        )
        rows = max(1, int(2**30 // row_bytes))
        for start in range(0, tensor.shape[0], rows):
            writable[start : start + rows].copy_(tensor[start : start + rows])
        # msync equivalent: fsync writes back the pages the copy just dirtied, so the data is on
        # disk and a later release_page_cache has clean pages to drop
        flush_fd = os.open(path, os.O_RDWR)
        try:
            os.fsync(flush_fd)
        finally:
            os.close(flush_fd)
        logger.info(
            f"share_memory: spilled {n_bytes / 2**30:.1f} GiB to {path} instead of /dev/shm"
        )
        return writable
    except Exception as e:  # noqa: BLE001
        # Unlink the partial file. mkstemp has already created it, so a failure in the mapping,
        # the copy or the flush would otherwise leave its bytes on disk with nothing referencing
        # them. That matters most in the case that causes the failure: the spill filesystem is
        # finite and node features are tens of GiB per replica, so a couple of failed attempts can
        # exhaust the disk and push the tensors back into RAM -- turning a recoverable spill
        # failure into the OOM this whole mechanism exists to avoid
        if path is not None:
            try:
                os.unlink(path)
                logger.info(f"share_memory: removed partial spill file {path}")
            except OSError:
                logger.warning(
                    f"share_memory: could not remove partial spill file {path}"
                )
        logger.error(
            f"share_memory: spill failed ({type(e).__name__}: {e}); using shared memory"
        )
        return None


def _shared_memory_backing_dir() -> str:
    """Where torch puts shared storages on Linux: ``/dev/shm``, under BOTH sharing strategies.

    Exists to point the free-space check below at the right mount: over-allocating tmpfs succeeds
    and then SIGBUSes on the first write, so checking the wrong filesystem makes the guard
    worthless. Measured rather than inferred, because the obvious reading of the docs is wrong. Under
    ``file_system`` torch calls ``_new_using_filename_cpu``, which sounds like an ordinary temporary
    file and is not: the storage lands in ``/dev/shm`` (observed:
    ``/dev/shm/torch_<pid>_<random>_0``, 16,000,064 bytes for a 16 MB tensor), and ``TMPDIR`` receives
    only a 4 KiB ``torch-shm-dir-*`` directory holding the libshm manager's socket. Under
    ``file_descriptor`` the segment is ``shm_open``ed and immediately unlinked, so nothing is visible
    in the directory listing while it still consumes the mount's capacity. Returning ``TMPDIR``
    for ``file_system`` would size the check against the wrong filesystem in precisely the case
    the check exists for.
    """
    return "/dev/shm"


def _shared_memory_shortfall(n_bytes: int) -> Optional[str]:
    """A human-readable reason a shared allocation of ``n_bytes`` would not fit, or None if it fits.

    Returns None when the check cannot be made, so an unreadable mount does not block an allocation
    that would have worked. The caller must still handle a failure from the allocation itself: this
    cannot close the gap between checking and allocating, during which another process on the same
    mount can take the space.
    """
    directory = _shared_memory_backing_dir()
    try:
        stats = os.statvfs(directory)
    except OSError:
        return None
    free_bytes = stats.f_bavail * stats.f_frsize
    # A margin, because this process is not the only user of that mount and the check races anything
    # else allocating on it.
    needed = n_bytes + _SHARED_MEMORY_RESERVE_BYTES
    if free_bytes >= needed:
        return None
    return (
        f"{directory} has {_human_bytes(free_bytes)} free, short of the "
        f"{_human_bytes(n_bytes)} requested plus a {_human_bytes(_SHARED_MEMORY_RESERVE_BYTES)} "
        f"reserve"
    )


def _human_bytes(n_bytes: int) -> str:
    """A byte count in the unit that makes it readable, so small tensors do not log as "0.0 GiB"."""
    if n_bytes >= 2**30:
        return f"{n_bytes / 2**30:.1f} GiB"
    if n_bytes >= 2**20:
        return f"{n_bytes / 2**20:.1f} MiB"
    return f"{n_bytes} B"


def _random_access_headroom_reserve_bytes() -> int:
    """Cgroup headroom to leave free when choosing memory over disk for a scattered destination."""
    return _env_bytes(
        "GIGL_RANDOM_ACCESS_HEADROOM_RESERVE_BYTES",
        _DEFAULT_RANDOM_ACCESS_RESERVE_BYTES,
    )


def _memory_is_the_better_home(n_bytes: int) -> Optional[str]:
    """None if ``n_bytes`` should go to memory rather than disk, else why it cannot.

    Two independent limits, and both have to hold. ``/dev/shm`` has its own size, usually half of
    RAM; the cgroup is what actually kills the process. Checking only the mount is not enough in a
    container -- the mount is sized off the machine spec while the cgroup limit can sit well below
    it.
    """
    shortfall = _shared_memory_shortfall(n_bytes)
    if shortfall is not None:
        return shortfall
    reserve = _random_access_headroom_reserve_bytes()
    available = available_memory_bytes()
    if available < n_bytes + reserve:
        return (
            f"only {_human_bytes(available)} of cgroup headroom, short of the "
            f"{_human_bytes(n_bytes)} requested plus a {_human_bytes(reserve)} reserve"
        )
    return None


def allocate_preshared(
    shape: tuple[int, ...], dtype: torch.dtype, random_access: bool = False
) -> torch.Tensor:
    """Allocate a large tensor in its FINAL home, so ``share_memory_()`` cannot duplicate it.

    GLT's ``Graph.__init__`` shares the topology unconditionally, and for an anonymous tensor that
    copies every byte into ``/dev/shm`` -- the tensor exists twice during the copy, fatal for a
    tens-of-GiB CSR near the memory limit. Both homes here are immune to that copy: a spill file
    (consumers that check :func:`is_disk_backed` leave it alone), or POSIX shared memory allocated
    directly (``share_memory_()`` then finds it already shared and does nothing).

    Disk-first for a destination written in order. A scatter destination prefers memory: an 8-byte
    write to an uncached file page is a 4 KiB read-modify-write, repeated over the same pages, so
    a file-backed scatter destination behaves like a hung run. With ``random_access=True`` a file
    is only the checked, loudly-logged fallback for when memory will not hold the tensor.

    Below the spill threshold returns a plain tensor. Uninitialised in all cases, like
    ``torch.empty``.

    Args:
        shape: Shape of the tensor.
        dtype: Dtype of the tensor.
        random_access: Set when the caller will write to scattered offsets rather than stream
            through in order; memory is then preferred and disk the checked fallback.
    """
    n_bytes = math.prod(shape) * torch.empty(0, dtype=dtype).element_size()

    prefer_memory = random_access and _memory_is_the_better_home(n_bytes) is None
    if not prefer_memory:
        if random_access and n_bytes >= _spill_min_bytes() and 0 not in shape:
            # Said out loud, with the reason: entering this branch is the run's most consequential
            # placement decision, and a silently disk-backed scatter destination is a run that
            # appears hung. A caller that detects the disk backing and switches to a banded
            # scatter writes each destination page once instead of once per chunk -- slower than
            # memory but bounded, and its own progress lines say how it is going
            logger.warning(
                f"share_memory: {_human_bytes(n_bytes)} {tuple(shape)} {dtype} will be scattered "
                f"into but must live on disk -- {_memory_is_the_better_home(n_bytes)}. The caller "
                f"is expected to use a banded writer; watch its progress lines rather than "
                f"assuming a hang"
            )
        on_disk = allocate_disk_backed(shape, dtype)
        if on_disk is not None:
            return on_disk

    if n_bytes < _spill_min_bytes() or 0 in shape:
        return torch.empty(shape, dtype=dtype)

    shortfall = _shared_memory_shortfall(n_bytes)
    if shortfall is not None:
        # Raised, not worked around, because there is NO safe fallback from here.
        #
        # Attempting the allocation is unsafe: sizing a tmpfs object does not reserve its pages, so
        # `_new_shared` for 31.3 GiB against a 64 MiB /dev/shm -- Docker's default -- SUCCEEDS, and
        # the first write past the limit takes SIGBUS, which no `except` can catch.
        #
        # Returning an anonymous tensor is equally unsafe, which is the part that is easy to get
        # wrong: the topology builder would then pick a plain in-memory topology, and
        # `Graph.__init__` calls GLT's UNGUARDED `share_memory_()` on it -- attempting the very same
        # oversized allocation on the very same full mount, a few seconds later and with no
        # attribution back to here. A warning-and-continue only moves the SIGBUS somewhere harder to
        # diagnose.
        #
        # So fail here, where the message can name both remedies. Reachable only for tensors at or
        # above the spill threshold, and only when the disk backend was unavailable too.
        raise RuntimeError(
            f"Cannot allocate {_human_bytes(n_bytes)} {tuple(shape)} {dtype} without risking SIGBUS: "
            f"{shortfall}. Continuing is not safe -- an anonymous tensor this size is relocated into "
            f"the same full mount by graphlearn_torch's Graph.__init__, a few seconds later and "
            f"with no attribution back to here. Either set GIGL_TENSOR_SPILL_DIR to a filesystem with room "
            f"(the intended configuration; the CSR is then file-backed and never copied), or raise "
            f"the container's shared-memory limit."
        )

    # `torch.empty(...).share_memory_()` would defeat the purpose: it allocates anonymously and then
    # copies, which is the duplication being avoided. Allocating the shared storage first means the
    # bytes are only ever written once.
    try:
        storage = torch.UntypedStorage._new_shared(n_bytes)
        tensor = torch.empty(0, dtype=dtype)
        tensor.set_(storage, 0, tuple(shape), _contiguous_strides(tuple(shape)))
        logger.info(
            f"share_memory: allocated {_human_bytes(n_bytes)} {tuple(shape)} {dtype} directly in "
            f"shared memory, so share_memory_() will not copy it"
        )
        return tensor
    except (AttributeError, OSError, RuntimeError) as e:
        # Falling back IS safe here, unlike the shortfall branch above, and the difference is the
        # whole point: capacity was just checked and found sufficient, so the duplicate that GLT's
        # `share_memory_()` will make has somewhere to go. This branch is for the mechanism being
        # unavailable -- `_new_shared` is private API and could be renamed -- not for the mount being
        # full. Costs a duplicate, not a SIGBUS.
        logger.warning(
            f"share_memory: could not preallocate {_human_bytes(n_bytes)} in shared memory "
            f"({type(e).__name__}: {e}); the mount has room, so a later share_memory_() will "
            f"duplicate it rather than fail"
        )
        return torch.empty(shape, dtype=dtype)


def has_live_mapping(path: str) -> bool:
    """Whether THIS process still maps ``path``, read from ``/proc/self/maps``.

    The kernel's own list of this process's mappings, so it sees every mapping over the file no
    matter who created it. False when the list cannot be read (non-Linux), matching the rest of
    this module: spilling is effectively Linux-only and refuses gracefully elsewhere.
    """
    real = os.path.realpath(path)
    try:
        with open("/proc/self/maps") as maps:
            return any(line.rstrip("\n").endswith(f" {real}") for line in maps)
    except OSError:
        return False


def release_page_cache_by_path(path: str) -> bool:
    """Request eviction of a spill file's page cache when nothing maps it any more.

    The counterpart to :func:`release_page_cache` for a tensor already dropped: ``del`` unmaps it
    but leaves the file's pages charged to the cgroup until the kernel reclaims them.

    Refuses when a mapping is still live: ``FADV_DONTNEED`` silently skips pages present in any
    page table, so it would return success having freed nothing. Use :func:`release_page_cache` on
    the tensor in that case, which unmaps first.

    Returns True if the eviction was REQUESTED (``FADV_DONTNEED`` is advisory and reports no
    count); False if a mapping is still live, the file could not be opened, or the platform lacks
    ``posix_fadvise``.
    """
    if has_live_mapping(path):
        logger.warning(
            f"share_memory: refusing to drop page cache for {path} -- a tensor still maps it, so "
            f"FADV_DONTNEED would free nothing. Drop every reference first, or call "
            f"release_page_cache(tensor) to unmap before discarding."
        )
        return False
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError as e:
        logger.warning(f"share_memory: could not open {path} to drop its cache: {e}")
        return False
    try:
        os.fsync(fd)
        _fadvise_dontneed(fd)
    except OSError as e:
        logger.info(f"share_memory: could not drop page cache for {path}: {e}")
        return False
    finally:
        os.close(fd)
    return True


def _madvise_dontneed(address: int, n_bytes: int, path: str) -> bool:
    """``MADV_DONTNEED`` on ``[address, address + n_bytes)``, dropping this process's page-table entries.

    Safe only on a file-backed mapping whose dirty pages were just written back: the pages remain
    in the file and refault on next access. On an anonymous mapping the same call would discard
    the data, which is why this helper is private to :func:`release_page_cache`.
    """
    dontneed = getattr(mmap, "MADV_DONTNEED", None)
    if dontneed is None:
        logger.info(
            f"share_memory: cannot unmap pages for {path} (no MADV_DONTNEED on this platform); "
            f"its pages stay charged to the cgroup until the kernel reclaims them"
        )
        return False
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.madvise(
        ctypes.c_void_p(address), ctypes.c_size_t(n_bytes), ctypes.c_int(dontneed)
    )
    if result != 0:
        reason = os.strerror(ctypes.get_errno())
        logger.info(
            f"share_memory: MADV_DONTNEED failed for {path} ({reason}); page-cache eviction "
            f"would be a no-op while the pages remain mapped, so nothing was dropped"
        )
        return False
    return True


def release_page_cache(tensor: torch.Tensor) -> bool:
    """Write a disk-backed tensor's dirty pages out and ask the kernel to drop them from the cache.

    A spilled tensor's pages are reclaimable but still charged to cgroup v2's ``memory.current``
    while resident; dropping them releases the charge. The mapping stays valid and refaults on
    next access, so this is worth it only for data written now and not read until much later.
    Operates on the tensor's whole backing file, so views over the same file lose their cached
    pages too.

    Three steps, in this order, and every one is load-bearing:

    1. ``fsync`` -- writes back the pages dirtied through the mapping; dirty pages cannot be
       dropped at all.
    2. ``MADV_DONTNEED`` on the mapping -- removes this process's page-table entries. Without it
       step 3 silently skips every still-mapped page and reports success having freed nothing.
    3. ``POSIX_FADV_DONTNEED`` -- discards the now-unmapped clean page cache.

    Returns True when every step COMPLETED -- ``FADV_DONTNEED`` is advisory, so completion is not
    proof of eviction; measure residency with ``mincore`` to know (the tests do). False means the
    sequence was incomplete, though a failure after step 2 has already dropped this process's
    page-table entries.
    """
    if not is_disk_backed(tensor):
        return False
    storage = tensor.untyped_storage()
    path = storage.filename
    assert path is not None  # is_disk_backed just checked

    try:
        fd = os.open(path, os.O_RDWR)
    except OSError as e:
        logger.warning(f"share_memory: could not open {path} to drop cache: {e}")
        return False
    try:
        os.fsync(fd)  # step 1: dirty pages cannot be dropped, so this is not optional
        # step 2 must precede step 3, and MADV_DONTNEED is only safe because the mapping is
        # file-backed and its dirty pages were just written back (on an anonymous mapping it
        # would discard the data)
        if not _madvise_dontneed(storage.data_ptr(), storage.nbytes(), path):
            return False
        _fadvise_dontneed(fd)  # step 3
    except OSError as e:
        logger.info(
            f"share_memory: could not drop page cache for {path} ({e}); its pages stay "
            f"charged to the cgroup until the kernel reclaims them"
        )
        return False
    finally:
        os.close(fd)
    logger.info(
        f"share_memory: unmapped and requested eviction of {_human_bytes(storage.nbytes())} of "
        f"page cache for {path}; the mapping stays valid and refaults on next read"
    )
    return True


def _rebuild_spilled_tensor(
    path: str,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    storage_offset: int,
    storage_nbytes: int,
) -> torch.Tensor:
    """The receiving half of :func:`_reduce_spill_aware`: re-map the file, no bytes copied.

    Raises:
        FileNotFoundError: If the spill file is gone -- ``from_file`` would otherwise CREATE it
            and silently hand back zero-filled, unreserved storage.
        ValueError: If the file is too small for the storage it is supposed to hold, for the same
            reason.
    """
    actual_bytes = os.path.getsize(path)
    if actual_bytes < storage_nbytes:
        raise ValueError(
            f"Spill file {path} is {actual_bytes} bytes but the pickled tensor needs "
            f"{storage_nbytes}. Refusing to map it."
        )
    # cast because the stub types from_file's return as the storage base class, which set_ rejects
    storage = cast(
        torch.UntypedStorage,
        torch.UntypedStorage.from_file(path, shared=True, nbytes=storage_nbytes),
    )
    tensor = torch.empty(0, dtype=dtype)
    tensor.set_(storage, storage_offset, shape, stride)
    return tensor


def _reduce_spill_aware(tensor: torch.Tensor) -> tuple:
    """Pickle a disk-backed tensor as its path; hand everything else to torch's own reducer.

    Torch's reduction for CPU tensors moves the bytes into ``/dev/shm`` -- for a spilled tensor
    that copies it back into RAM and undoes the spill. A storage with a ``filename`` was created
    by ``from_file(shared=True)``, whose documented contract is already "all changes are written
    to the file", so shipping the path preserves its sharing semantics: the receiver maps the same
    file with the same offset, shape and strides. Everything else -- no filename, CUDA, sparse,
    grad-carrying, quantized, named, conj/neg views whose flag a raw re-map would drop -- is
    reduced by torch exactly as if this module were not loaded.

    Gated on spilling being enabled, so the feature stays opt-in: a process that never set
    ``GIGL_TENSOR_SPILL_DIR`` pickles its own ``from_file`` tensors exactly as torch does today
    (spawn children inherit the environment, so the gate agrees on both sides of a boundary).
    """
    if (
        is_tensor_spilling_enabled()
        and tensor.layout is torch.strided
        and tensor.device.type == "cpu"
        and not tensor.requires_grad
        and not tensor.is_quantized
        and not tensor.has_names()
        and not tensor.is_conj()
        and not tensor.is_neg()
    ):
        storage = tensor.untyped_storage()
        if storage.filename is not None:
            return (
                _rebuild_spilled_tensor,
                (
                    storage.filename,
                    tensor.dtype,
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                    tensor.storage_offset(),
                    storage.nbytes(),
                ),
            )
    return _torch_reduce_tensor(tensor)


# Replaces torch's entry for torch.Tensor in the ForkingPickler dispatch table (keyed by exact
# type, so subclasses like torch.nn.Parameter keep their own reducers). Module import is the only
# hook that reliably precedes any pickling in both the sending and receiving process: the sender
# has imported this module to create or receive a spilled tensor, and the receiver imports it when
# unpickling resolves _rebuild_spilled_tensor
ForkingPickler.register(torch.Tensor, _reduce_spill_aware)


def share_memory(
    entity: Optional[
        Union[
            torch.Tensor,
            PartitionBook,
            dict[_KeyType, torch.Tensor],
            dict[_KeyType, PartitionBook],
        ]
    ],
) -> None:
    """
    Based on GraphLearn-for-PyTorch's `share_memory` implementation, with additional support for handling empty tensors with share_memory.
        https://github.com/alibaba/graphlearn-for-pytorch/blob/main/graphlearn_torch/python/utils/tensor.py#L88

    Calling `share_memory_()` on an empty tensor may cause processes to hang, although the root cause of this is currently unknown. As a result,
    we opt to not move empty tensors to shared memory if they are provided.

    When calling `share_memory` on a RangePartitionBook, we don't need to move the partition bounds to shared memory, since GLT doesn't natively
    provide a ForkingPickler registration method for the `RangePartitionBook`, and the cost of not moving this to shared memory is minimal,
    since the size of this tensor is very small, being equal in length to the number of machines.

    This function never spills to disk, and a tensor that is ALREADY disk-backed is left alone:
    ``share_memory_()`` on an mmap-backed tensor copies every byte into ``/dev/shm``, quietly
    undoing a spill made elsewhere. Spilling belongs to :func:`share_memory_for_ipc` and
    :func:`spill_tensor_to_disk`.

    Args:
        entity (Optional[Union[torch.Tensor, PartitionBook, dict[_KeyType, torch.Tensor], dict[_KeyType, PartitionBook]]]):
            Homogeneous or heterogeneous entity of tensors which is being moved to shared memory
    """

    if entity is None or isinstance(entity, RangePartitionBook):
        return
    elif isinstance(entity, abc.Mapping):
        for key in list(entity.keys()):
            share_memory(entity[key])
        return
    else:
        # If the tensor has a dimension which is 0, it is an empty tensor. As a result, we don't move this
        # to shared_memory, since share_memory_() is unsafe on empty tensors, which may cause processes to hang.
        if 0 in entity.shape:
            return
        if is_disk_backed(entity):
            logger.info(
                f"share_memory: leaving a disk-backed tensor {tuple(entity.shape)} "
                f"{entity.dtype} where it is; sharing it would copy "
                f"{entity.numel() * entity.element_size() / 2**30:.1f} GiB into /dev/shm"
            )
            return
        entity.share_memory_()


def share_memory_for_ipc(
    entity: dict[_KeyType, torch.Tensor],
) -> dict[_KeyType, torch.Tensor]:
    """Prepare a mapping of tensors to cross a process boundary, spilling instead of copying.

    Same intent as :func:`share_memory`, but for values that will be **pickled to another
    process**: each value large enough is spilled first, and a spilled tensor pickles as its file
    path (see :func:`_reduce_spill_aware`), so its bytes never transit ``/dev/shm``. Tensors that
    are not spilled (spilling disabled, below the size threshold) go to POSIX shared memory
    exactly as before, which pickles by handle already.

    Returns a NEW dict; the input is left alone, since the caller usually still holds the tensors
    and dropping them is its decision.
    """
    prepared: dict[_KeyType, torch.Tensor] = {}
    for key, value in entity.items():
        spilled = spill_tensor_to_disk(value)
        if spilled is not None:
            prepared[key] = spilled
            continue
        share_memory(value)
        prepared[key] = value
    return prepared
