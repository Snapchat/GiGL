import errno
import mmap
import os
import tempfile
import weakref
from collections import abc
from dataclasses import dataclass
from typing import Any, Optional, TypeVar, Union, cast

import torch
from graphlearn_torch.partition import PartitionBook, RangePartitionBook

from gigl.common.logger import Logger
from gigl.utils.host_memory import available_memory_bytes

logger = Logger()

_KeyType = TypeVar("_KeyType")  # Generic Key Type

# Directory for spilling large tensors to disk instead of POSIX shared memory. When unset,
# behaviour is exactly as before (everything goes to /dev/shm).
#
# Read from the ENVIRONMENT rather than passed as an argument on purpose: the dataset is
# built inside `mp.spawn` (gigl/distributed/dataset_factory.py:468), which starts a fresh
# interpreter. Module-level state set by a parent process does NOT survive that, but the
# environment does -- it is inherited by the spawned child. An earlier attempt to control
# this by monkey-patching from the trainer process was a silent no-op for exactly this
# reason.
_SPILL_DIR_ENV = "GIGL_TENSOR_SPILL_DIR"
_SPILL_MIN_BYTES_ENV = "GIGL_TENSOR_SPILL_MIN_BYTES"
# Set by whichever process cleans the spill directory, so its descendants know not to.
_SPILL_PREPARED_ENV = "GIGL_TENSOR_SPILL_DIR_PREPARED"
_DEFAULT_SPILL_MIN_BYTES = 2 * 2**30  # 2 GiB
# Headroom left on the shared-memory mount when preallocating there. Other tensors, torch's own
# queues and the sampling workers' channels all share it, and a shared storage that overruns the
# mount takes SIGBUS on write rather than failing at allocation.
_SHARED_MEMORY_RESERVE_BYTES = 2 * 2**30  # 2 GiB
# Cgroup headroom to leave free when a scattered destination is placed in memory. Only what is
# allocated AFTER this check runs belongs here -- at the measured worst case (the largest CSC of
# a ~1B-node graph): the direct scatter's full-size cursor clone (indptr[:-1], 7.2 GiB anonymous)
# and ~1 GiB of chunk/sort transients, ~8.2 GiB, so 16 leaves ~7.8 GiB of allocator/race margin.
# The freshly written indptr's 7.2 GiB of dirty pages are NOT itemised: they are already charged
# when this check reads the headroom. NOT covered: the next edge type; its own allocation re-runs
# this check against whatever headroom is left, and falls back to the banded disk path if the
# answer is no
_DEFAULT_RANDOM_ACCESS_RESERVE_BYTES = 16 * 2**30  # 16 GiB

_NUMPY_DTYPES = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.uint8: "uint8",
}


def prepare_spill_dir() -> None:
    """Clear leftover spill files, ONCE per run, before anything in the run can spill.

    Cleanup is done at the START of a run, not with ``atexit``. The tensors are spilled inside the
    ``mp.spawn`` dataset-building child (``dataset_factory._build_dataset_process``), and that
    child **exits before the trainer uses the dataset** -- so unlinking on its exit would delete
    files the trainer and its sampling workers still have mapped by path. Removing stale files up
    front bounds disk use without that hazard.

    Exactly one process may do this, and it must be a process that starts before any spilling
    one. Per-process age-based cleanup is NOT a substitute and is actively wrong: with sequential
    loading, the edge child spills and exits, then the node child starts, sees the edge files as
    older than itself, and deletes files the parent has not mapped yet. The marker below is an
    environment variable precisely because ``spawn`` children inherit the environment, so a child
    can tell that its parent already did this.

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

# (start address, byte length, weak reference to the backing array, descriptor) for every live spill
# mapping in this process. See _register_mapping for why this cannot be read back off the tensor.
_spilled_mappings: list[tuple[int, int, "weakref.ref[Any]", "SpilledTensorHandle"]] = []


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


@dataclass(frozen=True)
class SpilledTensorHandle:
    """Where a spilled tensor lives, with no tensor attached, so it is cheap to pickle.

    This is what crosses a process boundary in place of the tensor. Sending the tensor instead
    silently undoes the spill: torch's multiprocessing reduction does not consider a numpy-owned
    storage shareable, so pickling copies every byte into RAM-backed ``/dev/shm`` (measured: Shmem
    +383 MiB on a 383 MiB tensor). Call :func:`load_spilled_tensor` on the far side.
    """

    path: str
    dtype: torch.dtype
    shape: tuple[int, ...]

    def load(self) -> torch.Tensor:
        """Re-map the tensor in the current process."""
        return load_spilled_tensor(self.path, self.dtype, self.shape)


@dataclass(frozen=True)
class SpilledTensor:
    """A tensor living in a file, plus everything needed to re-map it elsewhere.

    ``tensor`` is an mmap view; ``handle`` is the part that is safe to send elsewhere.
    """

    tensor: torch.Tensor
    path: str
    dtype: torch.dtype
    shape: tuple[int, ...]

    @property
    def handle(self) -> SpilledTensorHandle:
        return SpilledTensorHandle(path=self.path, dtype=self.dtype, shape=self.shape)


def load_spilled_tensor(
    path: str, dtype: torch.dtype, shape: tuple[int, ...]
) -> torch.Tensor:
    """Re-map a spilled tensor in another process, without copying its bytes anywhere.

    The counterpart to :func:`spill_tensor_to_disk`. Pass ``SpilledTensor``'s path, dtype and
    shape through whatever IPC channel is available and call this on the far side.

    Raises:
        ValueError: If ``dtype`` has no numpy equivalent, or the file is the wrong size for
            ``shape`` -- which would otherwise be read as silently wrong data.
    """
    import numpy as np

    np_dtype = _NUMPY_DTYPES.get(dtype)
    if np_dtype is None:
        raise ValueError(f"Cannot map a spilled tensor of dtype {dtype}")
    expected_bytes = int(np.prod(shape)) * torch.empty(0, dtype=dtype).element_size()
    actual_bytes = os.path.getsize(path)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Spill file {path} is {actual_bytes} bytes but {shape} of {dtype} needs "
            f"{expected_bytes}. Refusing to map it."
        )
    return _map_spill_file(path, np_dtype, tuple(shape))


def _register_mapping(
    array: Any, tensor: torch.Tensor, handle: SpilledTensorHandle
) -> None:
    """Remember that ``tensor``'s bytes live in a file, so ``share_memory`` can refuse to copy them.

    Needed because the fact is not recoverable from the tensor: ``tensor.numpy().base`` is the
    tensor itself, and ``untyped_storage().filename`` is None for a storage adopted from numpy
    (both measured). Without a registry, ``share_memory_()`` on a spilled tensor silently relocates
    it into ``/dev/shm`` -- measured at +16,128 kB for a 16 MiB tensor -- undoing the spill at a
    point far from where it was made.

    The reference to ``array`` is weak, so an entry disappears when the mapping does. Views of a
    spilled tensor point inside the recorded range and are recognised too.
    """
    nbytes = tensor.numel() * tensor.element_size()
    _spilled_mappings[:] = [
        entry for entry in _spilled_mappings if entry[2]() is not None
    ]
    _spilled_mappings.append((tensor.data_ptr(), nbytes, weakref.ref(array), handle))


def is_disk_backed(tensor: torch.Tensor) -> bool:
    """Whether ``tensor`` (or a view of one) is an mmap over a spill file rather than RAM."""
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
        return False
    pointer = tensor.data_ptr()
    for start, nbytes, reference, _ in _spilled_mappings:
        if reference() is not None and start <= pointer < start + nbytes:
            return True
    return False


def disk_backed_handle(tensor: torch.Tensor) -> Optional[SpilledTensorHandle]:
    """The spill descriptor for a tensor that IS an entire spill mapping, else None.

    A partial view does not qualify: a descriptor names a whole file, so it cannot describe a
    slice of one. Lets a caller re-send an already-spilled tensor by path instead of spilling a
    second copy of it.

    Requires the tensor to be CONTIGUOUS with no storage offset, not merely to have the right start
    address and element count. A transpose of a square mmap tensor has the same pointer, shape and
    numel as the original but reads the file in the wrong order, so matching on those alone would
    hand the receiver transposed data described as untransposed -- silently wrong values rather than
    an error.
    """
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
        return None
    if not tensor.is_contiguous() or tensor.storage_offset() != 0:
        return None
    for start, nbytes, reference, handle in _spilled_mappings:
        if (
            reference() is not None
            and start == tensor.data_ptr()
            and nbytes == tensor.numel() * tensor.element_size()
            and tuple(tensor.shape) == tuple(handle.shape)
            and tensor.dtype == handle.dtype
        ):
            return handle
    return None


def _map_spill_file(path: str, np_dtype: str, shape: tuple[int, ...]) -> torch.Tensor:
    """mmap a spill file as a torch tensor, writable when the file permits it.

    Prefers ``r+`` over ``r``. A read-only mapping is PROT_READ, and ``torch.from_numpy`` does not
    enforce that -- an in-place write on the resulting tensor would reach the mapping and take a
    SIGSEGV rather than raising. Downstream code (partitioning, label handling) is not audited for
    in-place mutation of loaded tensors, so a writable mapping trades an unbounded crash risk for
    dirtied page-cache pages, which is the cheaper failure. Unwritten pages stay clean and
    evictable either way, which is the point of spilling.
    """
    import numpy as np

    try:
        mapped = np.memmap(path, dtype=np_dtype, mode="r+", shape=shape)
    except OSError:
        mapped = np.memmap(path, dtype=np_dtype, mode="r", shape=shape)
    array = np.asarray(mapped)
    tensor = torch.from_numpy(array)
    _register_mapping(
        array, tensor, SpilledTensorHandle(path=path, dtype=tensor.dtype, shape=shape)
    )
    return tensor


def spill_tensor_to_disk(tensor: torch.Tensor) -> Optional[SpilledTensor]:
    """Write ``tensor`` to disk and return a :class:`SpilledTensor`, or ``None`` to keep it.

    ``None`` covers every reason not to spill -- spilling disabled, tensor below the threshold,
    unsupported dtype, IO failure -- so callers can treat it as "keep what you had".

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
    # Already living in a file -- most likely allocated by allocate_disk_backed rather than filled in
    # memory and spilled. Copying it to a second file would double the disk use and the time for no
    # benefit, so hand back a descriptor for the file it is already in.
    existing = disk_backed_handle(tensor)
    if existing is not None:
        logger.info(
            f"share_memory: {tuple(tensor.shape)} {tensor.dtype} is already on disk at "
            f"{existing.path}; not writing a second copy"
        )
        return SpilledTensor(
            tensor=tensor,
            path=existing.path,
            dtype=existing.dtype,
            shape=existing.shape,
        )
    if tensor.numel() * tensor.element_size() < _spill_min_bytes():
        return None
    _ensure_spill_dir_prepared(spill_dir)
    return _spill_to_mmap(tensor, spill_dir)


def _reserve_file_blocks(fd: int, n_bytes: int, spill_dir: str) -> None:
    """Reserve every block of a spill file now, or raise ``OSError``.

    Running out of space must be an error HERE, not a SIGBUS on some later page write -- a signal,
    not an exception, which no ``try`` can catch. A filesystem that cannot reserve at all
    (``posix_fallocate`` unsupported) raises too: proceeding with an unreserved mapping would keep
    the SIGBUS window open, so the caller falls back to memory instead.
    """
    while True:
        try:
            os.posix_fallocate(fd, 0, n_bytes)
            return
        except InterruptedError:
            continue  # EINTR: the syscall was interrupted, not refused
        except AttributeError as no_fallocate:
            raise OSError(
                errno.ENOSYS,
                f"posix_fallocate is unavailable on this platform, so a "
                f"{n_bytes / 2**30:.1f} GiB mapping under {spill_dir} cannot be made SIGBUS-safe",
            ) from no_fallocate
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
    is mandatory. ``np.memmap(mode="w+")`` gives the file its full apparent size without allocating
    it, so a filesystem that fills later fails at the moment a page is written -- and a write to a
    mapping that cannot be backed raises **SIGBUS**, which is a signal, not an exception: it
    bypasses every ``try`` here and kills the process with no traceback. Reserving turns that into
    an ``OSError`` at allocation time, where it can be reported and fallen back on; a filesystem
    that cannot reserve at all gets None rather than an unreserved mapping.

    Measured cost of writing a 251-column fp32 matrix through the mapping instead of RAM: 46.8 s vs
    15.1 s at 8M rows, so ~3.1x, i.e. roughly 6 minutes at the production shape by linear
    extrapolation -- an estimate, not an upper bound, since the production pattern scatters ~938k
    rows per chunk under cgroup dirty-page throttling.

    What it buys: the destination's own bytes, ~56.42 GiB for a rank's node feature matrix, stop
    being unreclaimable anonymous memory. The mapped pages are still charged to the cgroup, but as
    page cache the kernel can reclaim them under pressure instead of OOM-killing. Freed chunk arenas
    retained by the allocator are unaffected, so this does not remove the whole measured 1.63x peak.

    Returns None whenever a file-backed buffer is not available or not worth it -- spilling disabled,
    below the size threshold, dtype with no numpy equivalent, no room to reserve, reservation
    unsupported by the filesystem, IO failure -- so callers can simply fall back to ``torch.empty``.
    """
    spill_dir = _spill_dir()
    if spill_dir is None:
        return None
    if 0 in shape:
        return None
    np_dtype = _NUMPY_DTYPES.get(dtype)
    if np_dtype is None:
        return None
    element_size = torch.empty(0, dtype=dtype).element_size()
    n_bytes = element_size
    for extent in shape:
        n_bytes *= extent
    if n_bytes < _spill_min_bytes():
        return None
    _ensure_spill_dir_prepared(spill_dir)

    import numpy as np

    path: Optional[str] = None
    try:
        os.makedirs(spill_dir, exist_ok=True)
        fd, path = tempfile.mkstemp(dir=spill_dir, prefix=_SPILL_PREFIX, suffix=".bin")
        try:
            _reserve_file_blocks(fd, n_bytes, spill_dir)
        finally:
            os.close(fd)
        # "r+", NOT "w+": the file is already sized by the reservation, and numpy opens "w+" as
        # `w+b`, which TRUNCATES the file and hands back every block just reserved -- leaving it
        # sparse again and the SIGBUS risk exactly where it was
        mapped = np.memmap(
            path,
            dtype=np_dtype,
            mode="r+",
            shape=tuple(shape),
        )
        array = np.asarray(mapped)
        tensor = torch.from_numpy(array)
        _register_mapping(
            array,
            tensor,
            SpilledTensorHandle(path=path, dtype=dtype, shape=tuple(shape)),
        )
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


def _spill_to_mmap(tensor: torch.Tensor, spill_dir: str) -> Optional[SpilledTensor]:
    """Write ``tensor`` to a file under ``spill_dir`` and return an mmap view of it.

    Returns None if the tensor cannot be spilled (unsupported dtype, reservation refused by the
    filesystem, IO failure), in which case the caller should keep the tensor it already had.

    Why this helps: ``/dev/shm`` is tmpfs, i.e. RAM. Node features for a ~1B-node graph approach
    a terabyte at fp32; even split across replicas that is the largest resident tensor, and
    ``share_memory_()`` costs 1x in tmpfs plus a transient 2x in RSS. Backing the bytes with
    a real file turns them into evictable page cache instead.

    NOTE the mapping only stays file-backed within this process. Torch's multiprocessing reduction
    copies a numpy-owned storage into shared memory when the tensor is pickled, so anything that
    ships this tensor to another process undoes the spill. Ship ``SpilledTensor.path`` and call
    :func:`load_spilled_tensor` on the far side instead.
    """
    np_dtype = _NUMPY_DTYPES.get(tensor.dtype)
    if np_dtype is None:
        logger.warning(
            f"share_memory: dtype {tensor.dtype} not spillable, using shared memory"
        )
        return None
    import numpy as np

    path: Optional[str] = None
    try:
        os.makedirs(spill_dir, exist_ok=True)
        n_bytes = tensor.numel() * tensor.element_size()
        # mkstemp, not a name derived from shape and byte count. A derived name collides for two
        # tensors with the same shape and element width -- including different dtypes of the same
        # width -- and reopening that path with mode="w+" TRUNCATES the file still backing the
        # first tensor's live read-only mapping, silently replacing its contents. Nothing would
        # raise; the features would simply be wrong.
        fd, path = tempfile.mkstemp(dir=spill_dir, prefix=_SPILL_PREFIX, suffix=".bin")
        try:
            # Same SIGBUS-safety rule as allocate_disk_backed: every block is reserved before the
            # copy starts, so a filesystem that fills mid-copy fails as OSError here, not as a
            # signal on some later page write
            _reserve_file_blocks(fd, n_bytes, spill_dir)
        finally:
            os.close(fd)
        # "r+", not "w+": the reservation already sized the file, and "w+" would truncate the
        # reserved blocks away
        writable = np.memmap(path, dtype=np_dtype, mode="r+", shape=tuple(tensor.shape))
        # Copy in ~1 GiB slices so we never hold a second full-size buffer.
        row_bytes = (
            max(1, int(tensor[0].numel()) * tensor.element_size())
            if tensor.dim() > 1
            else tensor.element_size()
        )
        rows = max(1, int(2**30 // row_bytes))
        for start in range(0, tensor.shape[0], rows):
            writable[start : start + rows] = tensor[start : start + rows].numpy()
        writable.flush()
        del writable
        logger.info(
            f"share_memory: spilled {n_bytes / 2**30:.1f} GiB to {path} instead of /dev/shm"
        )
        return SpilledTensor(
            tensor=_map_spill_file(path, np_dtype, tuple(tensor.shape)),
            path=path,
            dtype=tensor.dtype,
            shape=tuple(tensor.shape),
        )
    except Exception as e:  # noqa: BLE001
        # Unlink the partial file. mkstemp has already created it, so a failure in the mmap, the
        # copy, the flush or the remap would otherwise leave its bytes on disk with nothing
        # referencing them. That matters most in the case that causes the failure: the spill
        # filesystem is finite and node features are tens of GiB per replica, so a couple of failed
        # attempts can exhaust the disk and push the tensors back into RAM -- turning a recoverable
        # spill failure into the OOM this whole mechanism exists to avoid
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

    Measured rather than inferred, because the obvious reading of the docs is wrong. Under
    ``file_system`` torch calls ``_new_using_filename_cpu``, which sounds like an ordinary temporary
    file and is not: the storage lands in ``/dev/shm`` (observed:
    ``/dev/shm/torch_<pid>_<random>_0``, 16,000,064 bytes for a 16 MB tensor), and ``TMPDIR`` receives
    only a 4 KiB ``torch-shm-dir-*`` directory holding the libshm manager's socket. Under
    ``file_descriptor`` the segment is ``shm_open``ed and immediately unlinked, so nothing is visible
    in the directory listing while it still consumes the mount's capacity.

    An earlier version of this returned ``TMPDIR`` for ``file_system``, which would have sized the
    check against the wrong filesystem in precisely the case the check exists for.
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

    Args:
        shape: Shape of the tensor.
        dtype: Dtype of the tensor.
        random_access: Set when the caller will write to SCATTERED offsets rather than stream
            through in order. Then memory is preferred over a file and disk is a fallback, because
            a file is catastrophically slower under that pattern -- see below. Defaults to False,
            which keeps the disk-first behaviour that saves the most memory.

    PLACEMENT
        Disk-first is right for a destination written in order: one pass, sequential, and the bytes
        become reclaimable page cache instead of unreclaimable anonymous memory.

        It is badly wrong for a destination written by scatter. An 8-byte write to an uncached file
        page costs a 4 KiB read-modify-write, and a chunked CSR build makes hundreds of passes over
        the same millions of pages of a tens-of-GiB ``indices`` (measured: 251 passes over 8.2M
        pages, 2.06 BILLION page touches, ~19 h at 60k IOPS). A production-scale run sat in exactly
        that loop for over an hour at 4% CPU with memory flat, having logged nothing since the
        allocation. The measured 3.1x cost of a file-backed destination came from a FEATURE matrix,
        ~1000 contiguous bytes per row written once; carrying that ratio across to a scatter was an
        extrapolation across a change in kind, not degree.

        ``random_access=True`` therefore prefers shared memory, and falls back to a file only when
        memory genuinely will not hold it -- logging the expected slowdown when it does, because a
        silent fall back to disk here is a run that appears to hang.

    WHY PRE-SHARED AT ALL
    GLT's ``Graph.__init__`` calls ``Topology.share_memory_()`` unconditionally, and for an ordinary
    anonymous tensor that copies every byte into ``/dev/shm``. The CSR therefore exists TWICE for the
    duration of the copy. A production-scale run was killed one second after logging that its CSR was
    built: indices 31.3 GiB + indptr 7.2 GiB duplicating to 77 GiB on top of an 84 GiB baseline,
    against a limit under 160 GiB. No log line sits between the two, which is why it looked like the
    CSR build itself.

    Two backends, both immune to that copy:

    * a file under ``GIGL_TENSOR_SPILL_DIR`` -- consumers that check :func:`is_disk_backed` before
      sharing leave the tensor alone, and the bytes are reclaimable file pages rather than
      unreclaimable anonymous memory (still cgroup-charged while resident);
    * POSIX shared memory allocated directly -- ``share_memory_()`` then finds it already shared and
      does nothing (verified: ``data_ptr`` unchanged across the call).

    Below the spill threshold it returns a plain tensor, where a duplicate costs little.

    Uninitialised in both cases, like ``torch.empty``.
    """
    n_bytes = torch.empty(0, dtype=dtype).element_size()
    for extent in shape:
        n_bytes *= extent

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
        # wrong: `_build_topology_without_edge_ids` would then pick a plain `Topology`, and
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
            f"the same full mount by graphlearn_torch's Graph.__init__, and that copy has OOM-killed "
            f"real runs. Either set GIGL_TENSOR_SPILL_DIR to a filesystem with room "
            f"(the intended configuration; the CSR is then file-backed and never copied), or raise "
            f"the container's shared-memory limit."
        )

    # `torch.empty(...).share_memory_()` would defeat the purpose: it allocates anonymously and then
    # copies, which is the duplication being avoided. Allocating the shared storage first means the
    # bytes are only ever written once.
    try:
        storage = torch.UntypedStorage._new_shared(n_bytes)
        tensor = torch.empty(0, dtype=dtype)
        # Contiguous strides, computed rather than left to be inferred: the typed overload of `set_`
        # that takes a size also requires a stride.
        stride: list[int] = [1] * len(shape)
        for axis in range(len(shape) - 2, -1, -1):
            stride[axis] = stride[axis + 1] * int(shape[axis + 1])
        tensor.set_(storage, 0, tuple(shape), tuple(stride))
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
    """Whether some tensor in this process still maps ``path``.

    Reads the spill registry, whose entries hold a weakref to the ``np.memmap``: a dead weakref means
    the last tensor over that file was collected and the mapping is gone.
    """
    return any(
        reference() is not None and handle.path == path
        for _, _, reference, handle in _spilled_mappings
    )


def release_page_cache_by_path(path: str) -> bool:
    """Request eviction of a spill file's page cache when nothing maps it any more.

    The counterpart to :func:`release_page_cache` for a tensor that has already been dropped. Once no
    mapping remains there are no page-table entries to clear, so ``FADV_DONTNEED`` alone is
    sufficient -- the reason the tensor version needs ``MADV_DONTNEED`` first does not apply.

    This is what a consumed input needs. ``del`` on an mmap-backed tensor unmaps it but leaves the
    file's pages charged to the cgroup until the kernel reclaims them, so an input and an output of
    the same size are both charged while the output is being written.

    Refuses when a mapping is still live, because ``FADV_DONTNEED`` SILENTLY SKIPS pages that are in
    any page table -- it would return success having freed nothing. One stale reference (a tuple
    built for a call, a name the caller forgot to drop) is enough to turn this into a no-op, and
    without this check the caller logs a saving it did not get. Use :func:`release_page_cache` on the
    tensor itself in that case, which unmaps first.

    Returns True if the eviction was REQUESTED -- ``FADV_DONTNEED`` is advisory and reports no
    count, so "requested" is all success can mean without measuring residency. False if a mapping
    is still live, the file could not be opened, or the platform lacks ``posix_fadvise``.
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
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError) as e:
        logger.info(f"share_memory: could not drop page cache for {path}: {e}")
        return False
    finally:
        os.close(fd)
    return True


def release_page_cache(tensor: torch.Tensor) -> bool:
    """Write a disk-backed tensor's dirty pages out and ask the kernel to drop them from the cache.

    Moving a tensor to a file changes the KIND of memory it occupies, not whether it is charged:
    cgroup v2 counts page cache in ``memory.current``, so a 56.4 GiB feature matrix on the spill
    filesystem still counts against the container's limit -- reclaimable rather than OOM-triggering,
    but counted. A production-scale run died with exactly that 56.4 GiB charged alongside the CSR
    build's anonymous allocations, within a few GiB of its limit.

    Dropping the cache once the file is written releases the charge as the pages go (writeback can
    briefly delay stragglers; the fsync below bounds that). The mapping stays
    valid; pages fault back in on next access, so this costs a cold first read and nothing else.
    Worth doing only for a tensor that is written now and not read until much later -- the
    partitioned node features, which are written before the graph build and not touched again until
    sampling starts.

    Three steps, in this order, and every one is load-bearing:

    1. ``msync`` through the backing memmap -- dirty pages cannot be dropped at all.
    2. ``MADV_DONTNEED`` on the mapping -- removes this process's page-table entries.
    3. ``fsync`` then ``POSIX_FADV_DONTNEED`` -- discards the now-unmapped clean page cache.

    Step 2 is the one that is easy to omit and fatal to omit. Linux implements ``FADV_DONTNEED`` via
    ``invalidate_mapping_pages()``, which SKIPS any page currently mapped into a page table, so
    calling it alone on a live mapping leaves every page resident and charged while returning
    success. ``MADV_DONTNEED`` is safe here only because the mapping is file-backed and has just been
    flushed; on an anonymous mapping it would discard the data outright.

    Returns True when every eviction step COMPLETED, which is not the same as proof that every page
    went: ``FADV_DONTNEED`` is advisory and reports no count, and another process holding live PTEs on
    the same file can keep pages resident. False means the sequence was incomplete -- not disk-backed,
    no live mapping, or a failing step -- and note that a failure after step 2 leaves this process's
    PTEs already dropped, so False does not mean nothing changed. To know what was actually evicted,
    measure residency with ``mincore``; the tests do exactly that.
    """
    handle = disk_backed_handle(tensor)
    if handle is None:
        return False

    # The numpy memmap that owns the mapping. `np.asarray(memmap)` keeps the memmap as its base, and
    # torch keeps that array alive, so it is still reachable through the registry.
    backing = None
    for _, _, reference, registered in _spilled_mappings:
        if registered is handle:
            array = reference()
            backing = getattr(array, "base", None) if array is not None else None
            break
    if backing is None or not hasattr(backing, "flush"):
        logger.warning(
            f"share_memory: no live mapping found for {handle.path}; cannot drop its cache"
        )
        return False

    try:
        backing.flush()  # msync: dirty pages cannot be dropped, so this is not optional.
    except (OSError, ValueError) as e:
        logger.warning(f"share_memory: could not msync {handle.path}: {e}")
        return False

    # STEP ORDER MATTERS, and getting it wrong makes this whole function a silent no-op.
    #
    # POSIX_FADV_DONTNEED alone does nothing here: Linux implements it via
    # invalidate_mapping_pages(), which SKIPS any page currently mapped into a page table. With the
    # tensor still mapped, every one of those pages stays resident and stays charged to the cgroup --
    # while the call returns success. So the PTEs have to go first.
    #
    # MADV_DONTNEED on a shared FILE mapping only drops this process's page-table entries; the next
    # access refaults from the file. (On an anonymous mapping it would discard the data -- this is
    # only safe because the mapping is file-backed and has just been msync'd.)
    mapping = getattr(backing, "_mmap", None)
    if mapping is None or not hasattr(mapping, "madvise"):
        logger.info(
            f"share_memory: cannot unmap pages for {handle.path} (no madvise available); its "
            f"pages stay charged to the cgroup until the kernel reclaims them"
        )
        return False
    try:
        mapping.madvise(mmap.MADV_DONTNEED)
    except (AttributeError, OSError, ValueError) as e:
        logger.info(
            f"share_memory: MADV_DONTNEED failed for {handle.path} ({e}); page-cache eviction "
            f"would be a no-op while the pages remain mapped, so nothing was dropped"
        )
        return False

    try:
        fd = os.open(handle.path, os.O_RDWR)
    except OSError as e:
        logger.warning(
            f"share_memory: could not reopen {handle.path} to drop cache: {e}"
        )
        return False
    try:
        os.fsync(fd)
        # length 0 means "to end of file".
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError) as e:
        logger.info(
            f"share_memory: could not drop page cache for {handle.path} ({e}); its pages stay "
            f"charged to the cgroup until the kernel reclaims them"
        )
        return False
    finally:
        os.close(fd)
    n_bytes = tensor.numel() * tensor.element_size()
    logger.info(
        f"share_memory: unmapped and requested eviction of {_human_bytes(n_bytes)} of page cache "
        f"for {handle.path}; the mapping stays valid and refaults on next read"
    )
    return True


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

    This function does NOT spill to disk, deliberately. Its callers hand the result to another
    process, and a spill whose path is not carried alongside it is undone the moment the tensor is
    pickled -- torch copies a numpy-owned storage into ``/dev/shm``, so spilling here would cost a
    full disk write AND the full RAM copy it was meant to avoid. Spilling belongs where a
    descriptor travels with the tensor: :func:`share_memory_for_ipc` and
    :func:`spill_tensor_to_disk`.

    A tensor that is ALREADY disk-backed is left alone for the same reason. ``share_memory_()`` on
    such a tensor copies every byte into ``/dev/shm`` -- measured at +16,128 kB for a 16 MiB
    tensor -- which would quietly undo a spill made earlier and elsewhere.

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
) -> dict[_KeyType, Union[torch.Tensor, SpilledTensorHandle]]:
    """Prepare a mapping of tensors to cross a process boundary, spilling instead of copying.

    Same intent as :func:`share_memory`, but for values that will be **pickled to another
    process**: a spilled tensor is replaced by a :class:`SpilledTensorHandle` rather than by its
    mmap view. Sending the view would copy every byte back into ``/dev/shm``, which is RAM, and
    silently undo the spill -- the reason this function exists separately.

    Tensors that are not spilled (spilling disabled, below the size threshold, unsupported dtype)
    go to POSIX shared memory exactly as before, which pickles by handle already.

    Returns a NEW dict; the input is left alone, since the caller usually still holds the tensors
    and dropping them is its decision.
    """
    prepared: dict[_KeyType, Union[torch.Tensor, SpilledTensorHandle]] = {}
    for key, value in entity.items():
        already_spilled = disk_backed_handle(value)
        if already_spilled is not None:
            # Spilled earlier by something else; send that file's descriptor rather than writing a
            # second copy of the same bytes.
            prepared[key] = already_spilled
            continue
        spilled = spill_tensor_to_disk(value)
        if spilled is not None:
            prepared[key] = spilled.handle
            # Nothing here keeps the mmap view alive on purpose: the far side re-maps from the
            # path, and holding a second mapping in this process buys nothing.
            continue
        share_memory(value)
        prepared[key] = value
    return prepared


def resolve_spilled_handles(
    value: Union[
        torch.Tensor,
        SpilledTensorHandle,
        dict[_KeyType, Union[torch.Tensor, SpilledTensorHandle]],
        None,
    ],
) -> Any:
    """Turn any :class:`SpilledTensorHandle` back into a tensor, mapped in this process.

    The receiving half of :func:`share_memory_for_ipc`. Accepts a bare value or a mapping of
    values -- matching how the loader ships either a single tensor (homogeneous) or a dict keyed
    by node/edge type (heterogeneous) -- and passes anything that is already a tensor straight
    through.
    """
    if value is None:
        return None
    if isinstance(value, SpilledTensorHandle):
        tensor = value.load()
        logger.info(
            f"share_memory: mapped spilled tensor {tuple(value.shape)} {value.dtype} from "
            f"{value.path} ({tensor.numel() * tensor.element_size() / 2**30:.1f} GiB, no copy)"
        )
        return tensor
    if isinstance(value, abc.Mapping):
        by_type = cast(dict[Any, Union[torch.Tensor, SpilledTensorHandle]], value)
        return {key: resolve_spilled_handles(item) for key, item in by_type.items()}
    return value
