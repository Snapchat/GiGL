import errno
import gc
import io
import os
import pickle
import tempfile
import unittest
from collections import abc
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from typing import Optional, Union
from unittest import mock

import torch
import torch.multiprocessing as mp
from graphlearn_torch.partition import RangePartitionBook
from parameterized import param, parameterized
from torch.testing import assert_close

from gigl.src.common.types.graph_data import NodeType
from gigl.utils import share_memory as share_memory_module
from gigl.utils.share_memory import (
    allocate_disk_backed,
    allocate_preshared,
    has_live_mapping,
    is_disk_backed,
    load_spilled_tensor,
    prepare_spill_dir,
    release_page_cache,
    release_page_cache_by_path,
    share_memory,
    share_memory_for_ipc,
    spill_tensor_to_disk,
)
from tests.test_assets.test_case import TestCase


def _spill_file_path(tensor: torch.Tensor) -> str:
    """The file behind a disk-backed tensor, read off its storage."""
    path = tensor.untyped_storage().filename
    assert path is not None, "the tensor is not disk-backed"
    return path


def _forking_pickle(value: object) -> bytes:
    """Serialize the way multiprocessing does, so the spill-aware reducer applies.

    ``pickle.dumps`` uses the plain pickler, which does not consult ForkingPickler's dispatch
    table -- tensors then serialize by value and every size assertion would be meaningless.
    """
    buffer = io.BytesIO()
    ForkingPickler(buffer).dump(value)
    return buffer.getvalue()


class ShareMemoryTest(TestCase):
    @parameterized.expand(
        [
            param(
                "Test share_memory when provided entity is None",
                entity=None,
            ),
            param(
                "Test share_memory when provided entity is homogeneous",
                entity=torch.ones(10),
            ),
            param(
                "Test share_memory when provided entity is heterogeneous",
                entity={
                    NodeType("user"): torch.ones(10),
                    NodeType("item"): torch.ones(20) * 2,
                },
            ),
            param(
                "Test share_memory with range partition book",
                entity=RangePartitionBook(
                    partition_ranges=[(0, 3), (3, 5)], partition_idx=0
                ),
            ),
        ]
    )
    def test_share_memory(
        self,
        _,
        entity: Optional[
            Union[torch.Tensor, RangePartitionBook, dict[NodeType, torch.Tensor]]
        ],
    ):
        # The -> None contract is public API: returning the entity would be a signature change
        self.assertIsNone(share_memory(entity=entity))
        if isinstance(entity, torch.Tensor):
            self.assertTrue(entity.is_shared())
        elif isinstance(entity, RangePartitionBook):
            # If we have a range partition book, we don't move the partition bounds to shared memory, as the shape of this tensor
            # is very small (being at equal in length to the number of machines) and GLT doesn't natively provide support for
            # serializing a range partition book.
            self.assertFalse(entity.partition_bounds.is_shared())
        elif isinstance(entity, abc.Mapping):
            for entity_tensor in entity.values():
                self.assertTrue(entity_tensor.is_shared())

    def test_share_empty_memory(self):
        # If tensors are empty, they should not be moved to shared_memory, as this may lead to transient failures, which may cause processes to hang.

        # 1D Empty Tensor
        empty_1d_tensor = torch.empty(0)
        share_memory(empty_1d_tensor)

        self.assertFalse(empty_1d_tensor.is_shared())

        # 2D Empty Tensor
        empty_2d_tensor = torch.empty((5, 0))
        share_memory(empty_2d_tensor)
        self.assertFalse(empty_2d_tensor.is_shared())


def _receive_in_child(
    in_queue: "mp.Queue",
    result_queue: "mp.Queue",
) -> None:
    """Receive a prepared mapping in a spawned child and report what arrived, plus the Shmem delta.

    Runs in a separate process on purpose: the failure this guards against only exists across a
    process boundary, where torch's own reduction would copy the bytes into ``/dev/shm``. The
    mapping travels through the QUEUE rather than as a spawn argument so the deserialization
    happens inside the measurement window below.
    """
    before = _shmem_kib()
    received = in_queue.get(timeout=120)
    after = _shmem_kib()
    result_queue.put(
        {
            key: (
                tuple(tensor.shape),
                str(tensor.dtype),
                tensor.sum().item(),
                is_disk_backed(tensor),
            )
            for key, tensor in received.items()
        }
    )
    result_queue.put(after - before)
    result_queue.close()
    result_queue.join_thread()


def _spill_in_child(result_queue: "mp.Queue") -> None:
    """Spill a tensor in a spawned child, send it back (it pickles by path), then exit.

    Mirrors what a tensor-loading child does, including exiting before the parent uses the file.
    """
    prepared = share_memory_for_ipc(
        {NodeType("user"): torch.arange(100_000, dtype=torch.float32)}
    )
    result_queue.put(prepared[NodeType("user")])
    result_queue.close()
    result_queue.join_thread()


def _share_memory_in_child(
    in_queue: "mp.Queue",
    result_queue: "mp.Queue",
) -> None:
    """Receive a spilled tensor, pass it to share_memory, and report whether it got copied to RAM."""
    tensor = in_queue.get(timeout=120)
    before = _shmem_kib()
    share_memory(tensor)
    result_queue.put((_shmem_kib() - before, is_disk_backed(tensor)))
    result_queue.close()
    result_queue.join_thread()


def _resident_page_fraction(tensor: torch.Tensor) -> Optional[float]:
    """Fraction of a tensor's pages currently resident, via ``mincore``, or None if unavailable.

    The only way to tell a real eviction from a call that returned success and did nothing. Reading
    the tensor to check would itself refault every page, so residency must be sampled through the
    kernel rather than by touching memory.
    """
    import ctypes
    import ctypes.util

    page_size = os.sysconf("SC_PAGE_SIZE")
    length = tensor.numel() * tensor.element_size()
    pages = (length + page_size - 1) // page_size
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        mincore = libc.mincore
    except (OSError, AttributeError):
        return None
    mincore.restype = ctypes.c_int
    mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
    vector = ctypes.create_string_buffer(pages)
    # mincore requires a page-aligned address; a whole mapping's data_ptr always is.
    address = tensor.data_ptr()
    if address % page_size:
        return None
    if mincore(ctypes.c_void_p(address), ctypes.c_size_t(length), vector) != 0:
        return None
    return sum(byte & 1 for byte in vector.raw) / pages


def _fallocate_reflected_in_st_blocks(directory: str) -> bool:
    """Whether this filesystem both supports ``posix_fallocate`` and shows it in ``st_blocks``.

    Two separate capabilities, and both are needed before a block-count assertion means anything.
    Probed on the directory under test rather than assumed, because tests run on whatever the
    developer or CI happens to mount.
    """
    fd, path = tempfile.mkstemp(dir=directory)
    probe_bytes = 1 << 20
    # getattr because the symbol does not exist off Linux, where this file must still type-check
    fallocate = getattr(os, "posix_fallocate", None)
    if fallocate is None:
        os.close(fd)
        os.unlink(path)
        return False
    try:
        fallocate(fd, 0, probe_bytes)
        return os.stat(path).st_blocks * 512 >= probe_bytes
    except OSError:
        return False
    finally:
        os.close(fd)
        try:
            os.unlink(path)
        except OSError:
            pass


def _shmem_kib() -> int:
    """POSIX shared memory in use, from /proc/meminfo.

    Read from meminfo rather than by listing ``/dev/shm``: torch's ``file_descriptor`` sharing
    strategy unlinks its segment immediately, so the directory looks empty while the pages are very
    much resident -- a directory scan reports a false negative here.
    """
    with open("/proc/meminfo") as meminfo:
        for line in meminfo:
            if line.startswith("Shmem:"):
                return int(line.split()[1])
    raise AssertionError("no Shmem line in /proc/meminfo")


@unittest.skipUnless(
    hasattr(os, "posix_fallocate"),
    "tensor spilling requires posix_fallocate, absent on this platform",
)
class ShareMemoryForIpcTest(TestCase):
    def setUp(self) -> None:
        self._spill_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._spill_dir.cleanup)
        self._env = mock.patch.dict(
            os.environ,
            {
                "GIGL_TENSOR_SPILL_DIR": self._spill_dir.name,
                # Well below the tensors here, so they spill; the production default is 2 GiB.
                "GIGL_TENSOR_SPILL_MIN_BYTES": str(64 * 1024),
            },
        )
        self._env.start()
        self.addCleanup(self._env.stop)
        # The prepared-marker is process-wide and deliberately survives across calls in production
        # (spawn children inherit it, which is what stops a later sibling from deleting an earlier
        # sibling's live spill files). That makes it test-visible state: any earlier test that goes
        # through share_memory_for_ipc sets it, and prepare_spill_dir then returns early, so
        # test_prepare_removes_files_from_previous_runs would find its stale file still there.
        # mock.patch.dict overrides keys but does not remove ones it was not given, so clear it
        # explicitly. Each test gets a fresh temp spill dir, so there is never anything live to
        # protect here.
        os.environ.pop("GIGL_TENSOR_SPILL_DIR_PREPARED", None)

    def test_spilled_tensors_pickle_by_path_not_by_value(self):
        entity = {
            NodeType("user"): torch.arange(100_000, dtype=torch.float32),
            # Under the threshold, so it should stay a tensor in shared memory.
            NodeType("item"): torch.arange(10, dtype=torch.float32),
        }
        prepared = share_memory_for_ipc(entity)

        big = prepared[NodeType("user")]
        self.assertTrue(is_disk_backed(big))
        small = prepared[NodeType("item")]
        self.assertFalse(is_disk_backed(small))
        self.assertTrue(small.is_shared())

        # The whole point: what crosses the boundary must be small, whatever the tensor's size.
        # Serialized the way multiprocessing serializes, where the spill-aware reducer applies.
        self.assertLess(len(_forking_pickle({NodeType("user"): big})), 4096)

    def test_a_pickled_spilled_tensor_remaps_the_same_file(self):
        prepared = share_memory_for_ipc(
            {NodeType("user"): torch.arange(100_000, dtype=torch.float32)}
        )
        original = prepared[NodeType("user")]

        received = pickle.loads(_forking_pickle(original))

        assert_close(received, original)
        self.assertEqual(_spill_file_path(received), _spill_file_path(original))
        # A shared mapping of the same file, not a copy: a write on one side is visible on the
        # other, which is from_file's documented contract and what path-shipping must preserve.
        received[42] = -1.0
        self.assertEqual(original[42].item(), -1.0)
        original[42] = 42.0

    def test_a_pickled_view_keeps_its_offset_and_strides(self):
        """Shipping a view by path alone would hand the receiver the wrong bytes.

        A square tensor's transpose has the same storage, pointer, shape and numel as the
        original -- only the strides differ. A slice differs only in offset and shape. Both must
        round-trip by value, not just by file.
        """
        prepared = share_memory_for_ipc(
            {
                NodeType("user"): torch.arange(256 * 256, dtype=torch.float32).reshape(
                    256, 256
                )
            }
        )
        mapped = prepared[NodeType("user")]

        transposed = pickle.loads(_forking_pickle(mapped.t()))
        assert_close(transposed, mapped.t())

        sliced = pickle.loads(_forking_pickle(mapped[3:17, 5:9]))
        assert_close(sliced, mapped[3:17, 5:9])

        # Neither round-trip may write a second spill file: the view travels as (path, offset,
        # shape, strides) over the one existing file.
        self.assertEqual(
            len(list(Path(self._spill_dir.name).glob("spill_*.bin"))),
            1,
        )

    def test_a_stale_pickle_fails_loudly_instead_of_fabricating_zeros(self):
        """``from_file`` CREATES a missing file, so a stale recipe must be refused, not mapped.

        If the spill file is unlinked while a pickled tensor is in flight, mapping the path would
        silently hand the receiver a fresh zero-filled file -- corrupt data with the SIGBUS
        reservation gone too. Missing and truncated files must both raise, and the failed rebuild
        must not leave a recreated file behind.
        """
        prepared = share_memory_for_ipc(
            {NodeType("user"): torch.arange(100_000, dtype=torch.float32)}
        )
        original = prepared[NodeType("user")]
        path = _spill_file_path(original)
        blob = _forking_pickle(original)

        with open(path, "r+b") as spill_file:
            spill_file.truncate(1024)
        with self.assertRaises(ValueError):
            pickle.loads(blob)

        os.unlink(path)
        with self.assertRaises(FileNotFoundError):
            pickle.loads(blob)
        self.assertFalse(
            os.path.exists(path), "the failed rebuild recreated the missing spill file"
        )

    def test_quantized_tensors_are_refused(self):
        """A quantized tensor's quantizer lives beside its storage, so raw bytes cannot rebuild it.

        Mapping one anyway produces a tensor that trips internal torch assertions on first use;
        both entry points must decline so the caller keeps a working in-memory tensor.
        """
        self.assertIsNone(allocate_disk_backed((20_000, 8), torch.qint8))
        quantized = torch.quantize_per_tensor(torch.ones(100_000), 0.1, 0, torch.qint8)
        self.assertIsNone(spill_tensor_to_disk(quantized))

    def test_the_reducer_is_inert_when_spilling_is_disabled(self):
        """Opt-in means opt-in: without ``GIGL_TENSOR_SPILL_DIR``, torch's own reduction runs.

        A user's unrelated ``from_file`` tensor must pickle exactly as it does without this module
        loaded -- torch copies it into shared memory, so the received tensor has no ``filename``.
        With spilling enabled the same tensor ships by path and keeps it.
        """
        side_file = os.path.join(self._spill_dir.name, "not_a_spill.bin")
        tensor = torch.from_file(side_file, shared=True, size=1000, dtype=torch.float32)
        tensor.fill_(7.0)
        strategy = mp.get_sharing_strategy()
        # file_system, because torch's fd-passing reduction cannot round-trip inside one process
        mp.set_sharing_strategy("file_system")
        self.addCleanup(mp.set_sharing_strategy, strategy)

        # Enabled first: path-shipping does not mutate the sender. Torch's reduction does -- it
        # relocates the sender's storage into /dev/shm in place, stripping the filename -- so the
        # disabled round-trip must come last
        received_enabled = pickle.loads(_forking_pickle(tensor))
        with mock.patch.dict(os.environ, {"GIGL_TENSOR_SPILL_DIR": ""}):
            received_disabled = pickle.loads(_forking_pickle(tensor))

        self.assertEqual(received_enabled.untyped_storage().filename, side_file)
        assert_close(received_enabled, torch.full((1000,), 7.0))
        assert_close(received_disabled, torch.full((1000,), 7.0))
        self.assertIsNone(received_disabled.untyped_storage().filename)

    def test_a_named_tensor_is_delegated_to_torch_which_refuses_it(self):
        """Shipping a named tensor by path would drop its names silently; torch raises instead.

        Delegation preserves that explicit refusal rather than swallowing it.
        """
        prepared = share_memory_for_ipc(
            {NodeType("user"): torch.arange(100_000, dtype=torch.float32)}
        )
        named = prepared[NodeType("user")].reshape(1000, 100).refine_names("row", "col")

        with self.assertRaisesRegex(RuntimeError, "[Nn]amed"):
            _forking_pickle(named)

    def test_tensors_resolve_to_the_same_values_in_another_process(self):
        entity = {
            NodeType("user"): torch.arange(100_000, dtype=torch.float32),
            NodeType("item"): torch.arange(10, dtype=torch.float32),
        }
        expected = {
            key: (tuple(t.shape), str(t.dtype), t.sum().item(), key == NodeType("user"))
            for key, t in entity.items()
        }
        prepared = share_memory_for_ipc(entity)

        ctx = mp.get_context("spawn")
        in_queue = ctx.Queue()
        result_queue = ctx.Queue()
        child = ctx.Process(target=_receive_in_child, args=(in_queue, result_queue))
        child.start()
        in_queue.put(prepared)
        got = result_queue.get(timeout=120)
        shmem_delta_kib = result_queue.get(timeout=120)
        child.join(timeout=120)

        self.assertEqual(child.exitcode, 0)
        self.assertEqual(got, expected)
        # 400 KB of features must not appear in RAM-backed shared memory on the far side. The
        # bound is loose because the whole machine's Shmem is being sampled, not just ours.
        self.assertLess(shmem_delta_kib, 200)

    def test_share_memory_leaves_a_disk_backed_tensor_alone(self):
        """``share_memory_()`` relocating an mmap view into /dev/shm would undo the spill.

        Checked in a separate process, where the tensor arrives by path: after ``share_memory`` it
        must still be disk-backed and RAM-backed shared memory must not have grown.
        """
        prepared = share_memory_for_ipc(
            {NodeType("user"): torch.arange(1_000_000, dtype=torch.float32)}
        )

        ctx = mp.get_context("spawn")
        in_queue = ctx.Queue()
        result_queue = ctx.Queue()
        child = ctx.Process(
            target=_share_memory_in_child, args=(in_queue, result_queue)
        )
        child.start()
        in_queue.put(prepared[NodeType("user")])
        shmem_delta_kib, still_disk_backed = result_queue.get(timeout=120)
        child.join(timeout=120)

        self.assertEqual(child.exitcode, 0)
        self.assertTrue(
            still_disk_backed, "share_memory relocated a disk-backed tensor"
        )
        self.assertLess(shmem_delta_kib, 512)

    def test_share_memory_still_shares_an_ordinary_tensor(self):
        tensor = torch.arange(1000, dtype=torch.float32)

        share_memory(tensor)

        self.assertTrue(tensor.is_shared())

    def test_allocate_disk_backed_returns_a_writable_file_backed_buffer(self):
        """The destination IS the file, so the bytes never occupy anonymous memory at all.

        Spilling has to fill memory first and then copy it out; a buffer that is assembled
        incrementally can skip that entirely.
        """
        buffer = allocate_disk_backed((5000, 8), torch.float32)

        assert buffer is not None
        self.assertEqual(buffer.shape, (5000, 8))
        self.assertEqual(buffer.dtype, torch.float32)
        self.assertTrue(is_disk_backed(buffer))
        # Writable, and scattered advanced-index assignment lands correctly -- that is exactly how
        # the range partitioner fills it.
        rows = torch.tensor([4999, 0, 2500])
        buffer[rows] = torch.arange(3, dtype=torch.float32).unsqueeze(1).repeat(1, 8)
        assert_close(buffer[4999], torch.zeros(8))
        assert_close(buffer[0], torch.ones(8))
        assert_close(buffer[2500], torch.full((8,), 2.0))
        self.assertEqual(
            len(list(Path(self._spill_dir.name).glob("spill_*.bin"))),
            1,
        )

    def test_dtypes_without_numpy_equivalents_are_spillable(self):
        """The mapping is torch-native, so torch-only dtypes like bfloat16 spill too."""
        buffer = allocate_disk_backed((20_000, 8), torch.bfloat16)

        assert buffer is not None
        self.assertTrue(is_disk_backed(buffer))
        buffer.fill_(2.0)
        assert_close(buffer[123], torch.full((8,), 2.0, dtype=torch.bfloat16))

    @parameterized.expand(
        [
            param(
                "allocate path",
                act=lambda: allocate_disk_backed((20_000, 8), torch.float32),
            ),
            param(
                "spill copy path",
                act=lambda: spill_tensor_to_disk(torch.randn(20_000, 8)),
            ),
        ]
    )
    def test_every_block_is_reserved_before_the_file_is_mapped(self, _, act):
        """The ORDER is the SIGBUS guard, so it is pinned deterministically.

        A mapping made before the reservation is sparse for as long as the gap lasts, and a write
        into a sparse page on a full filesystem dies as SIGBUS -- a signal no ``except`` can catch.
        Asserted on the call order rather than on block counts, which vary by filesystem.
        """
        calls: list[str] = []
        real_fallocate = os.posix_fallocate
        real_from_file = torch.UntypedStorage.from_file

        def fallocate_spy(fd: int, offset: int, n_bytes: int) -> None:
            calls.append("reserve")
            real_fallocate(fd, offset, n_bytes)

        def from_file_spy(path: str, shared: bool = False, nbytes: int = 0):
            calls.append("map")
            return real_from_file(path, shared=shared, nbytes=nbytes)

        with (
            mock.patch("os.posix_fallocate", side_effect=fallocate_spy),
            mock.patch.object(
                torch.UntypedStorage, "from_file", side_effect=from_file_spy
            ),
        ):
            produced = act()

        assert produced is not None
        self.assertEqual(calls, ["reserve", "map"])

    def test_the_reserved_blocks_are_still_allocated_after_mapping(self):
        """``from_file`` must not truncate the reservation away when it maps the file.

        Skipped rather than failed where ``posix_fallocate`` is unsupported or where ``st_blocks``
        does not reflect reservations -- compressed, copy-on-write, network and emulated filesystems
        all account for blocks differently, so ``st_blocks`` is not a portable oracle even though
        the reservation itself is mandatory.
        """
        if not _fallocate_reflected_in_st_blocks(self._spill_dir.name):
            self.skipTest(
                "this filesystem does not report reserved extents through st_blocks"
            )
        rows, columns = 20_000, 8
        expected_bytes = rows * columns * 4

        buffer = allocate_disk_backed((rows, columns), torch.float32)

        assert buffer is not None
        stat = os.stat(_spill_file_path(buffer))
        self.assertEqual(stat.st_size, expected_bytes)
        # st_blocks counts 512-byte units actually allocated; a sparse file reports far fewer than
        # its apparent size.
        self.assertGreaterEqual(
            stat.st_blocks * 512,
            expected_bytes,
            "the file is sparse -- the fallocate reservation was discarded by the mapping",
        )

    def test_a_filesystem_that_cannot_reserve_gets_memory_not_a_file(self):
        """Reservation is mandatory: a refused ``posix_fallocate`` means no file-backed tensor.

        An unreserved mapping takes SIGBUS -- a signal no ``except`` can catch -- if the filesystem
        fills, so both refusal flavours must yield None (the caller falls back to memory) and must
        leave no partial file behind. Covers the allocate path and the spill-copy path, which share
        the reservation.
        """
        for refusal in (errno.ENOSPC, errno.EOPNOTSUPP):
            with self.subTest(errno=errno.errorcode[refusal]):
                with mock.patch(
                    "os.posix_fallocate",
                    side_effect=OSError(refusal, os.strerror(refusal)),
                ):
                    buffer = allocate_disk_backed((20_000, 8), torch.float32)
                    spilled = spill_tensor_to_disk(torch.randn(20_000, 8))
                self.assertIsNone(buffer)
                self.assertIsNone(spilled)
                leftovers = [
                    entry.name
                    for entry in os.scandir(self._spill_dir.name)
                    if entry.name.startswith("spill_")
                ]
                self.assertEqual(
                    leftovers, [], "a refused reservation left a file behind"
                )

    def test_release_page_cache_unmaps_before_discarding(self):
        """The ORDER is the whole mechanism, so it is pinned deterministically.

        ``FADV_DONTNEED`` on a live mapping is a no-op that reports success -- Linux skips pages held
        in a page table. So ``MADV_DONTNEED`` must come first, and after the write-back because
        dirty pages cannot be dropped at all. A residency test alone would not catch a reordering
        that happens to work on one kernel.
        """
        buffer = allocate_disk_backed((40_000, 8), torch.float32)
        assert buffer is not None
        calls: list[str] = []

        def record_madvise(address: int, n_bytes: int, path: str) -> bool:
            calls.append("madvise")
            return True

        with (
            mock.patch.object(
                os, "fsync", side_effect=lambda fd: calls.append("fsync")
            ),
            mock.patch.object(
                share_memory_module, "_madvise_dontneed", side_effect=record_madvise
            ),
            mock.patch.object(
                os, "posix_fadvise", side_effect=lambda *a: calls.append("fadvise")
            ),
        ):
            released = release_page_cache(buffer)

        self.assertTrue(released)
        self.assertEqual(calls, ["fsync", "madvise", "fadvise"])

    def test_release_page_cache_reports_failure_when_it_cannot_unmap(self):
        """A no-op must report False, not True.

        Returning True after calling only ``FADV_DONTNEED`` would drop nothing while the pages are
        mapped, and the caller would log a saving it did not get.
        """
        buffer = allocate_disk_backed((40_000, 8), torch.float32)
        assert buffer is not None

        with mock.patch.object(
            share_memory_module, "_madvise_dontneed", return_value=False
        ):
            self.assertFalse(release_page_cache(buffer))

    def test_release_page_cache_actually_reduces_residency(self):
        """Measured, not assumed: ``mincore`` before and after.

        Proving the data survived is not enough -- it survives even when eviction silently fails --
        and touching every page to check repopulates the cache before anything can observe it, so
        residency has to be read BEFORE the refault.
        """
        rows, columns = (
            200_000,
            32,
        )  # 25.6 MB, enough that partial eviction is unambiguous
        buffer = allocate_disk_backed((rows, columns), torch.float32)
        assert buffer is not None
        expected = (
            torch.arange(rows, dtype=torch.float32).unsqueeze(1).repeat(1, columns)
        )
        buffer.copy_(expected)
        resident_before = _resident_page_fraction(buffer)
        if resident_before is None:
            self.skipTest("mincore unavailable on this platform")
        self.assertGreater(
            resident_before,
            0.5,
            "the buffer should be resident right after being written",
        )

        released = release_page_cache(buffer)
        resident_after = _resident_page_fraction(buffer)

        self.assertTrue(released)
        assert resident_after is not None
        self.assertLess(
            resident_after,
            0.1,
            f"pages stayed resident ({resident_after:.0%}) -- eviction did not happen",
        )
        # And only now touch it, which must refault the real data back.
        assert_close(buffer, expected)

    def test_release_page_cache_drops_resident_pages_and_keeps_the_data(self):
        """Dropping the cache must remove the cgroup charge without losing the bytes.

        Moving a tensor to a file makes its pages reclaimable but leaves them CHARGED to
        ``memory.current``, so a tens-of-GiB feature matrix still counts against the limit while
        later phases allocate. Here the simpler observable is that the values survive a refault.
        """
        rows, columns = 40_000, 8
        buffer = allocate_disk_backed((rows, columns), torch.float32)
        assert buffer is not None
        expected = (
            torch.arange(rows, dtype=torch.float32).unsqueeze(1).repeat(1, columns)
        )
        buffer.copy_(expected)

        dropped = release_page_cache(buffer)

        self.assertTrue(
            dropped, "the buffer was disk-backed, so the cache should have been dropped"
        )
        # The mapping is still valid and the data must refault intact -- dropping cache must never
        # be a data-loss operation.
        assert_close(buffer, expected)
        # And the flush must have reached the file, not just the cache: read it through a fresh
        # mapping that shares nothing with the original.
        remapped = load_spilled_tensor(
            _spill_file_path(buffer), buffer.dtype, tuple(buffer.shape)
        )
        assert_close(remapped, expected)

    def test_release_page_cache_declines_for_an_ordinary_tensor(self):
        self.assertFalse(release_page_cache(torch.arange(1000, dtype=torch.float32)))

    def test_release_by_path_refuses_while_a_tensor_still_maps_the_file(self):
        """A live mapping must be reported as a failure, not silently freeing nothing.

        ``FADV_DONTNEED`` skips pages that are in any page table, so calling it on a still-mapped
        file returns success having freed nothing -- the same trap that made ``release_page_cache``
        a no-op before ``MADV_DONTNEED`` was added. Here the caller cannot unmap on our behalf, so
        the only honest answer is False.
        """
        buffer = allocate_disk_backed((40_000, 8), torch.float32)
        assert buffer is not None
        path = _spill_file_path(buffer)
        buffer.fill_(1.0)

        self.assertTrue(has_live_mapping(path))
        self.assertFalse(
            release_page_cache_by_path(path),
            "a still-mapped file cannot have its cache dropped by fadvise alone",
        )

        del buffer
        gc.collect()

        self.assertFalse(has_live_mapping(path))
        self.assertTrue(
            release_page_cache_by_path(path),
            "with the mapping gone, fadvise is sufficient",
        )

    def test_preshared_raises_when_the_buffer_cannot_fit_the_mount(self):
        """A shared allocation larger than the mount must RAISE -- there is no safe fallback.

        Attempting it: sizing a tmpfs object does not reserve its pages, so `_new_shared` against a
        64 MiB /dev/shm -- Docker's default -- returns a storage and the first write past the limit
        takes SIGBUS, which no `except` can catch. Falling back to an anonymous tensor: GLT's
        unguarded `Topology.share_memory_()` then attempts the SAME oversized allocation on the SAME
        full mount a few seconds later, so the SIGBUS is merely deferred to somewhere less
        attributable. The n_bytes here (76 MiB) exceeds the mocked mount outright, not just the
        reserve.
        """
        tiny_mount = os.statvfs_result(
            (4096, 4096, 16384, 16384, 16384, 0, 0, 0, 0, 255)
        )  # 64 MiB total, 64 MiB free

        with (
            mock.patch.dict(os.environ, {"GIGL_TENSOR_SPILL_DIR": ""}),
            mock.patch.object(os, "statvfs", return_value=tiny_mount),
            self.assertRaisesRegex(RuntimeError, "SIGBUS"),
        ):
            allocate_preshared((10_000_000,), torch.int64)  # 76 MiB > 64 MiB mount

    def test_preshared_raises_when_only_the_reserve_is_breached(self):
        """The reserve is part of the contract: room for the tensor but not the reserve still raises,
        because torch's queues and the sampling channels allocate on the same mount."""
        tiny_mount = os.statvfs_result(
            (4096, 4096, 16384, 16384, 16384, 0, 0, 0, 0, 255)
        )

        with (
            mock.patch.dict(os.environ, {"GIGL_TENSOR_SPILL_DIR": ""}),
            mock.patch.object(os, "statvfs", return_value=tiny_mount),
            self.assertRaisesRegex(RuntimeError, "reserve"),
        ):
            allocate_preshared(
                (200_000,), torch.int64
            )  # 1.5 MiB, fits; reserve does not

    def test_preshared_survives_an_unreadable_mount(self):
        """statvfs failing must not block an allocation that would have worked."""
        with (
            mock.patch.dict(
                os.environ,
                {
                    "GIGL_TENSOR_SPILL_DIR": "",
                    "GIGL_TENSOR_SPILL_MIN_BYTES": str(4 * 1024),
                },
            ),
            mock.patch.object(os, "statvfs", side_effect=OSError("no such mount")),
        ):
            tensor = allocate_preshared((200_000,), torch.int64)

        self.assertTrue(tensor.is_shared())

    def test_preshared_uses_shared_memory_when_the_mount_has_room(self):
        with (
            mock.patch.dict(
                os.environ,
                {
                    "GIGL_TENSOR_SPILL_DIR": "",
                    "GIGL_TENSOR_SPILL_MIN_BYTES": str(4 * 1024),
                },
            ),
            # "has room" must be part of the FIXTURE, not an accident of the host: even 1.6 MB
            # needs n_bytes + the 2 GiB reserve free, so on a container with Docker's default
            # 64 MiB /dev/shm this test would exercise the failure branch instead
            mock.patch.object(
                share_memory_module, "_shared_memory_shortfall", return_value=None
            ),
        ):
            tensor = allocate_preshared((200_000,), torch.int64)

        self.assertTrue(tensor.is_shared())

    def test_release_by_path_reports_failure_for_a_missing_file(self):
        self.assertFalse(
            release_page_cache_by_path(os.path.join(self._spill_dir.name, "absent.bin"))
        )

    def test_allocate_disk_backed_declines_when_too_small_or_disabled(self):
        # Below GIGL_TENSOR_SPILL_MIN_BYTES (64 KiB here): not worth a file.
        self.assertIsNone(allocate_disk_backed((4, 4), torch.float32))
        # Zero-sized: nothing to back.
        self.assertIsNone(allocate_disk_backed((0, 8), torch.float32))
        with mock.patch.dict(os.environ, {"GIGL_TENSOR_SPILL_DIR": ""}):
            self.assertIsNone(allocate_disk_backed((5000, 8), torch.float32))

    def test_an_already_disk_backed_buffer_is_not_spilled_again(self):
        """Spilling a tensor that already lives in a spill file must reuse that file: copying it to
        a second one doubles both the disk use and the time for no benefit."""
        buffer = allocate_disk_backed((5000, 8), torch.float32)
        assert buffer is not None

        spilled = spill_tensor_to_disk(buffer)

        self.assertIs(spilled, buffer)
        self.assertEqual(
            len(list(Path(self._spill_dir.name).glob("spill_*.bin"))),
            1,
            "a second copy of the same bytes was written",
        )

    def test_an_already_spilled_tensor_is_not_spilled_twice(self):
        first = share_memory_for_ipc(
            {NodeType("user"): torch.arange(100_000, dtype=torch.float32)}
        )
        tensor = first[NodeType("user")]

        second = share_memory_for_ipc({NodeType("user"): tensor})

        self.assertIs(second[NodeType("user")], tensor)
        self.assertEqual(
            len(list(Path(self._spill_dir.name).glob("spill_*.bin"))),
            1,
            "the same bytes were written to a second spill file",
        )

    def test_a_failed_spill_leaves_no_partial_file(self):
        """A failure after ``mkstemp`` must not leave orphaned bytes on the spill filesystem.

        The spill filesystem is finite and node features are tens of GiB per replica, so a couple of
        abandoned partial writes exhaust the disk, spilling stops working, and the tensors go back
        into RAM -- converting a recoverable spill failure into the OOM the spill exists to prevent.
        """
        with mock.patch.object(
            os, "fsync", side_effect=OSError("no space left on device")
        ):
            spilled = spill_tensor_to_disk(torch.arange(100_000, dtype=torch.float32))

        self.assertIsNone(spilled, "a failed spill must report failure")
        self.assertEqual(
            list(Path(self._spill_dir.name).glob("spill_*.bin")),
            [],
            "a partial spill file was left behind",
        )

    def test_prepare_removes_files_from_previous_runs(self):
        stale = Path(self._spill_dir.name) / "spill_from_a_previous_run.bin"
        stale.write_bytes(b"\0" * 1024)
        unrelated = Path(self._spill_dir.name) / "someone_elses_file.bin"
        unrelated.write_bytes(b"\0" * 16)

        prepare_spill_dir()

        self.assertFalse(stale.exists())
        # Only files this module could have written are fair game.
        self.assertTrue(unrelated.exists())

    def test_a_later_sibling_does_not_delete_an_earlier_sibling_spill(self):
        """The exact ordering that sequential loading produces, in real processes.

        The edge child spills and EXITS; then the node child starts, and only then does it spill.
        An age-based cleanup in the second child deletes the first child's file -- which the parent
        has not mapped yet, because it maps only after every child has joined. This test exists
        because an in-process version of it cannot reproduce that ordering and passed while the bug
        was live.
        """
        prepare_spill_dir()
        ctx = mp.get_context("spawn")

        first_queue = ctx.Queue()
        first = ctx.Process(target=_spill_in_child, args=(first_queue,))
        first.start()
        first_tensor = first_queue.get(timeout=120)
        first.join(timeout=120)
        self.assertEqual(first.exitcode, 0)
        first_path = _spill_file_path(first_tensor)
        self.assertTrue(
            Path(first_path).exists(),
            "the first child's spill vanished on its own",
        )

        second_queue = ctx.Queue()
        second = ctx.Process(target=_spill_in_child, args=(second_queue,))
        second.start()
        second_tensor = second_queue.get(timeout=120)
        second.join(timeout=120)
        self.assertEqual(second.exitcode, 0)

        self.assertNotEqual(first_path, _spill_file_path(second_tensor))
        self.assertTrue(
            Path(first_path).exists(),
            "the second child deleted the first child's live spill file",
        )
        # And the parent -- which received both tensors after their writers exited -- still gets
        # the data.
        assert_close(first_tensor, torch.arange(100_000, dtype=torch.float32))
        assert_close(second_tensor, torch.arange(100_000, dtype=torch.float32))


if __name__ == "__main__":
    from absl.testing import absltest

    absltest.main()
