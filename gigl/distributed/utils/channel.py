import ctypes
import ctypes.util
import os
import weakref
from functools import cached_property
from itertools import count
from typing import Any, Optional

from graphlearn_torch.channel import SampleMessage, ShmChannel

from gigl.common.metrics.metrics_interface import OpsMetricPublisher
from gigl.src.common.utils.metrics_service_provider import get_metrics_service_instance


class SizedShmChannel(ShmChannel):
    """Extends ShmChannel with queue size method `qsize()` by attaching to the channels memory region and inspecting the C++ struct layout.

    TODO: Revisit direct memory inspection vs. custom C++ channel implementation. Current solution
    inspects GLT's underlying C++ memory layout for simplicity, avoiding the overhead of porting GLT's
    full C++ queue code into GiGL. Revisit and implement a native C++ channel with public size methods if:
        (a) Deeper channel monitoring is needed (e.g., % queue filled in bytes).
        (b) We roll a custom IPC queue for other architectural reasons.
        (c) An upstream GLT release breaks the struct memory layout (tests should catch this).
    """

    # Class-level handle cached across all instances in the current process
    _libc: Optional[ctypes.CDLL] = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._finalizer: Optional[weakref.finalize] = None

    def __len__(self) -> int:
        """The number of `SampleMessage` items currently in the channel."""
        return self.qsize()

    def qsize(self) -> int:
        """The number of `SampleMessage` items currently in the channel."""
        # ShmQueueMeta Memory Layout in Shared Memory (64-bit Architecture)
        # Reference: https://github.com/alibaba/graphlearn-for-pytorch/blob/88ff111ac0d9e45c6c9d2d18cfc5883dca07e9f9/graphlearn_torch/include/shm_queue.h#L65
        # ==================================================================================
        # Byte Offset | C++ Member Variable | Type     | Size    | Description
        # ------------+---------------------+----------+---------+--------------------------
        # 00 - 07     | max_block_num_      | size_t   | 8 bytes | Capacity (max block count)
        # 08 - 15     | max_buf_size_       | size_t   | 8 bytes | Max buffer size in bytes
        # 16 - 23     | block_meta_offset_  | size_t   | 8 bytes | Offset to BlockMeta array
        # 24 - 31     | data_buf_offset_    | size_t   | 8 bytes | Offset to raw data buffer
        # 32 - 39     | write_block_id_     | size_t   | 8 bytes | Total messages enqueued <-- READ HERE
        # 40 - 47     | read_block_id_      | size_t   | 8 bytes | Total messages dequeued <-- READ HERE
        # 48 - 55     | alloc_offset_       | size_t   | 8 bytes | Internal ring write ptr
        # 56 - ...    | released_offset_    | size_t   | 8 bytes | Internal ring release ptr
        # ==================================================================================
        ptr = self._shm_ptr
        write_block_id = ctypes.c_size_t.from_address(ptr + 32).value
        read_block_id = ctypes.c_size_t.from_address(ptr + 40).value
        return write_block_id - read_block_id

    @cached_property
    def _shm_ptr(self) -> int:
        # Obtain the shmid from the underlying C++ SampleQueue instance
        shmid = self._queue.__getstate__()

        # Attach to the shared memory segment in the current process
        libc = self._get_libc()
        ptr = libc.shmat(shmid, None, 0)  # shmat returns (void *)(-1) on failure
        if ptr == ctypes.c_void_p(-1).value or ptr is None:
            err_num = ctypes.get_errno()
            error_msg = os.strerror(err_num)
            raise RuntimeError(f"shmat failed for shmid={shmid}: {error_msg}")

        # Register automatic cleanup when this object is GC'd in this process
        self._finalizer = weakref.finalize(self, libc.shmdt, ptr)
        return ptr

    @classmethod
    def _get_libc(cls) -> ctypes.CDLL:
        if cls._libc is None:
            libc_path = ctypes.util.find_library("c")
            if libc_path is None:
                raise RuntimeError(
                    "Failed to locate standard C library ('libc') via ctypes.util.find_library('c')."
                )
            libc = ctypes.CDLL(libc_path, use_errno=True)

            # void *shmat(int shmid, const void *shmaddr, int shmflg);
            libc.shmat.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
            libc.shmat.restype = ctypes.c_void_p

            # int shmdt(const void *shmaddr);
            libc.shmdt.argtypes = [ctypes.c_void_p]
            libc.shmdt.restype = ctypes.c_int

            cls._libc = libc

        return cls._libc

    def __getstate__(self) -> dict[str, Any]:
        # Invalidate cached pointer and finalizer across process boundaries
        state = self.__dict__.copy()
        state.pop("_shm_ptr", None)
        state.pop("_finalizer", None)
        return state


class MonitoredShmChannel(SizedShmChannel):
    # Counts instantiations of this class, per process.
    # This is needed so we can generate unique channel names for each instance within the same process.
    # NOTE: This is per-class, not per-instance.
    _counter = count(0)

    def __init__(self, channel_name: str, *args, **kwargs) -> None:
        """Monitored variant of SizedShmChannel that integrates with GiGL metrics service and records queue size on recv() as a gauge.

        Args:
            channel_name: Prefix for published metrics. Must be unique across
                processes to disambiguate channels owned by different dataloaders.
                Multiple instances within the same process are automatically
                disambiguated by an internally appended sequence ID (e.g., `id0`).
            *args: Positional arguments forwarded directly to `ShmChannel`.
            **kwargs: Keyword arguments forwarded directly to `ShmChannel`.

        Example:
            Passing `channel_name="my_channel_pid_12345"` publishes queue size for
            the first instance as `my_channel_pid_12345_id0_qsize`.
        """
        super().__init__(*args, **kwargs)
        self._channel_name = f"{channel_name}_id{next(self._counter)}"

    def recv(self, *args, **kwargs) -> SampleMessage:
        publisher: Optional[OpsMetricPublisher] = get_metrics_service_instance()
        if publisher is None:
            raise RuntimeError(
                "Failed to record channel metrics in MonitoredShmChannel: the metric publisher "
                "could not be retrieved. Check logs for metrics class construction errors."
            )
        publisher.add_gauge(f"{self._channel_name}_qsize", self.qsize())
        return super().recv(*args, **kwargs)
