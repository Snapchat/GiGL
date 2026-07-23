import ctypes
import os

from graphlearn_torch.channel import SampleMessage, ShmChannel

from gigl.src.common.utils.metrics_service_provider import get_metrics_service_instance

_libc = ctypes.CDLL("libc.so.6", use_errno=True)

# void *shmat(int shmid, const void *shmaddr, int shmflg);
_libc.shmat.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
_libc.shmat.restype = ctypes.c_void_p

# int shmdt(const void *shmaddr);
_libc.shmdt.argtypes = [ctypes.c_void_p]
_libc.shmdt.restype = ctypes.c_int


class SizedShmChannel(ShmChannel):
    """A ShmChannel subclass that adds direct shared-memory size inspection."""

    def qsize(self) -> int:
        # Obtain the shmid from the underlying C++ SampleQueue instance
        shmid = self._queue.__getstate__()

        # Attach to the shared memory segment in the current process
        ptr = _libc.shmat(shmid, None, 0)  # shmat returns (void *)(-1) on failure
        if ptr == ctypes.c_void_p(-1).value or ptr is None:
            err_num = ctypes.get_errno()
            error_msg = os.strerror(err_num)
            raise RuntimeError(
                f"shmat failed for shmid={shmid} with errno {err_num}: {error_msg}"
            )

        try:
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
            # 56 - 63     | released_offset_    | size_t   | 8 bytes | Internal ring release ptr
            # 64 - ...    | sem_t locks         | sem_t    | N bytes | POSIX semaphores
            # ==================================================================================
            write_block_id = ctypes.c_size_t.from_address(ptr + 32).value
            read_block_id = ctypes.c_size_t.from_address(ptr + 40).value
            return write_block_id - read_block_id
        finally:
            _libc.shmdt(ptr)


class MonitoredShmChannel(SizedShmChannel):
    """Monitored variant of SizedShmChannel that automatically records qsize on recv() as a gauge."""

    def __init__(self, channel_name: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._metric = f"{channel_name}.qsize"

    def recv(self, *args, **kwargs) -> SampleMessage:
        publisher = get_metrics_service_instance()
        publisher.add_gauge(self._metric, self.qsize())
        try:
            return super().recv(*args, **kwargs)
        finally:
            publisher.flush_metrics()  # TODO(jchmura): Can we afford this on each recv()?
