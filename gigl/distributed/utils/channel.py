from functools import cached_property
from itertools import count
from typing import Any, Optional

from gigl_core import ShmQueueProbe
from graphlearn_torch.channel import SampleMessage, ShmChannel

from gigl.common.metrics.metrics_interface import OpsMetricPublisher
from gigl.src.common.utils.metrics_service_provider import get_metrics_service_instance


class SizedShmChannel(ShmChannel):
    """Extends ShmChannel with a queue-depth method `qsize()`.

    GLT's ShmChannel exposes only `empty()`, so there is no way to ask how many messages a channel is
    holding. The depth is computed in C++ by `gigl_core.ShmQueueProbe`, which attaches to the same
    shared-memory segment and reads GLT's enqueue/dequeue counters using field offsets the compiler
    derives from GLT's own headers.

    The reported size is an instantaneous approximation: producers and consumers mutate the counters
    concurrently and the probe holds none of GLT's locks. Use it for metrics, not for control flow.
    """

    def __len__(self) -> int:
        """The number of `SampleMessage` items currently in the channel."""
        return self.qsize()

    def qsize(self) -> int:
        """The number of `SampleMessage` items currently in the channel."""
        return self._probe.qsize()

    @cached_property
    def _probe(self) -> ShmQueueProbe:
        # GLT pickles SampleQueue as its raw System V shmid, so __getstate__ is how the shmid is
        # obtained; the C++ ShmId() accessor is not otherwise exposed to Python.
        return ShmQueueProbe(self._queue.__getstate__())

    def __getstate__(self) -> dict[str, Any]:
        # A shared-memory attachment belongs to the process that made it, so drop the cached probe and
        # let the receiving process attach its own on first use.
        state = self.__dict__.copy()
        state.pop("_probe", None)
        return state


class MonitoredShmChannel(SizedShmChannel):
    """Monitored variant of SizedShmChannel that integrats with GiGL metrics_service and records queue size on recv() as a gauge."""

    # Counts instantiations of this class, per process.
    # This is needed so we can generate unique channel names for each instance within the same process.
    # NOTE: This is per-class, not per-instance.
    _counter = count(0)

    def __init__(self, channel_name: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._channel_name = f"{channel_name}_id{next(self._counter)}"
        self._publisher: Optional[OpsMetricPublisher] = get_metrics_service_instance()

    def recv(self, *args, **kwargs) -> SampleMessage:
        if self._publisher is not None:
            self._publisher.add_gauge(f"{self._channel_name}_qsize", self.qsize())
        return super().recv(*args, **kwargs)
