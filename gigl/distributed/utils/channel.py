from graphlearn_torch.channel import SampleMessage, ShmChannel

from gigl.src.common.utils.metrics_service_provider import (
    get_metrics_service_instance,
)


#class MonitoredShmChannel(ShmChannel):
#    """Monitored variant of ShmChannel that automatically records enqueue/dequeue metrics."""
#
#    def __init__(self, channel_name: str, *args, **kwargs) -> None:
#        super().__init__(*args, **kwargs)
#        self._enqueued_metric = f"{channel_name}.enqueued"
#        self._dequeued_metric = f"{channel_name}.dequeued"
#
#    def send(self, msg: SampleMessage, **kwargs) -> None:
#        super().send(msg, **kwargs)
#        publisher = get_metrics_service_instance()
#        publisher.add_count(self._enqueued_metric, 1)
#
#    def recv(self, *args, **kwargs) -> SampleMessage:
#        msg = super().recv(*args, **kwargs)
#        publisher = get_metrics_service_instance()
#        publisher.add_count(self._dequeued_metric, 1)
#        return msg
#
#    def flush_metrics(self):
#        publisher = get_metrics_service_instance()
#        publisher.flush_metrics()
#
import multiprocessing


class MonitoredShmChannel(ShmChannel):
    def __init__(self, channel_name: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._metric = f"{channel_name}.queue_size"
        self._shared_qsize = multiprocessing.Value("i", 0)

    def send(self, msg: SampleMessage, **kwargs) -> None:
        super().send(msg, **kwargs)

        with self._shared_qsize.get_lock():
            self._shared_qsize.value += 1
            qsize = self._shared_qsize.value

        publisher = get_metrics_service_instance()
        publisher.add_gauge(self._metric_name, qsize)

    def recv(self, *args, **kwargs) -> SampleMessage:
        msg = super().recv(*args, **kwargs)

        with self._shared_qsize.get_lock():
            self._shared_qsize.value = max(0, self._shared_qsize.value - 1)
            qsize = self._shared_qsize.value

        publisher = get_metrics_service_instance()
        publisher.add_gauge(self._metric_name, qsize)
        return msg
