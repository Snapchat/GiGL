from typing import Any

from graphlearn_torch.channel import ChannelBase, SampleMessage

from gigl.src.common.utils.metrics_service_provider import (
    get_metrics_service_instance,
)


class MonitoredChannel(ChannelBase):
    """
    Decorator that wraps any glt.ChannelBase implementation to record queue metrics.
    """

    def __init__(self, channel: ChannelBase, metric_prefix: str):
        self._channel = channel
        self._enqueued_metric = f"{metric_prefix}.enqueued"
        self._dequeued_metric = f"{metric_prefix}.dequeued"

    def send(self, msg: SampleMessage, **kwargs) -> None:
        self._channel.send(msg, **kwargs)
        publisher = get_metrics_service_instance()
        publisher.add_count(self._enqueued_metric, 1)

    def recv(self, **kwargs) -> SampleMessage:
        msg = self._channel.recv(**kwargs)
        publisher = get_metrics_service_instance()
        publisher.add_count(self._dequeued_metric, 1)
        return msg

    def flush_metrics(self):
        publisher = get_metrics_service_instance()
        publisher.flush_metrics()
        publisher.add_count(self._reset_metric, 1)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._channel, name)
