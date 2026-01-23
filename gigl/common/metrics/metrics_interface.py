from abc import ABC, abstractmethod


class OpsMetricPublisher(ABC):
    """
    Interface to implement to publish metrics to some monitoring system.

    To publish metrics to a monitoring system, you need to implement a class with this interface
    and expose the class path in the task config file:
    ```
    metrics_config:
        metrics_cls_path: "path.to.the.class.that.implements.OpsMetricPublisher"
    ```
    """

    @abstractmethod
    def add_count(self, metric_name: str, count: int):
        pass

    @abstractmethod
    def add_timer(self, metric_name: str, timer: int):
        pass

    @abstractmethod
    def add_level(self, metric_name: str, level: int):
        pass

    @abstractmethod
    def add_gauge(self, metric_name: str, gauge: float):
        pass

    @abstractmethod
    def flush_metrics(self):
        pass
