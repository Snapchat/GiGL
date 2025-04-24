from abc import ABC, abstractmethod


class OpsMetricPublisher(ABC):
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
