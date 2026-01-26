from gigl.common.metrics.metrics_interface import OpsMetricPublisher


class NopMetricsPublisher(OpsMetricPublisher):
    def __init__(self):
        self.timers = {}
        self.counts = {}
        self.levels = {}
        self.gauges = {}

    def add_count(self, metric_name: str, count: int):
        pass

    def add_timer(self, metric_name: str, timer: int):
        pass

    def add_level(self, metric_name: str, level: int):
        pass

    def add_gauge(self, metric_name: str, gauge: float):
        pass

    def flush_metrics(self):
        pass
