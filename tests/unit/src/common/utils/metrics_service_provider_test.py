import tempfile

from gigl.common import LocalUri
from gigl.common.metrics.base_metrics import NopMetricsPublisher
from gigl.common.utils.proto_utils import ProtoUtils
from gigl.src.common.utils import metrics_service_provider
from gigl.src.common.utils.metrics_service_provider import (
    get_metrics_service_instance,
    initialize_metrics,
)
from snapchat.research.gbml import gbml_config_pb2
from tests.test_assets.test_case import TestCase


class MetricsServiceProviderTest(TestCase):
    def setUp(self) -> None:
        self.proto_utils = ProtoUtils()
        self.tmp_dir = tempfile.TemporaryDirectory()
        self._original_metrics_instance = metrics_service_provider._metrics_instance

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()
        metrics_service_provider._metrics_instance = self._original_metrics_instance

    def _write_task_config(self, config: gbml_config_pb2.GbmlConfig) -> LocalUri:
        uri = LocalUri.join(self.tmp_dir.name, "task_config.yaml")
        self.proto_utils.write_proto_to_yaml(proto=config, uri=uri)
        return uri

    def test_no_custom_class_returns_true_and_uses_nop(self) -> None:
        """initialize_metrics returns True and sets NopMetricsPublisher when no metrics class is configured."""
        config = gbml_config_pb2.GbmlConfig()
        uri = self._write_task_config(config)

        result = initialize_metrics(task_config_uri=uri, service_name="test_service")

        self.assertTrue(result)
        self.assertIsInstance(get_metrics_service_instance(), NopMetricsPublisher)

    def test_valid_custom_class_returns_true(self) -> None:
        """initialize_metrics returns True and uses the custom class when it is a valid OpsMetricPublisher."""
        config = gbml_config_pb2.GbmlConfig()
        config.metrics_config.metrics_cls_path = (
            "gigl.common.metrics.base_metrics.NopMetricsPublisher"
        )
        uri = self._write_task_config(config)

        result = initialize_metrics(task_config_uri=uri, service_name="test_service")

        self.assertTrue(result)
        self.assertIsInstance(get_metrics_service_instance(), NopMetricsPublisher)

    def test_invalid_custom_class_returns_false_and_falls_back_to_nop(self) -> None:
        """initialize_metrics returns False and falls back to NopMetricsPublisher when the class path does not exist."""
        config = gbml_config_pb2.GbmlConfig()
        config.metrics_config.metrics_cls_path = "gigl.does.not.exist.MetricsClass"
        uri = self._write_task_config(config)

        result = initialize_metrics(task_config_uri=uri, service_name="test_service")

        self.assertFalse(result)
        self.assertIsInstance(get_metrics_service_instance(), NopMetricsPublisher)

    def test_get_metrics_service_instance_raises_before_initialization(self) -> None:
        """get_metrics_service_instance raises RuntimeError when called before initialize_metrics."""
        with self.assertRaises(RuntimeError):
            get_metrics_service_instance()
