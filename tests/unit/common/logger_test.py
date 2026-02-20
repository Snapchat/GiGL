import logging
from unittest.mock import patch

from gigl.common.logger import Logger, _GcpJsonFormatter
from tests.test_assets.test_case import TestCase


class LoggerModeSelectionTest(TestCase):
    """Verify that ``Logger`` selects the correct handler/formatter based on environment."""

    def tearDown(self) -> None:
        # Clean up any loggers created during tests to avoid handler leaks.
        logging.Logger.manager.loggerDict.clear()
        logging.root.handlers.clear()

    @patch("gigl.common.logger._is_gcp_environment", return_value=True)
    def test_gcp_mode_uses_gcp_json_formatter(self, _mock_is_gcp: object) -> None:
        logger = Logger(name="test_gcp_mode")
        handlers = logger.logger.handlers
        self.assertTrue(len(handlers) > 0, "Expected at least one handler")
        self.assertIsInstance(handlers[0].formatter, _GcpJsonFormatter)

    @patch("gigl.common.logger._is_gcp_environment", return_value=False)
    def test_local_mode_uses_standard_formatter(self, _mock_is_gcp: object) -> None:
        logger = Logger(name="test_local_mode")
        handlers = logger.logger.handlers
        self.assertTrue(len(handlers) > 0, "Expected at least one handler")
        self.assertNotIsInstance(handlers[0].formatter, _GcpJsonFormatter)
        self.assertIsInstance(handlers[0].formatter, logging.Formatter)
