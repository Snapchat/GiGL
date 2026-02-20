import json
import logging
import sys
from unittest.mock import patch

from gigl.common.logger import Logger, _GCP_LABELS_RECORD_ATTR, _GcpJsonFormatter
from tests.test_assets.test_case import TestCase


class GcpJsonFormatterTest(TestCase):
    def setUp(self) -> None:
        self.formatter = _GcpJsonFormatter()

    def _make_record(
        self,
        message: str = "test message",
        level: int = logging.INFO,
        exc_info: object = None,
    ) -> logging.LogRecord:
        """Create a ``LogRecord`` with deterministic source location."""
        record = logging.LogRecord(
            name="test",
            level=level,
            pathname="test_file.py",
            lineno=42,
            msg=message,
            args=None,
            exc_info=exc_info,  # type: ignore[arg-type]
        )
        record.funcName = "test_func"
        return record

    def test_basic_info_message_produces_valid_json(self) -> None:
        record = self._make_record()
        output = self.formatter.format(record)
        parsed = json.loads(output)

        self.assertEqual(parsed["severity"], "INFO")
        self.assertEqual(parsed["message"], "test message")
        self.assertIn("time", parsed)
        self.assertIn("logging.googleapis.com/sourceLocation", parsed)

    def test_severity_mapping_for_all_levels(self) -> None:
        levels = {
            logging.DEBUG: "DEBUG",
            logging.INFO: "INFO",
            logging.WARNING: "WARNING",
            logging.ERROR: "ERROR",
            logging.CRITICAL: "CRITICAL",
        }
        for python_level, expected_severity in levels.items():
            with self.subTest(level=python_level):
                record = self._make_record(level=python_level)
                parsed = json.loads(self.formatter.format(record))
                self.assertEqual(parsed["severity"], expected_severity)

    def test_output_is_single_line(self) -> None:
        record = self._make_record()
        output = self.formatter.format(record)
        self.assertEqual(output.count("\n"), 0)

    def test_time_field_is_iso_8601(self) -> None:
        record = self._make_record()
        parsed = json.loads(self.formatter.format(record))
        time_str = parsed["time"]
        # ISO 8601 with timezone: contains 'T' separator and '+' offset
        self.assertIn("T", time_str)
        self.assertIn("+", time_str)

    def test_extra_fields_appear_under_labels(self) -> None:
        record = self._make_record()
        setattr(record, _GCP_LABELS_RECORD_ATTR, {"custom_key": "custom_value"})
        parsed = json.loads(self.formatter.format(record))

        labels = parsed["logging.googleapis.com/labels"]
        self.assertEqual(labels["custom_key"], "custom_value")

    def test_no_labels_key_when_no_extras(self) -> None:
        record = self._make_record()
        parsed = json.loads(self.formatter.format(record))
        self.assertNotIn("logging.googleapis.com/labels", parsed)

    def test_exception_traceback_in_message(self) -> None:
        try:
            raise ValueError("boom")
        except ValueError:
            exc_info = sys.exc_info()

        record = self._make_record(exc_info=exc_info)
        parsed = json.loads(self.formatter.format(record))

        self.assertIn("ValueError: boom", parsed["message"])
        self.assertIn("Traceback", parsed["message"])

    def test_source_location_fields(self) -> None:
        record = self._make_record()
        parsed = json.loads(self.formatter.format(record))
        source = parsed["logging.googleapis.com/sourceLocation"]

        self.assertEqual(source["file"], "test_file.py")
        self.assertEqual(source["line"], 42)
        self.assertEqual(source["function"], "test_func")


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
