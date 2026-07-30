import copy
import logging
import os
import pickle
from unittest import mock

from absl.testing import absltest

from gigl.common.logger import Logger, _is_cloud_logging_disabled
from tests.test_assets.test_case import TestCase


class TestLogger(TestCase):
    def test_undefined_attribute_raises_attribute_error(self) -> None:
        # __getattr__ must surface a clean AttributeError for unknown attributes
        # rather than recursing forever (the wrapped logger is stored as
        # ``self.logger``, not ``self._logger``).
        logger = Logger(name="test_undefined_attr")
        with self.assertRaises(AttributeError):
            logger.this_attribute_does_not_exist

    def test_delegates_to_wrapped_logger(self) -> None:
        logger = Logger(name="test_delegation")
        # ``level`` is defined on the wrapped logging.Logger, reached via __getattr__.
        self.assertEqual(logger.level, logger.logger.level)

    def test_deepcopy_round_trips(self) -> None:
        logger = Logger(name="test_deepcopy")
        clone = copy.deepcopy(logger)
        self.assertIsInstance(clone, Logger)
        clone.info("deepcopy works")

    def test_pickle_round_trips(self) -> None:
        logger = Logger(name="test_pickle")
        clone = pickle.loads(pickle.dumps(logger))
        self.assertIsInstance(clone, Logger)
        clone.info("pickle works")

    def test_disabling_cloud_logging_uses_console_handler(self) -> None:
        # KUBERNETES_SERVICE_HOST alone would send records to Google Cloud Logging as
        # GCP JSON; GIGL_DISABLE_CLOUD_LOGGING must win and give the console format.
        with mock.patch.dict(
            os.environ,
            {
                "KUBERNETES_SERVICE_HOST": "10.0.0.1",
                "GIGL_DISABLE_CLOUD_LOGGING": "1",
            },
        ):
            logger = Logger(name="test_cloud_logging_disabled")

        handlers = logger.logger.handlers
        self.assertEqual(len(handlers), 1)
        self.assertIsInstance(handlers[0], logging.StreamHandler)
        record = logging.LogRecord(
            name="test_cloud_logging_disabled",
            level=logging.INFO,
            pathname="export.py",
            lineno=197,
            msg="upload took 8.82 seconds",
            args=None,
            exc_info=None,
            func="_flush",
        )
        formatted = handlers[0].format(record)
        self.assertIn("[INFO] : upload took 8.82 seconds", formatted)
        self.assertIn("(export.py:_flush:197)", formatted)

    def test_cloud_logging_stays_enabled_for_falsy_values(self) -> None:
        for value in ("", "0", "false", "False"):
            with self.subTest(value=value):
                with mock.patch.dict(os.environ, {"GIGL_DISABLE_CLOUD_LOGGING": value}):
                    self.assertFalse(_is_cloud_logging_disabled())


if __name__ == "__main__":
    absltest.main()
