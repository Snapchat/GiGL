import copy
import pickle

from absl.testing import absltest

from gigl.common.logger import Logger
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


if __name__ == "__main__":
    absltest.main()
