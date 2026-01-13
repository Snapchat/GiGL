import os
import unittest
from unittest import mock

from gigl.env.distributed import GIGL_COMPONENT_ENV_KEY
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.env import get_component


class TestGetComponent(unittest.TestCase):
    """Test suite for get_component function."""

    @mock.patch.dict(os.environ, {GIGL_COMPONENT_ENV_KEY: GiGLComponents.Trainer.value})
    def test_get_component_valid_value(self):
        """Test get_component returns correct component when env var is valid."""
        result = get_component()
        self.assertEqual(result, GiGLComponents.Trainer)

    @mock.patch.dict(os.environ, {GIGL_COMPONENT_ENV_KEY: "invalid_component"})
    def test_get_component_invalid_value(self):
        """Test get_component raises ValueError when env var is invalid."""
        with self.assertRaises(ValueError):
            get_component()

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_get_component_not_set(self):
        """Test get_component raises KeyError when env var is not set."""
        with self.assertRaises(KeyError):
            get_component()


if __name__ == "__main__":
    unittest.main()
