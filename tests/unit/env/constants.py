import os
from unittest.mock import patch

from gigl.env.constants import is_env_flag_enabled
from tests.test_assets.test_case import TestCase


class EnvFlagEnabledTest(TestCase):
    def test_env_flag_disabled(self):
        env_var = "TEST_ENV_VAR"
        with patch.dict(os.environ, {}):
            self.assertFalse(is_env_flag_enabled(env_var))

    def test_env_flag_enabled_1(self):
        env_var = "TEST_ENV_VAR"
        with patch.dict(os.environ, {env_var: "1"}):
            self.assertTrue(is_env_flag_enabled(env_var))

    def test_env_flag_enabled_True(self):
        env_var = "TEST_ENV_VAR"
        with patch.dict(os.environ, {env_var: "True"}):
            self.assertTrue(is_env_flag_enabled(env_var))

    def test_env_flag_enabled_true(self):
        env_var = "TEST_ENV_VAR"
        with patch.dict(os.environ, {env_var: "true"}):
            self.assertTrue(is_env_flag_enabled(env_var))

    def test_env_flag_enabled_true_with_whitespace(self):
        env_var = "TEST_ENV_VAR"
        with patch.dict(os.environ, {env_var: "  true "}):
            self.assertTrue(is_env_flag_enabled(env_var))
