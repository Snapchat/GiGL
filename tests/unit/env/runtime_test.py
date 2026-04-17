"""Tests for ``gigl.env.runtime`` execution-environment detection."""

import os
from unittest.mock import patch

from absl.testing import absltest

from gigl.env.runtime import RuntimeEnv, get_runtime_env, is_ray_runtime
from tests.test_assets.test_case import TestCase


class TestIsRayRuntime(TestCase):
    """Exercises each branch of the ``is_ray_runtime`` priority chain."""

    def test_gigl_ray_runtime_authoritative_signal_wins(self) -> None:
        with patch.dict(os.environ, {"GIGL_RAY_RUNTIME": "1"}, clear=True):
            self.assertTrue(is_ray_runtime())

    def test_ray_dashboard_address_triggers_ray(self) -> None:
        with patch.dict(
            os.environ,
            {"RAY_DASHBOARD_ADDRESS": "http://10.0.0.1:8265"},
            clear=True,
        ):
            self.assertTrue(is_ray_runtime())

    def test_ray_address_triggers_ray(self) -> None:
        with patch.dict(
            os.environ, {"RAY_ADDRESS": "ray://10.0.0.1:10001"}, clear=True
        ):
            self.assertTrue(is_ray_runtime())

    def test_gigl_ray_runtime_must_be_one(self) -> None:
        # Only the literal string "1" is treated as the authoritative signal.
        with patch.dict(os.environ, {"GIGL_RAY_RUNTIME": "0"}, clear=True):
            # Cannot `ray.init` in a unit test; rely on the ImportError fallback
            # path returning False when ray itself is unavailable. If ray *is*
            # installed, fall through to is_initialized() which must be False.
            self.assertFalse(is_ray_runtime())


class TestGetRuntimeEnv(TestCase):
    """Exercises each branch of ``get_runtime_env``."""

    def test_gigl_ray_runtime_returns_ray(self) -> None:
        with patch.dict(os.environ, {"GIGL_RAY_RUNTIME": "1"}, clear=True):
            self.assertEqual(get_runtime_env(), RuntimeEnv.RAY)

    def test_cloud_ml_job_id_returns_vertex_ai(self) -> None:
        with patch.dict(os.environ, {"CLOUD_ML_JOB_ID": "12345"}, clear=True):
            self.assertEqual(get_runtime_env(), RuntimeEnv.VERTEX_AI)

    def test_aip_model_dir_returns_vertex_ai(self) -> None:
        with patch.dict(os.environ, {"AIP_MODEL_DIR": "gs://bucket/model"}, clear=True):
            self.assertEqual(get_runtime_env(), RuntimeEnv.VERTEX_AI)

    def test_no_env_returns_unknown(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_runtime_env(), RuntimeEnv.UNKNOWN)
            self.assertFalse(is_ray_runtime())


if __name__ == "__main__":
    absltest.main()
