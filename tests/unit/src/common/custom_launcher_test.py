"""Unit tests for ``gigl.src.common.custom_launcher``."""

from typing import Optional
from unittest.mock import MagicMock, patch

from absl.testing import absltest

from gigl.common import Uri
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.custom_launcher import launch_custom
from snapchat.research.gbml import gigl_resource_config_pb2
from tests.test_assets import custom_launcher_fixtures
from tests.test_assets.test_case import TestCase

# Fixture referenced by its canonical dotted path so ``import_obj`` and the
# test module both bind to the same module object (and thus the same call
# list).
_FAKE_LAUNCHER_PATH = (
    "tests.test_assets.custom_launcher_fixtures.fake_launcher_callable"
)
_NOT_CALLABLE_PATH = "tests.test_assets.custom_launcher_fixtures.NOT_CALLABLE_SENTINEL"


class TestLaunchCustom(TestCase):
    """Exercises ``launch_custom`` dispatch, guards, and arg forwarding."""

    def setUp(self) -> None:
        super().setUp()
        custom_launcher_fixtures.FAKE_LAUNCHER_CALLS.clear()

    def _build_config(
        self,
        launcher_fn: str,
        launcher_args: Optional[dict[str, str]] = None,
    ) -> gigl_resource_config_pb2.CustomResourceConfig:
        return gigl_resource_config_pb2.CustomResourceConfig(
            launcher_fn=launcher_fn,
            launcher_args=launcher_args or {},
        )

    def test_dispatches_with_expected_kwargs(self) -> None:
        config = self._build_config(
            launcher_fn=_FAKE_LAUNCHER_PATH,
            launcher_args={"cluster_size": "4", "image": "gcr.io/p/img:tag"},
        )
        task_uri = Uri("gs://bucket/task.yaml")
        resource_uri = Uri("gs://bucket/resource.yaml")

        launch_custom(
            custom_resource_config=config,
            applied_task_identifier="job-42",
            task_config_uri=task_uri,
            resource_config_uri=resource_uri,
            process_command="python -m gigl.src.training.v2.glt_trainer",
            process_runtime_args={"lr": "0.01"},
            cpu_docker_uri="gcr.io/p/cpu:tag",
            cuda_docker_uri="gcr.io/p/cuda:tag",
            component=GiGLComponents.Trainer,
            is_dry_run=True,
        )

        calls = custom_launcher_fixtures.FAKE_LAUNCHER_CALLS
        self.assertEqual(len(calls), 1)
        call_kwargs = calls[0]
        self.assertEqual(call_kwargs["applied_task_identifier"], "job-42")
        self.assertEqual(call_kwargs["task_config_uri"], task_uri)
        self.assertEqual(call_kwargs["resource_config_uri"], resource_uri)
        self.assertEqual(
            call_kwargs["process_command"],
            "python -m gigl.src.training.v2.glt_trainer",
        )
        self.assertEqual(call_kwargs["process_runtime_args"], {"lr": "0.01"})
        # launcher_args is materialized from the proto map as a plain dict.
        self.assertEqual(
            call_kwargs["launcher_args"],
            {"cluster_size": "4", "image": "gcr.io/p/img:tag"},
        )
        self.assertIsInstance(call_kwargs["launcher_args"], dict)
        self.assertEqual(call_kwargs["cpu_docker_uri"], "gcr.io/p/cpu:tag")
        self.assertEqual(call_kwargs["cuda_docker_uri"], "gcr.io/p/cuda:tag")
        self.assertEqual(call_kwargs["component"], GiGLComponents.Trainer)
        self.assertTrue(call_kwargs["is_dry_run"])

    def test_is_dry_run_defaults_to_false(self) -> None:
        config = self._build_config(launcher_fn=_FAKE_LAUNCHER_PATH)

        launch_custom(
            custom_resource_config=config,
            applied_task_identifier="job-43",
            task_config_uri=Uri("gs://bucket/task.yaml"),
            resource_config_uri=Uri("gs://bucket/resource.yaml"),
            process_command="python -m foo",
            process_runtime_args={},
            cpu_docker_uri=None,
            cuda_docker_uri=None,
            component=GiGLComponents.Inferencer,
        )

        calls = custom_launcher_fixtures.FAKE_LAUNCHER_CALLS
        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0]["is_dry_run"])

    def test_empty_launcher_fn_raises_value_error(self) -> None:
        config = self._build_config(launcher_fn="")

        with self.assertRaises(ValueError):
            launch_custom(
                custom_resource_config=config,
                applied_task_identifier="job",
                task_config_uri=Uri("gs://bucket/task.yaml"),
                resource_config_uri=Uri("gs://bucket/resource.yaml"),
                process_command="python -m foo",
                process_runtime_args={},
                cpu_docker_uri=None,
                cuda_docker_uri=None,
                component=GiGLComponents.Trainer,
            )

    def test_non_callable_launcher_raises_type_error(self) -> None:
        config = self._build_config(launcher_fn=_NOT_CALLABLE_PATH)

        with self.assertRaises(TypeError):
            launch_custom(
                custom_resource_config=config,
                applied_task_identifier="job",
                task_config_uri=Uri("gs://bucket/task.yaml"),
                resource_config_uri=Uri("gs://bucket/resource.yaml"),
                process_command="python -m foo",
                process_runtime_args={},
                cpu_docker_uri=None,
                cuda_docker_uri=None,
                component=GiGLComponents.Trainer,
            )

    def test_invalid_component_raises_value_error(self) -> None:
        config = self._build_config(launcher_fn=_FAKE_LAUNCHER_PATH)

        # Any non-Trainer, non-Inferencer component must be rejected before
        # the launcher is invoked.
        with self.assertRaises(ValueError):
            launch_custom(
                custom_resource_config=config,
                applied_task_identifier="job",
                task_config_uri=Uri("gs://bucket/task.yaml"),
                resource_config_uri=Uri("gs://bucket/resource.yaml"),
                process_command="python -m foo",
                process_runtime_args={},
                cpu_docker_uri=None,
                cuda_docker_uri=None,
                component=GiGLComponents.DataPreprocessor,
            )
        self.assertEqual(len(custom_launcher_fixtures.FAKE_LAUNCHER_CALLS), 0)

    def test_logs_launcher_arg_keys_not_values(self) -> None:
        # Confirm launcher_args keys (but not values) appear in the log line,
        # so launcher_args cannot leak secrets into logs.
        config = self._build_config(
            launcher_fn=_FAKE_LAUNCHER_PATH,
            launcher_args={"secret_token": "s3cr3t", "cluster_size": "4"},
        )

        mock_logger = MagicMock()
        with patch("gigl.src.common.custom_launcher.logger", new=mock_logger):
            launch_custom(
                custom_resource_config=config,
                applied_task_identifier="job",
                task_config_uri=Uri("gs://bucket/task.yaml"),
                resource_config_uri=Uri("gs://bucket/resource.yaml"),
                process_command="python -m foo",
                process_runtime_args={},
                cpu_docker_uri=None,
                cuda_docker_uri=None,
                component=GiGLComponents.Trainer,
            )

        mock_logger.info.assert_called_once()
        (log_line,), _ = mock_logger.info.call_args
        self.assertIn("cluster_size", log_line)
        self.assertIn("secret_token", log_line)
        self.assertNotIn("s3cr3t", log_line)


if __name__ == "__main__":
    absltest.main()
