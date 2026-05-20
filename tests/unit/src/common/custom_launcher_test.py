"""Unit tests for ``gigl.src.common.custom_launcher``."""

import os
from unittest.mock import MagicMock, patch

from absl.testing import absltest

from gigl.common import Uri
from gigl.common.constants import (
    DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU,
    DEFAULT_GIGL_RELEASE_SRC_IMAGE_CUDA,
)
from gigl.env.constants import (
    GIGL_APPLIED_TASK_IDENTIFIER_ENV_KEY,
    GIGL_COMPONENT_ENV_KEY,
    GIGL_CPU_DOCKER_URI_ENV_KEY,
    GIGL_CUDA_DOCKER_URI_ENV_KEY,
    GIGL_RESOURCE_CONFIG_URI_ENV_KEY,
    GIGL_TASK_CONFIG_URI_ENV_KEY,
)
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.custom_launcher import launch_custom
from snapchat.research.gbml import gigl_resource_config_pb2
from tests.test_assets.test_case import TestCase


class TestLaunchCustom(TestCase):
    """Exercises ``launch_custom`` subprocess dispatch and guards.

    The launcher takes ``command`` and ``args[]`` from the proto
    verbatim (no template substitution) and shells out via
    ``subprocess.run``. Tests patch ``subprocess.run`` to capture the
    composed shell line without actually spawning processes.
    """

    def _build_config(
        self,
        command: str,
        args: list[str] | None = None,
    ) -> gigl_resource_config_pb2.CustomLauncherConfig:
        return gigl_resource_config_pb2.CustomLauncherConfig(
            command=command,
            args=args or [],
        )

    @patch("gigl.src.common.custom_launcher.subprocess.run")
    def test_dispatches_subprocess_with_literal_command_and_args(
        self, mock_run: MagicMock
    ) -> None:
        config = self._build_config(
            command="python -m my.cli",
            args=["--foo=bar", "--baz=qux"],
        )
        launch_custom(
            custom_launcher_config=config,
            applied_task_identifier="job-42",
            task_config_uri=Uri("gs://bucket/task.yaml"),
            resource_config_uri=Uri("gs://bucket/resource.yaml"),
            cpu_docker_uri="gcr.io/p/cpu:tag",
            cuda_docker_uri="gcr.io/p/cuda:tag",
            component=GiGLComponents.Trainer,
        )

        mock_run.assert_called_once()
        shell_line = mock_run.call_args.args[0]
        self.assertIn("python -m my.cli", shell_line)
        self.assertIn("--foo=bar", shell_line)
        self.assertIn("--baz=qux", shell_line)
        # subprocess invoked with shell=True and check=True.
        self.assertTrue(mock_run.call_args.kwargs.get("shell", False))
        self.assertTrue(mock_run.call_args.kwargs.get("check", False))

    @patch("gigl.src.common.custom_launcher.subprocess.run")
    def test_empty_command_raises_value_error(self, mock_run: MagicMock) -> None:
        config = self._build_config(command="", args=["ignored"])
        with self.assertRaises(ValueError):
            launch_custom(
                custom_launcher_config=config,
                applied_task_identifier="job",
                task_config_uri=Uri("gs://bucket/task.yaml"),
                resource_config_uri=Uri("gs://bucket/resource.yaml"),
                cpu_docker_uri=None,
                cuda_docker_uri=None,
                component=GiGLComponents.Trainer,
            )
        mock_run.assert_not_called()

    @patch("gigl.src.common.custom_launcher.subprocess.run")
    def test_invalid_component_raises_value_error(self, mock_run: MagicMock) -> None:
        config = self._build_config(command="echo")
        with self.assertRaises(ValueError):
            launch_custom(
                custom_launcher_config=config,
                applied_task_identifier="job",
                task_config_uri=Uri("gs://bucket/task.yaml"),
                resource_config_uri=Uri("gs://bucket/resource.yaml"),
                cpu_docker_uri=None,
                cuda_docker_uri=None,
                component=GiGLComponents.DataPreprocessor,
            )
        mock_run.assert_not_called()

    @patch("gigl.src.common.custom_launcher.subprocess.run")
    def test_args_with_spaces_are_shell_quoted(self, mock_run: MagicMock) -> None:
        config = self._build_config(command="echo", args=["a b c", "--name=with space"])
        launch_custom(
            custom_launcher_config=config,
            applied_task_identifier="job",
            task_config_uri=Uri("gs://bucket/task.yaml"),
            resource_config_uri=Uri("gs://bucket/resource.yaml"),
            cpu_docker_uri=None,
            cuda_docker_uri=None,
            component=GiGLComponents.Trainer,
        )
        shell_line = mock_run.call_args.args[0]
        # shlex.quote wraps tokens with spaces in single quotes so the
        # shell sees one argv element per proto args[] entry.
        self.assertIn("'a b c'", shell_line)
        self.assertIn("'--name=with space'", shell_line)

    @patch("gigl.src.common.custom_launcher.subprocess.run")
    def test_dispatch_sets_gigl_env_vars(self, mock_run: MagicMock) -> None:
        config = self._build_config(command="python -m my.cli")
        launch_custom(
            custom_launcher_config=config,
            applied_task_identifier="job-42",
            task_config_uri=Uri("gs://bucket/task.yaml"),
            resource_config_uri=Uri("gs://bucket/resource.yaml"),
            cpu_docker_uri="gcr.io/p/cpu:tag",
            cuda_docker_uri="gcr.io/p/cuda:tag",
            component=GiGLComponents.Trainer,
        )
        env = mock_run.call_args.kwargs["env"]
        self.assertEqual(env[GIGL_APPLIED_TASK_IDENTIFIER_ENV_KEY], "job-42")
        self.assertEqual(env[GIGL_TASK_CONFIG_URI_ENV_KEY], "gs://bucket/task.yaml")
        self.assertEqual(
            env[GIGL_RESOURCE_CONFIG_URI_ENV_KEY], "gs://bucket/resource.yaml"
        )
        self.assertEqual(env[GIGL_CPU_DOCKER_URI_ENV_KEY], "gcr.io/p/cpu:tag")
        self.assertEqual(env[GIGL_CUDA_DOCKER_URI_ENV_KEY], "gcr.io/p/cuda:tag")
        # component is exported via .name (the enum member identifier).
        self.assertEqual(env[GIGL_COMPONENT_ENV_KEY], "Trainer")

    @patch("gigl.src.common.custom_launcher.subprocess.run")
    def test_dispatch_defaults_optional_uris_to_release_images(
        self, mock_run: MagicMock
    ) -> None:
        config = self._build_config(command="echo")
        launch_custom(
            custom_launcher_config=config,
            applied_task_identifier="job",
            task_config_uri=Uri("gs://bucket/task.yaml"),
            resource_config_uri=Uri("gs://bucket/resource.yaml"),
            cpu_docker_uri=None,
            cuda_docker_uri=None,
            component=GiGLComponents.Inferencer,
        )
        env = mock_run.call_args.kwargs["env"]
        # When the caller passes None for a docker URI, the env var
        # falls back to the public release image so receivers always
        # see a usable URI.
        self.assertEqual(
            env[GIGL_CPU_DOCKER_URI_ENV_KEY], DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU
        )
        self.assertEqual(
            env[GIGL_CUDA_DOCKER_URI_ENV_KEY], DEFAULT_GIGL_RELEASE_SRC_IMAGE_CUDA
        )
        self.assertEqual(env[GIGL_COMPONENT_ENV_KEY], "Inferencer")

    @patch("gigl.src.common.custom_launcher.subprocess.run")
    def test_dispatch_isolates_subprocess_env_from_parent(
        self, mock_run: MagicMock
    ) -> None:
        sentinel_key = "GIGL_TEST_PARENT_ENV_SENTINEL"
        sentinel_value = "preserved-value"
        try:
            os.environ[sentinel_key] = sentinel_value
            snapshot = dict(os.environ)
            config = self._build_config(command="echo")
            launch_custom(
                custom_launcher_config=config,
                applied_task_identifier="job",
                task_config_uri=Uri("gs://bucket/task.yaml"),
                resource_config_uri=Uri("gs://bucket/resource.yaml"),
                cpu_docker_uri="gcr.io/p/cpu:tag",
                cuda_docker_uri="gcr.io/p/cuda:tag",
                component=GiGLComponents.Trainer,
            )
            # Parent os.environ is untouched; none of the GIGL_* keys
            # leak into it.
            self.assertEqual(dict(os.environ), snapshot)
            for key in (
                GIGL_APPLIED_TASK_IDENTIFIER_ENV_KEY,
                GIGL_TASK_CONFIG_URI_ENV_KEY,
                GIGL_RESOURCE_CONFIG_URI_ENV_KEY,
                GIGL_CPU_DOCKER_URI_ENV_KEY,
                GIGL_CUDA_DOCKER_URI_ENV_KEY,
                GIGL_COMPONENT_ENV_KEY,
            ):
                self.assertNotIn(key, os.environ)
            # Inherited parent env entries reach the subprocess env.
            env = mock_run.call_args.kwargs["env"]
            self.assertEqual(env.get(sentinel_key), sentinel_value)
        finally:
            os.environ.pop(sentinel_key, None)


if __name__ == "__main__":
    absltest.main()
