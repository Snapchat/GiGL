"""Unit tests for ``gigl.src.common.custom_launcher``."""

import os
from unittest.mock import MagicMock, patch

from absl.testing import absltest

from gigl.common import Uri
from gigl.env.custom_launcher import (
    GIGL_APPLIED_TASK_IDENTIFIER_ENV_KEY,
    GIGL_COMPONENT_ENV_KEY,
    GIGL_CPU_DOCKER_URI_ENV_KEY,
    GIGL_CUDA_DOCKER_URI_ENV_KEY,
    GIGL_PROCESS_COMMAND_ENV_KEY,
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
            process_command="ignored",
            process_runtime_args={"ignored": "v"},
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
                process_command="",
                process_runtime_args={},
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
                process_command="echo 'hello, world!",
                process_runtime_args={},
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
            process_command="",
            process_runtime_args={},
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
            process_command="python -m my.cli",
            process_runtime_args={},
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
        self.assertEqual(env[GIGL_PROCESS_COMMAND_ENV_KEY], "python -m my.cli")
        self.assertEqual(env[GIGL_CPU_DOCKER_URI_ENV_KEY], "gcr.io/p/cpu:tag")
        self.assertEqual(env[GIGL_CUDA_DOCKER_URI_ENV_KEY], "gcr.io/p/cuda:tag")
        # component is exported via .name (the enum member identifier).
        self.assertEqual(env[GIGL_COMPONENT_ENV_KEY], "Trainer")

    @patch("gigl.src.common.custom_launcher.subprocess.run")
    def test_dispatch_omits_optional_uris_when_none(self, mock_run: MagicMock) -> None:
        config = self._build_config(command="echo")
        launch_custom(
            custom_launcher_config=config,
            applied_task_identifier="job",
            task_config_uri=Uri("gs://bucket/task.yaml"),
            resource_config_uri=Uri("gs://bucket/resource.yaml"),
            process_command="echo",
            process_runtime_args={},
            cpu_docker_uri=None,
            cuda_docker_uri=None,
            component=GiGLComponents.Inferencer,
        )
        env = mock_run.call_args.kwargs["env"]
        # Optional URIs must be omitted entirely (not stringified to "None"
        # nor set to ""), so receivers see env.get(KEY) is None.
        self.assertNotIn(GIGL_CPU_DOCKER_URI_ENV_KEY, env)
        self.assertNotIn(GIGL_CUDA_DOCKER_URI_ENV_KEY, env)
        # Required keys are still present.
        self.assertEqual(env[GIGL_COMPONENT_ENV_KEY], "Inferencer")

    @patch("gigl.src.common.custom_launcher.subprocess.run")
    def test_dispatch_does_not_mutate_parent_os_environ(
        self, mock_run: MagicMock
    ) -> None:
        # Pre-condition: none of the GIGL_* keys leak into the parent.
        snapshot = dict(os.environ)
        config = self._build_config(command="echo")
        launch_custom(
            custom_launcher_config=config,
            applied_task_identifier="job",
            task_config_uri=Uri("gs://bucket/task.yaml"),
            resource_config_uri=Uri("gs://bucket/resource.yaml"),
            process_command="echo",
            process_runtime_args={},
            cpu_docker_uri="gcr.io/p/cpu:tag",
            cuda_docker_uri="gcr.io/p/cuda:tag",
            component=GiGLComponents.Trainer,
        )
        self.assertEqual(dict(os.environ), snapshot)
        for key in (
            GIGL_APPLIED_TASK_IDENTIFIER_ENV_KEY,
            GIGL_TASK_CONFIG_URI_ENV_KEY,
            GIGL_RESOURCE_CONFIG_URI_ENV_KEY,
            GIGL_PROCESS_COMMAND_ENV_KEY,
            GIGL_CPU_DOCKER_URI_ENV_KEY,
            GIGL_CUDA_DOCKER_URI_ENV_KEY,
            GIGL_COMPONENT_ENV_KEY,
        ):
            self.assertNotIn(key, os.environ)

    @patch("gigl.src.common.custom_launcher.subprocess.run")
    def test_dispatch_preserves_inherited_env(self, mock_run: MagicMock) -> None:
        sentinel_key = "GIGL_TEST_PARENT_ENV_SENTINEL"
        sentinel_value = "preserved-value"
        try:
            os.environ[sentinel_key] = sentinel_value
            config = self._build_config(command="echo")
            launch_custom(
                custom_launcher_config=config,
                applied_task_identifier="job",
                task_config_uri=Uri("gs://bucket/task.yaml"),
                resource_config_uri=Uri("gs://bucket/resource.yaml"),
                process_command="echo",
                process_runtime_args={},
                cpu_docker_uri=None,
                cuda_docker_uri=None,
                component=GiGLComponents.Trainer,
            )
            env = mock_run.call_args.kwargs["env"]
            self.assertEqual(env.get(sentinel_key), sentinel_value)
        finally:
            os.environ.pop(sentinel_key, None)


if __name__ == "__main__":
    absltest.main()
