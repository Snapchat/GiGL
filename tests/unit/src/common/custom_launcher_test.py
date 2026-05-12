"""Unit tests for ``gigl.src.common.custom_launcher``."""

import os
from unittest.mock import MagicMock, patch

from absl.testing import absltest

from gigl.common import Uri
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.custom_launcher import launch_custom
from snapchat.research.gbml import gigl_resource_config_pb2
from tests.test_assets.test_case import TestCase


class TestLaunchCustom(TestCase):
    """Exercises ``launch_custom`` subprocess dispatch and guards.

    The launcher takes ``command`` and ``args[]`` from the proto
    verbatim (no template substitution) and shells out via
    ``subprocess.run``. Tests patch ``subprocess.run`` to capture the
    composed shell line and the per-call ``env`` dict without actually
    spawning processes.
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
                process_command="",
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
    def test_unsubstituted_placeholder_passes_through_verbatim(
        self, mock_run: MagicMock
    ) -> None:
        # The launcher performs no template substitution: any
        # placeholder in command/args reaches subprocess unchanged.
        # Consumers that want runtime context should read the GIGL_*
        # env vars the launcher injects (see module docstring).
        config = self._build_config(
            command="python -m my.cli",
            args=["--foo=${gigl:bar}"],
        )
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
        # The placeholder is preserved verbatim inside the shell-quoted
        # arg.
        self.assertIn("${gigl:bar}", shell_line)

    @patch("gigl.src.common.custom_launcher.subprocess.run")
    def test_env_dict_carries_gigl_keys(self, mock_run: MagicMock) -> None:
        # All six GIGL_* keys land on the subprocess env dict, sourced
        # from the corresponding kwargs.
        config = self._build_config(command="echo")
        launch_custom(
            custom_launcher_config=config,
            applied_task_identifier="job-99",
            task_config_uri=Uri("gs://bucket/task.yaml"),
            resource_config_uri=Uri("gs://bucket/resource.yaml"),
            process_command="",
            process_runtime_args={},
            cpu_docker_uri="gcr.io/p/cpu:tag",
            cuda_docker_uri="gcr.io/p/cuda:tag",
            component=GiGLComponents.Inferencer,
        )
        env = mock_run.call_args.kwargs["env"]
        self.assertEqual(env["GIGL_TASK_CONFIG_URI"], "gs://bucket/task.yaml")
        self.assertEqual(env["GIGL_RESOURCE_CONFIG_URI"], "gs://bucket/resource.yaml")
        self.assertEqual(env["GIGL_COMPONENT"], "Inferencer")
        self.assertEqual(env["GIGL_APPLIED_TASK_IDENTIFIER"], "job-99")
        self.assertEqual(env["GIGL_CUDA_DOCKER_IMAGE"], "gcr.io/p/cuda:tag")
        self.assertEqual(env["GIGL_CPU_DOCKER_IMAGE"], "gcr.io/p/cpu:tag")

    @patch("gigl.src.common.custom_launcher.subprocess.run")
    def test_env_dict_inherits_parent_env(self, mock_run: MagicMock) -> None:
        # The parent process's env (PATH, GCP creds, etc.) is forwarded
        # via os.environ.copy(); the launcher only overlays GIGL_*.
        sentinel_key = "CUSTOM_LAUNCHER_TEST_SENTINEL"
        with patch.dict(os.environ, {sentinel_key: "outer-value"}, clear=False):
            launch_custom(
                custom_launcher_config=self._build_config(command="echo"),
                applied_task_identifier="job",
                task_config_uri=Uri("gs://bucket/task.yaml"),
                resource_config_uri=Uri("gs://bucket/resource.yaml"),
                process_command="",
                process_runtime_args={},
                cpu_docker_uri=None,
                cuda_docker_uri=None,
                component=GiGLComponents.Trainer,
            )
        env = mock_run.call_args.kwargs["env"]
        self.assertEqual(env[sentinel_key], "outer-value")

    @patch("gigl.src.common.custom_launcher.subprocess.run")
    def test_parent_environ_not_mutated(self, mock_run: MagicMock) -> None:
        # Critical invariant: the launcher must not mutate the parent
        # process's os.environ. Ensures concurrent launch_custom calls
        # don't race or leak GIGL_* values into the parent process
        # (and into any sibling subprocess that started later but
        # inherited os.environ before the next launch_custom call).
        for key in (
            "GIGL_TASK_CONFIG_URI",
            "GIGL_RESOURCE_CONFIG_URI",
            "GIGL_COMPONENT",
            "GIGL_APPLIED_TASK_IDENTIFIER",
            "GIGL_CUDA_DOCKER_IMAGE",
            "GIGL_CPU_DOCKER_IMAGE",
        ):
            self.assertNotIn(
                key,
                os.environ,
                f"test prerequisite: {key!r} must be unset on os.environ before the call",
            )

        launch_custom(
            custom_launcher_config=self._build_config(command="echo"),
            applied_task_identifier="job",
            task_config_uri=Uri("gs://bucket/task.yaml"),
            resource_config_uri=Uri("gs://bucket/resource.yaml"),
            process_command="",
            process_runtime_args={},
            cpu_docker_uri="gcr.io/p/cpu:tag",
            cuda_docker_uri="gcr.io/p/cuda:tag",
            component=GiGLComponents.Trainer,
        )

        for key in (
            "GIGL_TASK_CONFIG_URI",
            "GIGL_RESOURCE_CONFIG_URI",
            "GIGL_COMPONENT",
            "GIGL_APPLIED_TASK_IDENTIFIER",
            "GIGL_CUDA_DOCKER_IMAGE",
            "GIGL_CPU_DOCKER_IMAGE",
        ):
            self.assertNotIn(
                key,
                os.environ,
                f"{key!r} leaked into parent os.environ after launch_custom",
            )

    @patch("gigl.src.common.custom_launcher.subprocess.run")
    def test_none_image_uris_become_empty_strings(self, mock_run: MagicMock) -> None:
        # Optional cuda/cpu docker URIs collapse to "" rather than the
        # literal "None" string. Subprocess CLIs reading these env vars
        # treat empty as "not provided".
        launch_custom(
            custom_launcher_config=self._build_config(command="echo"),
            applied_task_identifier="job",
            task_config_uri=Uri("gs://bucket/task.yaml"),
            resource_config_uri=Uri("gs://bucket/resource.yaml"),
            process_command="",
            process_runtime_args={},
            cpu_docker_uri=None,
            cuda_docker_uri=None,
            component=GiGLComponents.Trainer,
        )
        env = mock_run.call_args.kwargs["env"]
        self.assertEqual(env["GIGL_CUDA_DOCKER_IMAGE"], "")
        self.assertEqual(env["GIGL_CPU_DOCKER_IMAGE"], "")


if __name__ == "__main__":
    absltest.main()
