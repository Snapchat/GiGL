"""Unit tests for ``gigl.src.common.custom_launcher``."""

from unittest.mock import MagicMock, patch

from absl.testing import absltest

from gigl.common import Uri
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.custom_launcher import launch_custom
from snapchat.research.gbml import gigl_resource_config_pb2
from tests.test_assets.test_case import TestCase


class TestLaunchCustom(TestCase):
    """Exercises ``launch_custom`` subprocess dispatch and guards.

    The launcher resolves ``${gigl:*}`` placeholders in the proto's
    ``command`` / ``args`` fields against the runtime kwargs and shells
    out via ``subprocess.run``. Tests patch ``subprocess.run`` to
    capture the resolved shell line without actually spawning processes.
    """

    def _build_config(
        self,
        command: str,
        args: list[str] | None = None,
    ) -> gigl_resource_config_pb2.CustomResourceConfig:
        return gigl_resource_config_pb2.CustomResourceConfig(
            command=command,
            args=args or [],
        )

    @patch("gigl.src.common.custom_launcher.subprocess.run")
    def test_dispatches_subprocess_with_resolved_command_and_args(
        self, mock_run: MagicMock
    ) -> None:
        config = self._build_config(
            command="python -m my.cli",
            args=[
                "--task_config_uri=${gigl:task_config_uri}",
                "--component=${gigl:component}",
                "--cuda=${gigl:cuda_docker_image}",
                "--applied_task_identifier=${gigl:applied_task_identifier}",
            ],
        )
        launch_custom(
            custom_resource_config=config,
            applied_task_identifier="job-42",
            task_config_uri=Uri("gs://bucket/task.yaml"),
            resource_config_uri=Uri("gs://bucket/resource.yaml"),
            process_command="ignored",
            process_runtime_args={"ignored": "v"},
            cpu_docker_uri="gcr.io/p/cpu:tag",
            cuda_docker_uri="gcr.io/p/cuda:tag",
            component=GiGLComponents.Trainer,
            is_dry_run=False,
        )

        mock_run.assert_called_once()
        shell_line = mock_run.call_args.args[0]
        self.assertIn("python -m my.cli", shell_line)
        self.assertIn("--task_config_uri=gs://bucket/task.yaml", shell_line)
        # component should resolve to the Title-case name (matching CLI
        # argparse choices), NOT the lowercase enum value.
        self.assertIn("--component=Trainer", shell_line)
        self.assertIn("--cuda=gcr.io/p/cuda:tag", shell_line)
        self.assertIn("--applied_task_identifier=job-42", shell_line)
        # No leftover ${gigl:*} placeholders in the shell line.
        self.assertNotIn("${gigl:", shell_line)
        # subprocess invoked with shell=True and check=True.
        self.assertTrue(mock_run.call_args.kwargs.get("shell", False))
        self.assertTrue(mock_run.call_args.kwargs.get("check", False))

    @patch("gigl.src.common.custom_launcher.subprocess.run")
    def test_is_dry_run_skips_subprocess(self, mock_run: MagicMock) -> None:
        config = self._build_config(command="echo", args=["hi"])
        launch_custom(
            custom_resource_config=config,
            applied_task_identifier="job",
            task_config_uri=Uri("gs://bucket/task.yaml"),
            resource_config_uri=Uri("gs://bucket/resource.yaml"),
            process_command="",
            process_runtime_args={},
            cpu_docker_uri=None,
            cuda_docker_uri=None,
            component=GiGLComponents.Trainer,
            is_dry_run=True,
        )
        mock_run.assert_not_called()

    @patch("gigl.src.common.custom_launcher.subprocess.run")
    def test_is_dry_run_defaults_to_false(self, mock_run: MagicMock) -> None:
        config = self._build_config(command="echo", args=[])
        launch_custom(
            custom_resource_config=config,
            applied_task_identifier="job-43",
            task_config_uri=Uri("gs://bucket/task.yaml"),
            resource_config_uri=Uri("gs://bucket/resource.yaml"),
            process_command="",
            process_runtime_args={},
            cpu_docker_uri=None,
            cuda_docker_uri=None,
            component=GiGLComponents.Inferencer,
        )
        mock_run.assert_called_once()

    @patch("gigl.src.common.custom_launcher.subprocess.run")
    def test_empty_command_raises_value_error(self, mock_run: MagicMock) -> None:
        config = self._build_config(command="", args=["ignored"])
        with self.assertRaises(ValueError):
            launch_custom(
                custom_resource_config=config,
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
                custom_resource_config=config,
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
        config = self._build_config(
            command="echo", args=["a b c", "--name=with space"]
        )
        launch_custom(
            custom_resource_config=config,
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
    def test_process_command_and_runtime_args_are_not_plumbed(
        self, mock_run: MagicMock
    ) -> None:
        # Confirm the resolver dict does not carry process_command or
        # process_runtime_args — consumers re-derive them from
        # ${gigl:task_config_uri} on the receiving side.
        config = self._build_config(command="python", args=["-m", "foo"])
        launch_custom(
            custom_resource_config=config,
            applied_task_identifier="job",
            task_config_uri=Uri("gs://bucket/task.yaml"),
            resource_config_uri=Uri("gs://bucket/resource.yaml"),
            process_command="should-not-appear",
            process_runtime_args={"unused_lr": "0.42"},
            cpu_docker_uri=None,
            cuda_docker_uri=None,
            component=GiGLComponents.Trainer,
        )
        shell_line = mock_run.call_args.args[0]
        self.assertNotIn("should-not-appear", shell_line)
        self.assertNotIn("unused_lr", shell_line)
        self.assertNotIn("0.42", shell_line)

    @patch("gigl.src.common.custom_launcher.subprocess.run")
    def test_logs_resolved_shell_line(self, mock_run: MagicMock) -> None:
        config = self._build_config(command="echo", args=["${gigl:component}"])
        mock_logger = MagicMock()
        with patch("gigl.src.common.custom_launcher.logger", new=mock_logger):
            launch_custom(
                custom_resource_config=config,
                applied_task_identifier="job",
                task_config_uri=Uri("gs://bucket/task.yaml"),
                resource_config_uri=Uri("gs://bucket/resource.yaml"),
                process_command="",
                process_runtime_args={},
                cpu_docker_uri=None,
                cuda_docker_uri=None,
                component=GiGLComponents.Inferencer,
                is_dry_run=False,
            )
        mock_logger.info.assert_called_once()
        (log_line,), _ = mock_logger.info.call_args
        self.assertIn("Inferencer", log_line)
        self.assertIn("dry_run=False", log_line)


if __name__ == "__main__":
    absltest.main()
