"""Unit tests for gigl.utils.tensorboard_writer."""

from unittest.mock import patch

from absl.testing import absltest

from gigl.utils import tensorboard_writer as tensorboard_writer_module
from gigl.utils.tensorboard_writer import TensorBoardWriter
from tests.test_assets.test_case import TestCase

_TB_RESOURCE = "projects/my-project/locations/us-central1/tensorboards/42"
_EXPERIMENT = "my-experiment"
_RUN = "my-job-name-20260507-120000"


class TestTensorBoardWriter(TestCase):
    """Tests for the TensorBoardWriter class."""

    def test_create_returns_noop_when_disabled(self) -> None:
        """Disabled (non-chief) writers must not touch aiplatform at all."""
        with (
            patch("google.cloud.aiplatform.init") as mock_init,
            patch("google.cloud.aiplatform.start_run") as mock_start_run,
            patch("google.cloud.aiplatform.log_time_series_metrics") as mock_log,
            patch("google.cloud.aiplatform.end_run") as mock_end,
        ):
            writer = TensorBoardWriter.create(
                resource_name=None,
                experiment_name=None,
                experiment_run_name=_RUN,
                enabled=False,
            )
            writer.log({"Loss/train": 1.0}, step=0)
            writer.close()

        mock_init.assert_not_called()
        mock_start_run.assert_not_called()
        mock_log.assert_not_called()
        mock_end.assert_not_called()

    def test_create_initializes_aiplatform_and_starts_run(self) -> None:
        with (
            patch("google.cloud.aiplatform.init") as mock_init,
            patch("google.cloud.aiplatform.start_run") as mock_start_run,
        ):
            TensorBoardWriter.create(
                resource_name=_TB_RESOURCE,
                experiment_name=_EXPERIMENT,
                experiment_run_name=_RUN,
                enabled=True,
            )

        mock_init.assert_called_once_with(
            project="my-project",
            location="us-central1",
            experiment=_EXPERIMENT,
            experiment_tensorboard=_TB_RESOURCE,
        )
        mock_start_run.assert_called_once_with(_RUN, resume=False)

    def test_create_raises_when_enabled_and_resource_name_missing(self) -> None:
        with (
            patch("google.cloud.aiplatform.init") as mock_init,
            patch("google.cloud.aiplatform.start_run") as mock_start_run,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                TensorBoardWriter.create(
                    resource_name=None,
                    experiment_name=_EXPERIMENT,
                    experiment_run_name=_RUN,
                    enabled=True,
                )

        self.assertIn("resource_name", str(ctx.exception))
        mock_init.assert_not_called()
        mock_start_run.assert_not_called()

    def test_create_raises_when_enabled_and_experiment_name_missing(self) -> None:
        with (
            patch("google.cloud.aiplatform.init") as mock_init,
            patch("google.cloud.aiplatform.start_run") as mock_start_run,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                TensorBoardWriter.create(
                    resource_name=_TB_RESOURCE,
                    experiment_name=None,
                    experiment_run_name=_RUN,
                    enabled=True,
                )

        self.assertIn("experiment_name", str(ctx.exception))
        mock_init.assert_not_called()
        mock_start_run.assert_not_called()

    def test_create_raises_when_enabled_and_run_name_missing(self) -> None:
        with (
            patch("google.cloud.aiplatform.init") as mock_init,
            patch("google.cloud.aiplatform.start_run") as mock_start_run,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                TensorBoardWriter.create(
                    resource_name=_TB_RESOURCE,
                    experiment_name=_EXPERIMENT,
                    experiment_run_name="",
                    enabled=True,
                )

        self.assertIn("experiment_run_name", str(ctx.exception))
        mock_init.assert_not_called()
        mock_start_run.assert_not_called()

    def test_create_raises_on_invalid_resource_name(self) -> None:
        with (
            patch("google.cloud.aiplatform.init") as mock_init,
            patch("google.cloud.aiplatform.start_run") as mock_start_run,
        ):
            with self.assertRaises(ValueError) as ctx:
                TensorBoardWriter.create(
                    resource_name="not-a-valid-resource-name",
                    experiment_name=_EXPERIMENT,
                    experiment_run_name=_RUN,
                    enabled=True,
                )

        self.assertIn("resource_name", str(ctx.exception))
        mock_init.assert_not_called()
        mock_start_run.assert_not_called()

    def test_create_logs_named_experiment_url_on_start(self) -> None:
        """The named-experiment URL is logged so engineers can find the TB
        page from trainer stdout.
        """
        with (
            patch("google.cloud.aiplatform.init"),
            patch("google.cloud.aiplatform.start_run"),
            patch.object(tensorboard_writer_module.logger, "info") as mock_info,
        ):
            TensorBoardWriter.create(
                resource_name=_TB_RESOURCE,
                experiment_name=_EXPERIMENT,
                experiment_run_name=_RUN,
                enabled=True,
            )

        url_logs = [
            call.args[0]
            for call in mock_info.call_args_list
            if "View TensorBoard" in call.args[0]
        ]
        self.assertEqual(len(url_logs), 1)
        self.assertIn(_EXPERIMENT, url_logs[0])
        self.assertIn("tensorboards+42", url_logs[0])
        self.assertIn("us-central1", url_logs[0])

    def test_log_forwards_to_log_time_series_metrics(self) -> None:
        with patch("google.cloud.aiplatform.log_time_series_metrics") as mock_log:
            writer = TensorBoardWriter(active=True)
            writer.log({"Loss/train": 1.5, "Loss/val": 2.0}, step=10)

        mock_log.assert_called_once_with({"Loss/train": 1.5, "Loss/val": 2.0}, step=10)

    def test_log_is_noop_when_inactive(self) -> None:
        with patch("google.cloud.aiplatform.log_time_series_metrics") as mock_log:
            writer = TensorBoardWriter(active=False)
            writer.log({"Loss/train": 1.0}, step=0)

        mock_log.assert_not_called()

    def test_log_is_noop_after_close(self) -> None:
        with (
            patch("google.cloud.aiplatform.end_run"),
            patch("google.cloud.aiplatform.log_time_series_metrics") as mock_log,
        ):
            writer = TensorBoardWriter(active=True)
            writer.close()
            writer.log({"Loss/train": 1.0}, step=0)

        mock_log.assert_not_called()

    def test_context_manager_ends_run(self) -> None:
        with patch("google.cloud.aiplatform.end_run") as mock_end:
            with TensorBoardWriter(active=True):
                pass

        mock_end.assert_called_once_with()

    def test_close_is_idempotent(self) -> None:
        with patch("google.cloud.aiplatform.end_run") as mock_end:
            writer = TensorBoardWriter(active=True)
            writer.close()
            writer.close()

        mock_end.assert_called_once_with()

    def test_close_on_inactive_writer_does_not_raise(self) -> None:
        with patch("google.cloud.aiplatform.end_run") as mock_end:
            writer = TensorBoardWriter(active=False)
            writer.close()
            writer.close()  # Idempotent on no-op writer.

        mock_end.assert_not_called()


if __name__ == "__main__":
    absltest.main()
