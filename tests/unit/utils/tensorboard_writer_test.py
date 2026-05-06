"""Unit tests for gigl.utils.tensorboard_writer."""

import os
from unittest.mock import Mock, patch

from absl.testing import absltest

from gigl.utils import tensorboard_writer as tensorboard_writer_module
from gigl.utils.tensorboard_writer import TensorBoardWriter
from tests.test_assets.test_case import TestCase


class TestTensorBoardWriter(TestCase):
    """Tests for the TensorBoardWriter class."""

    def test_from_env_returns_noop_when_disabled(self) -> None:
        # When disabled (e.g. non-chief rank), env var state is irrelevant
        # and no TF writer is constructed.
        with patch.dict(
            os.environ,
            {"AIP_TENSORBOARD_LOG_DIR": "gs://vertex-managed/logs"},
            clear=True,
        ):
            with patch(
                "gigl.utils.tensorboard_writer.tf.summary.create_file_writer"
            ) as mock_create_file_writer:
                writer = TensorBoardWriter.from_env(enabled=False)
                writer.log({"Loss/train": 1.0}, step=0)
                writer.close()

        mock_create_file_writer.assert_not_called()

    def test_from_env_uses_parent_log_dir_when_no_run_name(self) -> None:
        with patch.dict(
            os.environ,
            {"AIP_TENSORBOARD_LOG_DIR": "gs://vertex-managed/logs"},
            clear=True,
        ):
            with patch(
                "gigl.utils.tensorboard_writer.tf.summary.create_file_writer"
            ) as mock_create_file_writer:
                TensorBoardWriter.from_env()

        mock_create_file_writer.assert_called_once_with("gs://vertex-managed/logs")

    def test_from_env_uses_run_name_subdir_when_set(self) -> None:
        """Writer points TF at the subdir so the SDK uploader sees a distinct run."""
        with patch.dict(
            os.environ,
            {
                "AIP_TENSORBOARD_LOG_DIR": "gs://vertex-managed/logs",
                "GIGL_TENSORBOARD_RUN_NAME": "my-run",
            },
            clear=True,
        ):
            with patch(
                "gigl.utils.tensorboard_writer.tf.summary.create_file_writer"
            ) as mock_create_file_writer:
                TensorBoardWriter.from_env()

        mock_create_file_writer.assert_called_once_with(
            "gs://vertex-managed/logs/my-run"
        )

    def test_from_env_raises_when_env_var_missing(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "gigl.utils.tensorboard_writer.tf.summary.create_file_writer"
            ) as mock_create_file_writer:
                with self.assertRaises(RuntimeError):
                    TensorBoardWriter.from_env()

        mock_create_file_writer.assert_not_called()

    @patch("gigl.utils.tensorboard_writer.tf.summary.scalar")
    def test_log_writes_each_metric_at_step_and_flushes(
        self, mock_summary_scalar
    ) -> None:
        underlying_writer = Mock()
        underlying_writer.as_default.return_value.__enter__ = Mock(return_value=None)
        underlying_writer.as_default.return_value.__exit__ = Mock(return_value=None)
        with patch(
            "gigl.utils.tensorboard_writer.tf.summary.create_file_writer",
            return_value=underlying_writer,
        ):
            writer = TensorBoardWriter(log_dir="gs://logs/")
            writer.log({"Loss/train": 1.5, "Loss/val": 2.0}, step=10)

        self.assertEqual(mock_summary_scalar.call_count, 2)
        mock_summary_scalar.assert_any_call("Loss/train", 1.5, step=10)
        mock_summary_scalar.assert_any_call("Loss/val", 2.0, step=10)
        underlying_writer.flush.assert_called_once()

    @patch("gigl.utils.tensorboard_writer.tf.summary.scalar")
    def test_log_is_noop_when_writer_disabled(self, mock_summary_scalar) -> None:
        with patch(
            "gigl.utils.tensorboard_writer.tf.summary.create_file_writer"
        ) as mock_create_file_writer:
            writer = TensorBoardWriter(log_dir=None)
            writer.log({"Loss/train": 1.0}, step=0)

        mock_create_file_writer.assert_not_called()
        mock_summary_scalar.assert_not_called()

    def test_context_manager_closes_writer(self) -> None:
        underlying_writer = Mock()
        with patch(
            "gigl.utils.tensorboard_writer.tf.summary.create_file_writer",
            return_value=underlying_writer,
        ):
            with TensorBoardWriter(log_dir="gs://logs/"):
                pass

        underlying_writer.close.assert_called_once()

    def test_close_is_idempotent(self) -> None:
        underlying_writer = Mock()
        with patch(
            "gigl.utils.tensorboard_writer.tf.summary.create_file_writer",
            return_value=underlying_writer,
        ):
            writer = TensorBoardWriter(log_dir="gs://logs/")
            writer.close()
            writer.close()

        underlying_writer.close.assert_called_once()

    def test_close_on_noop_writer_does_not_raise(self) -> None:
        writer = TensorBoardWriter(log_dir=None)
        writer.close()
        writer.close()  # Idempotent on no-op writer.


class TestTensorBoardWriterUploader(TestCase):
    """Tests for the chief-rank ``aiplatform.start_upload_tb_log`` hook."""

    _LOG_DIR = "gs://vertex-managed/logs"
    _TB_RESOURCE = "projects/my-project/locations/us-central1/tensorboards/42"
    _EXPERIMENT = "my-comparison"

    def test_uploader_starts_when_all_env_vars_present(self) -> None:
        """Uploader watches the parent log dir; writer points at the run-name subdir."""
        with patch.dict(
            os.environ,
            {
                "AIP_TENSORBOARD_LOG_DIR": self._LOG_DIR,
                "GIGL_TENSORBOARD_RESOURCE_NAME": self._TB_RESOURCE,
                "GIGL_TENSORBOARD_EXPERIMENT_NAME": self._EXPERIMENT,
                "GIGL_TENSORBOARD_RUN_NAME": "my-run",
            },
            clear=True,
        ):
            with patch(
                "gigl.utils.tensorboard_writer.tf.summary.create_file_writer"
            ) as mock_create_file_writer:
                with (
                    patch("google.cloud.aiplatform.start_upload_tb_log") as mock_start,
                    patch("google.cloud.aiplatform.init") as mock_init,
                    patch("google.cloud.aiplatform.end_upload_tb_log") as mock_end,
                ):
                    writer = TensorBoardWriter.from_env()
                    writer.close()

        mock_create_file_writer.assert_called_once_with(f"{self._LOG_DIR}/my-run")
        mock_init.assert_called_once_with(project="my-project", location="us-central1")
        # Uploader watches the PARENT log dir so the run-name subdir is
        # discovered as a TensorboardRun via os.path.relpath.
        mock_start.assert_called_once_with(
            tensorboard_id="42",
            tensorboard_experiment_name=self._EXPERIMENT,
            logdir=self._LOG_DIR,
        )
        mock_end.assert_called_once()

    def test_uploader_does_not_start_when_only_log_dir_set(self) -> None:
        with patch.dict(
            os.environ,
            {"AIP_TENSORBOARD_LOG_DIR": self._LOG_DIR},
            clear=True,
        ):
            with patch("gigl.utils.tensorboard_writer.tf.summary.create_file_writer"):
                with (
                    patch("google.cloud.aiplatform.start_upload_tb_log") as mock_start,
                    patch("google.cloud.aiplatform.end_upload_tb_log") as mock_end,
                ):
                    writer = TensorBoardWriter.from_env()
                    writer.close()

        mock_start.assert_not_called()
        mock_end.assert_not_called()

    def test_invalid_tb_resource_name_raises(self) -> None:
        with patch.dict(
            os.environ,
            {
                "AIP_TENSORBOARD_LOG_DIR": self._LOG_DIR,
                "GIGL_TENSORBOARD_RESOURCE_NAME": "not-a-valid-resource-name",
                "GIGL_TENSORBOARD_EXPERIMENT_NAME": self._EXPERIMENT,
            },
            clear=True,
        ):
            with patch("gigl.utils.tensorboard_writer.tf.summary.create_file_writer"):
                with self.assertRaises(ValueError) as ctx:
                    TensorBoardWriter.from_env()

        self.assertIn("GIGL_TENSORBOARD_RESOURCE_NAME", str(ctx.exception))

    def test_uploader_skipped_for_disabled_writer(self) -> None:
        """Non-chief ranks (enabled=False) skip both the writer and uploader."""
        with patch.dict(
            os.environ,
            {
                "AIP_TENSORBOARD_LOG_DIR": self._LOG_DIR,
                "GIGL_TENSORBOARD_RESOURCE_NAME": self._TB_RESOURCE,
                "GIGL_TENSORBOARD_EXPERIMENT_NAME": self._EXPERIMENT,
            },
            clear=True,
        ):
            with patch("google.cloud.aiplatform.start_upload_tb_log") as mock_start:
                writer = TensorBoardWriter.from_env(enabled=False)
                writer.close()

        mock_start.assert_not_called()

    def test_uploader_logs_named_experiment_url_on_start(self) -> None:
        """The named-experiment URL is logged so engineers can find the TB
        page without the (now-absent) Vertex AI job-page button.
        """
        with patch.dict(
            os.environ,
            {
                "AIP_TENSORBOARD_LOG_DIR": self._LOG_DIR,
                "GIGL_TENSORBOARD_RESOURCE_NAME": self._TB_RESOURCE,
                "GIGL_TENSORBOARD_EXPERIMENT_NAME": self._EXPERIMENT,
            },
            clear=True,
        ):
            with patch("gigl.utils.tensorboard_writer.tf.summary.create_file_writer"):
                with (
                    patch("google.cloud.aiplatform.start_upload_tb_log"),
                    patch("google.cloud.aiplatform.init"),
                    patch("google.cloud.aiplatform.end_upload_tb_log"),
                    patch.object(tensorboard_writer_module.logger, "info") as mock_info,
                ):
                    writer = TensorBoardWriter.from_env()
                    writer.close()

        info_calls = [call.args[0] for call in mock_info.call_args_list]
        url_log = next((msg for msg in info_calls if "View TensorBoard" in msg), None)
        self.assertIsNotNone(url_log)
        self.assertIn(self._EXPERIMENT, url_log)
        self.assertIn("tensorboards+42", url_log)
        self.assertIn("us-central1", url_log)

    def test_uploader_failure_after_writer_construction_closes_writer(self) -> None:
        """If start_upload_tb_log raises, the TF file writer is closed and
        the exception propagates — no leaked uploader thread, no half-built
        writer.
        """
        underlying_writer = Mock()
        with patch.dict(
            os.environ,
            {
                "AIP_TENSORBOARD_LOG_DIR": self._LOG_DIR,
                "GIGL_TENSORBOARD_RESOURCE_NAME": self._TB_RESOURCE,
                "GIGL_TENSORBOARD_EXPERIMENT_NAME": self._EXPERIMENT,
            },
            clear=True,
        ):
            with patch(
                "gigl.utils.tensorboard_writer.tf.summary.create_file_writer",
                return_value=underlying_writer,
            ):
                with (
                    patch(
                        "google.cloud.aiplatform.start_upload_tb_log",
                        side_effect=RuntimeError("boom"),
                    ),
                    patch("google.cloud.aiplatform.init"),
                ):
                    with self.assertRaises(RuntimeError):
                        TensorBoardWriter.from_env()

        underlying_writer.close.assert_called_once()


if __name__ == "__main__":
    absltest.main()
