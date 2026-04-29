"""Unit tests for gigl.src.common.utils.tensorboard."""

import os
from unittest.mock import Mock, patch

from absl.testing import absltest

from gigl.common import UriFactory
from gigl.src.common.utils.tensorboard import TensorBoardWriter
from tests.test_assets.test_case import TestCase


class TestTensorBoardWriter(TestCase):
    """Tests for the TensorBoardWriter class."""

    def test_from_uri_returns_noop_when_disabled(self) -> None:
        configured_uri = UriFactory.create_uri("gs://config/logs/")
        with patch(
            "gigl.src.common.utils.tensorboard.tf.summary.create_file_writer"
        ) as mock_create_file_writer:
            writer = TensorBoardWriter.from_uri(configured_uri, enabled=False)
            writer.log({"Loss/train": 1.0}, step=0)
            writer.close()

        mock_create_file_writer.assert_not_called()

    def test_from_uri_prefers_vertex_env_var(self) -> None:
        configured_uri = UriFactory.create_uri("gs://config/logs/")
        with patch.dict(
            os.environ,
            {"AIP_TENSORBOARD_LOG_DIR": "gs://vertex-managed/logs"},
            clear=False,
        ):
            with patch(
                "gigl.src.common.utils.tensorboard.tf.summary.create_file_writer"
            ) as mock_create_file_writer:
                TensorBoardWriter.from_uri(configured_uri)

        mock_create_file_writer.assert_called_once_with("gs://vertex-managed/logs")

    def test_from_uri_falls_back_to_configured_uri(self) -> None:
        configured_uri = UriFactory.create_uri("gs://config/logs/")
        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "gigl.src.common.utils.tensorboard.tf.summary.create_file_writer"
            ) as mock_create_file_writer:
                TensorBoardWriter.from_uri(configured_uri)

        mock_create_file_writer.assert_called_once_with(configured_uri.uri)

    def test_from_uri_returns_noop_when_no_uri_anywhere(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "gigl.src.common.utils.tensorboard.tf.summary.create_file_writer"
            ) as mock_create_file_writer:
                writer = TensorBoardWriter.from_uri(configured_uri=None)
                writer.log({"Loss/train": 1.0}, step=0)
                writer.close()

        mock_create_file_writer.assert_not_called()

    @patch("gigl.src.common.utils.tensorboard.tf.summary.scalar")
    def test_log_writes_each_metric_at_step_and_flushes(
        self, mock_summary_scalar
    ) -> None:
        underlying_writer = Mock()
        underlying_writer.as_default.return_value.__enter__ = Mock(return_value=None)
        underlying_writer.as_default.return_value.__exit__ = Mock(return_value=None)
        with patch(
            "gigl.src.common.utils.tensorboard.tf.summary.create_file_writer",
            return_value=underlying_writer,
        ):
            writer = TensorBoardWriter(log_dir="gs://logs/")
            writer.log({"Loss/train": 1.5, "Loss/val": 2.0}, step=10)

        self.assertEqual(mock_summary_scalar.call_count, 2)
        mock_summary_scalar.assert_any_call("Loss/train", 1.5, step=10)
        mock_summary_scalar.assert_any_call("Loss/val", 2.0, step=10)
        underlying_writer.flush.assert_called_once()

    @patch("gigl.src.common.utils.tensorboard.tf.summary.scalar")
    def test_log_is_noop_when_writer_disabled(self, mock_summary_scalar) -> None:
        with patch(
            "gigl.src.common.utils.tensorboard.tf.summary.create_file_writer"
        ) as mock_create_file_writer:
            writer = TensorBoardWriter(log_dir=None)
            writer.log({"Loss/train": 1.0}, step=0)

        mock_create_file_writer.assert_not_called()
        mock_summary_scalar.assert_not_called()

    def test_context_manager_closes_writer(self) -> None:
        underlying_writer = Mock()
        with patch(
            "gigl.src.common.utils.tensorboard.tf.summary.create_file_writer",
            return_value=underlying_writer,
        ):
            with TensorBoardWriter(log_dir="gs://logs/"):
                pass

        underlying_writer.close.assert_called_once()

    def test_close_is_idempotent(self) -> None:
        underlying_writer = Mock()
        with patch(
            "gigl.src.common.utils.tensorboard.tf.summary.create_file_writer",
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


if __name__ == "__main__":
    absltest.main()
