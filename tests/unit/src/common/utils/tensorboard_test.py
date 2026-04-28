"""Unit tests for gigl.src.common.utils.tensorboard."""

import os
from unittest.mock import Mock, patch

from absl.testing import absltest

from gigl.common import UriFactory
from gigl.src.common.utils.tensorboard import (
    VERTEX_TENSORBOARD_LOG_DIR_ENV_KEY,
    create_tensorboard_writer,
    resolve_tensorboard_log_dir,
    write_tensorboard_scalar,
)
from tests.test_assets.test_case import TestCase


class TestTensorboardUtils(TestCase):
    """Tests for shared TensorBoard helpers."""

    def test_resolve_tensorboard_log_dir_prefers_vertex_env(self) -> None:
        configured_tensorboard_log_uri = UriFactory.create_uri(
            "gs://perm-assets/job/trainer/logs/"
        )

        with patch.dict(
            os.environ,
            {VERTEX_TENSORBOARD_LOG_DIR_ENV_KEY: "gs://vertex-managed/logs"},
            clear=False,
        ):
            resolved_log_dir = resolve_tensorboard_log_dir(
                configured_tensorboard_log_uri=configured_tensorboard_log_uri
            )

        self.assertEqual(resolved_log_dir, "gs://vertex-managed/logs")

    @patch("gigl.src.common.utils.tensorboard.tf.summary.create_file_writer")
    def test_create_tensorboard_writer_uses_configured_uri_when_vertex_env_missing(
        self,
        mock_create_file_writer,
    ) -> None:
        configured_tensorboard_log_uri = UriFactory.create_uri(
            "gs://perm-assets/job/trainer/logs/"
        )
        writer = object()
        mock_create_file_writer.return_value = writer

        created_writer = create_tensorboard_writer(
            should_log_to_tensorboard=True,
            configured_tensorboard_log_uri=configured_tensorboard_log_uri,
            should_write_events=True,
        )

        self.assertIs(created_writer, writer)
        mock_create_file_writer.assert_called_once_with(
            configured_tensorboard_log_uri.uri
        )

    @patch("gigl.src.common.utils.tensorboard.tf.summary.create_file_writer")
    def test_create_tensorboard_writer_skips_non_chief_process(
        self,
        mock_create_file_writer,
    ) -> None:
        created_writer = create_tensorboard_writer(
            should_log_to_tensorboard=True,
            configured_tensorboard_log_uri=UriFactory.create_uri(
                "gs://perm-assets/job/trainer/logs/"
            ),
            should_write_events=False,
        )

        self.assertIsNone(created_writer)
        mock_create_file_writer.assert_not_called()

    @patch("gigl.src.common.utils.tensorboard.tf.summary.scalar")
    def test_write_tensorboard_scalar_flushes_writer(self, mock_summary_scalar) -> None:
        writer = Mock()
        writer.as_default.return_value.__enter__ = Mock(return_value=None)
        writer.as_default.return_value.__exit__ = Mock(return_value=None)

        write_tensorboard_scalar(
            writer=writer,
            tag="Loss/train",
            value=1.5,
            step=10,
        )

        mock_summary_scalar.assert_called_once_with("Loss/train", 1.5, step=10)
        writer.flush.assert_called_once()


if __name__ == "__main__":
    absltest.main()
