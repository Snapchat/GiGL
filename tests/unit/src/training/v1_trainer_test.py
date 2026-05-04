"""Unit tests for v1 Trainer — verifies tensorboard_experiment_name forwarding."""

from unittest.mock import MagicMock, patch

from gigl.common import UriFactory
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.training.v1.trainer import Trainer
from snapchat.research.gbml import gbml_config_pb2
from snapchat.research.gbml import gigl_resource_config_pb2
from tests.test_assets.test_case import TestCase


def _make_resource_config_wrapper_with_single_pool() -> MagicMock:
    """Return a GiglResourceConfigWrapper mock backed by a VertexAiResourceConfig."""
    vertex_ai_config = gigl_resource_config_pb2.VertexAiResourceConfig(
        machine_type="n1-standard-8",
        num_replicas=1,
        timeout=7200,
    )
    mock_wrapper = MagicMock()
    mock_wrapper.trainer_config = vertex_ai_config
    mock_wrapper.vertex_ai_trainer_region = "us-central1"
    return mock_wrapper


def _make_gbml_config_pb_wrapper(experiment_name: str = "my-comparison") -> MagicMock:
    """Return a GbmlConfigPbWrapper mock with tensorboard_experiment_name set."""
    trainer_config_proto = gbml_config_pb2.GbmlConfig.TrainerConfig(
        tensorboard_experiment_name=experiment_name,
    )

    mock_wrapper = MagicMock()
    mock_wrapper.trainer_config = trainer_config_proto
    # Ensure tensorboard_logs_uri is empty so UriFactory is not called.
    mock_wrapper.shared_config.trained_model_metadata.tensorboard_logs_uri = ""
    return mock_wrapper


class TestV1TrainerExperimentNameForwarding(TestCase):
    """Tests that v1 Trainer forwards tensorboard_experiment_name to the launcher."""

    @patch("gigl.src.training.v1.trainer.launch_single_pool_job")
    @patch("gigl.src.training.v1.trainer.GbmlConfigPbWrapper")
    @patch("gigl.src.training.v1.trainer.get_resource_config")
    def test_single_pool_forwards_experiment_name(
        self,
        mock_get_resource_config,
        mock_gbml_config_cls,
        mock_launch_single_pool_job,
    ) -> None:
        """launch_single_pool_job receives tensorboard_experiment_name='my-comparison'."""
        mock_get_resource_config.return_value = (
            _make_resource_config_wrapper_with_single_pool()
        )
        mock_gbml_config_cls.get_gbml_config_pb_wrapper_from_uri.return_value = (
            _make_gbml_config_pb_wrapper("my-comparison")
        )

        trainer = Trainer()
        trainer.run(
            applied_task_identifier=AppliedTaskIdentifier("test-job"),
            task_config_uri=UriFactory.create_uri("gs://bucket/task.yaml"),
            resource_config_uri=UriFactory.create_uri("gs://bucket/resource.yaml"),
        )

        mock_launch_single_pool_job.assert_called_once()
        call_kwargs = mock_launch_single_pool_job.call_args.kwargs
        self.assertEqual(call_kwargs["tensorboard_experiment_name"], "my-comparison")

    @patch("gigl.src.training.v1.trainer.launch_single_pool_job")
    @patch("gigl.src.training.v1.trainer.GbmlConfigPbWrapper")
    @patch("gigl.src.training.v1.trainer.get_resource_config")
    def test_single_pool_empty_experiment_name_becomes_none(
        self,
        mock_get_resource_config,
        mock_gbml_config_cls,
        mock_launch_single_pool_job,
    ) -> None:
        """Empty string tensorboard_experiment_name is coerced to None."""
        mock_get_resource_config.return_value = (
            _make_resource_config_wrapper_with_single_pool()
        )
        mock_gbml_config_cls.get_gbml_config_pb_wrapper_from_uri.return_value = (
            _make_gbml_config_pb_wrapper("")  # proto default empty string
        )

        trainer = Trainer()
        trainer.run(
            applied_task_identifier=AppliedTaskIdentifier("test-job"),
            task_config_uri=UriFactory.create_uri("gs://bucket/task.yaml"),
            resource_config_uri=UriFactory.create_uri("gs://bucket/resource.yaml"),
        )

        mock_launch_single_pool_job.assert_called_once()
        call_kwargs = mock_launch_single_pool_job.call_args.kwargs
        self.assertIsNone(call_kwargs["tensorboard_experiment_name"])
