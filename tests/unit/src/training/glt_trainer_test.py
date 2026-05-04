"""Unit tests for GLTTrainer — verifies tensorboard_experiment_name forwarding."""

from unittest.mock import MagicMock, patch

from gigl.common import UriFactory
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.training.v2.glt_trainer import GLTTrainer
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


def _make_resource_config_wrapper_with_graph_store() -> MagicMock:
    """Return a GiglResourceConfigWrapper mock backed by a VertexAiGraphStoreConfig."""
    compute_pool = gigl_resource_config_pb2.VertexAiResourceConfig(
        machine_type="n1-standard-16",
        num_replicas=1,
    )
    storage_pool = gigl_resource_config_pb2.VertexAiResourceConfig(
        machine_type="n1-highmem-32",
        num_replicas=1,
    )
    graph_store_config = gigl_resource_config_pb2.VertexAiGraphStoreConfig(
        compute_pool=compute_pool,
        graph_store_pool=storage_pool,
        compute_cluster_local_world_size=4,
    )
    mock_wrapper = MagicMock()
    mock_wrapper.trainer_config = graph_store_config
    return mock_wrapper


def _make_gbml_config_pb_wrapper(experiment_name: str = "my-comparison") -> MagicMock:
    """Return a GbmlConfigPbWrapper mock with tensorboard_experiment_name set."""
    trainer_config_proto = gbml_config_pb2.GbmlConfig.TrainerConfig(
        command="python -m gigl.src.training.v2.glt_trainer",
        tensorboard_experiment_name=experiment_name,
    )

    mock_wrapper = MagicMock()
    mock_wrapper.trainer_config = trainer_config_proto
    # Ensure tensorboard_logs_uri is empty so UriFactory is not called.
    mock_wrapper.shared_config.trained_model_metadata.tensorboard_logs_uri = ""
    return mock_wrapper


class TestGltTrainerExperimentNameForwarding(TestCase):
    """Tests that GLTTrainer forwards tensorboard_experiment_name to the launcher."""

    @patch("gigl.src.training.v2.glt_trainer.launch_single_pool_job")
    @patch("gigl.src.training.v2.glt_trainer.GbmlConfigPbWrapper")
    @patch("gigl.src.training.v2.glt_trainer.get_resource_config")
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

        trainer = GLTTrainer()
        trainer.run(
            applied_task_identifier=AppliedTaskIdentifier("test-job"),
            task_config_uri=UriFactory.create_uri("gs://bucket/task.yaml"),
            resource_config_uri=UriFactory.create_uri("gs://bucket/resource.yaml"),
        )

        mock_launch_single_pool_job.assert_called_once()
        call_kwargs = mock_launch_single_pool_job.call_args.kwargs
        self.assertEqual(call_kwargs["tensorboard_experiment_name"], "my-comparison")

    @patch("gigl.src.training.v2.glt_trainer.launch_graph_store_enabled_job")
    @patch("gigl.src.training.v2.glt_trainer.GbmlConfigPbWrapper")
    @patch("gigl.src.training.v2.glt_trainer.get_resource_config")
    def test_graph_store_forwards_experiment_name(
        self,
        mock_get_resource_config,
        mock_gbml_config_cls,
        mock_launch_graph_store_enabled_job,
    ) -> None:
        """launch_graph_store_enabled_job receives tensorboard_experiment_name='my-comparison'."""
        mock_get_resource_config.return_value = (
            _make_resource_config_wrapper_with_graph_store()
        )
        mock_gbml_config_cls.get_gbml_config_pb_wrapper_from_uri.return_value = (
            _make_gbml_config_pb_wrapper("my-comparison")
        )

        trainer = GLTTrainer()
        trainer.run(
            applied_task_identifier=AppliedTaskIdentifier("test-job"),
            task_config_uri=UriFactory.create_uri("gs://bucket/task.yaml"),
            resource_config_uri=UriFactory.create_uri("gs://bucket/resource.yaml"),
        )

        mock_launch_graph_store_enabled_job.assert_called_once()
        call_kwargs = mock_launch_graph_store_enabled_job.call_args.kwargs
        self.assertEqual(call_kwargs["tensorboard_experiment_name"], "my-comparison")

    @patch("gigl.src.training.v2.glt_trainer.launch_single_pool_job")
    @patch("gigl.src.training.v2.glt_trainer.GbmlConfigPbWrapper")
    @patch("gigl.src.training.v2.glt_trainer.get_resource_config")
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

        trainer = GLTTrainer()
        trainer.run(
            applied_task_identifier=AppliedTaskIdentifier("test-job"),
            task_config_uri=UriFactory.create_uri("gs://bucket/task.yaml"),
            resource_config_uri=UriFactory.create_uri("gs://bucket/resource.yaml"),
        )

        mock_launch_single_pool_job.assert_called_once()
        call_kwargs = mock_launch_single_pool_job.call_args.kwargs
        self.assertIsNone(call_kwargs["tensorboard_experiment_name"])
