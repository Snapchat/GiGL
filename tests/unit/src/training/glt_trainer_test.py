"""Unit tests for GLTTrainer dispatch.

The trainer no longer extracts ``tensorboard_experiment_name`` from
``GbmlConfig``; that field now lives on ``VertexAiResourceConfig`` and the
launcher reads it directly. These tests confirm the trainer dispatches to
the right launcher based on ``trainer_config`` type.
"""

from unittest.mock import MagicMock, patch

from gigl.common import UriFactory
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.training.v2.glt_trainer import GLTTrainer
from snapchat.research.gbml import gbml_config_pb2, gigl_resource_config_pb2
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


def _make_gbml_config_pb_wrapper() -> MagicMock:
    """Return a minimal GbmlConfigPbWrapper mock for trainer dispatch."""
    trainer_config_proto = gbml_config_pb2.GbmlConfig.TrainerConfig(
        command="python -m gigl.src.training.v2.glt_trainer",
    )
    mock_wrapper = MagicMock()
    mock_wrapper.trainer_config = trainer_config_proto
    mock_wrapper.shared_config.trained_model_metadata.tensorboard_logs_uri = ""
    return mock_wrapper


class TestGltTrainerDispatch(TestCase):
    """Tests that GLTTrainer dispatches to the correct launcher entry point."""

    @patch("gigl.src.training.v2.glt_trainer.launch_single_pool_job")
    @patch("gigl.src.training.v2.glt_trainer.GbmlConfigPbWrapper")
    @patch("gigl.src.training.v2.glt_trainer.get_resource_config")
    def test_single_pool_resource_config_dispatches_to_single_pool_launcher(
        self,
        mock_get_resource_config: MagicMock,
        mock_gbml_config_cls: MagicMock,
        mock_launch_single_pool_job: MagicMock,
    ) -> None:
        mock_get_resource_config.return_value = (
            _make_resource_config_wrapper_with_single_pool()
        )
        mock_gbml_config_cls.get_gbml_config_pb_wrapper_from_uri.return_value = (
            _make_gbml_config_pb_wrapper()
        )

        GLTTrainer().run(
            applied_task_identifier=AppliedTaskIdentifier("test-job"),
            task_config_uri=UriFactory.create_uri("gs://bucket/task.yaml"),
            resource_config_uri=UriFactory.create_uri("gs://bucket/resource.yaml"),
        )

        mock_launch_single_pool_job.assert_called_once()

    @patch("gigl.src.training.v2.glt_trainer.launch_graph_store_enabled_job")
    @patch("gigl.src.training.v2.glt_trainer.GbmlConfigPbWrapper")
    @patch("gigl.src.training.v2.glt_trainer.get_resource_config")
    def test_graph_store_resource_config_dispatches_to_graph_store_launcher(
        self,
        mock_get_resource_config: MagicMock,
        mock_gbml_config_cls: MagicMock,
        mock_launch_graph_store_enabled_job: MagicMock,
    ) -> None:
        mock_get_resource_config.return_value = (
            _make_resource_config_wrapper_with_graph_store()
        )
        mock_gbml_config_cls.get_gbml_config_pb_wrapper_from_uri.return_value = (
            _make_gbml_config_pb_wrapper()
        )

        GLTTrainer().run(
            applied_task_identifier=AppliedTaskIdentifier("test-job"),
            task_config_uri=UriFactory.create_uri("gs://bucket/task.yaml"),
            resource_config_uri=UriFactory.create_uri("gs://bucket/resource.yaml"),
        )

        mock_launch_graph_store_enabled_job.assert_called_once()
