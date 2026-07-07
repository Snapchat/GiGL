"""Tests for ``gigl.src.training.v2.glt_trainer`` Vertex AI dispatch."""

from typing import Final
from unittest.mock import patch

from absl.testing import absltest

from gigl.common import Uri
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.types.pb_wrappers.gigl_resource_config import (
    GiglResourceConfigWrapper,
)
from gigl.src.training.v2.glt_trainer import GLTTrainer
from snapchat.research.gbml import gbml_config_pb2, gigl_resource_config_pb2
from tests.test_assets.test_case import TestCase

_PROCESS_COMMAND: Final[str] = "python -m gigl.src.training.v2.glt_trainer"
_STORAGE_COMMAND: Final[str] = "python -m gigl.distributed.graph_store.storage_main"


def _build_shared_resource_config() -> gigl_resource_config_pb2.SharedResourceConfig:
    return gigl_resource_config_pb2.SharedResourceConfig(
        resource_labels={
            "env": "test",
            "cost_resource_group_tag": "unittest_COMPONENT",
            "cost_resource_group": "gigl_test",
        },
        common_compute_config=(
            gigl_resource_config_pb2.SharedResourceConfig.CommonComputeConfig(
                project="test-project",
                region="us-central1",
                temp_assets_bucket="gs://test-temp-bucket",
                temp_regional_assets_bucket="gs://test-temp-regional-bucket",
                perm_assets_bucket="gs://test-perm-bucket",
                temp_assets_bq_dataset_name="test_temp_dataset",
                embedding_bq_dataset_name="test_embeddings_dataset",
                gcp_service_account_email="test-sa@test-project.iam.gserviceaccount.com",
                dataflow_runner="DataflowRunner",
            )
        ),
    )


def _build_resource_config_with_vertex_ai_trainer() -> (
    gigl_resource_config_pb2.GiglResourceConfig
):
    return gigl_resource_config_pb2.GiglResourceConfig(
        shared_resource_config=_build_shared_resource_config(),
        trainer_resource_config=gigl_resource_config_pb2.TrainerResourceConfig(
            vertex_ai_trainer_config=gigl_resource_config_pb2.VertexAiResourceConfig(
                machine_type="n1-standard-8",
                num_replicas=1,
            ),
        ),
    )


def _build_resource_config_with_graph_store_trainer() -> (
    gigl_resource_config_pb2.GiglResourceConfig
):
    return gigl_resource_config_pb2.GiglResourceConfig(
        shared_resource_config=_build_shared_resource_config(),
        trainer_resource_config=gigl_resource_config_pb2.TrainerResourceConfig(
            vertex_ai_graph_store_trainer_config=(
                gigl_resource_config_pb2.VertexAiGraphStoreConfig(
                    compute_pool=gigl_resource_config_pb2.VertexAiResourceConfig(
                        machine_type="n1-standard-8",
                        num_replicas=1,
                    ),
                    graph_store_pool=gigl_resource_config_pb2.VertexAiResourceConfig(
                        machine_type="n1-highmem-16",
                        num_replicas=1,
                    ),
                )
            ),
        ),
    )


def _build_gbml_config_with_trainer_command() -> gbml_config_pb2.GbmlConfig:
    return gbml_config_pb2.GbmlConfig(
        trainer_config=gbml_config_pb2.GbmlConfig.TrainerConfig(
            command=_PROCESS_COMMAND,
            trainer_args={"lr": "0.01", "epochs": "5"},
            graph_store_storage_config=(
                gbml_config_pb2.GbmlConfig.GraphStoreStorageConfig(
                    command=_STORAGE_COMMAND,
                    storage_args={"dataset_uri": "gs://bucket/dataset"},
                )
            ),
        ),
    )


class TestGLTTrainerVertexAiDispatch(TestCase):
    """Asserts GLT trainer forwards raw job names to launcher helpers."""

    @patch("gigl.src.training.v2.glt_trainer.launch_single_pool_job")
    @patch(
        "gigl.src.training.v2.glt_trainer.GbmlConfigPbWrapper"
        ".get_gbml_config_pb_wrapper_from_uri"
    )
    @patch("gigl.src.training.v2.glt_trainer.get_resource_config")
    def test_single_pool_dispatch_passes_raw_job_name(
        self,
        mock_get_resource_config,
        mock_get_gbml,
        mock_launch_single_pool_job,
    ) -> None:
        mock_get_resource_config.return_value = GiglResourceConfigWrapper(
            resource_config=_build_resource_config_with_vertex_ai_trainer()
        )
        mock_get_gbml.return_value = GbmlConfigPbWrapper(
            gbml_config_pb=_build_gbml_config_with_trainer_command()
        )

        task_uri = Uri("gs://bucket/task.yaml")
        resource_uri = Uri("gs://bucket/resource.yaml")
        GLTTrainer().run(
            applied_task_identifier=AppliedTaskIdentifier("job_77"),
            task_config_uri=task_uri,
            resource_config_uri=resource_uri,
            cpu_docker_uri="gcr.io/p/cpu:tag",
            cuda_docker_uri="gcr.io/p/cuda:tag",
        )

        mock_launch_single_pool_job.assert_called_once()
        call_kwargs = mock_launch_single_pool_job.call_args.kwargs
        self.assertEqual(call_kwargs["job_name"], "job_77")
        self.assertNotIn("applied_task_identifier", call_kwargs)
        self.assertEqual(call_kwargs["component"], GiGLComponents.Trainer)
        self.assertEqual(call_kwargs["task_config_uri"], task_uri)
        self.assertEqual(call_kwargs["resource_config_uri"], resource_uri)
        self.assertEqual(call_kwargs["process_command"], _PROCESS_COMMAND)
        self.assertEqual(
            dict(call_kwargs["process_runtime_args"]),
            {"lr": "0.01", "epochs": "5"},
        )

    @patch("gigl.src.training.v2.glt_trainer.launch_graph_store_enabled_job")
    @patch(
        "gigl.src.training.v2.glt_trainer.GbmlConfigPbWrapper"
        ".get_gbml_config_pb_wrapper_from_uri"
    )
    @patch("gigl.src.training.v2.glt_trainer.get_resource_config")
    def test_graph_store_dispatch_passes_raw_job_name(
        self,
        mock_get_resource_config,
        mock_get_gbml,
        mock_launch_graph_store_enabled_job,
    ) -> None:
        mock_get_resource_config.return_value = GiglResourceConfigWrapper(
            resource_config=_build_resource_config_with_graph_store_trainer()
        )
        mock_get_gbml.return_value = GbmlConfigPbWrapper(
            gbml_config_pb=_build_gbml_config_with_trainer_command()
        )

        GLTTrainer().run(
            applied_task_identifier=AppliedTaskIdentifier("job_88"),
            task_config_uri=Uri("gs://bucket/task.yaml"),
            resource_config_uri=Uri("gs://bucket/resource.yaml"),
        )

        mock_launch_graph_store_enabled_job.assert_called_once()
        call_kwargs = mock_launch_graph_store_enabled_job.call_args.kwargs
        self.assertEqual(call_kwargs["job_name"], "job_88")
        self.assertNotIn("applied_task_identifier", call_kwargs)
        self.assertEqual(call_kwargs["component"], GiGLComponents.Trainer)
        self.assertEqual(call_kwargs["storage_command"], _STORAGE_COMMAND)
        self.assertEqual(
            dict(call_kwargs["storage_args"]),
            {"dataset_uri": "gs://bucket/dataset"},
        )


if __name__ == "__main__":
    absltest.main()
