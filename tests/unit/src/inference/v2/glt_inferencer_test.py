"""Tests for ``gigl.src.inference.v2.glt_inferencer`` Vertex AI dispatch."""

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
from gigl.src.inference.v2.glt_inferencer import GLTInferencer
from snapchat.research.gbml import gbml_config_pb2, gigl_resource_config_pb2
from tests.test_assets.test_case import TestCase

_PROCESS_COMMAND: Final[str] = "python -m gigl.src.inference.v2.glt_inferencer"
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


def _build_resource_config_with_vertex_ai_inferencer() -> (
    gigl_resource_config_pb2.GiglResourceConfig
):
    return gigl_resource_config_pb2.GiglResourceConfig(
        shared_resource_config=_build_shared_resource_config(),
        inferencer_resource_config=gigl_resource_config_pb2.InferencerResourceConfig(
            vertex_ai_inferencer_config=gigl_resource_config_pb2.VertexAiResourceConfig(
                machine_type="n1-standard-8",
                num_replicas=1,
            ),
        ),
    )


def _build_resource_config_with_graph_store_inferencer() -> (
    gigl_resource_config_pb2.GiglResourceConfig
):
    return gigl_resource_config_pb2.GiglResourceConfig(
        shared_resource_config=_build_shared_resource_config(),
        inferencer_resource_config=gigl_resource_config_pb2.InferencerResourceConfig(
            vertex_ai_graph_store_inferencer_config=(
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


def _build_resource_config_with_custom_inferencer() -> (
    gigl_resource_config_pb2.GiglResourceConfig
):
    return gigl_resource_config_pb2.GiglResourceConfig(
        shared_resource_config=_build_shared_resource_config(),
        inferencer_resource_config=gigl_resource_config_pb2.InferencerResourceConfig(
            custom_inferencer_config=gigl_resource_config_pb2.CustomLauncherConfig(
                command="python -m my.custom.inferencer",
                args=["--foo=bar", "--noskip_inference"],
            ),
        ),
    )


def _build_gbml_config_with_inferencer_command() -> gbml_config_pb2.GbmlConfig:
    return gbml_config_pb2.GbmlConfig(
        inferencer_config=gbml_config_pb2.GbmlConfig.InferencerConfig(
            command=_PROCESS_COMMAND,
            inferencer_args={"batch_size": "64"},
            graph_store_storage_config=(
                gbml_config_pb2.GbmlConfig.GraphStoreStorageConfig(
                    command=_STORAGE_COMMAND,
                    storage_args={"dataset_uri": "gs://bucket/dataset"},
                )
            ),
        ),
    )


class TestGLTInferencerVertexAiDispatch(TestCase):
    """Asserts GLT inferencer forwards raw job names to launcher helpers."""

    @patch("gigl.src.inference.v2.glt_inferencer.launch_single_pool_job")
    @patch(
        "gigl.src.inference.v2.glt_inferencer.GbmlConfigPbWrapper"
        ".get_gbml_config_pb_wrapper_from_uri"
    )
    @patch("gigl.src.inference.v2.glt_inferencer.get_resource_config")
    def test_single_pool_dispatch_passes_raw_job_name(
        self,
        mock_get_resource_config,
        mock_get_gbml,
        mock_launch_single_pool_job,
    ) -> None:
        mock_get_resource_config.return_value = GiglResourceConfigWrapper(
            resource_config=_build_resource_config_with_vertex_ai_inferencer()
        )
        mock_get_gbml.return_value = GbmlConfigPbWrapper(
            gbml_config_pb=_build_gbml_config_with_inferencer_command()
        )

        task_uri = Uri("gs://bucket/task.yaml")
        resource_uri = Uri("gs://bucket/resource.yaml")
        GLTInferencer().run(
            applied_task_identifier=AppliedTaskIdentifier("job_99"),
            task_config_uri=task_uri,
            resource_config_uri=resource_uri,
            cpu_docker_uri="gcr.io/p/cpu:tag",
            cuda_docker_uri="gcr.io/p/cuda:tag",
        )

        mock_launch_single_pool_job.assert_called_once()
        call_kwargs = mock_launch_single_pool_job.call_args.kwargs
        self.assertEqual(call_kwargs["job_name"], "job_99")
        self.assertNotIn("applied_task_identifier", call_kwargs)
        self.assertEqual(call_kwargs["component"], GiGLComponents.Inferencer)
        self.assertEqual(call_kwargs["task_config_uri"], task_uri)
        self.assertEqual(call_kwargs["resource_config_uri"], resource_uri)
        self.assertEqual(call_kwargs["process_command"], _PROCESS_COMMAND)
        self.assertEqual(
            dict(call_kwargs["process_runtime_args"]),
            {"batch_size": "64"},
        )

    @patch("gigl.src.inference.v2.glt_inferencer.launch_graph_store_enabled_job")
    @patch(
        "gigl.src.inference.v2.glt_inferencer.GbmlConfigPbWrapper"
        ".get_gbml_config_pb_wrapper_from_uri"
    )
    @patch("gigl.src.inference.v2.glt_inferencer.get_resource_config")
    def test_graph_store_dispatch_passes_raw_job_name(
        self,
        mock_get_resource_config,
        mock_get_gbml,
        mock_launch_graph_store_enabled_job,
    ) -> None:
        mock_get_resource_config.return_value = GiglResourceConfigWrapper(
            resource_config=_build_resource_config_with_graph_store_inferencer()
        )
        mock_get_gbml.return_value = GbmlConfigPbWrapper(
            gbml_config_pb=_build_gbml_config_with_inferencer_command()
        )

        GLTInferencer().run(
            applied_task_identifier=AppliedTaskIdentifier("job_100"),
            task_config_uri=Uri("gs://bucket/task.yaml"),
            resource_config_uri=Uri("gs://bucket/resource.yaml"),
        )

        mock_launch_graph_store_enabled_job.assert_called_once()
        call_kwargs = mock_launch_graph_store_enabled_job.call_args.kwargs
        self.assertEqual(call_kwargs["job_name"], "job_100")
        self.assertNotIn("applied_task_identifier", call_kwargs)
        self.assertEqual(call_kwargs["component"], GiGLComponents.Inferencer)
        self.assertEqual(call_kwargs["storage_command"], _STORAGE_COMMAND)
        self.assertEqual(
            dict(call_kwargs["storage_args"]),
            {"dataset_uri": "gs://bucket/dataset"},
        )

    @patch("gigl.src.inference.v2.glt_inferencer.launch_single_pool_job")
    @patch("gigl.src.inference.v2.glt_inferencer.launch_graph_store_enabled_job")
    @patch("gigl.src.inference.v2.glt_inferencer.launch_custom")
    @patch("gigl.src.inference.v2.glt_inferencer.get_resource_config")
    def test_custom_launcher_dispatch(
        self,
        mock_get_resource_config,
        mock_launch_custom,
        mock_launch_graph_store_enabled_job,
        mock_launch_single_pool_job,
    ) -> None:
        resource_config = _build_resource_config_with_custom_inferencer()
        mock_get_resource_config.return_value = GiglResourceConfigWrapper(
            resource_config=resource_config
        )

        task_uri = Uri("gs://bucket/task.yaml")
        resource_uri = Uri("gs://bucket/resource.yaml")
        GLTInferencer().run(
            applied_task_identifier=AppliedTaskIdentifier("job_custom"),
            task_config_uri=task_uri,
            resource_config_uri=resource_uri,
            cpu_docker_uri="gcr.io/p/cpu:tag",
            cuda_docker_uri="gcr.io/p/cuda:tag",
        )

        mock_launch_custom.assert_called_once()
        call_kwargs = mock_launch_custom.call_args.kwargs
        self.assertEqual(
            call_kwargs["custom_launcher_config"],
            resource_config.inferencer_resource_config.custom_inferencer_config,
        )
        self.assertEqual(
            call_kwargs["applied_task_identifier"],
            AppliedTaskIdentifier("job_custom"),
        )
        self.assertEqual(call_kwargs["task_config_uri"], task_uri)
        self.assertEqual(call_kwargs["resource_config_uri"], resource_uri)
        self.assertEqual(call_kwargs["cpu_docker_uri"], "gcr.io/p/cpu:tag")
        self.assertEqual(call_kwargs["cuda_docker_uri"], "gcr.io/p/cuda:tag")
        self.assertEqual(call_kwargs["component"], GiGLComponents.Inferencer)
        mock_launch_single_pool_job.assert_not_called()
        mock_launch_graph_store_enabled_job.assert_not_called()


if __name__ == "__main__":
    absltest.main()
