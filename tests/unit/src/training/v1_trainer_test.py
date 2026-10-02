from unittest.mock import patch

from absl.testing import absltest

from gigl.common import Uri
from gigl.env.constants import (
    GIGL_COMPONENT_ENV_KEY,
    GIGL_CPU_DOCKER_URI_ENV_KEY,
    GIGL_CUDA_DOCKER_URI_ENV_KEY,
)
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.pb_wrappers.gigl_resource_config import (
    GiglResourceConfigWrapper,
)
from gigl.src.training.v1.trainer import Trainer
from snapchat.research.gbml import gigl_resource_config_pb2
from tests.test_assets.test_case import TestCase


def _build_resource_config_with_vertex_ai_trainer() -> (
    gigl_resource_config_pb2.GiglResourceConfig
):
    return gigl_resource_config_pb2.GiglResourceConfig(
        shared_resource_config=gigl_resource_config_pb2.SharedResourceConfig(
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
        ),
        trainer_resource_config=gigl_resource_config_pb2.TrainerResourceConfig(
            vertex_ai_trainer_config=gigl_resource_config_pb2.VertexAiResourceConfig(
                machine_type="n1-standard-8",
                num_replicas=1,
            ),
        ),
    )


class TrainerV1Test(TestCase):
    @patch("gigl.src.training.v1.trainer.VertexAIService")
    @patch("gigl.src.training.v1.trainer.get_resource_config")
    def test_vertex_ai_training_process_receives_runtime_image_context(
        self,
        mock_get_resource_config,
        mock_vertex_ai_service,
    ) -> None:
        mock_get_resource_config.return_value = GiglResourceConfigWrapper(
            resource_config=_build_resource_config_with_vertex_ai_trainer()
        )

        cpu_docker_uri = "gcr.io/project/cpu:tag"
        cuda_docker_uri = "gcr.io/project/cuda:tag"
        Trainer().run(
            applied_task_identifier=AppliedTaskIdentifier("job_1"),
            task_config_uri=Uri("gs://bucket/task.yaml"),
            resource_config_uri=Uri("gs://bucket/resource.yaml"),
            cpu_docker_uri=cpu_docker_uri,
            cuda_docker_uri=cuda_docker_uri,
        )

        mock_vertex_ai_service.return_value.launch_job.assert_called_once()
        job_config = mock_vertex_ai_service.return_value.launch_job.call_args.kwargs[
            "job_config"
        ]
        self.assertIn(f"--cpu_docker_uri={cpu_docker_uri}", job_config.args)
        self.assertIn(f"--cuda_docker_uri={cuda_docker_uri}", job_config.args)

        env_vars = {
            env_var.name: env_var.value for env_var in job_config.environment_variables
        }
        self.assertEqual(env_vars[GIGL_CPU_DOCKER_URI_ENV_KEY], cpu_docker_uri)
        self.assertEqual(env_vars[GIGL_CUDA_DOCKER_URI_ENV_KEY], cuda_docker_uri)
        self.assertEqual(env_vars[GIGL_COMPONENT_ENV_KEY], GiGLComponents.Trainer.name)


if __name__ == "__main__":
    absltest.main()
