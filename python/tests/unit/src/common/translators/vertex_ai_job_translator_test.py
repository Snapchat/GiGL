import unittest

from google.cloud.aiplatform_v1.types import Scheduling, env_var

from gigl.common import UriFactory
from gigl.common.services.vertex_ai import VertexAiJobConfig
from gigl.src.common.translators.vertex_ai_job_translator import build_job_config
from snapchat.research.gbml.gigl_resource_config_pb2 import VertexAiResourceConfig


class TestGetJobConfigFromVertexAiResourceConfig(unittest.TestCase):
    """Test suite for get_job_config_from_vertex_ai_resource_config function."""

    def test_training_job_config_with_all_options(self):
        """Test creating a training job config with all options set."""
        vertex_ai_resource_config = VertexAiResourceConfig(
            machine_type="n1-standard-8",
            gpu_type="nvidia-tesla-v100",
            gpu_limit=2,
            num_replicas=4,
            timeout=7200,
            scheduling_strategy="SPOT",
        )

        test_env_vars = [
            env_var.EnvVar(name="TEST_VAR", value="test_value"),
        ]

        job_name = "gigl_train_test_task"
        task_config_uri = UriFactory.create_uri("gs://test-bucket/task_config.yaml")
        resource_config_uri = UriFactory.create_uri(
            "gs://test-bucket/resource_config.yaml"
        )
        actual_config = build_job_config(
            job_name=job_name,
            task_config_uri=task_config_uri,
            resource_config_uri=resource_config_uri,
            command_str="python -m gigl.train",
            args={"custom_arg": "custom_value"},
            use_cuda=True,
            container_uri="gcr.io/test-project/test-container:latest",
            vertex_ai_resource_config=vertex_ai_resource_config,
            env_vars=test_env_vars,
            labels={"test_label": "test_value"},
        )

        # Create expected config
        expected_config = VertexAiJobConfig(
            job_name=job_name,
            container_uri="gcr.io/test-project/test-container:latest",
            command=["python", "-m", "gigl.train"],
            args=[
                f"--job_name={job_name}",
                f"--task_config_uri={task_config_uri}",
                f"--resource_config_uri={resource_config_uri}",
                "--use_cuda",
                "--custom_arg=custom_value",
            ],
            environment_variables=test_env_vars,
            machine_type=vertex_ai_resource_config.machine_type,
            accelerator_type="NVIDIA_TESLA_V100",  # Test this since we do transformations on the name
            accelerator_count=vertex_ai_resource_config.gpu_limit,
            replica_count=vertex_ai_resource_config.num_replicas,
            boot_disk_type="pd-ssd",
            boot_disk_size_gb=100,
            timeout_s=vertex_ai_resource_config.timeout,
            enable_web_access=True,
            scheduling_strategy=Scheduling.Strategy.SPOT,  # type: ignore
            labels={"test_label": "test_value"},
        )

        self.assertEqual(actual_config, expected_config)


if __name__ == "__main__":
    unittest.main()
