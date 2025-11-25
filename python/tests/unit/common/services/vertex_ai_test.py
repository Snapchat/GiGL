"""Unit tests for vertex_ai.py.

Note: Tests for VertexAIService class methods (launch_job, run_pipeline, etc.)
are located in python/tests/integration/common/services/vertex_ai_test.py since
they require actual GCP resources and network calls.

This file contains unit tests for standalone utility functions that can be tested
in isolation.
"""

import unittest

from google.cloud.aiplatform_v1.types import Scheduling, env_var

from gigl.common import UriFactory
from gigl.common.constants import GIGL_ROOT_DIR
from gigl.common.services.vertex_ai import VertexAiJobConfig, build_job_config
from snapchat.research.gbml.gigl_resource_config_pb2 import VertexAiResourceConfig


class TestGetJobConfigFromVertexAiResourceConfig(unittest.TestCase):
    """Test suite for get_job_config_from_vertex_ai_resource_config function."""

    def setUp(self) -> None:
        """Set up common test fixtures."""
        self.job_name = "test_task"
        self.task_config_uri = UriFactory.create_uri(
            "gs://test-bucket/task_config.yaml"
        )
        # Use actual unittest resource config that exists in the repo
        self.resource_config_uri = (
            UriFactory.create_uri(GIGL_ROOT_DIR)
            / "deployment"
            / "configs"
            / "unittest_resource_config.yaml"
        )
        self.container_uri = "gcr.io/test-project/test-container:latest"
        self.command_str = "python -m gigl.train"

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

        actual_config = build_job_config(
            job_name=self.job_name,
            is_inference=False,
            task_config_uri=self.task_config_uri,
            resource_config_uri=self.resource_config_uri,
            command_str=self.command_str,
            args={"custom_arg": "custom_value"},
            run_on_cpu=False,
            container_uri=self.container_uri,
            vertex_ai_resource_config=vertex_ai_resource_config,
            env_vars=test_env_vars,
            labels={"test_label": "test_value"},
        )

        # Create expected config
        expected_config = VertexAiJobConfig(
            job_name=f"gigl_train_{self.job_name}",
            container_uri=self.container_uri,
            command=["python", "-m", "gigl.train"],
            args=[
                f"--job_name={self.job_name}",
                f"--task_config_uri={self.task_config_uri}",
                f"--resource_config_uri={self.resource_config_uri}",
                "--use_cuda",
                "--custom_arg=custom_value",
            ],
            environment_variables=test_env_vars,
            machine_type="n1-standard-8",
            accelerator_type="NVIDIA_TESLA_V100",
            accelerator_count=2,
            replica_count=4,
            boot_disk_type="pd-ssd",
            boot_disk_size_gb=100,
            timeout_s=7200,
            enable_web_access=True,
            scheduling_strategy=Scheduling.Strategy.SPOT,  # type: ignore
            labels={"test_label": "test_value"},
        )

        self.assertEqual(actual_config, expected_config)

    def test_inference_job_config_cpu_minimal(self):
        """Test creating an inference job config for CPU with minimal options."""
        vertex_ai_resource_config = VertexAiResourceConfig(
            machine_type="n1-standard-4",
            gpu_type="nvidia-tesla-t4",
            gpu_limit=1,
            num_replicas=1,
            # No timeout or scheduling_strategy set
        )

        actual_config = build_job_config(
            job_name=self.job_name,
            is_inference=True,
            task_config_uri=self.task_config_uri,
            resource_config_uri=self.resource_config_uri,
            command_str="  python -m gigl.infer  ",  # Test whitespace handling
            args={},
            run_on_cpu=True,
            container_uri=self.container_uri,
            vertex_ai_resource_config=vertex_ai_resource_config,
            env_vars=[],
        )

        # Create expected config
        expected_config = VertexAiJobConfig(
            job_name=f"gigl_infer_{self.job_name}",
            container_uri=self.container_uri,
            command=["python", "-m", "gigl.infer"],
            args=[
                f"--job_name={self.job_name}",
                f"--task_config_uri={self.task_config_uri}",
                f"--resource_config_uri={self.resource_config_uri}",
            ],
            environment_variables=[],
            machine_type="n1-standard-4",
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1,
            replica_count=1,
            boot_disk_type="pd-ssd",
            boot_disk_size_gb=100,
            labels=None,
            timeout_s=None,
            enable_web_access=True,
            scheduling_strategy=None,
        )

        self.assertEqual(actual_config, expected_config)


if __name__ == "__main__":
    unittest.main()
