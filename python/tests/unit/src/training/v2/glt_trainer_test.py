import unittest
from unittest.mock import MagicMock, patch

from gigl.common import UriFactory
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.training.v2.glt_trainer import GLTTrainer
from snapchat.research.gbml.gigl_resource_config_pb2 import VertexAiResourceConfig


class GLTTrainerTest(unittest.TestCase):
    """Test cases for GLTTrainer region override functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.applied_task_identifier = AppliedTaskIdentifier("test-job")
        self.task_config_uri = UriFactory.create_uri(
            "gs://TEST BUCKET/task-config.yaml"
        )
        self.resource_config_uri = UriFactory.create_uri(
            "gs://TEST BUCKET/resource-config.yaml"
        )
        self.trainer = GLTTrainer()

        # Common test data
        self.test_project = "test-project"
        self.test_default_region = "us-central1"
        self.test_override_region = "us-west1"
        self.test_service_account = "test@test-project.iam.gserviceaccount.com"
        self.test_staging_bucket = "gs://TEST STAGING BUCKET"

    @patch("gigl.src.training.v2.glt_trainer.get_resource_config")
    @patch(
        "gigl.src.training.v2.glt_trainer.GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri"
    )
    @patch("gigl.src.training.v2.glt_trainer.VertexAIService")
    def test_region_override_is_applied_when_set(
        self, mock_vertex_ai_service, mock_gbml_config, mock_get_resource_config
    ):
        """Test that region override is used when gcp_region_override is set."""
        # Arrange
        mock_resource_config_wrapper = MagicMock()
        mock_resource_config_wrapper.project = self.test_project
        mock_resource_config_wrapper.region = self.test_default_region
        mock_resource_config_wrapper.service_account_email = self.test_service_account
        mock_resource_config_wrapper.temp_assets_regional_bucket_path.uri = (
            self.test_staging_bucket
        )
        mock_resource_config_wrapper.get_resource_labels.return_value = {}

        # Create inferencer config with region override
        mock_trainer_config = VertexAiResourceConfig()
        mock_trainer_config.gcp_region_override = self.test_override_region
        mock_trainer_config.machine_type = "n1-standard-4"
        mock_trainer_config.gpu_type = ""  # CPU training
        mock_trainer_config.num_replicas = 1
        mock_resource_config_wrapper.trainer_config = mock_trainer_config

        mock_get_resource_config.return_value = mock_resource_config_wrapper

        # Mock GBML config
        mock_gbml_config_wrapper = MagicMock()
        mock_gbml_config_wrapper.trainer_config.command = "python test.py"
        mock_gbml_config_wrapper.trainer_config.trainer_args = {}
        mock_gbml_config.return_value = mock_gbml_config_wrapper

        # Mock VertexAI service
        mock_vertex_service_instance = MagicMock()
        mock_vertex_ai_service.return_value = mock_vertex_service_instance

        # Act
        self.trainer.run(
            applied_task_identifier=self.applied_task_identifier,
            task_config_uri=self.task_config_uri,
            resource_config_uri=self.resource_config_uri,
        )

        # Assert
        # Verify VertexAIService was created with the override region
        mock_vertex_ai_service.assert_called_once_with(
            project=self.test_project,
            location=self.test_override_region,  # Should use override region
            service_account=self.test_service_account,
            staging_bucket=self.test_staging_bucket,
        )

        # Verify launch_job was called
        mock_vertex_service_instance.launch_job.assert_called_once()

    @patch("gigl.src.training.v2.glt_trainer.get_resource_config")
    @patch(
        "gigl.src.training.v2.glt_trainer.GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri"
    )
    @patch("gigl.src.training.v2.glt_trainer.VertexAIService")
    def test_region_fallback_when_no_override(
        self, mock_vertex_ai_service, mock_gbml_config, mock_get_resource_config
    ):
        """Test that default region is used when gcp_region_override is not set."""
        # Arrange
        mock_resource_config_wrapper = MagicMock()
        mock_resource_config_wrapper.project = self.test_project
        mock_resource_config_wrapper.region = self.test_default_region
        mock_resource_config_wrapper.service_account_email = self.test_service_account
        mock_resource_config_wrapper.temp_assets_regional_bucket_path.uri = (
            self.test_staging_bucket
        )
        mock_resource_config_wrapper.get_resource_labels.return_value = {}

        # Create inferencer config without region override
        mock_trainer_config = VertexAiResourceConfig()
        mock_trainer_config.gcp_region_override = ""  # No override
        mock_trainer_config.machine_type = "n1-standard-4"
        mock_trainer_config.gpu_type = ""  # CPU training
        mock_trainer_config.num_replicas = 1
        mock_resource_config_wrapper.trainer_config = mock_trainer_config

        mock_get_resource_config.return_value = mock_resource_config_wrapper

        # Mock GBML config
        mock_gbml_config_wrapper = MagicMock()
        mock_gbml_config_wrapper.trainer_config.command = "python test.py"
        mock_gbml_config_wrapper.trainer_config.trainer_args = {}
        mock_gbml_config.return_value = mock_gbml_config_wrapper

        # Mock VertexAI service
        mock_vertex_service_instance = MagicMock()
        mock_vertex_ai_service.return_value = mock_vertex_service_instance

        # Act
        self.trainer.run(
            applied_task_identifier=self.applied_task_identifier,
            task_config_uri=self.task_config_uri,
            resource_config_uri=self.resource_config_uri,
        )

        # Assert
        # Verify VertexAIService was created with the default region
        mock_vertex_ai_service.assert_called_once_with(
            project=self.test_project,
            location=self.test_default_region,  # Should use default region
            service_account=self.test_service_account,
            staging_bucket=self.test_staging_bucket,
        )

        # Verify launch_job was called
        mock_vertex_service_instance.launch_job.assert_called_once()


if __name__ == "__main__":
    unittest.main()
