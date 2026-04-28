"""Unit tests for gigl.common.services.vertex_ai."""

from unittest.mock import Mock, patch

from absl.testing import absltest

from gigl.common.services.vertex_ai import VertexAIService, VertexAiJobConfig
from tests.test_assets.test_case import TestCase


class TestVertexAIService(TestCase):
    """Tests for Vertex AI CustomJob submission plumbing."""

    @patch("gigl.common.services.vertex_ai.aiplatform.CustomJob")
    @patch("gigl.common.services.vertex_ai.aiplatform.init")
    def test_submit_job_passes_tensorboard_and_base_output_dir(
        self,
        mock_aiplatform_init,
        mock_custom_job_class,
    ) -> None:
        mock_job = Mock()
        mock_job.resource_name = "projects/test/locations/us-central1/customJobs/123"
        mock_job.name = "123"
        mock_custom_job_class.return_value = mock_job

        service = VertexAIService(
            project="test-project",
            location="us-central1",
            service_account="svc@test-project.iam.gserviceaccount.com",
            staging_bucket="gs://test-staging-bucket",
        )

        job_config = VertexAiJobConfig(
            job_name="test-job",
            container_uri="gcr.io/test/image:latest",
            command=["python", "-m", "trainer"],
            base_output_dir="gs://test-perm-bucket/test-job/trainer",
            tensorboard_resource_name=(
                "projects/test-project/locations/us-central1/tensorboards/123"
            ),
        )

        service.launch_job(job_config=job_config)

        mock_aiplatform_init.assert_called_once_with(
            project="test-project",
            location="us-central1",
            staging_bucket="gs://test-staging-bucket",
        )
        mock_custom_job_class.assert_called_once()
        _, custom_job_kwargs = mock_custom_job_class.call_args
        self.assertEqual(
            custom_job_kwargs["base_output_dir"],
            job_config.base_output_dir,
        )
        mock_job.submit.assert_called_once()
        _, submit_kwargs = mock_job.submit.call_args
        self.assertEqual(
            submit_kwargs["tensorboard"],
            job_config.tensorboard_resource_name,
        )


if __name__ == "__main__":
    absltest.main()
