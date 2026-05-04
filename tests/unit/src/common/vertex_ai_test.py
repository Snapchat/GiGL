"""Unit tests for gigl.common.services.vertex_ai."""

from unittest.mock import Mock, patch

from absl.testing import absltest

from gigl.common.services.vertex_ai import VertexAiJobConfig, VertexAIService
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
        self.assertNotIn("experiment", submit_kwargs)


    def test_vertex_ai_job_config_carries_experiment_name(self) -> None:
        cfg = VertexAiJobConfig(
            job_name="job",
            container_uri="gcr.io/p/img:tag",
            command=["python", "-m", "x"],
            tensorboard_resource_name="projects/p/locations/us/tensorboards/1",
            tensorboard_experiment_name="my-comparison",
        )
        self.assertEqual(cfg.tensorboard_experiment_name, "my-comparison")

    @patch("gigl.common.services.vertex_ai.aiplatform.Experiment")
    @patch("gigl.common.services.vertex_ai.aiplatform.CustomJob")
    @patch("gigl.common.services.vertex_ai.aiplatform.init")
    def test_submit_job_uses_experiment_when_set(
        self,
        mock_aiplatform_init: Mock,
        mock_custom_job_class: Mock,
        mock_experiment_cls: Mock,
    ) -> None:
        """When tensorboard_experiment_name is set, submit uses experiment= and experiment_run= instead of tensorboard=."""
        mock_exp = Mock()
        mock_exp.get_backing_tensorboard_resource.return_value = Mock(
            resource_name="projects/test/locations/us-central1/tensorboards/123"
        )
        mock_experiment_cls.get.return_value = mock_exp

        mock_job = Mock()
        mock_job.resource_name = "projects/test/locations/us-central1/customJobs/456"
        mock_job.name = "456"
        mock_custom_job_class.return_value = mock_job

        service = VertexAIService(
            project="test-project",
            location="us-central1",
            service_account="svc@test-project.iam.gserviceaccount.com",
            staging_bucket="gs://test-staging-bucket",
        )

        job_config = VertexAiJobConfig(
            job_name="test-job-exp",
            container_uri="gcr.io/test/image:latest",
            command=["python", "-m", "trainer"],
            base_output_dir="gs://test-perm-bucket/test-job/trainer",
            tensorboard_resource_name="projects/test/locations/us-central1/tensorboards/123",
            tensorboard_experiment_name="my-comparison",
        )

        service.launch_job(job_config=job_config)

        mock_job.submit.assert_called_once()
        submit_kwargs = mock_job.submit.call_args.kwargs
        self.assertEqual(submit_kwargs["experiment"], "my-comparison")
        self.assertEqual(submit_kwargs["experiment_run"], job_config.job_name)
        self.assertNotIn("tensorboard", submit_kwargs)

    @patch("gigl.common.services.vertex_ai.aiplatform.CustomJob")
    @patch("gigl.common.services.vertex_ai.aiplatform.init")
    def test_submit_job_raises_when_experiment_name_set_but_no_tb_resource(
        self,
        mock_aiplatform_init: Mock,
        mock_custom_job_class: Mock,
    ) -> None:
        """When tensorboard_experiment_name is set but tensorboard_resource_name is empty, raises ValueError."""
        mock_job = Mock()
        mock_custom_job_class.return_value = mock_job

        service = VertexAIService(
            project="test-project",
            location="us-central1",
            service_account="svc@test-project.iam.gserviceaccount.com",
            staging_bucket="gs://test-staging-bucket",
        )

        job_config = VertexAiJobConfig(
            job_name="test-job-no-tb",
            container_uri="gcr.io/test/image:latest",
            command=["python", "-m", "trainer"],
            base_output_dir="gs://test-perm-bucket/test-job/trainer",
            tensorboard_resource_name="",
            tensorboard_experiment_name="my-comparison",
        )

        with self.assertRaises(ValueError) as ctx:
            service.launch_job(job_config=job_config)

        self.assertIn("tensorboard_resource_name", str(ctx.exception))


class TestEnsureExperimentWithBackingTb(TestCase):
    """Tests for VertexAIService._ensure_experiment_with_backing_tb."""

    _TB_RESOURCE_NAME = "projects/p/locations/us-central1/tensorboards/42"
    _EXPERIMENT_NAME = "my-experiment"

    def _make_service(self, mock_init: Mock) -> VertexAIService:
        return VertexAIService(
            project="test-project",
            location="us-central1",
            service_account="svc@test.iam.gserviceaccount.com",
            staging_bucket="gs://test-bucket",
        )

    @patch("gigl.common.services.vertex_ai.aiplatform.Experiment")
    @patch("gigl.common.services.vertex_ai.aiplatform.init")
    def test_experiment_does_not_exist_creates_and_assigns(
        self,
        mock_init: Mock,
        mock_experiment_class: Mock,
    ) -> None:
        """When the experiment doesn't exist, creates it and assigns backing TB."""
        mock_experiment_class.get.return_value = None
        mock_new_experiment = Mock()
        mock_experiment_class.create.return_value = mock_new_experiment

        service = self._make_service(mock_init)
        service._ensure_experiment_with_backing_tb(
            self._EXPERIMENT_NAME, self._TB_RESOURCE_NAME
        )

        mock_experiment_class.get.assert_called_once_with(self._EXPERIMENT_NAME)
        mock_experiment_class.create.assert_called_once_with(self._EXPERIMENT_NAME)
        mock_new_experiment.assign_backing_tensorboard.assert_called_once_with(
            self._TB_RESOURCE_NAME
        )

    @patch("gigl.common.services.vertex_ai.aiplatform.Experiment")
    @patch("gigl.common.services.vertex_ai.aiplatform.init")
    def test_experiment_exists_no_backing_tb_assigns(
        self,
        mock_init: Mock,
        mock_experiment_class: Mock,
    ) -> None:
        """When the experiment exists with no backing TB, assigns the backing TB."""
        mock_existing_experiment = Mock()
        mock_existing_experiment.get_backing_tensorboard_resource.return_value = None
        mock_experiment_class.get.return_value = mock_existing_experiment

        service = self._make_service(mock_init)
        service._ensure_experiment_with_backing_tb(
            self._EXPERIMENT_NAME, self._TB_RESOURCE_NAME
        )

        mock_experiment_class.create.assert_not_called()
        mock_existing_experiment.assign_backing_tensorboard.assert_called_once_with(
            self._TB_RESOURCE_NAME
        )

    @patch("gigl.common.services.vertex_ai.aiplatform.Experiment")
    @patch("gigl.common.services.vertex_ai.aiplatform.init")
    def test_experiment_exists_different_backing_tb_raises(
        self,
        mock_init: Mock,
        mock_experiment_class: Mock,
    ) -> None:
        """When the experiment exists with a different backing TB, raises ValueError."""
        mock_backing = Mock()
        mock_backing.resource_name = "projects/p/locations/us-central1/tensorboards/99"
        mock_existing_experiment = Mock()
        mock_existing_experiment.get_backing_tensorboard_resource.return_value = (
            mock_backing
        )
        mock_experiment_class.get.return_value = mock_existing_experiment

        service = self._make_service(mock_init)
        with self.assertRaises(ValueError) as ctx:
            service._ensure_experiment_with_backing_tb(
                self._EXPERIMENT_NAME, self._TB_RESOURCE_NAME
            )

        self.assertIn("backing tensorboard", str(ctx.exception).lower())

    @patch("gigl.common.services.vertex_ai.aiplatform.Experiment")
    @patch("gigl.common.services.vertex_ai.aiplatform.init")
    def test_experiment_exists_matching_backing_tb_is_noop(
        self,
        mock_init: Mock,
        mock_experiment_class: Mock,
    ) -> None:
        """When the experiment exists with the correct backing TB, does nothing."""
        mock_backing = Mock()
        mock_backing.resource_name = self._TB_RESOURCE_NAME
        mock_existing_experiment = Mock()
        mock_existing_experiment.get_backing_tensorboard_resource.return_value = (
            mock_backing
        )
        mock_experiment_class.get.return_value = mock_existing_experiment

        service = self._make_service(mock_init)
        # Should not raise and should not call assign or create
        service._ensure_experiment_with_backing_tb(
            self._EXPERIMENT_NAME, self._TB_RESOURCE_NAME
        )

        mock_experiment_class.create.assert_not_called()
        mock_existing_experiment.assign_backing_tensorboard.assert_not_called()


if __name__ == "__main__":
    absltest.main()
