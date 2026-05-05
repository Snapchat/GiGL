"""Unit tests for gigl.common.services.vertex_ai."""

from unittest.mock import Mock, patch

from absl.testing import absltest

from gigl.common.services.vertex_ai import (
    VertexAiJobConfig,
    VertexAIService,
    _build_tensorboard_experiment_url,
)
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

    @patch("gigl.common.services.vertex_ai.aiplatform.CustomJob")
    @patch("gigl.common.services.vertex_ai.aiplatform.init")
    def test_submit_job_passes_tensorboard_with_or_without_experiment_name(
        self,
        mock_aiplatform_init: Mock,
        mock_custom_job_class: Mock,
    ) -> None:
        """``tensorboard=`` is always passed when a TB resource is set, so the
        VAI job page's "Open TensorBoard" link works. The chief-rank uploader
        (driven by injected env vars) handles cross-job comparison separately.
        """
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
        self.assertEqual(
            submit_kwargs["tensorboard"],
            "projects/test/locations/us-central1/tensorboards/123",
        )
        self.assertNotIn("experiment", submit_kwargs)
        self.assertNotIn("experiment_run", submit_kwargs)

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


class TestSubmitJobValidatesExperimentName(TestCase):
    """Tests that _submit_job validates the user-supplied experiment name."""

    @patch("gigl.common.services.vertex_ai.aiplatform.CustomJob")
    @patch("gigl.common.services.vertex_ai.aiplatform.init")
    def test_invalid_experiment_name_raises(
        self,
        mock_aiplatform_init: Mock,
        mock_custom_job_class: Mock,
    ) -> None:
        """User-supplied tensorboard_experiment_name must match Vertex's regex."""
        mock_job = Mock()
        mock_custom_job_class.return_value = mock_job

        service = VertexAIService(
            project="test-project",
            location="us-central1",
            service_account="svc@test-project.iam.gserviceaccount.com",
            staging_bucket="gs://test-staging-bucket",
        )

        job_config = VertexAiJobConfig(
            job_name="any-job",
            container_uri="gcr.io/test/image:latest",
            command=["python", "-m", "trainer"],
            base_output_dir="gs://test-perm-bucket/run/trainer",
            tensorboard_resource_name="projects/test/locations/us-central1/tensorboards/123",
            tensorboard_experiment_name="Invalid_Experiment_Name",
        )

        with self.assertRaises(ValueError) as ctx:
            service.launch_job(job_config=job_config)

        self.assertIn("tensorboard_experiment_name", str(ctx.exception))


class TestBuildTensorboardExperimentUrl(TestCase):
    """Tests for the small URL-builder helper used in submit-time logging."""

    def test_builds_url_for_well_formed_resource_name(self) -> None:
        url = _build_tensorboard_experiment_url(
            tensorboard_resource_name="projects/p/locations/us-central1/tensorboards/42",
            experiment_id="my-exp",
        )
        self.assertEqual(
            url,
            "https://us-central1.tensorboard.googleusercontent.com/experiment/"
            "projects+p+locations+us-central1+tensorboards+42+experiments+my-exp",
        )

    def test_returns_none_for_malformed_resource_name(self) -> None:
        # A stray bad TB resource name should never break submission — the
        # caller falls back to no URL log.
        self.assertIsNone(
            _build_tensorboard_experiment_url(
                tensorboard_resource_name="not-a-resource-name",
                experiment_id="my-exp",
            )
        )


class TestSubmitJobLogsTensorboardUrls(TestCase):
    """Tests that _submit_job logs both per-job and cross-job TB URLs."""

    @patch("gigl.common.services.vertex_ai.logger.info")
    @patch("gigl.common.services.vertex_ai.aiplatform.CustomJob")
    @patch("gigl.common.services.vertex_ai.aiplatform.init")
    def test_logs_both_urls_when_experiment_name_set(
        self,
        mock_aiplatform_init: Mock,
        mock_custom_job_class: Mock,
        mock_logger_info: Mock,
    ) -> None:
        mock_job = Mock()
        mock_job.resource_name = "projects/p/locations/us-central1/customJobs/9876"
        mock_job.name = "9876"  # numeric job ID
        mock_custom_job_class.return_value = mock_job

        service = VertexAIService(
            project="p",
            location="us-central1",
            service_account="svc@p.iam.gserviceaccount.com",
            staging_bucket="gs://staging",
        )
        job_config = VertexAiJobConfig(
            job_name="my-job",
            container_uri="gcr.io/p/img",
            command=["python", "-m", "trainer"],
            base_output_dir="gs://staging/my-job/trainer",
            tensorboard_resource_name="projects/p/locations/us-central1/tensorboards/42",
            tensorboard_experiment_name="my-exp",
        )

        service.launch_job(job_config=job_config)

        emitted = " ".join(call.args[0] for call in mock_logger_info.call_args_list)
        # Per-job URL keyed on the job's numeric ID.
        self.assertIn(
            "experiments+9876",
            emitted,
        )
        # Cross-job URL keyed on the user-supplied experiment name.
        self.assertIn(
            "experiments+my-exp",
            emitted,
        )

    @patch("gigl.common.services.vertex_ai.logger.info")
    @patch("gigl.common.services.vertex_ai.aiplatform.CustomJob")
    @patch("gigl.common.services.vertex_ai.aiplatform.init")
    def test_logs_only_per_job_url_when_no_experiment_name(
        self,
        mock_aiplatform_init: Mock,
        mock_custom_job_class: Mock,
        mock_logger_info: Mock,
    ) -> None:
        mock_job = Mock()
        mock_job.resource_name = "projects/p/locations/us-central1/customJobs/9876"
        mock_job.name = "9876"
        mock_custom_job_class.return_value = mock_job

        service = VertexAIService(
            project="p",
            location="us-central1",
            service_account="svc@p.iam.gserviceaccount.com",
            staging_bucket="gs://staging",
        )
        job_config = VertexAiJobConfig(
            job_name="my-job",
            container_uri="gcr.io/p/img",
            command=["python", "-m", "trainer"],
            base_output_dir="gs://staging/my-job/trainer",
            tensorboard_resource_name="projects/p/locations/us-central1/tensorboards/42",
        )

        service.launch_job(job_config=job_config)

        emitted = " ".join(call.args[0] for call in mock_logger_info.call_args_list)
        self.assertIn("experiments+9876", emitted)
        self.assertNotIn("cross-job comparison", emitted)


if __name__ == "__main__":
    absltest.main()
