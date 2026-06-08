from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import ANY, patch

from absl.testing import absltest

from gigl.common import GcsUri, LocalUri
from gigl.common.logger import Logger
from gigl.orchestration.kubeflow.kfp_orchestrator import KfpOrchestrator
from gigl.src.common.types import AppliedTaskIdentifier
from tests.test_assets.test_case import TestCase

logger = Logger()


class KfpOrchestratorTest(TestCase):
    @patch("gigl.orchestration.kubeflow.kfp_orchestrator.FileLoader")
    def test_compile_uploads_compiled_yaml(self, MockFileLoader):
        mock_file_loader = MockFileLoader.return_value
        mock_file_loader.load_file.return_value = None

        dst_compiled_pipeline_path = GcsUri(
            "gs://SOME NON EXISTING BUCKET/ NON EXISTING FILE"
        )
        KfpOrchestrator.compile(
            cuda_container_image="SOME NONEXISTENT IMAGE 1",
            cpu_container_image="SOME NONEXISTENT IMAGE 2",
            dataflow_container_image="SOME NONEXISTENT IMAGE 3",
            dst_compiled_pipeline_path=dst_compiled_pipeline_path,
        )
        mock_file_loader.load_file.assert_called_once_with(
            file_uri_src=ANY, file_uri_dst=dst_compiled_pipeline_path
        )

    def test_compile_uses_glt_backend_pipeline_parameter(self):
        with TemporaryDirectory() as temp_dir:
            dst_compiled_pipeline_path = LocalUri(Path(temp_dir) / "pipeline.yaml")
            KfpOrchestrator.compile(
                cuda_container_image="SOME NONEXISTENT IMAGE 1",
                cpu_container_image="SOME NONEXISTENT IMAGE 2",
                dataflow_container_image="SOME NONEXISTENT IMAGE 3",
                dst_compiled_pipeline_path=dst_compiled_pipeline_path,
            )

            compiled_pipeline_yaml = Path(dst_compiled_pipeline_path.uri).read_text()

        self.assertIn("should_use_glt_backend", compiled_pipeline_yaml)
        self.assertNotIn(
            "check-glt-backend-eligibility-component", compiled_pipeline_yaml
        )

    @patch(
        "gigl.orchestration.kubeflow.kfp_orchestrator.resolve_should_use_glt_backend"
    )
    @patch("gigl.orchestration.kubeflow.kfp_orchestrator.VertexAIService")
    @patch("gigl.orchestration.kubeflow.kfp_orchestrator.get_resource_config")
    @patch("gigl.orchestration.kubeflow.kfp_orchestrator.FileLoader")
    def test_run_passes_resolved_glt_backend_param(
        self,
        MockFileLoader,
        mock_get_resource_config,
        MockVertexAIService,
        mock_resolve_should_use_glt_backend,
    ):
        mock_file_loader = MockFileLoader.return_value
        mock_file_loader.does_uri_exist.return_value = True
        mock_resolve_should_use_glt_backend.return_value = True
        mock_get_resource_config.return_value = SimpleNamespace(
            project="test-project",
            region="us-central1",
            service_account_email="test@test-project.iam.gserviceaccount.com",
            temp_assets_regional_bucket_path=GcsUri("gs://test-bucket"),
        )
        mock_vertex_ai_service = MockVertexAIService.return_value
        mock_vertex_ai_service.run_pipeline.return_value = "test-run"

        with TemporaryDirectory() as temp_dir:
            compiled_pipeline_path = LocalUri(Path(temp_dir) / "pipeline.yaml")
            Path(compiled_pipeline_path.uri).write_text(
                """
root:
  inputDefinitions:
    parameters:
      should_use_glt_backend:
        parameterType: BOOLEAN
"""
            )
            task_config_uri = GcsUri("gs://test-bucket/task_config.yaml")
            run = KfpOrchestrator().run(
                applied_task_identifier=AppliedTaskIdentifier("test_job"),
                task_config_uri=task_config_uri,
                resource_config_uri=GcsUri("gs://test-bucket/resource_config.yaml"),
                compiled_pipeline_path=compiled_pipeline_path,
            )

        self.assertEqual(run, "test-run")
        mock_resolve_should_use_glt_backend.assert_called_once_with(
            task_config_uri=task_config_uri
        )
        mock_vertex_ai_service.run_pipeline.assert_called_once()
        run_keyword_args = mock_vertex_ai_service.run_pipeline.call_args.kwargs[
            "run_keyword_args"
        ]
        self.assertTrue(run_keyword_args["should_use_glt_backend"])


if __name__ == "__main__":
    absltest.main()
