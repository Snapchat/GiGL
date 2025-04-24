import unittest
from unittest.mock import ANY, patch

from gigl.common import GcsUri
from gigl.common.logger import Logger
from gigl.orchestration.kubeflow.kfp_orchestrator import KfpOrchestrator

logger = Logger()


class KfpOrchestratorTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
