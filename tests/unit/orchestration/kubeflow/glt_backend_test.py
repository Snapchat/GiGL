from unittest.mock import patch

from absl.testing import absltest

from gigl.common import GcsUri
from gigl.orchestration.kubeflow.utils.glt_backend import resolve_should_use_glt_backend
from tests.test_assets.test_case import TestCase


class GltBackendTest(TestCase):
    @patch("gigl.orchestration.kubeflow.utils.glt_backend.GbmlConfigPbWrapper")
    def test_resolve_should_use_glt_backend_reads_task_config(self, MockGbmlConfig):
        mock_config = MockGbmlConfig.get_gbml_config_pb_wrapper_from_uri.return_value
        mock_config.should_use_glt_backend = True

        task_config_uri = GcsUri("gs://test-bucket/task_config.yaml")

        self.assertTrue(resolve_should_use_glt_backend(task_config_uri=task_config_uri))
        MockGbmlConfig.get_gbml_config_pb_wrapper_from_uri.assert_called_once_with(
            gbml_config_uri=task_config_uri
        )


if __name__ == "__main__":
    absltest.main()
