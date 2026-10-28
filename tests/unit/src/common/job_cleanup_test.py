from unittest.mock import MagicMock, patch

from google.api_core.exceptions import PermissionDenied

from gigl.common import LocalUri
from gigl.src.common.job_cleanup import cancel_child_jobs
from tests.test_assets.test_case import TestCase

_RESOURCE_CONFIG_URI = LocalUri("/not/read/resource_config.yaml")


class CancelChildJobsTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.resource_config = MagicMock()
        self.resource_config.project = "p"
        self.resource_config.region = "us-central1"
        self.resource_config.service_account_email = "sa"
        self.resource_config.temp_assets_regional_bucket_path.uri = "gs://b"
        patches = {
            "get_resource_config": patch(
                "gigl.src.common.job_cleanup.get_resource_config",
                return_value=self.resource_config,
            ),
            "vertex": patch("gigl.src.common.job_cleanup.VertexAIService"),
            "dataflow": patch("gigl.src.common.job_cleanup.DataflowService"),
            "dataproc": patch("gigl.src.common.job_cleanup.DataprocService"),
        }
        self.mocks = {name: p.start() for name, p in patches.items()}
        for p in patches.values():
            self.addCleanup(p.stop)
        self.vertex = self.mocks["vertex"].return_value
        self.dataflow = self.mocks["dataflow"].return_value
        self.dataproc = self.mocks["dataproc"].return_value
        self.vertex.cancel_custom_jobs_named.return_value = []
        self.dataflow.cancel_jobs_with_name_prefix.return_value = []
        self.dataproc.delete_cluster_if_exists.return_value = False

    def test_cancels_every_child_kind_by_job_name(self) -> None:
        cancel_child_jobs(job_name="My_Job", resource_config_uri=_RESOURCE_CONFIG_URI)

        self.mocks["get_resource_config"].assert_called_once_with(
            resource_config_uri=_RESOURCE_CONFIG_URI
        )
        self.mocks["vertex"].assert_called_once_with(
            project="p",
            location="us-central1",
            service_account="sa",
            staging_bucket="gs://b",
        )
        self.vertex.cancel_custom_jobs_named.assert_called_once_with(
            ["gigl_train_My_Job", "gigl_infer_My_Job"]
        )
        self.mocks["dataflow"].assert_called_once_with(
            project="p", region="us-central1"
        )
        self.dataflow.cancel_jobs_with_name_prefix.assert_called_once_with(
            "gigl-my-job-"
        )
        self.mocks["dataproc"].assert_called_once_with(
            project_id="p", region="us-central1"
        )
        self.assertEqual(
            [c.args[0] for c in self.dataproc.delete_cluster_if_exists.call_args_list],
            ["sgs-My-Job", "split-My-Job"],
        )

    def test_continues_after_one_client_raises_and_raises_at_the_end(self) -> None:
        self.vertex.cancel_custom_jobs_named.side_effect = PermissionDenied("no")

        with self.assertRaisesRegex(RuntimeError, "custom jobs"):
            cancel_child_jobs(job_name="j", resource_config_uri=_RESOURCE_CONFIG_URI)

        self.dataflow.cancel_jobs_with_name_prefix.assert_called_once()
        self.assertEqual(self.dataproc.delete_cluster_if_exists.call_count, 2)

    def test_no_children_is_not_an_error(self) -> None:
        self.assertIsNone(
            cancel_child_jobs(job_name="j", resource_config_uri=_RESOURCE_CONFIG_URI)
        )
