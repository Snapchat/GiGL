import json
import os
from unittest.mock import call, patch

from absl.testing import absltest

from gigl.common import GcsUri
from gigl.common.services.vertex_ai import LEADER_WORKER_INTERNAL_IP_FILE_PATH_ENV_KEY
from gigl.common.utils.vertex_ai_context import (
    ClusterSpec,
    TaskInfo,
    connect_worker_pool,
    get_cluster_spec,
    get_host_name,
    get_leader_hostname,
    get_leader_port,
    get_rank,
    get_vertex_ai_job_id,
    get_world_size,
    is_currently_running_in_vertex_ai_job,
)
from tests.test_assets.test_case import TestCase


class TestVertexAIContext(TestCase):
    VAI_JOB_ENV = {"CLOUD_ML_JOB_ID": "test_job_id"}

    @patch.dict(os.environ, VAI_JOB_ENV)
    def test_is_currently_running_in_vertex_ai_job(self):
        self.assertTrue(is_currently_running_in_vertex_ai_job())

    @patch.dict(os.environ, VAI_JOB_ENV)
    def test_get_vertex_ai_job_id(self):
        self.assertEqual(get_vertex_ai_job_id(), "test_job_id")

    @patch.dict(os.environ, VAI_JOB_ENV | {"HOSTNAME": "test_hostname"})
    def test_get_host_name(self):
        self.assertEqual(get_host_name(), "test_hostname")

    @patch.dict(os.environ, VAI_JOB_ENV | {"MASTER_ADDR": "test_leader_hostname"})
    def test_get_leader_hostname(self):
        self.assertEqual(get_leader_hostname(), "test_leader_hostname")

    @patch.dict(os.environ, VAI_JOB_ENV | {"MASTER_PORT": "12345"})
    def test_get_leader_port(self):
        self.assertEqual(get_leader_port(), 12345)

    @patch.dict(os.environ, VAI_JOB_ENV | {"WORLD_SIZE": "4"})
    def test_get_world_size(self):
        self.assertEqual(get_world_size(), 4)

    @patch.dict(os.environ, VAI_JOB_ENV | {"RANK": "1"})
    def test_get_rank(self):
        self.assertEqual(get_rank(), 1)

    def test_throws_if_not_on_vai(self):
        with self.assertRaises(Exception):
            get_vertex_ai_job_id()
        with self.assertRaises(Exception):
            get_host_name()
        with self.assertRaises(Exception):
            get_leader_hostname()
        with self.assertRaises(Exception):
            get_leader_port()
        with self.assertRaises(Exception):
            get_world_size()
        with self.assertRaises(Exception):
            get_rank()

    @patch("subprocess.check_output", return_value=b"127.0.0.1")
    @patch("time.sleep", return_value=None)
    @patch("gigl.common.utils.gcs.GcsUtils.upload_from_string")
    @patch.dict(
        os.environ,
        {
            "RANK": "0",
            "WORLD_SIZE": "2",
            LEADER_WORKER_INTERNAL_IP_FILE_PATH_ENV_KEY: "gs://FAKE BUCKET DNE/some-file.txt",
            "CLOUD_ML_JOB_ID": "test_job_id",
        },
    )
    def test_connect_worker_pool_leader(self, mock_upload, mock_sleep, mock_subprocess):
        distributed_context = connect_worker_pool()
        self.assertEqual(distributed_context.main_worker_ip_address, "127.0.0.1")
        self.assertEqual(distributed_context.global_rank, 0)
        self.assertEqual(distributed_context.global_world_size, 2)
        mock_upload.assert_called_once_with(
            gcs_path=GcsUri("gs://FAKE BUCKET DNE/some-file.txt"), content="127.0.0.1"
        )

    @patch("gigl.common.utils.vertex_ai_context._ping_host_ip")
    @patch("subprocess.check_output", return_value=b"127.0.0.1")
    @patch("time.sleep", return_value=None)
    @patch("gigl.common.utils.gcs.GcsUtils.read_from_gcs", return_value="127.0.0.1")
    @patch("gigl.common.utils.gcs.GcsUtils.upload_from_string")
    @patch.dict(
        os.environ,
        {
            "RANK": "1",
            "WORLD_SIZE": "2",
            LEADER_WORKER_INTERNAL_IP_FILE_PATH_ENV_KEY: "gs://FAKE BUCKET DNE/some-file.txt",
            "CLOUD_ML_JOB_ID": "test_job_id",
        },
    )
    def test_connect_worker_pool_worker(
        self, mock_upload, mock_read, mock_sleep, mock_subprocess, mock_ping_host
    ):
        mock_ping_host.side_effect = [False, True]
        distributed_context = connect_worker_pool()
        self.assertEqual(distributed_context.main_worker_ip_address, "127.0.0.1")
        self.assertEqual(distributed_context.global_rank, 1)
        self.assertEqual(distributed_context.global_world_size, 2)
        mock_read.assert_has_calls(
            [
                call(GcsUri("gs://FAKE BUCKET DNE/some-file.txt")),
                call(GcsUri("gs://FAKE BUCKET DNE/some-file.txt")),
            ]
        )

    def test_parse_cluster_spec_success(self):
        """Test successful parsing of cluster specification."""
        cluster_spec_json = json.dumps(
            {
                "cluster": {
                    "workerpool0": ["replica0-0", "replica0-1"],
                    "workerpool1": ["replica1-0"],
                    "workerpool2": ["replica2-0"],
                    "workerpool3": ["replica3-0", "replica3-1"],
                },
                "task": {"type": "workerpool0", "index": 1, "trial": "trial-123"},
                "environment": "cloud",
                # NOTE: We intentionally omit the "job" field from the ClusterSpec.
                # But want to make sure we can handle it if it's present.
                "job": '{ "worker_pool_specs": [ {"machine_spec": {"machine_type": "n1-standard-4"}}]}',
            }
        )

        with patch.dict(
            os.environ, self.VAI_JOB_ENV | {"CLUSTER_SPEC": cluster_spec_json}
        ):
            cluster_spec = get_cluster_spec()
            expected_cluster_spec = ClusterSpec(
                cluster={
                    "workerpool0": ["replica0-0", "replica0-1"],
                    "workerpool1": ["replica1-0"],
                    "workerpool2": ["replica2-0"],
                    "workerpool3": ["replica3-0", "replica3-1"],
                },
                environment="cloud",
                task=TaskInfo(type="workerpool0", index=1, trial="trial-123"),
            )
            self.assertEqual(cluster_spec, expected_cluster_spec)

    def test_parse_cluster_spec_not_on_vai(self):
        """Test that function raises ValueError when not running in Vertex AI."""
        with self.assertRaises(ValueError) as context:
            get_cluster_spec()
        self.assertIn("Not running in a Vertex AI job", str(context.exception))

    def test_parse_cluster_spec_missing_cluster_spec(self):
        """Test that function raises ValueError when CLUSTER_SPEC is missing."""
        with patch.dict(os.environ, self.VAI_JOB_ENV):
            with self.assertRaises(ValueError) as context:
                get_cluster_spec()
            self.assertIn(
                "CLUSTER_SPEC not found in environment variables",
                str(context.exception),
            )

    def test_parse_cluster_spec_invalid_json(self):
        """Test that function raises JSONDecodeError for invalid JSON."""
        with patch.dict(
            os.environ, self.VAI_JOB_ENV | {"CLUSTER_SPEC": "invalid json"}
        ):
            with self.assertRaises(json.JSONDecodeError) as context:
                get_cluster_spec()


if __name__ == "__main__":
    absltest.main()
