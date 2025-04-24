import os
import unittest
from unittest.mock import call, patch

from gigl.common import GcsUri
from gigl.common.services.vertex_ai import LEADER_WORKER_INTERNAL_IP_FILE_PATH_ENV_KEY
from gigl.common.utils.vertex_ai_context import (
    DistributedContext,
    connect_worker_pool,
    get_host_name,
    get_leader_hostname,
    get_leader_port,
    get_rank,
    get_vertex_ai_job_id,
    get_world_size,
    is_currently_running_in_vertex_ai_job,
)
from gigl.distributed import DistributedContext


class TestVertexAIContext(unittest.TestCase):
    @patch.dict(os.environ, {"CLOUD_ML_JOB_ID": "test_job_id"})
    def test_is_currently_running_in_vertex_ai_job(self):
        self.assertTrue(is_currently_running_in_vertex_ai_job())

    @patch.dict(os.environ, {"CLOUD_ML_JOB_ID": "test_job_id"})
    def test_get_vertex_ai_job_id(self):
        self.assertEqual(get_vertex_ai_job_id(), "test_job_id")

    @patch.dict(os.environ, {"HOSTNAME": "test_hostname"})
    def test_get_host_name(self):
        self.assertEqual(get_host_name(), "test_hostname")

    @patch.dict(os.environ, {"MASTER_ADDR": "test_leader_hostname"})
    def test_get_leader_hostname(self):
        self.assertEqual(get_leader_hostname(), "test_leader_hostname")

    @patch.dict(os.environ, {"MASTER_PORT": "12345"})
    def test_get_leader_port(self):
        self.assertEqual(get_leader_port(), 12345)

    @patch.dict(os.environ, {"WORLD_SIZE": "4"})
    def test_get_world_size(self):
        self.assertEqual(get_world_size(), 4)

    @patch.dict(os.environ, {"RANK": "1"})
    def test_get_rank(self):
        self.assertEqual(get_rank(), 1)

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
        distributed_context: DistributedContext = connect_worker_pool()
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
        distributed_context: DistributedContext = connect_worker_pool()
        self.assertEqual(distributed_context.main_worker_ip_address, "127.0.0.1")
        self.assertEqual(distributed_context.global_rank, 1)
        self.assertEqual(distributed_context.global_world_size, 2)
        mock_read.assert_has_calls(
            [
                call(GcsUri("gs://FAKE BUCKET DNE/some-file.txt")),
                call(GcsUri("gs://FAKE BUCKET DNE/some-file.txt")),
            ]
        )


if __name__ == "__main__":
    unittest.main()
