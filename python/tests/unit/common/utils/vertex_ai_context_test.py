import os
import unittest
from unittest.mock import patch

from gigl.common.utils.vertex_ai_context import (
    DistributedContext,
    connect_worker_pool,
    get_host_name,
    get_leader_hostname,
    get_leader_port,
    get_local_world_size,
    get_rank,
    get_vertex_ai_job_id,
    get_world_size,
    is_currently_running_in_vertex_ai_job,
)
from gigl.distributed import DistributedContext


class TestVertexAIContext(unittest.TestCase):
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

    @patch.dict(os.environ, VAI_JOB_ENV | {"LOCAL_WORLD_SIZE": "4"})
    def test_get_local_world_size(self):
        self.assertEqual(get_local_world_size(), 4)

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
        with self.assertRaises(Exception):
            get_local_world_size()

    @patch("torch.distributed.init_process_group")
    @patch("torch.distributed.broadcast_object_list")
    @patch("subprocess.check_output", return_value=b"127.0.0.1")
    @patch("time.sleep", return_value=None)
    @patch.dict(
        os.environ,
        {
            "RANK": "0",
            "WORLD_SIZE": "2",
            "LOCAL_WORLD_SIZE": "4",
            "CLOUD_ML_JOB_ID": "test_job_id",
        },
    )
    def test_connect_worker_pool_leader(
        self,
        mock_sleep,
        mock_check_output,
        mock_broadcast_object_list,
        mock_init_process_group,
    ):
        distributed_context: DistributedContext = connect_worker_pool()
        self.assertEqual(distributed_context.main_worker_ip_address, "127.0.0.1")
        self.assertEqual(distributed_context.global_rank, 0)
        self.assertEqual(distributed_context.global_world_size, 2)
        self.assertEqual(distributed_context.local_world_size, 4)
        self.assertEqual(len(distributed_context.master_sampling_ports), 4)
        self.assertEqual(len(distributed_context.master_worker_ports), 4)

    @patch("torch.distributed.init_process_group")
    @patch("torch.distributed.broadcast_object_list")
    @patch("subprocess.check_output", return_value=b"127.0.0.1")
    @patch("time.sleep", return_value=None)
    @patch.dict(
        os.environ,
        {
            "RANK": "1",
            "WORLD_SIZE": "2",
            "LOCAL_WORLD_SIZE": "4",
            "CLOUD_ML_JOB_ID": "test_job_id",
        },
    )
    def test_connect_worker_pool_worker(
        self,
        mock_sleep,
        mock_check_output,
        mock_broadcast_object_list,
        mock_init_process_group,
    ):
        def _mock_broadcast_object_list(object_list, src):
            # Simulate broadcasting
            object_list[0] = ("127.0.0.1", 0, [1, 2, 3, 4], [5, 6, 7, 8])

        mock_broadcast_object_list.side_effect = _mock_broadcast_object_list
        distributed_context: DistributedContext = connect_worker_pool()
        self.assertEqual(distributed_context.main_worker_ip_address, "127.0.0.1")
        self.assertEqual(distributed_context.global_rank, 1)
        self.assertEqual(distributed_context.global_world_size, 2)
        self.assertEqual(distributed_context.local_world_size, 4)
        self.assertEqual(distributed_context.master_partitioning_port, 0)
        self.assertEqual(distributed_context.master_worker_ports, [1, 2, 3, 4])
        self.assertEqual(distributed_context.master_sampling_ports, [5, 6, 7, 8])


if __name__ == "__main__":
    unittest.main()
