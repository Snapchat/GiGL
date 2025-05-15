import os
import socket
import unittest
from unittest.mock import patch

from gigl.distributed import DistributedContext

_TEST_MAIN_WORKER_IP_ADDRESS = "localhost"
_TEST_GLOBAL_RANK = 0
_TEST_GLOBAL_WORLD_SIZE = 1
_TEST_LOCAL_WORLD_SIZE = 4


def _is_port_free(master_ip_address: str, port: int):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((master_ip_address, port))
            return True
        except socket.error:
            return False


class DistributedContextTest(unittest.TestCase):
    def test_distributed_context_manual(self):
        distributed_context = DistributedContext(
            main_worker_ip_address=_TEST_MAIN_WORKER_IP_ADDRESS,
            global_rank=_TEST_GLOBAL_RANK,
            global_world_size=_TEST_GLOBAL_WORLD_SIZE,
            local_world_size=_TEST_LOCAL_WORLD_SIZE,
        )
        self.assertEqual(
            distributed_context.main_worker_ip_address, _TEST_MAIN_WORKER_IP_ADDRESS
        )
        self.assertEqual(distributed_context.global_rank, _TEST_GLOBAL_RANK)
        self.assertEqual(distributed_context.global_world_size, _TEST_GLOBAL_WORLD_SIZE)
        self.assertEqual(distributed_context.local_world_size, _TEST_LOCAL_WORLD_SIZE)

        all_ports = (
            [distributed_context.master_partitioning_port]
            + list(distributed_context.local_rank_to_master_sampling_port.values())
            + list(distributed_context.local_rank_to_master_worker_port.values())
        )
        self.assertEqual(len(all_ports), len(set(all_ports)))
        self.assertEqual(len(all_ports), 2 * _TEST_LOCAL_WORLD_SIZE + 1)
        for port in all_ports:
            self.assertTrue(
                _is_port_free(
                    master_ip_address=distributed_context.main_worker_ip_address,
                    port=port,
                )
            )

    @patch.dict(os.environ, {"LOCAL_WORLD_SIZE": str(_TEST_LOCAL_WORLD_SIZE)})
    def test_distributed_context_inferred(self):
        distributed_context = DistributedContext(
            main_worker_ip_address=_TEST_MAIN_WORKER_IP_ADDRESS,
            global_rank=_TEST_GLOBAL_RANK,
            global_world_size=_TEST_GLOBAL_WORLD_SIZE,
        )
        self.assertEqual(
            distributed_context.main_worker_ip_address, _TEST_MAIN_WORKER_IP_ADDRESS
        )
        self.assertEqual(distributed_context.global_rank, _TEST_GLOBAL_RANK)
        self.assertEqual(distributed_context.global_world_size, _TEST_GLOBAL_WORLD_SIZE)
        self.assertEqual(distributed_context.local_world_size, _TEST_LOCAL_WORLD_SIZE)

        all_ports = (
            [distributed_context.master_partitioning_port]
            + list(distributed_context.local_rank_to_master_sampling_port.values())
            + list(distributed_context.local_rank_to_master_worker_port.values())
        )
        self.assertEqual(len(all_ports), len(set(all_ports)))
        self.assertEqual(len(all_ports), 2 * _TEST_LOCAL_WORLD_SIZE + 1)
        for port in all_ports:
            self.assertTrue(
                _is_port_free(
                    master_ip_address=distributed_context.main_worker_ip_address,
                    port=port,
                )
            )


if __name__ == "__main__":
    unittest.main()
