import socket
import unittest

from gigl.distributed.dist_context import get_free_ports

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


class GetFreePortTest(unittest.TestCase):
    def test_get_free_ports(self):
        (
            master_partitioning_port,
            master_sampling_ports,
            master_worker_ports,
        ) = get_free_ports(
            main_worker_ip_address=_TEST_MAIN_WORKER_IP_ADDRESS,
            local_world_size=_TEST_LOCAL_WORLD_SIZE,
        )
        all_ports = (
            [master_partitioning_port] + master_sampling_ports + master_worker_ports
        )
        self.assertEqual(len(all_ports), len(set(all_ports)))
        self.assertEqual(len(all_ports), 2 * _TEST_LOCAL_WORLD_SIZE + 1)
        for port in all_ports:
            self.assertTrue(
                _is_port_free(
                    master_ip_address=_TEST_MAIN_WORKER_IP_ADDRESS,
                    port=port,
                )
            )


if __name__ == "__main__":
    unittest.main()
