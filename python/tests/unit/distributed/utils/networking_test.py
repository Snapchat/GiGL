import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from parameterized import param, parameterized

from gigl.distributed.utils import get_free_master_ports, get_free_port


def run_in_dist_context(
    rank: int, world_size: int, init_process_group_init_method: str, num_ports: int
):
    # Initialize distributed process group
    dist.init_process_group(
        backend="gloo",
        init_method=init_process_group_init_method,
        world_size=world_size,
        rank=rank,
    )
    try:
        ports: list[int] = get_free_master_ports(num_ports=num_ports)
        # Check that all ranks see the same ports broadcasted from master (rank 0)
        gathered_ports = [
            torch.zeros(num_ports, dtype=torch.int32) for _ in range(world_size)
        ]
        dist.all_gather_object(gathered_ports, ports)
        assert len(gathered_ports) == world_size, (
            f"Expected {world_size} ports, but got {len(gathered_ports)}"
        )
        assert all(isinstance(p, int) and p > 0 for p in gathered_ports), (
            f"All gathered ports should be positive integers, but got: {gathered_ports}"
        )
        for rank in range(world_size):
            port_n = [gathered_ports[rank][port_num] for port_num in range(num_ports)]
            assert all(
                p == port_n[0] for p in port_n
            ), f"Ports not synchronized across ranks: {port_n}"
    finally:
        dist.destroy_process_group()


class TestGetFreeMasterPorts(unittest.TestCase):
    def tearDown(self):
        dist.destroy_process_group()

    @parameterized.expand(
        [
            param(
                "Test fetching 1 port for world_size = 1",
                num_ports=1,
                world_size=1,
            ),
            param(
                "Test fetching 1 port for world_size = 2",
                num_ports=1,
                world_size=2,
            ),
            param(
                "Test fetching 2 ports for world_size = 2",
                num_ports=2,
                world_size=2,
            ),
        ]
    )
    def test_get_free_master_ports_two_ranks(self, _, num_ports: int, world_size: int):
        port = get_free_port()
        init_process_group_init_method = f"tcp://127.0.0.1:{port}"
        mp.spawn(
            run_in_dist_context,
            args=(world_size, init_process_group_init_method, num_ports),
            nprocs=world_size,
        )

    def test_fails_if_process_group_not_initialized(self):
        with self.assertRaises(
            RuntimeError,
            msg="An error should be raised since the `dist.init_process_group` is not initialized"
        ):
            get_free_master_ports(num_ports=1)
