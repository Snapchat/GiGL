import subprocess
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from parameterized import param, parameterized

from gigl.distributed.utils import (
    get_free_port,
    get_free_ports_from_master_node,
    get_internal_ip_from_master_node,
)


def _test_fetching_free_ports_in_dist_context(
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
        free_ports: list[int] = get_free_ports_from_master_node(num_ports=num_ports)
        assert len(free_ports) == num_ports

        # Check that all ranks see the same ports broadcasted from master (rank 0)
        gathered_ports_across_ranks = [
            torch.zeros(num_ports, dtype=torch.int32) for _ in range(world_size)
        ]
        dist.all_gather_object(gathered_ports_across_ranks, free_ports)
        assert (
            len(gathered_ports_across_ranks) == world_size
        ), f"Expected {world_size} ports, but got {len(gathered_ports_across_ranks)}"
        ports_gathered_at_rank_0 = gathered_ports_across_ranks[0]
        assert (
            len(ports_gathered_at_rank_0) == num_ports
        ), "returned number of ports to match requested number of ports"
        assert all(
            port >= 0 for port in ports_gathered_at_rank_0
        ), "All ports should be non-negative integers"
        assert all(
            ports_gathered_at_rank_k == ports_gathered_at_rank_0
            for ports_gathered_at_rank_k in gathered_ports_across_ranks
        ), "All ranks should receive the same ports from master (rank 0)"
    finally:
        dist.destroy_process_group()


def _test_get_internal_ip_from_master_node_in_dist_context(
    rank: int, world_size: int, init_process_group_init_method: str, expected_ip: str
):
    # Initialize distributed process group
    dist.init_process_group(
        backend="gloo",
        init_method=init_process_group_init_method,
        world_size=world_size,
        rank=rank,
    )
    print(
        f"Rank {rank} initialized process group with init method: {init_process_group_init_method}"
    )
    try:
        master_ip = get_internal_ip_from_master_node()
        assert (
            master_ip == expected_ip
        ), f"Expected master IP to be {expected_ip}, but got {master_ip}"
    finally:
        dist.destroy_process_group()


class TestDistributedNetworkingUtils(unittest.TestCase):
    def tearDown(self):
        if dist.is_initialized():
            print("Destroying process group")
            # Ensure the process group is destroyed after each test
            # to avoid interference with subsequent tests
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
    def test_get_free_ports_from_master_node_two_ranks(
        self, _name, num_ports, world_size
    ):
        port = get_free_port()
        init_process_group_init_method = f"tcp://127.0.0.1:{port}"
        mp.spawn(
            fn=_test_fetching_free_ports_in_dist_context,
            args=(world_size, init_process_group_init_method, num_ports),
            nprocs=world_size,
        )

    def test_get_free_ports_from_master_fails_if_process_group_not_initialized(self):
        with self.assertRaises(
            AssertionError,
            msg="An error should be raised since the `dist.init_process_group` is not initialized",
        ):
            get_free_ports_from_master_node(num_ports=1)

    def test_get_internal_ip_from_master_node(self):
        port = get_free_port()
        init_process_group_init_method = f"tcp://127.0.0.1:{port}"
        expected_host_ip = subprocess.check_output(["hostname", "-i"]).decode().strip()
        world_size = 2
        mp.spawn(
            fn=_test_get_internal_ip_from_master_node_in_dist_context,
            args=(world_size, init_process_group_init_method, expected_host_ip),
            nprocs=world_size,
        )

    def test_get_internal_ip_from_master_node_fails_if_process_group_not_initialized(self):
        with self.assertRaises(
            AssertionError,
            msg="An error should be raised since the `dist.init_process_group` is not initialized",
        ):
            get_internal_ip_from_master_node()
