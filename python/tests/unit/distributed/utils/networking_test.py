import json
import os
import subprocess
import unittest
from typing import Optional
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from parameterized import param, parameterized

from gigl.distributed.utils import (
    GraphStoreInfo,
    get_free_ports_from_master_node,
    get_free_ports_from_node,
    get_graph_store_info,
    get_internal_ip_from_master_node,
    get_internal_ip_from_node,
)
from tests.test_assets.distributed.utils import get_process_group_init_method


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


def _test_fetching_free_ports_from_node(
    rank: int,
    world_size: int,
    init_process_group_init_method: str,
    num_ports: int,
    master_node_rank: int,
    ports: list[int],
):
    # Initialize distributed process group
    dist.init_process_group(
        backend="gloo",
        init_method=init_process_group_init_method,
        world_size=world_size,
        rank=rank,
    )
    try:
        if rank == master_node_rank:
            with patch(
                "gigl.distributed.utils.networking.get_free_ports", return_value=ports
            ):
                free_ports: list[int] = get_free_ports_from_node(
                    num_ports=num_ports, node_rank=master_node_rank
                )
        else:
            with patch(
                "gigl.distributed.utils.networking.get_free_ports",
                side_effect=Exception("Should not be called on non-master node"),
            ):
                free_ports = get_free_ports_from_node(
                    num_ports=num_ports, node_rank=master_node_rank
                )
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


def _test_get_internal_ip_from_node(
    rank: int,
    world_size: int,
    init_process_group_init_method: str,
    expected_ip: str,
    master_node_rank: int,
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
        if rank == master_node_rank:
            master_ip = get_internal_ip_from_node(node_rank=master_node_rank)
        else:
            with patch(
                "gigl.distributed.utils.networking.socket.gethostbyname",
                side_effect=Exception("Should not be called on non-master node"),
            ):
                master_ip = get_internal_ip_from_node(node_rank=master_node_rank)
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
    def _test_get_free_ports_from_master_node_two_ranks(
        self, _name, num_ports, world_size
    ):
        init_process_group_init_method = get_process_group_init_method()
        mp.spawn(
            fn=_test_fetching_free_ports_in_dist_context,
            args=(world_size, init_process_group_init_method, num_ports),
            nprocs=world_size,
        )

    @parameterized.expand(
        [
            param(
                "Test fetching 2 ports for world_size = 2 with master_node_rank = 0",
                num_ports=2,
                world_size=2,
                master_node_rank=0,
                ports=[1, 2],
            ),
            param(
                "Test fetching 2 ports for world_size = 2 with master_node_rank = 1",
                num_ports=2,
                world_size=2,
                master_node_rank=1,
                ports=[3, 4],
            ),
        ]
    )
    def _test_get_free_ports_from_master_node_two_ranks_custom_master_node_rank(
        self, _name, num_ports, world_size, master_node_rank, ports
    ):
        init_process_group_init_method = get_process_group_init_method()
        mp.spawn(
            fn=_test_fetching_free_ports_from_node,
            args=(
                world_size,
                init_process_group_init_method,
                num_ports,
                master_node_rank,
                ports,
            ),
            nprocs=world_size,
        )

    def _test_get_free_ports_from_master_fails_if_process_group_not_initialized(self):
        with self.assertRaises(
            AssertionError,
            msg="An error should be raised since the `dist.init_process_group` is not initialized",
        ):
            get_free_ports_from_master_node(num_ports=1)

    def _test_get_internal_ip_from_master_node(self):
        init_process_group_init_method = get_process_group_init_method()
        expected_host_ip = subprocess.check_output(["hostname", "-i"]).decode().strip()
        world_size = 2
        mp.spawn(
            fn=_test_get_internal_ip_from_master_node_in_dist_context,
            args=(world_size, init_process_group_init_method, expected_host_ip),
            nprocs=world_size,
        )

    @parameterized.expand(
        [
            param(
                "Getting internal IP from master node with master_node_rank = 0",
                world_size=2,
                master_node_rank=0,
            ),
            param(
                "Getting internal IP from master node with master_node_rank = 1",
                world_size=2,
                master_node_rank=1,
            ),
        ]
    )
    def _test_get_internal_ip_from_master_node_with_master_node_rank(
        self, _, world_size, master_node_rank
    ):
        init_process_group_init_method = get_process_group_init_method()
        expected_host_ip = subprocess.check_output(["hostname", "-i"]).decode().strip()
        mp.spawn(
            fn=_test_get_internal_ip_from_node,
            args=(
                world_size,
                init_process_group_init_method,
                expected_host_ip,
                master_node_rank,
            ),
            nprocs=world_size,
        )

    def _test_get_internal_ip_from_master_node_fails_if_process_group_not_initialized(
        self,
    ):
        with self.assertRaises(
            AssertionError,
            msg="An error should be raised since the `dist.init_process_group` is not initialized",
        ):
            get_internal_ip_from_master_node()


def _test_get_graph_store_info_in_dist_context(
    rank: int,
    world_size: int,
    init_process_group_init_method: str,
    storage_nodes: int,
    compute_nodes: int,
):
    """Test get_graph_store_info in a real distributed context."""
    # Initialize distributed process group
    dist.init_process_group(
        backend="gloo",
        init_method=init_process_group_init_method,
        world_size=world_size,
        rank=rank,
    )
    try:
        # Call get_graph_store_info
        graph_store_info = get_graph_store_info()

        # Verify the result is a GraphStoreInfo instance
        assert isinstance(
            graph_store_info, GraphStoreInfo
        ), "Result should be a GraphStoreInfo instance"
        # Verify cluster sizes
        assert (
            graph_store_info.num_storage_nodes == storage_nodes
        ), f"Expected {storage_nodes} storage nodes"
        assert (
            graph_store_info.num_compute_nodes == compute_nodes
        ), f"Expected {compute_nodes} compute nodes"
        assert (
            graph_store_info.num_cluster_nodes == storage_nodes + compute_nodes
        ), "Total nodes should be sum of storage and compute nodes"

        # Verify IP addresses are strings and not empty
        assert isinstance(
            graph_store_info.cluster_master_ip, str
        ), "Cluster master IP should be a string"
        assert (
            len(graph_store_info.cluster_master_ip) > 0
        ), "Cluster master IP should not be empty"
        assert isinstance(
            graph_store_info.storage_cluster_master_ip, str
        ), "Storage cluster master IP should be a string"
        assert (
            len(graph_store_info.storage_cluster_master_ip) > 0
        ), "Storage cluster master IP should not be empty"
        assert isinstance(
            graph_store_info.compute_cluster_master_ip, str
        ), "Compute cluster master IP should be a string"
        assert (
            len(graph_store_info.compute_cluster_master_ip) > 0
        ), "Compute cluster master IP should not be empty"

        # Verify ports are positive integers
        assert isinstance(
            graph_store_info.cluster_master_port, int
        ), "Cluster master port should be an integer"
        assert (
            graph_store_info.cluster_master_port > 0
        ), "Cluster master port should be positive"
        assert isinstance(
            graph_store_info.storage_cluster_master_port, int
        ), "Storage cluster master port should be an integer"
        assert (
            graph_store_info.storage_cluster_master_port > 0
        ), "Storage cluster master port should be positive"
        assert isinstance(
            graph_store_info.compute_cluster_master_port, int
        ), "Compute cluster master port should be an integer"
        assert (
            graph_store_info.compute_cluster_master_port > 0
        ), "Compute cluster master port should be positive"

        # Verify all ranks get the same result (since they should all get the same broadcasted values)
        gathered_info: list[Optional[GraphStoreInfo]] = [None] * world_size
        dist.all_gather_object(gathered_info, graph_store_info)

        # All ranks should have the same GraphStoreInfo
        for i, info in enumerate(gathered_info):
            assert info is not None
            assert (
                info.num_cluster_nodes == graph_store_info.num_cluster_nodes
            ), f"Rank {i} should have same cluster nodes"
            assert (
                info.num_storage_nodes == graph_store_info.num_storage_nodes
            ), f"Rank {i} should have same storage nodes"
            assert (
                info.num_compute_nodes == graph_store_info.num_compute_nodes
            ), f"Rank {i} should have same compute nodes"
            assert (
                info.cluster_master_ip == graph_store_info.cluster_master_ip
            ), f"Rank {i} should have same cluster master IP"
            assert (
                info.storage_cluster_master_ip
                == graph_store_info.storage_cluster_master_ip
            ), f"Rank {i} should have same storage master IP"
            assert (
                info.compute_cluster_master_ip
                == graph_store_info.compute_cluster_master_ip
            ), f"Rank {i} should have same compute master IP"
            assert (
                info.cluster_master_port == graph_store_info.cluster_master_port
            ), f"Rank {i} should have same cluster master port"
            assert (
                info.storage_cluster_master_port
                == graph_store_info.storage_cluster_master_port
            ), f"Rank {i} should have same storage master port"
            assert (
                info.compute_cluster_master_port
                == graph_store_info.compute_cluster_master_port
            ), f"Rank {i} should have same compute master port"

    finally:
        dist.destroy_process_group()


def _get_cluster_spec_for_test(worker_pool_sizes: list[int]) -> dict:
    cluster_spec: dict = {
        "environment": "cloud",
        "task": {
            "type": "workerpool0",
            "index": 0,
        },
        "cluster": {},
    }
    for i, worker_pool_size in enumerate(worker_pool_sizes):
        cluster_spec["cluster"][f"workerpool{i}"] = [
            f"workerpool{i}-{j}:2222" for j in range(worker_pool_size)
        ]
    return cluster_spec


class TestGetGraphStoreInfo(unittest.TestCase):
    """Test suite for get_graph_store_info function."""

    def tearDown(self):
        """Clean up after each test."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def _test_get_graph_store_info_fails_when_distributed_not_initialized(self):
        """Test that get_graph_store_info fails when distributed environment is not initialized."""
        with self.assertRaises(ValueError) as context:
            get_graph_store_info()

        self.assertIn(
            "Distributed environment must be initialized", str(context.exception)
        )

    def _test_get_graph_store_info_fails_when_not_running_in_vertex_ai_job(self):
        """Test that get_graph_store_info fails when not running in a Vertex AI job."""
        init_process_group_init_method = get_process_group_init_method()
        torch.distributed.init_process_group(
            backend="gloo",
            init_method=init_process_group_init_method,
            world_size=1,
            rank=0,
        )
        with self.assertRaises(ValueError) as context:
            get_graph_store_info()

        self.assertIn(
            "Must be running on a vertex AI job to get graph store cluster info!",
            str(context.exception),
        )

    @parameterized.expand(
        [
            param(
                "Test with 1 storage node and 1 compute node",
                storage_nodes=1,
                compute_nodes=1,
            ),
            param(
                "Test with 2 storage nodes and 1 compute nodes",
                storage_nodes=2,
                compute_nodes=1,
            ),
            param(
                "Test with 3 storage nodes and 2 compute nodes",
                storage_nodes=3,
                compute_nodes=2,
            ),
        ]
    )
    def test_get_graph_store_info_success_in_distributed_context(
        self, _name, storage_nodes, compute_nodes
    ):
        """Test successful execution of get_graph_store_info in a real distributed context."""
        init_process_group_init_method = get_process_group_init_method()
        world_size = storage_nodes + compute_nodes
        if compute_nodes == 1:
            worker_pool_sizes = [1, 0, storage_nodes]
        else:
            worker_pool_sizes = [1, compute_nodes - 1, storage_nodes]
        with patch.dict(
            os.environ,
            {
                "CLUSTER_SPEC": json.dumps(
                    _get_cluster_spec_for_test(worker_pool_sizes)
                ),
                "CLOUD_ML_JOB_ID": "test_job_id",
            },
            clear=False,
        ):
            mp.spawn(
                fn=_test_get_graph_store_info_in_dist_context,
                args=(
                    world_size,
                    init_process_group_init_method,
                    storage_nodes,
                    compute_nodes,
                ),
                nprocs=world_size,
            )
