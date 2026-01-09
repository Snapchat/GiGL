import collections
import os

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
import socket
import unittest
from unittest import mock

import torch
import torch.multiprocessing as mp
from torch_geometric.data import Data

from gigl.common import Uri
from gigl.common.logger import Logger
from gigl.distributed.distributed_neighborloader import DistNeighborLoader
from gigl.distributed.graph_store.compute import (
    init_compute_process,
    shutdown_compute_proccess,
)
from gigl.distributed.graph_store.remote_dist_dataset import RemoteDistDataset
from gigl.distributed.graph_store.storage_main import storage_node_process
from gigl.distributed.utils.neighborloader import shard_nodes_by_process
from gigl.distributed.utils.networking import get_free_ports
from gigl.env.distributed import (
    COMPUTE_CLUSTER_LOCAL_WORLD_SIZE_ENV_KEY,
    GraphStoreInfo,
)
from gigl.src.mocking.lib.versioning import get_mocked_dataset_artifact_metadata
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO,
)
from tests.test_assets.distributed.utils import assert_tensor_equality

logger = Logger()


def _assert_sampler_input(
    cluster_info: GraphStoreInfo,
    sampler_input: list[torch.Tensor],
    expected_sampler_input: dict[int, list[torch.Tensor]],
) -> None:
    rank_expected_sampler_input = expected_sampler_input[cluster_info.compute_node_rank]
    for i in range(cluster_info.compute_cluster_world_size):
        if i == torch.distributed.get_rank():
            logger.info(
                f"Verifying sampler input for rank {i} / {cluster_info.compute_cluster_world_size}"
            )
            logger.info(f"--------------------------------")
            assert len(sampler_input) == len(rank_expected_sampler_input)
            for j, expected in enumerate(rank_expected_sampler_input):
                assert_tensor_equality(sampler_input[j], expected)
            logger.info(
                f"{i} / {cluster_info.compute_cluster_world_size} compute node rank input nodes verified"
            )
        torch.distributed.barrier()

    torch.distributed.barrier()


def _run_client_process(
    client_rank: int,
    cluster_info: GraphStoreInfo,
    mp_sharing_dict: dict[str, torch.Tensor],
    expected_sampler_input: dict[int, list[torch.Tensor]],
) -> None:
    init_compute_process(client_rank, cluster_info, compute_world_backend="gloo")

    remote_dist_dataset = RemoteDistDataset(
        cluster_info=cluster_info,
        local_rank=client_rank,
        mp_sharing_dict=mp_sharing_dict,
    )
    assert (
        remote_dist_dataset.get_edge_dir() == "in"
    ), f"Edge direction must be 'in' for the test dataset. Got {remote_dist_dataset.get_edge_dir()}"
    assert (
        remote_dist_dataset.get_edge_feature_info() is not None
    ), "Edge feature info must not be None for the test dataset"
    assert (
        remote_dist_dataset.get_node_feature_info() is not None
    ), "Node feature info must not be None for the test dataset"
    ports = remote_dist_dataset.get_free_ports_on_storage_cluster(num_ports=2)
    assert len(ports) == 2, "Expected 2 free ports"
    if torch.distributed.get_rank() == 0:
        all_ports = [None] * torch.distributed.get_world_size()
    else:
        all_ports = None
    torch.distributed.gather_object(ports, all_ports)
    logger.info(f"All ports: {all_ports}")

    if torch.distributed.get_rank() == 0:
        assert isinstance(all_ports, list)
        for i, received_ports in enumerate(all_ports):
            assert (
                received_ports == ports
            ), f"Expected {ports} free ports, got {received_ports}"

    torch.distributed.barrier()
    logger.info("Verified that all ranks received the same free ports")

    sampler_input = remote_dist_dataset.get_node_ids()
    _assert_sampler_input(cluster_info, sampler_input, expected_sampler_input)

    # test "simple" case where we don't have mp sharing dict too
    simple_sampler_input = RemoteDistDataset(
        cluster_info=cluster_info,
        local_rank=client_rank,
        mp_sharing_dict=None,
    ).get_node_ids()
    _assert_sampler_input(cluster_info, simple_sampler_input, expected_sampler_input)
    torch.distributed.barrier()

    # Test the DistNeighborLoader
    loader = DistNeighborLoader(
        dataset=remote_dist_dataset,
        num_neighbors=[2, 2],
        pin_memory_device=torch.device("cpu"),
        input_nodes=sampler_input,
        num_workers=2,
        worker_concurrency=2,
    )
    count = 0
    for datum in loader:
        assert isinstance(datum, Data)
        count += 1
    torch.distributed.barrier()
    logger.info(f"Rank {torch.distributed.get_rank()} loaded {count} batches")
    count_tensor = torch.tensor(count, dtype=torch.int64)
    all_node_count = 0
    for rank_expected_sampler_input in expected_sampler_input.values():
        all_node_count += sum(len(nodes) for nodes in rank_expected_sampler_input)
    torch.distributed.all_reduce(count_tensor, op=torch.distributed.ReduceOp.SUM)
    assert (
        count_tensor.item() == all_node_count
    ), f"Expected {all_node_count} total nodes, got {count_tensor.item()}"
    shutdown_compute_proccess()


def _client_process(
    client_rank: int,
    cluster_info: GraphStoreInfo,
    expected_sampler_input: dict[int, list[torch.Tensor]],
) -> None:
    logger.info(
        f"Initializing client node {client_rank} / {cluster_info.num_compute_nodes}. OS rank: {os.environ['RANK']}, OS world size: {os.environ['WORLD_SIZE']}, local client rank: {client_rank}"
    )

    mp_context = torch.multiprocessing.get_context("spawn")
    mp_sharing_dict = torch.multiprocessing.Manager().dict()
    client_processes = []
    logger.info("Starting client processes")
    for i in range(cluster_info.num_processes_per_compute):
        client_process = mp_context.Process(
            target=_run_client_process,
            args=[
                i,  # client_rank
                cluster_info,  # cluster_info
                mp_sharing_dict,  # mp_sharing_dict
                expected_sampler_input,  # expected_sampler_input
            ],
        )
        client_processes.append(client_process)
    for client_process in client_processes:
        client_process.start()
    for client_process in client_processes:
        client_process.join()


def _run_server_processes(
    cluster_info: GraphStoreInfo,
    task_config_uri: Uri,
    is_inference: bool,
) -> None:
    logger.info(
        f"Initializing server processes. OS rank: {os.environ['RANK']}, OS world size: {os.environ['WORLD_SIZE']}"
    )
    storage_node_process(
        storage_rank=cluster_info.storage_node_rank,
        cluster_info=cluster_info,
        task_config_uri=task_config_uri,
        is_inference=is_inference,
        tf_record_uri_pattern=".*tfrecord",
        storage_world_backend="gloo",
    )


def _get_expected_input_nodes_by_rank(
    num_nodes: int, cluster_info: GraphStoreInfo
) -> dict[int, list[torch.Tensor]]:
    """Get the expected sampler input for each compute rank.

    We generate the expected sampler input for each compute rank by sharding the nodes across the compute ranks.
    We then append the generated nodes to the expected sampler input for each compute rank.
    Example for num_nodes = 16, num_processes_per_compute = 1, num_compute_nodes = 2, num_storage_nodes = 2:
    {
    0: # compute rank 0
    [
        [0, 1, 3, 4], # From storage rank 0
        [8, 9, 11, 12] # From storage rank 1
    ]
    1: # compute rank 1
    [
        [5, 6, 7, 8], # From storage rank 0
        [13, 14, 15, 16] # From storage rank 1
    ],
    }


    Args:
        num_nodes (int): The number of nodes in the graph.
        cluster_info (GraphStoreInfo): The cluster information.

    Returns:
        dict[int, list[torch.Tensor]]: The expected sampler input for each compute rank.
    """
    expected_sampler_input = collections.defaultdict(list)
    all_nodes = torch.arange(num_nodes, dtype=torch.int64)
    for server_rank in range(cluster_info.num_storage_nodes):
        server_node_start = server_rank * num_nodes // cluster_info.num_storage_nodes
        server_node_end = (
            (server_rank + 1) * num_nodes // cluster_info.num_storage_nodes
        )
        server_nodes = all_nodes[server_node_start:server_node_end]
        for compute_rank in range(cluster_info.num_compute_nodes):
            generated_nodes = shard_nodes_by_process(
                server_nodes, compute_rank, cluster_info.num_processes_per_compute
            )
            expected_sampler_input[compute_rank].append(generated_nodes)
    return dict(expected_sampler_input)


class TestUtils(unittest.TestCase):
    def test_graph_store_locally(self):
        # Simulating two server machine, two compute machines.
        # Each machine has one process.
        cora_supervised_info = get_mocked_dataset_artifact_metadata()[
            CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO.name
        ]
        task_config_uri = cora_supervised_info.frozen_gbml_config_uri
        (
            cluster_master_port,
            storage_cluster_master_port,
            compute_cluster_master_port,
            master_port,
            rpc_master_port,
            rpc_wait_port,
        ) = get_free_ports(num_ports=6)
        host_ip = socket.gethostbyname(socket.gethostname())
        cluster_info = GraphStoreInfo(
            num_storage_nodes=2,
            num_compute_nodes=2,
            num_processes_per_compute=2,
            cluster_master_ip=host_ip,
            storage_cluster_master_ip=host_ip,
            compute_cluster_master_ip=host_ip,
            cluster_master_port=cluster_master_port,
            storage_cluster_master_port=storage_cluster_master_port,
            compute_cluster_master_port=compute_cluster_master_port,
            rpc_master_port=rpc_master_port,
            rpc_wait_port=rpc_wait_port,
        )

        num_cora_nodes = 2708
        expected_sampler_input = _get_expected_input_nodes_by_rank(
            num_cora_nodes, cluster_info
        )

        ctx = mp.get_context("spawn")
        client_processes: list = []
        for i in range(cluster_info.num_compute_nodes):
            with mock.patch.dict(
                os.environ,
                {
                    "MASTER_ADDR": host_ip,
                    "MASTER_PORT": str(master_port),
                    "RANK": str(i),
                    "WORLD_SIZE": str(cluster_info.compute_cluster_world_size),
                    COMPUTE_CLUSTER_LOCAL_WORLD_SIZE_ENV_KEY: str(
                        cluster_info.num_processes_per_compute
                    ),
                },
                clear=False,
            ):
                client_process = ctx.Process(
                    target=_client_process,
                    args=[
                        i,  # client_rank
                        cluster_info,  # cluster_info
                        expected_sampler_input,  # expected_sampler_input
                    ],
                )
                client_process.start()
                client_processes.append(client_process)
        # Start server process
        server_processes = []
        for i in range(cluster_info.num_storage_nodes):
            with mock.patch.dict(
                os.environ,
                {
                    "MASTER_ADDR": host_ip,
                    "MASTER_PORT": str(master_port),
                    "RANK": str(i + cluster_info.num_compute_nodes),
                    "WORLD_SIZE": str(cluster_info.compute_cluster_world_size),
                    COMPUTE_CLUSTER_LOCAL_WORLD_SIZE_ENV_KEY: str(
                        cluster_info.num_processes_per_compute
                    ),
                },
                clear=False,
            ):
                server_process = ctx.Process(
                    target=_run_server_processes,
                    args=[
                        cluster_info,  # cluster_info
                        task_config_uri,  # task_config_uri
                        True,  # is_inference
                    ],
                )
                server_process.start()
                server_processes.append(server_process)

        for client_process in client_processes:
            client_process.join()
        for server_process in server_processes:
            server_process.join()
