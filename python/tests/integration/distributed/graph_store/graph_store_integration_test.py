import collections
import os
import unittest
from unittest import mock

import torch
import torch.multiprocessing as mp
from graphlearn_torch.distributed import init_client, shutdown_client

from gigl.common import Uri
from gigl.common.logger import Logger
from gigl.distributed.graph_store.remote_dist_dataset import RemoteDistDataset
from gigl.distributed.graph_store.storage_main import storage_node_process
from gigl.distributed.utils import get_free_port
from gigl.distributed.utils.neighborloader import shard_nodes_by_process
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


def _run_client_process(
    client_rank: int,
    cluster_info: GraphStoreInfo,
    expected_sampler_input: dict[int, list[torch.Tensor]],
) -> None:
    client_global_rank = (
        cluster_info.compute_node_rank * cluster_info.num_processes_per_compute
        + client_rank
    )
    logger.info(
        f"Initializing client process {client_global_rank} / {cluster_info.compute_cluster_world_size}. on {cluster_info.cluster_master_ip}:{cluster_info.cluster_master_port}. OS rank: {os.environ['RANK']}, local client rank: {client_rank} on port: {cluster_info.cluster_master_port}"
    )
    # TODO(kmonte): Add gigl.*.init_client as a helper function to do this.
    torch.distributed.init_process_group(
        backend="gloo",
        world_size=cluster_info.compute_cluster_world_size,
        rank=client_global_rank,
        init_method=f"tcp://{cluster_info.compute_cluster_master_ip}:{cluster_info.compute_cluster_master_port}",
        group_name="gigl_client_comms",
    )
    logger.info(
        f"Client {client_global_rank} / {cluster_info.compute_cluster_world_size} process group initialized"
    )
    init_client(
        num_servers=cluster_info.num_storage_nodes,
        num_clients=cluster_info.compute_cluster_world_size,
        client_rank=client_global_rank,
        master_addr=cluster_info.cluster_master_ip,
        master_port=cluster_info.cluster_master_port,
        client_group_name="gigl_client_rpc",
    )

    torch.distributed.barrier()
    remote_dist_dataset = RemoteDistDataset(
        local_rank=client_rank, cluster_info=cluster_info
    )
    sampler_input = remote_dist_dataset.get_node_ids()
    rank_expected_sampler_input = expected_sampler_input[client_global_rank]
    for i in range(cluster_info.compute_cluster_world_size):
        if i == client_global_rank:
            assert len(sampler_input) == len(rank_expected_sampler_input)
            for j, expected in enumerate(rank_expected_sampler_input):
                assert_tensor_equality(sampler_input[j], expected)
            logger.info(
                f"{client_global_rank} / {cluster_info.compute_cluster_world_size} Sampler input verified"
            )
        torch.distributed.barrier()

    torch.distributed.barrier()
    logger.info(
        f"{client_global_rank} / {cluster_info.compute_cluster_world_size} Shutting down client"
    )
    shutdown_client()


def _client_process(
    client_rank: int,
    cluster_info: GraphStoreInfo,
    expected_sampler_input: dict[int, list[torch.Tensor]],
) -> None:
    logger.info(
        f"Initializing client node {client_rank} / {cluster_info.num_compute_nodes}. OS rank: {os.environ['RANK']}, OS world size: {os.environ['WORLD_SIZE']}, local client rank: {client_rank}"
    )

    mp_context = torch.multiprocessing.get_context("spawn")
    client_processes = []
    for i in range(cluster_info.num_processes_per_compute):
        client_process = mp_context.Process(
            target=_run_client_process,
            args=[
                i,  # client_rank
                cluster_info,  # cluster_info
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
    )


def _get_expected_sampler_input(
    num_nodes: int, cluster_info: GraphStoreInfo
) -> dict[int, list[torch.Tensor]]:
    """Get the expected sampler input for each compute rank.

    We generate the expected sampler input for each compute rank by sharding the nodes across the compute ranks.
    We then append the generated nodes to the expected sampler input for each compute rank.
    Example for num_nodes = 16, num_processes_per_compute = 2, num_compute_nodes = 2, num_storage_nodes = 2:
    {
    0: # compute rank 0
    [
        [0, 1], # From storage rank 0
        [8, 9] # From storage rank 1
    ]
    1: # compute rank 1
    [
        [2, 3], # From storage rank 0
        [10, 11] # From storage rank 1
    ],
    2: # compute rank 2
    [
        [4, 5], # From storage rank 0
        [12, 13] # From storage rank 1
    ],
    3: # compute rank 3
    [
        [6, 7], # From storage rank 0
        [14, 15] # From storage rank 1
    ]
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
        for compute_rank in range(cluster_info.compute_cluster_world_size):
            generated_nodes = shard_nodes_by_process(
                server_nodes, compute_rank, cluster_info.compute_cluster_world_size
            )
            expected_sampler_input[compute_rank].append(generated_nodes)
    return expected_sampler_input


class TestUtils(unittest.TestCase):
    def test_graph_store_locally(self):
        # Simulating two server machine, two compute machines.
        # Each machine has one process.
        cora_supervised_info = get_mocked_dataset_artifact_metadata()[
            CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO.name
        ]
        task_config_uri = cora_supervised_info.frozen_gbml_config_uri
        cluster_info = GraphStoreInfo(
            num_storage_nodes=2,
            num_compute_nodes=2,
            num_processes_per_compute=2,
            cluster_master_ip="localhost",
            storage_cluster_master_ip="localhost",
            compute_cluster_master_ip="localhost",
            cluster_master_port=get_free_port(),
            storage_cluster_master_port=get_free_port(),
            compute_cluster_master_port=get_free_port(),
        )

        num_cora_nodes = 2708
        expected_sampler_input = _get_expected_sampler_input(
            num_cora_nodes, cluster_info
        )
        master_port = get_free_port()
        ctx = mp.get_context("spawn")
        client_processes: list = []
        for i in range(cluster_info.num_compute_nodes):
            with mock.patch.dict(
                os.environ,
                {
                    "MASTER_ADDR": "localhost",
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
                    "MASTER_ADDR": "localhost",
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
