import os

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # isort: skip
import collections
import unittest
from unittest import mock

import torch
import torch.multiprocessing as mp
from graphlearn_torch.distributed import init_client, shutdown_client

from gigl.common import Uri, UriFactory
from gigl.common.logger import Logger
from gigl.distributed.server_client.server_main import run_servers
from gigl.distributed.server_client.utils import get_sampler_input_for_inference
from gigl.distributed.utils import get_free_port
from gigl.distributed.utils.neighborloader import shard_nodes_by_process
from gigl.env.distributed import (
    GRAPH_STORE_PROCESSES_PER_COMPUTE_VAR_NAME,
    GRAPH_STORE_PROCESSES_PER_STORAGE_VAR_NAME,
    GraphStoreInfo,
)
from gigl.src.common.types.graph_data import NodeType
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.mocking.lib.versioning import get_mocked_dataset_artifact_metadata
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO,
)
from tests.test_assets.distributed.utils import assert_tensor_equality

logger = Logger()


def _run_client_process(
    client_rank: int,
    cluster_info: GraphStoreInfo,
    node_type: NodeType,
    expected_sampler_input: dict[int, list[torch.Tensor]],
) -> None:
    client_global_rank = (
        int(os.environ["RANK"]) * cluster_info.num_processes_per_compute + client_rank
    )
    logger.info(
        f"Initializing client {client_global_rank} / {cluster_info.compute_world_size}. on {cluster_info.cluster_master_ip}:{cluster_info.cluster_master_port}. OS rank: {os.environ['RANK']}, local client rank: {client_rank} on port: {cluster_info.cluster_master_port}"
    )
    torch.distributed.init_process_group(
        backend="gloo",
        world_size=cluster_info.compute_world_size,
        rank=client_global_rank,
        init_method=f"tcp://{cluster_info.compute_cluster_master_ip}:{cluster_info.compute_cluster_master_port}",
        group_name="gigl_client_comms",
    )
    init_client(
        num_servers=cluster_info.storage_world_size,
        num_clients=cluster_info.compute_world_size,
        client_rank=client_global_rank,
        master_addr=cluster_info.cluster_master_ip,
        master_port=cluster_info.cluster_master_port,
        client_group_name="gigl_client_rpc",
    )

    sampler_input = get_sampler_input_for_inference(
        client_global_rank,
        cluster_info,
        node_type,
    )

    rank_expected_sampler_input = expected_sampler_input[client_global_rank]
    for i in range(cluster_info.compute_world_size):
        if i == client_global_rank:
            assert len(sampler_input) == len(rank_expected_sampler_input)
            for j, expected in enumerate(rank_expected_sampler_input):
                assert_tensor_equality(sampler_input[j], expected)
            logger.info(
                f"{client_global_rank} / {cluster_info.compute_world_size} Sampler input verified"
            )
        torch.distributed.barrier()

    torch.distributed.barrier()
    logger.info(
        f"{client_global_rank} / {cluster_info.compute_world_size} Shutting down client"
    )
    shutdown_client()


def _client_process(
    client_rank: int,
    cluster_info: GraphStoreInfo,
    node_type: NodeType,
    expected_sampler_input: dict[int, list[torch.Tensor]],
) -> None:
    logger.info(
        f"Initializing client {client_rank} / {cluster_info.compute_world_size}. OS rank: {os.environ['RANK']}, OS world size: {os.environ['WORLD_SIZE']}, local client rank: {client_rank}"
    )
    # torch.distributed.init_process_group()
    logger.info(f"Client {client_rank} / {cluster_info.compute_world_size} initialized")

    # cluster_info = get_graph_store_info()
    mp_context = torch.multiprocessing.get_context("spawn")
    client_processes = []
    for i in range(cluster_info.num_processes_per_compute):
        client_process = mp_context.Process(
            target=_run_client_process,
            args=[
                i,  # client_rank
                cluster_info,  # cluster_info
                node_type,  # node_type
                expected_sampler_input,  # expected_sampler_input
            ],
        )
        client_processes.append(client_process)
    for client_process in client_processes:
        client_process.start()
    for client_process in client_processes:
        client_process.join()


def _run_server_processes(
    server_rank: int,
    cluster_info: GraphStoreInfo,
    task_config_uri: Uri,
    is_inference: bool,
) -> None:
    logger.info(
        f"Initializing server processes. OS rank: {os.environ['RANK']}, OS world size: {os.environ['WORLD_SIZE']}"
    )
    run_servers(
        server_rank=int(os.environ["RANK"]) - cluster_info.num_compute_nodes,
        cluster_info=cluster_info,
        task_config_uri=task_config_uri,
        is_inference=is_inference,
    )


class TestUtils(unittest.TestCase):
    def test_get_sampler_input_for_inference(self):
        # Simulating two server machine, two compute machines.
        # Each machine has one process.
        cora_supervised_info = get_mocked_dataset_artifact_metadata()[
            CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO.name
        ]
        task_config_uri = cora_supervised_info.frozen_gbml_config_uri
        task_config = GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
            gbml_config_uri=task_config_uri
        )
        cluster_info = GraphStoreInfo(
            num_cluster_nodes=4,
            num_storage_nodes=2,
            num_compute_nodes=2,
            num_processes_per_storage=1,
            num_processes_per_compute=2,
            cluster_master_ip="localhost",
            storage_cluster_master_ip="localhost",
            compute_cluster_master_ip="localhost",
            cluster_master_port=get_free_port(),
            storage_cluster_master_port=get_free_port(),
            compute_cluster_master_port=get_free_port(),
        )

        expected_sampler_input = collections.defaultdict(list)
        num_cora_nodes = 2708
        all_nodes = torch.arange(num_cora_nodes, dtype=torch.int64)
        all_nodes_generated_nodes = []
        for server_rank in range(cluster_info.storage_world_size):
            server_node_start = (
                server_rank * num_cora_nodes // cluster_info.storage_world_size
            )
            server_node_end = (
                (server_rank + 1) * num_cora_nodes // cluster_info.storage_world_size
            )
            server_nodes = all_nodes[server_node_start:server_node_end]
            logger.info(
                f"Server rank {server_rank} nodes: {server_node_start}-{server_node_end}"
            )
            for compute_rank in range(cluster_info.compute_world_size):
                generated_nodes = shard_nodes_by_process(
                    server_nodes, compute_rank, cluster_info.compute_world_size
                )
                all_nodes_generated_nodes.append(generated_nodes)
                expected_sampler_input[compute_rank].append(generated_nodes)

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
                    "WORLD_SIZE": str(cluster_info.cluster_world_size),
                    GRAPH_STORE_PROCESSES_PER_STORAGE_VAR_NAME: str(
                        cluster_info.num_processes_per_storage
                    ),
                    GRAPH_STORE_PROCESSES_PER_COMPUTE_VAR_NAME: str(
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
                        task_config.graph_metadata_pb_wrapper.homogeneous_node_type,  # node_type
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
                    "WORLD_SIZE": str(cluster_info.cluster_world_size),
                    GRAPH_STORE_PROCESSES_PER_STORAGE_VAR_NAME: str(
                        cluster_info.num_processes_per_storage
                    ),
                    GRAPH_STORE_PROCESSES_PER_COMPUTE_VAR_NAME: str(
                        cluster_info.num_processes_per_compute
                    ),
                },
                clear=False,
            ):
                server_process = ctx.Process(
                    target=_run_server_processes,
                    args=[
                        i,  # server_rank
                        cluster_info,  # cluster_info
                        UriFactory.create_uri(task_config_uri),  # task_config_uri
                        True,  # is_inference
                    ],
                )
                server_process.start()
                server_processes.append(server_process)

        for client_process in client_processes:
            client_process.join()
        for server_process in server_processes:
            server_process.join()
