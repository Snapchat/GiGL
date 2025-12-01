import os
import unittest
from unittest import mock

import torch
import torch.multiprocessing as mp

from gigl.common import Uri
from gigl.common.logger import Logger
from gigl.distributed.graph_store.compute import (
    init_compute_proccess,
    shutdown_compute_proccess,
)
from gigl.distributed.graph_store.storage_main import storage_node_process
from gigl.distributed.utils import get_free_port
from gigl.env.distributed import (
    COMPUTE_CLUSTER_LOCAL_WORLD_SIZE_ENV_KEY,
    GraphStoreInfo,
)
from gigl.src.mocking.lib.versioning import get_mocked_dataset_artifact_metadata
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO,
)

logger = Logger()


def _run_client_process(
    client_rank: int,
    cluster_info: GraphStoreInfo,
) -> None:
    client_global_rank = (
        cluster_info.compute_node_rank * cluster_info.num_processes_per_compute
        + client_rank
    )
    init_compute_proccess(client_rank, cluster_info)
    torch.distributed.barrier()
    logger.info(
        f"{client_global_rank} / {cluster_info.compute_cluster_world_size} Shutting down client"
    )
    shutdown_compute_proccess()


def _client_process(
    client_rank: int,
    cluster_info: GraphStoreInfo,
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
