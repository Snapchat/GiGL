import os
import time
from typing import Optional

import graphlearn_torch as glt
import torch

from gigl.common.logger import Logger
from gigl.distributed.utils.networking import get_free_ports_from_master_node
from gigl.env.distributed import GraphStoreInfo
#from gigl.distributed.dist_client import init_client, shutdown_client
from graphlearn_torch.distributed.dist_client import init_client, shutdown_client

logger = Logger()


def init_compute_process(
    local_rank: int,
    cluster_info: GraphStoreInfo,
    compute_world_backend: Optional[str] = None,
) -> None:
    """
    Initializes distributed setup for a compute node in a Graph Store cluster.

    Should be called *once* per compute process (e.g. one per process per compute node, once per cluster_info.compute_cluster_world_size)

    Args:
        local_rank (int): The local (process) rank on the compute node.
        cluster_info (GraphStoreInfo): The cluster information.
        compute_world_backend (Optional[str]): The backend for the compute Torch Distributed process group.

    Raises:
        ValueError: If the process group is already initialized.
    """
    if torch.distributed.is_initialized():
        raise ValueError(
            "Process group already initialized! When using the Graph Store, you should not call `torch.distributed.init_process_group` directly."
        )
    compute_cluster_rank = (
        cluster_info.compute_node_rank * cluster_info.num_processes_per_compute
        + local_rank
    )
    cluster_master_ip = cluster_info.storage_cluster_master_ip
    world_size = (
        cluster_info.compute_cluster_world_size + cluster_info.num_storage_nodes
    )
    # init_method = f"tcp://{cluster_master_ip}:{cluster_info.rpc_wait_port}"
    # timeout = timedelta(minutes=15)
    # logger.info(f"compute cluster rank {compute_cluster_rank} / {cluster_info.compute_cluster_world_size} will wait for server to be ready. PG rank: {compute_cluster_rank} / {world_size} and init method {init_method} and timeout {timeout}")
    # start_time = time.time()
    # torch.distributed.init_process_group(
    #     backend="gloo",
    #     world_size=world_size,
    #     rank=compute_cluster_rank,
    #     init_method=init_method,
    #     timeout=timeout,
    # )
    # end_time = time.time()
    # logger.info(f"compute cluster rank {compute_cluster_rank} / {cluster_info.compute_cluster_world_size} waited for {end_time - start_time} seconds to be ready")
    # torch.distributed.destroy_process_group()
    #time.sleep(90)  # Wait for the server to be ready
    logger.info(
        f"Initializing RPC client for compute node {compute_cluster_rank} / {cluster_info.compute_cluster_world_size} on {cluster_master_ip}:{cluster_info.rpc_master_port}."
        f" OS rank: {os.environ['RANK']}, local compute rank: {local_rank}"
        f" num_servers: {cluster_info.num_storage_nodes}, num_clients: {cluster_info.compute_cluster_world_size}"
    )
    init_client(
        num_servers=cluster_info.num_storage_nodes,
        num_clients=cluster_info.compute_cluster_world_size,
        client_rank=compute_cluster_rank,
        master_addr=cluster_master_ip,
        master_port=cluster_info.rpc_master_port,
        client_group_name="gigl_client_rpc",
    )

    logger.info(
        f"Initializing compute process group {compute_cluster_rank} / {cluster_info.compute_cluster_world_size}. on {cluster_info.compute_cluster_master_ip}:{cluster_info.compute_cluster_master_port} with backend {compute_world_backend}."
        f" OS rank: {os.environ['RANK']}, local client rank: {local_rank}"
    )
    torch.distributed.init_process_group(
        backend=compute_world_backend,
        world_size=cluster_info.compute_cluster_world_size,
        rank=compute_cluster_rank,
        init_method=f"tcp://{cluster_info.compute_cluster_master_ip}:{cluster_info.compute_cluster_master_port}",
    )
    logger.info(
        f"Free ports from master node: {get_free_ports_from_master_node(num_ports=3)}"
    )


def shutdown_compute_proccess() -> None:
    """
    Shuts down the distributed setup for a compute node in a Graph Store cluster.

    Should be called *once* per compute process (e.g. one per process per compute node, once per cluster_info.compute_cluster_world_size)

    Args:
        None
    """
    shutdown_client()
    torch.distributed.destroy_process_group()
