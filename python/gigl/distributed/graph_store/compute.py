import os
from typing import Optional

import graphlearn_torch as glt
import torch

from gigl.common.logger import Logger
from gigl.env.distributed import GraphStoreInfo

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
        f"Initializing RPC client for compute node {compute_cluster_rank} / {cluster_info.compute_cluster_world_size} on {cluster_info.cluster_master_ip}:{cluster_info.cluster_master_port}."
        f" OS rank: {os.environ['RANK']}, local compute rank: {local_rank}"
        f" num_servers: {cluster_info.num_storage_nodes}, num_clients: {cluster_info.compute_cluster_world_size}"
    )
    glt.distributed.init_client(
        num_servers=cluster_info.num_storage_nodes,
        num_clients=cluster_info.compute_cluster_world_size,
        client_rank=compute_cluster_rank,
        master_addr=cluster_info.cluster_master_ip,
        master_port=cluster_info.cluster_master_port,
        client_group_name="gigl_client_rpc",
    )


def shutdown_compute_proccess() -> None:
    """
    Shuts down the distributed setup for a compute node in a Graph Store cluster.

    Should be called *once* per compute process (e.g. one per process per compute node, once per cluster_info.compute_cluster_world_size)

    Args:
        None
    """
    glt.distributed.shutdown_client()
    torch.distributed.destroy_process_group()
