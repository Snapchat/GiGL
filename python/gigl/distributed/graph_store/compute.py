import os

import torch
from graphlearn_torch.distributed import init_client, shutdown_client

from gigl.common.logger import Logger
from gigl.env.distributed import GraphStoreInfo

logger = Logger()


def init_compute_proccess(node_local_rank: int, cluster_info: GraphStoreInfo) -> None:
    """
    Initializes distributed setup for a compute node in a Graph Store cluster.

    Should be called *once* per compute process (e.g. one per process per compute node, once per cluster_info.compute_cluster_world_size)

    Args:
        node_local_rank (int): The local (process) rank on the compute node.
        cluster_info (GraphStoreInfo): The cluster information.
    """
    compute_cluster_rank = (
        cluster_info.compute_node_rank * cluster_info.num_processes_per_compute
        + node_local_rank
    )
    logger.info(
        f"Initializing compute process group {compute_cluster_rank} / {cluster_info.compute_cluster_world_size}. on {cluster_info.compute_cluster_master_ip}:{cluster_info.compute_cluster_master_port}."
        f" OS rank: {os.environ['RANK']}, local client rank: {node_local_rank}"
    )
    torch.distributed.init_process_group(
        backend="gloo",
        world_size=cluster_info.compute_cluster_world_size,
        rank=compute_cluster_rank,
        init_method=f"tcp://{cluster_info.compute_cluster_master_ip}:{cluster_info.compute_cluster_master_port}",
        group_name="gigl_client_comms",
    )
    logger.info(
        f"Initializing RPC client for compute node {compute_cluster_rank} / {cluster_info.compute_cluster_world_size} on {cluster_info.master_ip}:{cluster_info.master_port}."
        f" OS rank: {os.environ['RANK']}, local compute rank: {node_local_rank}"
        f" num_servers: {cluster_info.num_storage_nodes}, num_clients: {cluster_info.compute_cluster_world_size}"
    )
    init_client(
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
    shutdown_client()
