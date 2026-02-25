import os
from typing import Any, Callable, Optional, TypeVar

import graphlearn_torch as glt
import torch
from graphlearn_torch.distributed.dist_context import DistRole
from graphlearn_torch.distributed.rpc import rpc_global_request_async

from gigl.common.logger import Logger
from gigl.distributed.graph_store.dist_server import _call_func_on_server
from gigl.distributed.utils.timing import TimingStats
from gigl.env.distributed import GraphStoreInfo
from datetime import timedelta

logger = Logger()

R = TypeVar("R")


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
    _ts = TimingStats.get_instance()
    with _ts.track("init_compute_process"):
        if torch.distributed.is_initialized():
            raise ValueError(
                "Process group already initialized! When using the Graph Store, you should not call `torch.distributed.init_process_group` directly."
            )
        compute_cluster_rank = (
            cluster_info.compute_node_rank * cluster_info.num_processes_per_compute
            + local_rank
        )
        cluster_master_ip = cluster_info.storage_cluster_master_ip
        logger.info(
            f"Initializing RPC client for compute node {compute_cluster_rank} / {cluster_info.compute_cluster_world_size} on {cluster_master_ip}:{cluster_info.rpc_master_port}."
            f" OS rank: {os.environ['RANK']}, local compute rank: {local_rank}"
            f" num_servers: {cluster_info.num_storage_nodes}, num_clients: {cluster_info.compute_cluster_world_size}"
        )
        # Initialize the GLT client before starting the Torch Distributed process group.
        # Otherwise, we saw intermittent hangs when initializing the client.
        with _ts.track("init_compute_process.init_client_rpc"):
            glt.distributed.init_client(
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
        with _ts.track("init_compute_process.init_process_group"):
            torch.distributed.init_process_group(
                backend=compute_world_backend,
                world_size=cluster_info.compute_cluster_world_size,
                rank=compute_cluster_rank,
                init_method=f"tcp://{cluster_info.compute_cluster_master_ip}:{cluster_info.compute_cluster_master_port}",
                timeout=timedelta(minutes=180)
            )


def shutdown_compute_proccess() -> None:
    """
    Shuts down the distributed setup for a compute node in a Graph Store cluster.

    Should be called *once* per compute process (e.g. one per process per compute node, once per cluster_info.compute_cluster_world_size)

    Args:
        None
    """
    with TimingStats.get_instance().track("shutdown_compute_proccess"):
        glt.distributed.shutdown_client()
        torch.distributed.destroy_process_group()


def async_request_server(
    server_rank: int,
    func: Callable[..., R],
    *args: Any,
    **kwargs: Any,
) -> torch.futures.Future[R]:
    """Perform an asynchronous request on a remote server, calling on the client side."""
    call_args = [func] + list(args)
    return rpc_global_request_async(
        target_role=DistRole.SERVER,
        role_rank=server_rank,
        func=_call_func_on_server,
        args=call_args,
        kwargs=kwargs,
    )


def request_server(
    server_rank: int,
    func: Callable[..., R],
    *args: Any,
    **kwargs: Any,
) -> R:
    """Perform a synchronous request on a remote server, calling on the client side."""
    future = async_request_server(server_rank, func, *args, **kwargs)
    return future.wait()
