import os
from typing import Any, Callable, Optional, TypeVar

import graphlearn_torch as glt
import torch
from graphlearn_torch.distributed.dist_context import DistRole, _set_client_context
from graphlearn_torch.distributed.rpc import rpc_global_request_async

from gigl.common.logger import Logger
from gigl.distributed.graph_store.dist_server import _call_func_on_server
from gigl.env.distributed import GraphStoreInfo

logger = Logger()

R = TypeVar("R")


def init_compute_process(
    local_rank: int,
    cluster_info: GraphStoreInfo,
    compute_world_backend: Optional[str] = None,
    rpc_timeout: int = 300,
) -> None:
    """
    Initializes distributed setup for a compute node in a Graph Store cluster.

    Should be called *once* per compute process (e.g. one per process per compute node, once per cluster_info.compute_cluster_world_size)

    Args:
        local_rank (int): The local (process) rank on the compute node.
        cluster_info (GraphStoreInfo): The cluster information.
        compute_world_backend (Optional[str]): The backend for the compute Torch Distributed process group.
        rpc_timeout (int): The max timeout in seconds for remote RPC requests.
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
    logger.info(
        f"Initializing RPC client for compute node {compute_cluster_rank} / {cluster_info.compute_cluster_world_size} on {cluster_master_ip}:{cluster_info.rpc_master_port}."
        f" OS rank: {os.environ['RANK']}, local compute rank: {local_rank}"
        f" num_servers: {cluster_info.num_storage_nodes}, num_clients: {cluster_info.compute_cluster_world_size}"
    )
    # Initialize the GLT client before starting the Torch Distributed process group.
    # Otherwise, we saw intermittent hangs when initializing the client.
    _init_client_rpc(
        num_servers=cluster_info.num_storage_nodes,
        num_clients=cluster_info.compute_cluster_world_size,
        client_rank=compute_cluster_rank,
        master_addr=cluster_master_ip,
        master_port=cluster_info.rpc_master_port,
        client_group_name="gigl_client_rpc",
        timeout=rpc_timeout,
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


def shutdown_compute_proccess() -> None:
    """
    Shuts down the distributed setup for a compute node in a Graph Store cluster.

    Should be called *once* per compute process (e.g. one per process per compute node, once per cluster_info.compute_cluster_world_size)

    Args:
        None
    """
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


def _init_client_rpc(
    num_servers: int,
    num_clients: int,
    client_rank: int,
    master_addr: str,
    master_port: int,
    num_rpc_threads: int = 4,
    client_group_name: Optional[str] = None,
    is_dynamic: bool = False,
    timeout: int = 180,
):
    """Initialize the current process as a client and establish connections
    with all other servers and clients. Note that this method should be called
    only in the server-client distribution mode.

    Based on https://github.com/alibaba/graphlearn-for-pytorch/blob/main/graphlearn_torch/python/distributed/dist_client.py#L24
    with the modification being to add the timeout parameter.

    Args:
      num_servers (int): Number of processes participating in the server group.
      num_clients (int): Number of processes participating in the client group.
      client_rank (int): Rank of the current process withing the client group (it
        should be a number between 0 and ``num_clients``-1).
      master_addr (str): The master TCP address for RPC connection between all
        servers and clients, the value of this parameter should be same for all
        servers and clients.
        This should be the an address in the storage cluster.
      master_port (int): The master TCP port for RPC connection between all
        servers and clients, the value of this parameter should be same for all
        servers and clients.
      num_rpc_threads (int): The number of RPC worker threads used for the
        current client. (Default: ``4``).
      client_group_name (str): A unique name of the client group that current
        process belongs to. If set to ``None``, a default name will be used.
        (Default: ``None``).
      is_dynamic (bool): Whether the world size is dynamic. (Default: ``False``).
    """
    if client_group_name:
        client_group_name = client_group_name.replace("-", "_")
    _set_client_context(num_servers, num_clients, client_rank, client_group_name)
    # Note that a client RPC agent will never serve remote requests,
    # thus setiting the number of rpc threads to ``1`` is enough.
    glt.distributed.init_rpc(
        master_addr,
        master_port,
        num_rpc_threads=num_rpc_threads,
        is_dynamic=is_dynamic,
        rpc_timeout=timeout,
    )
