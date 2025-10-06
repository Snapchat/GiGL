import os
import socket
from dataclasses import dataclass
from typing import Optional

import torch

from gigl.common.logger import Logger
from gigl.src.common.constants.distributed import (
    COMPUTE_CLUSTER_MASTER_KEY,
    COMPUTE_CLUSTER_NUM_NODES_KEY,
    STORAGE_CLUSTER_MASTER_KEY,
    STORAGE_CLUSTER_NUM_NODES_KEY,
)

logger = Logger()


def get_free_port() -> int:
    """
    Get a free port number.
    Note: If you call `get_free_port` multiple times, it can return the same port number if the port is still free.
    If you want multiple free ports before you init/use them, leverage `get_free_ports` instead.
    Returns:
        int: A free port number on the current machine.
    """
    return get_free_ports(num_ports=1)[0]


def get_free_ports(num_ports: int) -> list[int]:
    """
    Get a list of free port numbers.
    Note: If you call `get_free_ports` multiple times, it can return the same port number if the port is still free.
    Args:
        num_ports (int): Number of free ports to find.
    Returns:
        list[int]: A list of free port numbers on the current machine.
    """
    assert num_ports >= 1, "num_ports must be >= 1"
    ports: list[int] = []
    open_sockets: list[socket.socket] = []
    for _ in range(num_ports):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # OS assigns a free port; we want to keep it open until we have all ports so we only return unique ports
        s.bind(("", 0))
        open_sockets.append(s)
        ports.append(s.getsockname()[1])
    # Free up ports by closing the sockets
    for s in open_sockets:
        s.close()
    return ports


def get_free_ports_from_master_node(
    num_ports: int, _global_rank_override: Optional[int] = None
) -> list[int]:
    """
    Get free ports from master node, that can be used for communication between workers.
    Args:
        num_ports (int): Number of free ports to find.
        _global_rank_override (Optional[int]): Override for the global rank,
        useful for testing or if global rank is not accurately available.
    """
    return get_free_ports_from_node(
        num_ports, node_rank=0, _global_rank_override=_global_rank_override
    )


def get_free_ports_from_node(
    num_ports: int,
    node_rank: int,
    _global_rank_override: Optional[int] = None,
) -> list[int]:
    """
    Get free ports from a node, that can be used for communication between workers.
    Args:
        num_ports (int): Number of free ports to find.
        node_rank (int): Rank of the node, default is 0.
        _global_rank_override (Optional[int]): Override for the global rank,
            useful for testing or if global rank is not accurately available.
    Returns:
        list[int]: A list of free port numbers on the node.
    """
    # Ensure that the distributed environment is initialized
    assert (
        torch.distributed.is_initialized()
    ), "Distributed environment must be initialized to communicate free ports on a node"
    assert num_ports >= 1, "num_ports must be >= 1"

    rank = (
        torch.distributed.get_rank()
        if _global_rank_override is None
        else _global_rank_override
    )
    logger.info(
        f"Rank {rank} is requesting {num_ports} free ports from rank {node_rank} (node)"
    )
    if rank == node_rank:
        ports = get_free_ports(num_ports)
        logger.info(f"Rank {rank} found free ports: {ports}")
    else:
        ports = [0] * num_ports

    # Broadcast from master from rank 0 to all other ranks
    torch.distributed.broadcast_object_list(ports, src=node_rank)
    logger.info(f"Rank {rank} received ports: {ports}")
    return ports


def get_internal_ip_from_master_node(
    _global_rank_override: Optional[int] = None,
) -> str:
    """
    Get the internal IP address of the master node in a distributed setup.
    """
    return get_internal_ip_from_node(
        node_rank=0, _global_rank_override=_global_rank_override
    )


def get_internal_ip_from_node(
    node_rank: int,
    _global_rank_override: Optional[int] = None,
) -> str:
    """
    Get the internal IP address of the node in a distributed setup.
    This is useful for setting up RPC communication between workers where the default torch.distributed env:// setup is not enough.

    i.e. when using :py:obj:`gigl.distributed.dataset_factory`

    Returns:
        str: The internal IP address of the node.
    """
    assert (
        torch.distributed.is_initialized()
    ), "Distributed environment must be initialized"

    rank = (
        torch.distributed.get_rank()
        if _global_rank_override is None
        else _global_rank_override
    )
    logger.info(
        f"Rank {rank} is requesting internal ip address of node from rank {node_rank}"
    )

    ip_list: list[Optional[str]] = []
    if rank == node_rank:
        # Master node, return its own internal IP
        ip_list = [socket.gethostbyname(socket.gethostname())]
    else:
        # Other nodes will receive the master's IP via broadcast
        ip_list = [None]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.distributed.broadcast_object_list(ip_list, src=node_rank, device=device)
    node_ip = ip_list[0]
    logger.info(f"Rank {rank} received master internal IP: {node_ip}")
    assert node_ip is not None, "Could not retrieve master node's internal IP"
    return node_ip


def get_internal_ip_from_all_ranks() -> list[str]:
    """
    Get the internal IP addresses of all ranks in a distributed setup. Internal IPs are usually not accessible
    from the web. i.e. the machines will have to be on the same network or VPN to get the right address so each
    rank can communicate with each other.
    This is useful for setting up RPC communication between ranks where the default torch.distributed env:// setup is not enough.
    Or, if you are trying to run validation checks, get local world size for a specific node, etc.

    Returns:
        list[str]: A list of internal IP addresses of all ranks.
    """
    assert (
        torch.distributed.is_initialized()
    ), "Distributed environment must be initialized"

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    logger.info(f"Rank {rank} is requesting internal IP addresses from all ranks")

    ip_list: list[str] = [""] * world_size
    curr_rank_ip = socket.gethostbyname(socket.gethostname())
    torch.distributed.all_gather_object(ip_list, curr_rank_ip)

    logger.info(f"Rank {rank} received internal IPs: {ip_list}")
    assert all(ip for ip in ip_list), "Could not retrieve all ranks' internal IPs"

    return ip_list


@dataclass(frozen=True)
class GraphStoreInfo:
    """Information about a graph store cluster."""

    # Number of nodes in the whole cluster
    num_cluster_nodes: int
    # Number of nodes in the storage cluster
    num_storage_nodes: int
    # Number of nodes in the compute cluster
    num_compute_nodes: int

    # IP address of the master node for the whole cluster
    cluster_master_ip: str
    # IP address of the master node for the storage cluster
    storage_cluster_master_ip: str
    # IP address of the master node for the compute cluster
    compute_cluster_master_ip: str

    # Port of the master node for the whole cluster
    cluster_master_port: int
    # Port of the master node for the storage cluster
    storage_cluster_master_port: int
    # Port of the master node for the compute cluster
    compute_cluster_master_port: int


def get_graph_store_info() -> GraphStoreInfo:
    """
    Get the information about the graph store cluster.

    Returns:
        GraphStoreInfo: The information about the graph store cluster.

    Raises:
        ValueError: If a torch distributed environment is not initialized.
        ValueError: If the storage cluster master key or the compute cluster master key is not set as an environment variable.
    """
    if not torch.distributed.is_initialized():
        raise ValueError("Distributed environment must be initialized")

    if not STORAGE_CLUSTER_MASTER_KEY in os.environ:
        raise ValueError(
            f"{STORAGE_CLUSTER_MASTER_KEY} must be set as an environment variable"
        )
    if not COMPUTE_CLUSTER_MASTER_KEY in os.environ:
        raise ValueError(
            f"{COMPUTE_CLUSTER_MASTER_KEY} must be set as an environment variable"
        )
    if not STORAGE_CLUSTER_NUM_NODES_KEY in os.environ:
        raise ValueError(
            f"{STORAGE_CLUSTER_NUM_NODES_KEY} must be set as an environment variable"
        )
    if not COMPUTE_CLUSTER_NUM_NODES_KEY in os.environ:
        raise ValueError(
            f"{COMPUTE_CLUSTER_NUM_NODES_KEY} must be set as an environment variable"
        )

    storage_cluster_master_rank = int(os.environ[STORAGE_CLUSTER_MASTER_KEY])
    compute_cluster_master_rank = int(os.environ[COMPUTE_CLUSTER_MASTER_KEY])

    cluster_master_ip = get_internal_ip_from_master_node()
    # We assume that the storage cluster nodes come first.
    storage_cluster_master_ip = get_internal_ip_from_node(
        node_rank=storage_cluster_master_rank
    )
    compute_cluster_master_ip = get_internal_ip_from_node(
        node_rank=compute_cluster_master_rank
    )

    cluster_master_port = get_free_ports_from_node(num_ports=1, node_rank=0)[0]
    storage_cluster_master_port = get_free_ports_from_node(
        num_ports=1, node_rank=storage_cluster_master_rank
    )[0]
    compute_cluster_master_port = get_free_ports_from_node(
        num_ports=1, node_rank=compute_cluster_master_rank
    )[0]

    num_storage_nodes = int(os.environ[STORAGE_CLUSTER_NUM_NODES_KEY])
    num_compute_nodes = int(os.environ[COMPUTE_CLUSTER_NUM_NODES_KEY])

    return GraphStoreInfo(
        num_cluster_nodes=num_storage_nodes + num_compute_nodes,
        num_storage_nodes=num_storage_nodes,
        num_compute_nodes=num_compute_nodes,
        cluster_master_ip=cluster_master_ip,
        storage_cluster_master_ip=storage_cluster_master_ip,
        compute_cluster_master_ip=compute_cluster_master_ip,
        cluster_master_port=cluster_master_port,
        storage_cluster_master_port=storage_cluster_master_port,
        compute_cluster_master_port=compute_cluster_master_port,
    )
