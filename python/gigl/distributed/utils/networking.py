import socket
from typing import Optional

import torch

from gigl.common.logger import Logger

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
    num_ports=1, master_node_id: int = 0, _global_rank_override: Optional[int] = None
) -> list[int]:
    """
    Get free ports from master node, that can be used for communication between workers.
    Args:
        num_ports (int): Number of free ports to find.
        _global_rank_override (Optional[int]): Override for the global rank,
            useful for testing or if global rank is not accurately available.
    Returns:
        list[int]: A list of free port numbers on the master node.
    """
    # Ensure that the distributed environment is initialized
    assert (
        torch.distributed.is_initialized()
    ), "Distributed environment must be initialized to communicate free ports on master"
    assert num_ports >= 1, "num_ports must be >= 1"

    rank = (
        torch.distributed.get_rank()
        if _global_rank_override is None
        else _global_rank_override
    )
    logger.info(
        f"Rank {rank} is requesting {num_ports} free ports from rank {master_node_id} (master)"
    )
    ports: list[int]
    if rank == master_node_id:
        ports = get_free_ports(num_ports)
        logger.info(f"Rank {rank} found free ports: {ports}")
    else:
        ports = [0] * num_ports

    # Broadcast from master from rank 0 to all other ranks
    torch.distributed.broadcast_object_list(ports, src=master_node_id)
    logger.info(f"Rank {rank} received ports: {ports}")
    return ports


def get_internal_ip_from_master_node(
    _global_rank_override: Optional[int] = None,
) -> str:
    """
    Get the internal IP address of the master node in a distributed setup.
    This is useful for setting up RPC communication between workers where the default torch.distributed env:// setup is not enough.

    i.e. when using :py:obj:`gigl.distributed.dataset_factory`

    Returns:
        str: The internal IP address of the master node.
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
        f"Rank {rank} is requesting internal ip address of master node from rank 0 (master)"
    )

    master_ip_list: list[Optional[str]] = []
    if rank == 0:
        # Master node, return its own internal IP
        master_ip_list = [socket.gethostbyname(socket.gethostname())]
    else:
        # Other nodes will receive the master's IP via broadcast
        master_ip_list = [None]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.distributed.broadcast_object_list(master_ip_list, src=0, device=device)
    master_ip = master_ip_list[0]
    logger.info(f"Rank {rank} received master internal IP: {master_ip}")
    assert master_ip is not None, "Could not retrieve master node's internal IP"
    return master_ip


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


def get_ports_for_server_client_clusters(
    num_servers: int, num_clients: int
) -> tuple[int, int]:
    server_port = get_free_ports_from_master_node(1)[0]
    client_port = get_free_ports_from_master_node(1, master_node_id=num_servers)[0]
    return server_port, client_port
