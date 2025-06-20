import socket
from typing import List, Optional

import torch

from gigl.common.logger import Logger

logger = Logger()


def is_port_free(port: int, hostname: str = "localhost") -> bool:
    """
    Check if a port is free on for the provided host..
    Args:
        port (int): Port number to check.
        hostname (str): Hostname to check the port on. Defaults to 'localhost'.
    Returns:
        bool: If the port is free.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((hostname, port))
            return True
        except OSError:
            # If the port is already in use, an OSError will be raised
            return False


def get_free_port() -> int:
    """
    Get a free port number.
    Note: If you call `get_free_port` multiple times, it can return the same port number if the port is still free.
    If you want multiple free ports before you init/use them, leverage `get_free_ports` instead.
    Returns:
        int: A free port number on the current machine.
    """
    return get_free_ports(num_ports=1)[0]


def get_free_ports(num_ports: int) -> List[int]:
    """
    Get a list of free port numbers.
    Note: If you call `get_free_ports` multiple times, it can return the same port number if the port is still free.
    Args:
        num_ports (int): Number of free ports to find.
    Returns:
        List[int]: A list of free port numbers on the current machine.
    """
    assert num_ports >= 1, "num_ports must be >= 1"
    ports: List[int] = []
    open_sockets: List[socket.socket] = []
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
    num_ports=1, _global_rank_override: Optional[int] = None
) -> List[int]:
    """
    Get free ports from master node, that can be used for communication between workers.
    Args:
        num_ports (int): Number of free ports to find.
        _global_rank_override (Optional[int]): Override for the global rank,
            useful for testing or if global rank is not accurately available.
    Returns:
        List[int]: A list of free port numbers on the master node.
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
        f"Rank {rank} is requesting {num_ports} free ports from rank 0 (master)"
    )
    ports: List[int]
    if rank == 0:
        ports = get_free_ports(num_ports)
        logger.info(f"Rank {rank} found free ports: {ports}")
    else:
        ports = [0] * num_ports

    # Broadcast from master from rank 0 to all other ranks
    torch.distributed.broadcast_object_list(ports, src=0)
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

    master_ip_list: List[Optional[str]] = []
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
