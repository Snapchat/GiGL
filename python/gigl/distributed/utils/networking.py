import socket
from typing import Optional

import torch

from gigl.common.logger import Logger

logger = Logger()


def get_free_port() -> int:
    """
    Find a free port and return the socket (to keep open) and port number.
    Returns:
        int: A free port number on the current machine.
    """
    return get_free_ports(num_ports=1)[0]


def get_free_ports(num_ports: int) -> list[int]:
    """
    Get a list of free ports.
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
    num_ports=1, _global_rank_override: Optional[int] = None
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
        f"Rank {rank} is requesting {num_ports} free ports from rank 0 (master)"
    )
    ports: list[int]
    if rank == 0:
        ports = get_free_ports(num_ports)
        logger.info(f"Rank {rank} found free ports: {ports}")
    else:
        ports = [0] * num_ports

    # Wrap port in a tensor to communicate
    port_tensor = torch.tensor(
        ports, dtype=torch.int32, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    # Broadcast from master from rank 0 to all other ranks
    torch.distributed.broadcast(port_tensor, src=0)
    logger.info(f"Rank {rank} received ports: {port_tensor.tolist()}")
    return port_tensor.tolist()
