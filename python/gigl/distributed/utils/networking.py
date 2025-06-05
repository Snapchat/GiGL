import socket
from typing import Optional

import torch


def get_free_port() -> tuple[socket.socket, int]:
    """
    Find a free port and return the socket (to keep open) and port number.
    Returns:
        tuple[socket.socket, int]: A tuple containing the socket and the free port number.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))  # OS assigns a free port
    return s, s.getsockname()[1]


def get_free_master_ports(
    num_ports=1, _global_rank_override: Optional[int] = None
) -> list[int]:
    """
    Get a free port from master node, that can be used for communication between workers.
    """
    assert (
        torch.distributed.is_initialized()
    ), "Distributed environment must be initialized to communicate free ports on master"
    assert num_ports >= 1, "num_ports must be >= 1"

    rank = (
        torch.distributed.get_rank()
        if _global_rank_override is None
        else _global_rank_override
    )
    ports: list[int] = []
    if rank == 0:
        # Find a free port on the master node
        s, port = get_free_port()
        ports.append(port)
    else:
        ports.append(0)  # dummy for non rank 0 processes

    # Wrap port in a tensor to communicate
    port_tensor = torch.tensor(
        ports, dtype=torch.int32, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    # Broadcast from master from rank 0 to all other ranks
    torch.distributed.broadcast(port_tensor, src=0)
    return port_tensor.tolist()
