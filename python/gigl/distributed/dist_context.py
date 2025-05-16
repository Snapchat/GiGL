from dataclasses import dataclass
from typing import List

import graphlearn_torch as glt

from gigl.common.logger import Logger

logger = Logger()


@dataclass(frozen=True)
class DistributedContext:
    """
    GiGL Distributed Context
    """

    # Main Worker's IP Address for RPC communication
    main_worker_ip_address: str

    # Rank of machine
    global_rank: int

    # Total number of machines
    global_world_size: int

    # Total number of training or inference processes per machine
    local_world_size: int

    # Master port for partitioning
    master_partitioning_port: int

    # Master ports for training or inference, where master_worker_ports[i] indicates the master worker port for the ith local process rank
    master_worker_ports: List[int]

    # Master ports for sampling, where master_sampling_ports[i] indicates the master sampling port for the ith local process rank
    master_sampling_ports: List[int]


def get_free_ports(main_worker_ip_address: str, local_world_size: int):
    num_ports_required = local_world_size * 2 + 1
    free_ports: list[int] = []
    while len(free_ports) < num_ports_required:
        candidate_port = glt.utils.get_free_port(main_worker_ip_address)
        if candidate_port not in free_ports:
            free_ports.append(candidate_port)
    master_partitioning_port: int = free_ports[0]
    master_worker_ports: list[int] = free_ports[1 : local_world_size + 1]
    master_sampling_ports: list[int] = free_ports[local_world_size + 1 :]
    return master_partitioning_port, master_worker_ports, master_sampling_ports
