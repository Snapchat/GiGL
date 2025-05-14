import os
from dataclasses import dataclass, field

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

    # Total number of training or inference processes per machine. This defaults to the os environment var "LOCAL_WORLD_SIZE" if not explicitly provided.
    local_world_size: int = int(os.getenv("LOCAL_WORLD_SIZE", "1"))

    # Master port for partitioning
    master_partitioning_port: int = field(init=False)

    # Master port for training or inference
    master_worker_port: int = field(init=False)

    # Map of local process rank to the corresponding master port for sampling
    local_rank_to_master_sampling_port: dict[int, int] = field(init=False)

    def __post_init__(self):
        num_ports_required = self.local_world_size + 2
        free_ports: list[int] = []
        while len(free_ports) < num_ports_required:
            candidate_port = glt.utils.get_free_port(self.main_worker_ip_address)
            if candidate_port not in free_ports:
                free_ports.append(candidate_port)
        master_partitioning_port: int = free_ports[0]
        master_worker_port: int = free_ports[1]
        local_rank_to_master_sampling_port: dict[int, int] = {
            local_rank: free_ports[local_rank + 2]
            for local_rank in range(self.local_world_size)
        }
        object.__setattr__(self, "master_partitioning_port", master_partitioning_port)
        object.__setattr__(self, "master_worker_port", master_worker_port)
        object.__setattr__(
            self,
            "local_rank_to_master_sampling_port",
            local_rank_to_master_sampling_port,
        )
        logger.info(
            f"Initialized distributed context with main_worker_ip_address: {self.main_worker_ip_address}, global_rank: {self.global_rank}, global_world_size: {self.global_world_size}, local world size: {self.local_world_size},"
            f"master_partitioning_port: {master_partitioning_port}, master_worker_port: {master_worker_port}, and a mapping of local rank to master_sampling_port: {local_rank_to_master_sampling_port}"
        )
