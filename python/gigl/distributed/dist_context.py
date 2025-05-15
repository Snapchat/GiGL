import os
from dataclasses import dataclass, field

import graphlearn_torch as glt

from gigl.common.logger import Logger

logger = Logger()


def _get_local_world_size_from_env() -> int:
    if "LOCAL_WORLD_SIZE" in os.environ:
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        logger.info(f"Got local world size {local_world_size}")
    else:
        raise ValueError(
            "No environment variable found for `LOCAL_WORLD_SIZE` and no local world size explicitly provided."
        )


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
    local_world_size: int = field(
        default_factory=lambda: _get_local_world_size_from_env
    )

    # Master port for partitioning
    master_partitioning_port: int = field(init=False)

    # Map of local process rank to the corresponding master port for training or inference
    local_rank_to_master_worker_port: dict[int, int] = field(init=False)

    # Map of local process rank to the corresponding master port for sampling
    local_rank_to_master_sampling_port: dict[int, int] = field(init=False)

    def __post_init__(self):
        num_ports_required = self.local_world_size * 2 + 1
        free_ports: list[int] = []
        while len(free_ports) < num_ports_required:
            candidate_port = glt.utils.get_free_port(self.main_worker_ip_address)
            if candidate_port not in free_ports:
                free_ports.append(candidate_port)
        master_partitioning_port: int = free_ports[0]
        local_rank_to_master_worker_port: dict[int, int] = {
            local_rank: free_ports[local_rank + 1]
            for local_rank in range(self.local_world_size)
        }
        local_rank_to_master_sampling_port: dict[int, int] = {
            local_rank: free_ports[self.local_world_size + local_rank + 1]
            for local_rank in range(self.local_world_size)
        }
        object.__setattr__(self, "master_partitioning_port", master_partitioning_port)
        object.__setattr__(
            self, "local_rank_to_master_worker_port", local_rank_to_master_worker_port
        )
        object.__setattr__(
            self,
            "local_rank_to_master_sampling_port",
            local_rank_to_master_sampling_port,
        )
        logger.info(
            f"Initialized distributed context with main_worker_ip_address: {self.main_worker_ip_address}, global_rank: {self.global_rank}, global_world_size: {self.global_world_size}, local world size: {self.local_world_size},"
            f"master_partitioning_port: {master_partitioning_port}, mapping of local rank to master_worker_port: {local_rank_to_master_worker_port}, and a mapping of local rank to master_sampling_port: {local_rank_to_master_sampling_port}"
        )
