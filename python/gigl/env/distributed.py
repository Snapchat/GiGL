"""Information about distributed environments."""

from dataclasses import dataclass
from typing import Final

COMPUTE_CLUSTER_LOCAL_WORLD_SIZE_ENV_KEY: Final[
    str
] = "COMPUTE_CLUSTER_LOCAL_WORLD_SIZE"


@dataclass(frozen=True)
class DistributedContext:
    """
    GiGL Distributed Context
    """

    # TODO (mkolodner-sc): Investigate adding local rank and local world size

    # Main Worker's IP Address for RPC communication
    main_worker_ip_address: str

    # Rank of machine
    global_rank: int

    # Total number of machines
    global_world_size: int


@dataclass(frozen=True)
class GraphStoreInfo:
    """Information about a graph store cluster."""

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

    # Number of processes per compute machine
    # See documentation on the VertexAiGraphStoreConfig message for more details.
    # https://snapchat.github.io/GiGL/docs/api/snapchat/research/gbml/gigl_resource_config_pb2/index.html#snapchat.research.gbml.gigl_resource_config_pb2.VertexAiGraphStoreConfig
    num_processes_per_compute: int

    @property
    def num_cluster_nodes(self) -> int:
        return self.num_storage_nodes + self.num_compute_nodes

    @property
    def compute_cluster_world_size(self) -> int:
        return self.num_compute_nodes * self.num_processes_per_compute
