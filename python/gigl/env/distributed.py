"""Information about distributed environments."""

from dataclasses import dataclass


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
