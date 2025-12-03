import os
import socket
from typing import Optional

import torch

from gigl.common.logger import Logger
from gigl.common.utils.vertex_ai_context import (
    ClusterSpec,
    get_cluster_spec,
    is_currently_running_in_vertex_ai_job,
)
from gigl.env.distributed import (
    COMPUTE_CLUSTER_LOCAL_WORLD_SIZE_ENV_KEY,
    GraphStoreInfo,
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
        port = s.getsockname()[1]
        ports.append(port)
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
    logger.info(f"Rank {rank} received master node's internal IP: {node_ip}")
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


def get_graph_store_info() -> GraphStoreInfo:
    """
    Get the information about the graph store cluster.

    MUST be called with a torch.distributed process group initialized, for the *entire* training cluster.
    E.g. the process group *must* include both the compute and storage nodes.

    This function should only be called on clusters that are setup by GiGL.
    E.g. when GiGLResourceConfig.trainer_resource_config.vertex_ai_graph_store_trainer_config is set.

    Returns:
        GraphStoreInfo: The information about the graph store cluster.

    Raises:
        ValueError: If a torch distributed environment is not initialized.
        ValueError: If not running running in a supported environment.
    """
    # If we want to ever support other (non-VAI) environments,
    # we must switch here depending on the environment.
    if not is_currently_running_in_vertex_ai_job():
        raise ValueError("get_graph_store_info must be called in a Vertex AI job.")
    cluster_spec = get_cluster_spec()
    _validate_cluster_spec(cluster_spec)

    if "workerpool1" in cluster_spec.cluster:
        num_compute_nodes = len(cluster_spec.cluster["workerpool0"]) + len(
            cluster_spec.cluster["workerpool1"]
        )
    else:
        num_compute_nodes = len(cluster_spec.cluster["workerpool0"])
    num_storage_nodes = len(cluster_spec.cluster["workerpool2"])

    cluster_master_ip = get_internal_ip_from_master_node()
    # We assume that the compute cluster nodes come first, followed by the storage nodes.
    # Since the compute cluster nodes are the first in the cluster spec, we can use the cluster master IP for the compute cluster master IP.
    compute_cluster_master_ip = cluster_master_ip
    storage_cluster_master_ip = get_internal_ip_from_node(node_rank=num_compute_nodes)

    cluster_master_port, compute_cluster_master_port = get_free_ports_from_node(
        num_ports=2, node_rank=0
    )
    storage_cluster_master_port = get_free_ports_from_node(
        num_ports=1, node_rank=num_compute_nodes
    )[0]

    num_processes_per_compute = int(
        os.environ.get(COMPUTE_CLUSTER_LOCAL_WORLD_SIZE_ENV_KEY, "1")
    )

    return GraphStoreInfo(
        num_storage_nodes=num_storage_nodes,
        num_compute_nodes=num_compute_nodes,
        num_processes_per_compute=num_processes_per_compute,
        cluster_master_ip=cluster_master_ip,
        storage_cluster_master_ip=storage_cluster_master_ip,
        compute_cluster_master_ip=compute_cluster_master_ip,
        cluster_master_port=cluster_master_port,
        storage_cluster_master_port=storage_cluster_master_port,
        compute_cluster_master_port=compute_cluster_master_port,
    )


def _validate_cluster_spec(cluster_spec: ClusterSpec) -> None:
    """Validate the cluster spec is setup as we'd expect."""

    if len(cluster_spec.cluster["workerpool0"]) != 1:
        raise ValueError(
            f"Expected exactly one machine in workerpool0, but got {len(cluster_spec.cluster['workerpool0'])}"
        )

    # We want to ensure that the cluster is setup as we'd expect.
    # e.g. `[[compute0], [compute1, ..., computeN], [storage0, ..., storageN]]`
    # So we do this by checking that the task index matches up with the rank.
    env_rank = int(os.environ["RANK"])
    if cluster_spec.task.type == "workerpool0":
        offset = 0
    elif cluster_spec.task.type == "workerpool1":
        offset = len(cluster_spec.cluster["workerpool0"])
    elif cluster_spec.task.type == "workerpool2":
        if "workerpool1" in cluster_spec.cluster:
            offset = len(cluster_spec.cluster["workerpool0"]) + len(
                cluster_spec.cluster["workerpool1"]
            )
        else:
            offset = len(cluster_spec.cluster["workerpool0"])
    else:
        raise ValueError(
            f"Expected task type to be workerpool0, workerpool1, or workerpool2, but got {cluster_spec.task.type}"
        )

    if cluster_spec.task.index + offset != env_rank:
        raise ValueError(
            f"Expected task index to be {env_rank}, but got {cluster_spec.task.index + offset}"
        )
