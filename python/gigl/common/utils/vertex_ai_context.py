"""Utility functions to be used by machines running on Vertex AI."""

import os
import subprocess
from typing import List, Optional

import torch.distributed as dist

from gigl.common.logger import Logger
from gigl.distributed.dist_context import DistributedContext

logger = Logger()


_VAI_EXCEPTION = Exception("Not running in Vertex AI job.")


def is_currently_running_in_vertex_ai_job() -> bool:
    """
    Check if the code is running in a Vertex AI job.

    Returns:
        bool: True if running in a Vertex AI job, False otherwise.
    """
    return "CLOUD_ML_JOB_ID" in os.environ


def get_vertex_ai_job_id() -> str:
    """
    Get the Vertex AI job ID.
    Throws if not on Vertex AI.
    """
    if not is_currently_running_in_vertex_ai_job():
        raise _VAI_EXCEPTION
    return os.environ["CLOUD_ML_JOB_ID"]


def get_host_name() -> str:
    """
    Get the current machines hostname.
    Throws if not on Vertex AI.
    """
    if not is_currently_running_in_vertex_ai_job():
        raise _VAI_EXCEPTION
    return os.environ["HOSTNAME"]


def get_leader_hostname() -> str:
    """
    Hostname of the machine that will host the process with rank 0. It is used
    to synchronize the workers.

    VAI does not automatically set this for single-replica jobs, hence the
    default value of "localhost".
    Throws if not on Vertex AI.
    """
    if not is_currently_running_in_vertex_ai_job():
        raise _VAI_EXCEPTION
    return os.environ.get("MASTER_ADDR", "localhost")


def get_leader_port() -> int:
    """
    A free port on the machine that will host the process with rank 0.

    VAI does not automatically set this for single-replica jobs, hence the
    default value of 29500. This is a PyTorch convention:
    https://github.com/pytorch/pytorch/blob/main/torch/distributed/run.py#L585
    Throws if not on Vertex AI.
    """
    if not is_currently_running_in_vertex_ai_job():
        raise _VAI_EXCEPTION
    return int(os.environ.get("MASTER_PORT", 29500))


def get_local_world_size() -> int:
    """
    The total number of processes spun up on each VAI Machine. This is currently is manually set upon launching a VAI job manually for GLT processes.
    We should deprecate this in the future if we migrate VAI jobs to be spun up with torchrun instead.
    Throws if not on Vertex AI.
    """
    if not is_currently_running_in_vertex_ai_job():
        raise _VAI_EXCEPTION
    return int(os.environ.get("LOCAL_WORLD_SIZE", 1))


def get_world_size() -> int:
    """
    The total number of processes that VAI creates. Note that VAI only creates one process per machine.
    It is the user's responsibility to create multiple processes per machine.

    VAI does not automatically set this for single-replica jobs, hence the
    default value of 1.
    Throws if not on Vertex AI.
    """
    if not is_currently_running_in_vertex_ai_job():
        raise _VAI_EXCEPTION
    return int(os.environ.get("WORLD_SIZE", 1))


def get_rank() -> int:
    """
    Rank of the current VAI process, so they will know whether it is the master or a worker.
    Note: that VAI only creates one process per machine. It is the user's responsibility to
    create multiple processes per machine. Meaning, this function will only return one integer
    for the main process that VAI creates.

    VAI does not automatically set this for single-replica jobs, hence the
    default value of 0.
    Throws if not on Vertex AI.
    """
    if not is_currently_running_in_vertex_ai_job():
        raise _VAI_EXCEPTION
    return int(os.environ.get("RANK", 0))


def connect_worker_pool() -> DistributedContext:
    """
    Used to connect the worker pool. This function should be called by all workers
    to get the leader worker's internal IP address and to ensure that the workers
    can all communicate with the leader worker.
    """
    global_rank = get_rank()
    global_world_size = get_world_size()
    local_world_size = get_local_world_size()
    # Uses the VAI-set environment variables for `RANK`, `WORLD_SIZE`, `MASTER_IP_ADDRESS`, and `MASTER_PORT` for setting up the process group
    dist.init_process_group(backend="gloo")

    is_leader_worker = global_rank == 0
    broadcast_list: List[Optional[str]]
    if is_leader_worker:
        host_ip = subprocess.check_output(["hostname", "-i"]).decode().strip()
        broadcast_list = [host_ip]
    else:
        broadcast_list = [None]

    dist.broadcast_object_list(object_list=broadcast_list, src=0)
    main_worker_ip_address = broadcast_list[0]
    assert main_worker_ip_address is not None

    # Tears down the process group, since we no longer need it for establishing communication between machines
    dist.destroy_process_group()
    return DistributedContext(
        main_worker_ip_address=main_worker_ip_address,
        global_rank=global_rank,
        global_world_size=global_world_size,
        local_world_size=local_world_size,
    )
