"""Utility functions to be used by machines running on Vertex AI."""

import json
import logging
import os
import subprocess
import time
from dataclasses import asdict, dataclass

from graphlearn_torch.utils import get_free_port

from gigl.common import GcsUri
from gigl.common.logger import Logger
from gigl.common.services.vertex_ai import LEADER_WORKER_INTERNAL_IP_FILE_PATH_ENV_KEY
from gigl.common.utils.gcs import GcsUtils
from gigl.distributed import DistributedContext

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


@dataclass(frozen=True)
class _LeaderInfo:
    ip: str
    dataset_port: int


def connect_worker_pool() -> DistributedContext:
    """
    Used to connect the worker pool. This function should be called by all workers
    to get the leader worker's internal IP address and to ensure that the workers
    can all communicate with the leader worker.
    """

    global_rank = get_rank()
    global_world_size = get_world_size()

    is_leader_worker = global_rank == 0
    ip_file_uri = GcsUri(_get_leader_worker_internal_ip_file_path())
    gcs_utils = GcsUtils()
    host_ip: str
    if is_leader_worker:
        logger.info("Wait 180 seconds for the leader machine to settle down.")
        time.sleep(180)
        host_ip = subprocess.check_output(["hostname", "-i"]).decode().strip()
        dataset_port = get_free_port()
        logger.info(
            f"Writing host IP address ({host_ip}) and dataset port ({dataset_port}) to {ip_file_uri}"
        )
        leader_info = _LeaderInfo(ip=host_ip, dataset_port=dataset_port)
        gcs_utils.upload_from_string(
            gcs_path=ip_file_uri, content=json.dumps(asdict(leader_info))
        )
    else:
        max_retries = 60
        interval_s = 30
        for attempt_num in range(1, max_retries + 1):
            logger.info(
                f"Checking if {ip_file_uri} exists and reading HOST_IP (attempt {attempt_num})..."
            )
            try:
                leader_info_json = json.loads(gcs_utils.read_from_gcs(ip_file_uri))
                leader_info = _LeaderInfo(**leader_info_json)
                logging.info(f"Leader info: {leader_info}")
                host_ip = leader_info.ip
                logger.info(f"Pinging host ip ({host_ip}) ...")
                if _ping_host_ip(host_ip):
                    logger.info(f"Ping to host ip ({host_ip}) was successful.")
                    break
            except Exception as e:
                logger.info(e)
            logger.info(
                f"Retrieving host information and/or ping failed, retrying in {interval_s} seconds..."
            )
            time.sleep(interval_s)
        if attempt_num >= max_retries:
            logger.info(
                f"Failed to ping HOST_IP after {max_retries} attempts. Exiting."
            )
            raise Exception(f"Failed to ping HOST_IP after {max_retries} attempts.")

    return DistributedContext(
        main_worker_ip_address=host_ip,
        global_rank=global_rank,
        global_world_size=global_world_size,
        dataset_building_port=leader_info.dataset_port,
    )


def _get_leader_worker_internal_ip_file_path() -> str:
    """
    Get the file path to the leader worker's internal IP address.
    """
    assert is_currently_running_in_vertex_ai_job(), "Not running in Vertex AI job."
    internal_ip_file_path = os.getenv(LEADER_WORKER_INTERNAL_IP_FILE_PATH_ENV_KEY)
    assert internal_ip_file_path is not None, (
        f"Internal IP file path ({LEADER_WORKER_INTERNAL_IP_FILE_PATH_ENV_KEY}) "
        + f"not found in environment variables. {os.environ}"
    )

    return internal_ip_file_path


def _ping_host_ip(host_ip: str) -> bool:
    try:
        subprocess.check_output(["ping", "-c", "1", host_ip])
        return True
    except subprocess.CalledProcessError:
        return False
