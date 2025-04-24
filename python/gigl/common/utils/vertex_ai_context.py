"""Utility functions to be used by machines running on Vertex AI."""

import os
import subprocess
import time

from gigl.common import GcsUri
from gigl.common.logger import Logger
from gigl.common.services.vertex_ai import LEADER_WORKER_INTERNAL_IP_FILE_PATH_ENV_KEY
from gigl.common.utils.gcs import GcsUtils
from gigl.distributed import DistributedContext

logger = Logger()


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
    return os.environ["CLOUD_ML_JOB_ID"]


def get_host_name() -> str:
    """
    Get the current machines hostname.
    Throws if not on Vertex AI.
    """
    return os.environ["HOSTNAME"]


def get_leader_hostname() -> str:
    """
    Hostname of the machine that will host the process with rank 0. It is used
    to synchronize the workers.
    Throws if not on Vertex AI.
    """
    return os.environ["MASTER_ADDR"]


def get_leader_port() -> int:
    """
    A free port on the machine that will host the process with rank 0.
    Throws if not on Vertex AI.
    """
    return int(os.environ["MASTER_PORT"])


def get_world_size() -> int:
    """
    The total number of processes that VAI creates. Note that VAI only creates one process per machine.
    It is the user's responsibility to create multiple processes per machine.
    Throws if not on Vertex AI.
    """
    return int(os.environ["WORLD_SIZE"])


def get_rank() -> int:
    """
    Rank of the current VAI process, so they will know whether it is the master or a worker.
    Note: that VAI only creates one process per machine. It is the user's responsibility to
    create multiple processes per machine. Meaning, this function will only return one integer
    for the main process that VAI creates.
    Throws if not on Vertex AI.
    """
    return int(os.environ["RANK"])


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
        logger.info(f"Writing host IP address ({host_ip}) to {ip_file_uri}")
        gcs_utils.upload_from_string(gcs_path=ip_file_uri, content=host_ip)
    else:
        max_retries = 60
        interval_s = 30
        for attempt_num in range(1, max_retries + 1):
            logger.info(
                f"Checking if {ip_file_uri} exists and reading HOST_IP (attempt {attempt_num})..."
            )
            try:
                host_ip = gcs_utils.read_from_gcs(ip_file_uri)
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
