import os
from typing import Tuple

from gigl.common.logger import Logger
from gigl.distributed.dist_context import DistributedContext

logger = Logger()


def set_process_env_vars_for_torch_dist(
    process_number_on_current_machine: int,
    num_processes_on_current_machine: int,
    machine_context: DistributedContext,
    port: int = 29500,
) -> Tuple[int, int, int, int]:
    """
    This function sets the environment variables required for
    distributed training with PyTorch.  It assumes a multi-machine
    setup where each machine has a number of processes running.
    The number of machines and rendevous is determined by the
    `machine_context` provided.

    Args:
        process_number_on_current_machine (int): The process number on the current machine.
        num_processes_on_current_machine (int): The total number of processes on the current machine.
        machine_context (DistributedContext): The context containing information about the distributed setup.

    Returns:
        Tuple[int, int, int, int]: A tuple containing:
            - local_rank (int): The local rank of the process on the current machine.
            - rank (int): The global rank of the process across all machines.
            - local_world_size (int): The number of processes on the current machine.
            - world_size (int): The total number of processes across all machines.
    """
    # Set the environment variables for the current process
    # This is required for distributed training
    os.environ["LOCAL_RANK"] = str(process_number_on_current_machine)
    os.environ["RANK"] = str(
        machine_context.global_rank * num_processes_on_current_machine
        + process_number_on_current_machine
    )
    os.environ["WORLD_SIZE"] = str(
        num_processes_on_current_machine * machine_context.global_world_size
    )
    os.environ["LOCAL_WORLD_SIZE"] = str(num_processes_on_current_machine)
    os.environ["MASTER_ADDR"] = machine_context.main_worker_ip_address
    os.environ["MASTER_PORT"] = str(port)
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    world_size = int(os.environ["WORLD_SIZE"])
    return local_rank, rank, local_world_size, world_size
