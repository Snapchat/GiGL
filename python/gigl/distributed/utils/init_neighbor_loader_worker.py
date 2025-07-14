import time
from functools import lru_cache
from typing import Optional

import graphlearn_torch as glt
import psutil
import torch

from gigl.common.logger import Logger

logger = Logger()

_is_cpu_env_initialized = False


def get_process_group_name(process_rank: int) -> str:
    """
    Returns the name of the process group for the given process rank.
    Args:
        process_rank (int): The rank of the process.
    Returns:
        str: The name of the process group.
    """
    return f"distributed-process-{process_rank}"


# torch.set_num_interop_threads() can only be called once, otherwise we see:
# RuntimeError: Error: cannot set number of interop threads after parallel work has started or set_num_interop_threads called
# Since we don't need to re-setup the identical worker pools, etc, we can just "cache" this call.
# That way the "side-effects" of the call are only executed once.
@lru_cache(maxsize=1)
def init_neighbor_loader_worker(
    local_process_rank: int,
    local_process_world_size: int,
    machine_rank: int,
    machine_world_size: int,
    device: torch.device,
    should_use_cpu_workers: bool = False,
    num_cpu_threads: Optional[int] = None,
    process_start_gap_seconds: float = 60.0,
) -> None:
    """
    Sets up processes and torch device for initializing the GLT DistNeighborLoader, setting up RPC and worker groups to minimize
    the memory overhead and CPU contention. Returns the torch device which current worker is assigned to.
    Args:
        local_process_rank (int): Process number on the current machine
        local_process_world_size (int): Total number of processes on the current machine
        machine_rank (int): Rank of current machine
        machine_world_size (int): Total number of machines
        device (torch.device): The device where you want to load the data onto - i.e. where is your model?
        should_use_cpu_workers (bool): Whether we should do CPU training or inference.
        num_cpu_threads (Optional[int]): Number of cpu threads PyTorch should use for CPU training or inference.
            Must be set if should_use_cpu_workers is True.
        process_start_gap_seconds (float): Delay between each process for initializing neighbor loader. At large scales, it is recommended to set
            this value to be between 60 and 120 seconds -- otherwise multiple processes may attempt to initialize dataloaders at overlapping timesß,
            which can cause CPU memory OOM.
    """

    global _is_cpu_env_initialized

    # When initiating data loader(s), there will be a spike of memory usage lasting for ~30s.
    # The current hypothesis is making connections across machines require a lot of memory.
    # If we start all data loaders in all processes simultaneously, the spike of memory
    # usage will add up and cause CPU memory OOM. Hence, we initiate the data loaders group by group
    # to smooth the memory usage. The definition of group is discussed below.
    logger.info(
        f"---Machine {machine_rank} local process number {local_process_rank} preparing to sleep for {process_start_gap_seconds * local_process_rank} seconds"
    )
    time.sleep(process_start_gap_seconds * local_process_rank)
    logger.info(
        f"---Machine {machine_rank} local process number {local_process_rank} started"
    )
    if not should_use_cpu_workers:
        assert (
            torch.cuda.device_count() > 0
        ), f"Must have at least 1 GPU available for GPU Training or inference, got {torch.cuda.device_count()}"

    if should_use_cpu_workers:
        assert (
            num_cpu_threads is not None
        ), "Must provide number of cpu threads when using cpu workers"
        # Assign processes to disjoint physical cores. Since training or inference is computation
        # bound instead of I/O bound, logical core segmentation is not enough, as two
        # hyperthreads on the same physical core could still compete for resources.

        # Compute the range of physical cores the process should run on.
        total_physical_cores = psutil.cpu_count(logical=False)
        if total_physical_cores is None:
            raise ValueError("Was unable to determine the number of physical cores")
        physical_cores_per_process = total_physical_cores // local_process_world_size
        start_physical_core = local_process_rank * physical_cores_per_process
        end_physical_core = (
            total_physical_cores
            if local_process_rank == local_process_world_size - 1
            else start_physical_core + physical_cores_per_process
        )

        # Essentially we could only specify the logical cores the process should run
        # on, so we have to map physical cores to logical cores. For GCP machines,
        # logical cores are assigned to physical cores in a round robin manner, i.e.,
        # if there are 4 physical cores, logical cores 0, 1, 2, 3, will be assigned
        # to physical cores 0, 1, 2, 3. Logical core 4 will be assigned to physical
        # core 0, logical core 5 will be assigned to physical core 1, etc. However,
        # this mapping does not always hold. Some VM assigns logical cores 0 and 1 to
        # physical core 0, and assigns logical cores 2, 3 to physical core 1. We could
        # to check it by running `lscpu -p` command in the terminal.
        first_logical_core_range = list(range(start_physical_core, end_physical_core))
        second_logical_core_range = list(
            range(
                start_physical_core + total_physical_cores,
                end_physical_core + total_physical_cores,
            )
        )
        logical_cores = first_logical_core_range + second_logical_core_range

        if not _is_cpu_env_initialized:
            torch.set_num_interop_threads(num_cpu_threads)
            torch.set_num_threads(num_cpu_threads)

            # Set the logical cpu cores the current process shoud run on. Note
            # that the sampling process spawned by the process will inherit
            # this setting, meaning that sampling process will run on the same group
            # of logical cores. However, the sampling process is network bound so
            # it may not heavily compete resouce with model training or inference.
            p = psutil.Process()
            p.cpu_affinity(logical_cores)
            _is_cpu_env_initialized = True
        else:
            logger.info("Logical CPU cores has already been set for current process.")

    else:
        # Setting the default CUDA device for the current process to be the
        # device. Without it, there will be a process created on cuda:0 device, and
        # another process created on the device. Consequently, there will be
        # more processes running on cuda:0 than other cuda devices. The processes on
        # cuda:0 will compete for memory and could cause CUDA OOM.
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        logger.info(
            f"Machine {machine_rank} local rank {local_process_rank} uses device {torch.cuda.current_device()} by default"
        )

    # Only initialize the worker group if it is not already set up
    if glt.distributed.get_context() is None:
        # Group of workers. Each process is a worker. Each
        # worker will initiate one model and at least one data loader. Each data loader
        # will spawn several sampling processes (a.k.a. sampling workers).
        # Instead of combining all workers into one group, we define N groups where
        # N is the number of processes on each machine. Specifically, we have
        # Group 0: (Machine 0, process 0), (Machine 1, process 0),..., (Machine M, process 0)
        # Group 1: (Machine 0, process 1), (Machine 1, process 1),..., (Machine M, process 1)
        # ...
        # Group N-1: (Machine 0, process N-1), (Machine 1, process N-1),..., (Machine M, process N-1)
        # We do this as we want to start different groups in different times to smooth
        # the spike of memory usage as mentioned above.
        group_name = get_process_group_name(local_process_rank)
        logger.info(
            f"Init worker group with: world_size={machine_world_size}, rank={machine_rank}, group_name={group_name}, "
        )
        glt.distributed.init_worker_group(
            world_size=machine_world_size,
            rank=machine_rank,
            group_name=group_name,
        )
