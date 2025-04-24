import multiprocessing
import os


def get_available_cpus() -> int:
    """
    Get the number of available CPUs.

    Returns:
        int: The number of available CPUs.
    """
    return int(os.environ.get("K8_CPU_RESOURCE_REQUEST", multiprocessing.cpu_count()))
