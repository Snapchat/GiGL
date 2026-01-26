import os
from typing import Optional

import torch.distributed

from gigl.common.logger import Logger

logger = Logger()


def get_world_size() -> int:
    """
    This is automatically set by Kubeflow PyTorchJob launcher
    Returns:
        int: Total number of processes involved in distributed training
    """
    return int(os.environ.get("WORLD_SIZE", 1))


def get_rank() -> int:
    """
    This is automatically set by Kubeflow PyTorchJob launcher
    Returns:
        int: The index of the process involved in distributed training
    """
    return int(os.environ.get("RANK", 0))


def is_distributed_local_debug() -> bool:
    """
    For local debugging purpose only
    This sets necessary environment variables for distributed training at local machine
    Returns:
        bool: If True, then should_distribute early exit and enables distributed training
    """
    if not int(os.environ.get("DISTRIBUTED_LOCAL_DEBUG", 0)):
        return False
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29501")
    logger.info(
        f'Overriding local environment variables for debugging WORLD_SIZE={os.environ["WORLD_SIZE"]}, RANK={os.environ["RANK"]}, MASTER_ADDR={os.environ["MASTER_ADDR"]}, MASTER_PORT={os.environ["MASTER_PORT"]}'
    )
    return True


def should_distribute() -> bool:
    """
    Determines whether the process should be configured for distributed training.
    Returns:
        bool: True if the process is configured for distributed training
    """
    if is_distributed_local_debug():
        logger.info(f"Distributed training enabled for local debugging")
        return True
    should_distribute = torch.distributed.is_available() and get_world_size() > 1
    logger.info(f"Should we distribute training? {should_distribute}")
    return should_distribute


def get_distributed_backend(use_cuda: bool) -> Optional[str]:
    """
    Returns the distributed backend based on whether distributed training is enabled and whether CUDA is used.
    Args:
        use_cuda (bool): Whether CUDA is used for training
    Returns:
        Optional[str]: The distributed backend (NCCL or GLOO) if distributed training is enabled, None otherwise
    """
    if not should_distribute():
        return None
    return (
        torch.distributed.Backend.NCCL if use_cuda else torch.distributed.Backend.GLOO
    )


def is_distributed_available_and_initialized() -> bool:
    """
    Returns:
        bool: True if distributed training is available and initialized, False otherwise
    """
    return torch.distributed.is_available() and torch.distributed.is_initialized()
