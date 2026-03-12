import torch


def get_available_device(local_process_rank: int) -> torch.device:
    r"""Returns the available device for the current process.

    Args:
        local_process_rank (int): The local rank of the current process within a node.
    Returns:
        torch.device: The device to use.
    """
    device = torch.device(
        "cpu"
        if not torch.cuda.is_available()
        # If the number of processes are larger than the available GPU,
        # we assign each process to one GPU in a round robin manner.
        else f"cuda:{local_process_rank % torch.cuda.device_count()}"
    )
    return device


def get_device_from_process_group() -> torch.device:
    """
    Returns the device for the current process group.
    Args:
        None
    Raises:
        ValueError: If the distributed environment is not initialized.
    Returns:
        torch.device: The device to use.
    """
    if not torch.distributed.is_initialized():
        raise ValueError(
            "Distributed environment must be initialized to get device from process group"
        )
    return torch.device("cuda" if torch.distributed.get_backend() == "nccl" else "cpu")
