from typing import Callable, Optional

import torch

from gigl.distributed.utils import get_free_port


def assert_tensor_equality(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    dim: Optional[int] = None,
) -> None:
    """
    Asserts that the two provided tensors are equal to each other
    Args:
        tensor_a (torch.Tensor): First tensor which equality is being checked for
        tensor b (torch.Tensor): Second tensor which equality is being checked for
        dim (int): The dimension we are sorting over. If this value is None, we assume that the tensors must be an exact match. For a
            2D tensor, passing in a value of 1 will mean that the column order does not matter.
    """

    assert (
        tensor_a.dim() == tensor_b.dim()
    ), f"Provided tensors have different dimension {tensor_a.dim()} and {tensor_b.dim()}"

    # Exact match
    if dim is None:
        torch.testing.assert_close(tensor_a, tensor_b)
    else:
        # Sort along the specified dimension if provided
        if dim < 0 or dim >= tensor_a.dim():
            raise ValueError(
                f"Invalid dimension for sorting: {dim} provided tensor of dimension {tensor_a.dim()}"
            )

        # Sort the tensors along the specified dimension
        sorted_a, _ = torch.sort(tensor_a, dim=dim)
        sorted_b, _ = torch.sort(tensor_b, dim=dim)

        # Compare the sorted tensors
        torch.testing.assert_close(sorted_a, sorted_b)


def get_process_group_init_method(
    host: str = "localhost", port_picker: Callable[[], int] = get_free_port
) -> str:
    """
    Returns the initialization method for the process group.
    This is should be used with torch.distributed.init_process_group
    Args:
        host (str): The host address for the process group.
        port_picker (Callable[[], int]): A callable that returns a free port number.
    Returns:
        str: The initialization method for the process group.
    """
    return f"tcp://{host}:{port_picker()}"


def create_test_process_group() -> None:
    """
    Creates a single node process group for testing.
    Uses the "gloo" backend.
    """
    torch.distributed.init_process_group(
        backend="gloo",
        rank=0,
        world_size=1,
        init_method=get_process_group_init_method(),
    )


def destroy_test_process_group() -> None:
    """
    Destroys the test process group if it exists.
    """
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
