from typing import Optional

import torch


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
