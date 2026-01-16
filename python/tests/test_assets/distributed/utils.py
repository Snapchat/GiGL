from collections.abc import Mapping
from typing import Callable, Optional, overload

import torch

from gigl.distributed.utils import get_free_port
from gigl.src.common.types.graph_data import EdgeType, NodeType


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


class MockNodeAnchorLinkSplitter:
    """A mock splitter that returns hard-coded train/val/test splits for testing.

    This splitter is useful for unit tests where you want to control the exact
    split output without relying on the actual splitting logic.
    """

    def __init__(
        self,
        homogeneous_splits: Optional[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None,
        heterogeneous_splits: Optional[
            Mapping[NodeType, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = None,
        should_convert_labels_to_edges: bool = True,
    ):
        """Initialize the mock splitter with pre-defined splits.

        Args:
            homogeneous_splits: A tuple of (train, val, test) tensors for homogeneous graphs.
            heterogeneous_splits: A mapping from NodeType to (train, val, test) tensor tuples
                for heterogeneous graphs.
            should_convert_labels_to_edges: Whether labels should be converted to edges.
        """
        self._homogeneous_splits = homogeneous_splits
        self._heterogeneous_splits = heterogeneous_splits
        self._should_convert_labels_to_edges = should_convert_labels_to_edges

    @overload
    def __call__(
        self,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ...

    @overload
    def __call__(
        self,
        edge_index: Mapping[EdgeType, torch.Tensor],
    ) -> Mapping[NodeType, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        ...

    def __call__(
        self,
        edge_index: torch.Tensor | Mapping[EdgeType, torch.Tensor],
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | Mapping[NodeType, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ):
        """Return the pre-defined splits.

        Args:
            edge_index: The edge index (ignored, as splits are hard-coded).

        Returns:
            The pre-defined train/val/test splits.
        """
        if isinstance(edge_index, torch.Tensor):
            if self._homogeneous_splits is None:
                raise ValueError(
                    "MockNodeAnchorLinkSplitter was called with homogeneous edge_index "
                    "but no homogeneous_splits were provided."
                )
            return self._homogeneous_splits
        else:
            if self._heterogeneous_splits is None:
                raise ValueError(
                    "MockNodeAnchorLinkSplitter was called with heterogeneous edge_index "
                    "but no heterogeneous_splits were provided."
                )
            return self._heterogeneous_splits

    @property
    def should_convert_labels_to_edges(self) -> bool:
        return self._should_convert_labels_to_edges
