from typing import Callable, Optional

import torch

from gigl.distributed.graph_store.remote_dist_dataset import RemoteDistDataset
from gigl.distributed.utils import get_free_port, get_free_ports
from gigl.env.distributed import GraphStoreInfo
from gigl.src.common.types.graph_data import EdgeType


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


class MockGraphStoreInfo(GraphStoreInfo):
    """
    A mock wrapper around GraphStoreInfo that allows overriding the compute_node_rank property.

    The real GraphStoreInfo.compute_node_rank reads from environment variables (RANK),
    which makes it difficult to test. This mock allows setting the compute_node_rank
    directly during initialization.

    Args:
        real_info: The real GraphStoreInfo instance to delegate to for most properties.
        compute_node_rank: The compute node rank value to return instead of reading from env.

    Example:
        >>> real_info = GraphStoreInfo(num_storage_nodes=2, ...)
        >>> mock_info = MockGraphStoreInfo(real_info, compute_node_rank=0)
        >>> mock_info.compute_node_rank  # Returns 0 instead of reading from env
    """

    def __init__(self, real_info: GraphStoreInfo, compute_node_rank: int):
        self._real_info = real_info
        self._compute_node_rank = compute_node_rank

    @property
    def num_storage_nodes(self) -> int:
        return self._real_info.num_storage_nodes

    @property
    def num_compute_nodes(self) -> int:
        return self._real_info.num_compute_nodes

    @property
    def storage_cluster_master_ip(self) -> str:
        return self._real_info.storage_cluster_master_ip

    @property
    def num_processes_per_compute(self) -> int:
        return self._real_info.num_processes_per_compute

    @property
    def compute_node_rank(self) -> int:
        return self._compute_node_rank


class MockRemoteDistDataset(RemoteDistDataset):
    """
    A mock RemoteDistDataset for testing that doesn't make remote RPC calls.

    The real RemoteDistDataset makes remote calls to storage nodes via graphlearn_torch's
    request_server mechanism. This mock class overrides all remote-calling methods to
    return configurable mock values, enabling unit testing of code that depends on
    RemoteDistDataset without needing a real distributed cluster.

    Args:
        num_storage_nodes: Number of storage nodes in the mock cluster. Defaults to 2.
        num_compute_nodes: Number of compute nodes in the mock cluster. Defaults to 1.
        num_processes_per_compute: Number of processes per compute node. Defaults to 1.
        compute_node_rank: The rank of the compute node. Defaults to 0.
        edge_types: Optional list of edge types for heterogeneous graphs. Defaults to None.
        edge_dir: Edge direction, either "in" or "out". Defaults to "out".

    Example:
        >>> mock_dataset = MockRemoteDistDataset(
        ...     num_storage_nodes=2,
        ...     edge_types=[EdgeType("user", "knows", "user")],
        ... )
        >>> mock_dataset.get_edge_types()  # Returns the configured edge_types
        >>> mock_dataset.cluster_info.num_storage_nodes  # Returns 2
    """

    def __init__(
        self,
        num_storage_nodes: int = 2,
        num_compute_nodes: int = 1,
        num_processes_per_compute: int = 1,
        compute_node_rank: int = 0,
        edge_types: Optional[list[EdgeType]] = None,
        edge_dir: str = "out",
    ):
        # Create a mock GraphStoreInfo with placeholder values
        self._mock_cluster_info = GraphStoreInfo(
            num_storage_nodes=num_storage_nodes,
            num_compute_nodes=num_compute_nodes,
            cluster_master_ip="127.0.0.1",
            storage_cluster_master_ip="127.0.0.1",
            compute_cluster_master_ip="127.0.0.1",
            cluster_master_port=12345,
            storage_cluster_master_port=12346,
            compute_cluster_master_port=12347,
            num_processes_per_compute=num_processes_per_compute,
            rpc_master_port=12348,
            rpc_wait_port=12349,
        )
        self._mock_compute_node_rank = compute_node_rank
        self._mock_edge_types = edge_types
        self._mock_edge_dir = edge_dir
        # Don't call super().__init__() to avoid needing a real cluster connection

    @property
    def cluster_info(self) -> GraphStoreInfo:
        """Returns a MockGraphStoreInfo with the configured compute_node_rank."""
        return MockGraphStoreInfo(self._mock_cluster_info, self._mock_compute_node_rank)

    def get_node_feature_info(self):
        """Returns None (no node features configured)."""
        return None

    def get_edge_feature_info(self):
        """Returns None (no edge features configured)."""
        return None

    def get_edge_dir(self) -> str:
        """Returns the configured edge direction."""
        return self._mock_edge_dir

    def get_edge_types(self) -> Optional[list[EdgeType]]:
        """Returns the configured edge types."""
        return self._mock_edge_types

    def get_free_ports_on_storage_cluster(self, num_ports: int) -> list[int]:
        """Returns a list of mock port numbers starting at 20000."""
        return get_free_ports(num_ports=num_ports)
