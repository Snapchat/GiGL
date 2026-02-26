"""
Utility functions for computing node degrees in distributed graph settings.
"""

from collections import Counter
from typing import TYPE_CHECKING, Callable, Optional, Union

import torch
from graphlearn_torch.data import Graph
from torch_geometric.typing import EdgeType

import gigl.distributed.utils
from gigl.common.logger import Logger

if TYPE_CHECKING:
    from gigl.distributed.dist_dataset import DistDataset
    from gigl.distributed.graph_store.remote_dist_dataset import RemoteDistDataset

logger = Logger()


def compute_and_broadcast_degree_tensors(
    dataset: Union["DistDataset", "RemoteDistDataset"],
) -> Union[torch.Tensor, dict[EdgeType, torch.Tensor]]:
    """
    Compute node degrees from the graph partition and aggregate across all machines.

    For DistDataset: extracts topology locally and uses all-reduce.
    For RemoteDistDataset: fetches degrees from storage nodes and aggregates.

    Args:
        dataset: Either a DistDataset or RemoteDistDataset containing the graph data.

    Returns:
        Union[torch.Tensor, dict[EdgeType, torch.Tensor]]: The aggregated degree tensors.
            - For homogeneous graphs: A tensor of shape [num_nodes].
            - For heterogeneous graphs: A dict mapping EdgeType to degree tensors.

    Raises:
        ValueError: If the dataset graph is None or topology is unavailable.
        TypeError: If the dataset type is not supported.
    """
    dataset_class_name = type(dataset).__name__

    if dataset_class_name == "RemoteDistDataset":
        return _compute_degrees_from_remote_dataset(dataset)
    elif dataset_class_name == "DistDataset":
        return _compute_degrees_from_local_dataset(dataset)
    else:
        raise TypeError(
            f"Unsupported dataset type: {type(dataset)}. "
            f"Expected DistDataset or RemoteDistDataset."
        )


# =============================================================================
# Core Helper Functions
# =============================================================================


def _pad_to_size(tensor: torch.Tensor, target_size: int) -> torch.Tensor:
    """Pad tensor with zeros to reach target_size."""
    if tensor.size(0) >= target_size:
        return tensor
    padding = torch.zeros(
        target_size - tensor.size(0),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, padding])


def _clamp_to_int32(tensor: torch.Tensor) -> torch.Tensor:
    """Clamp tensor values to int32 max and convert dtype."""
    max_int32 = torch.iinfo(torch.int32).max
    return tensor.clamp(max=max_int32).to(torch.int32)


def _sum_tensors_with_padding(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Sum multiple tensors, padding shorter ones to match the longest."""
    if not tensors:
        return torch.tensor([], dtype=torch.int32)

    max_size = max(t.size(0) for t in tensors)
    result = torch.zeros(max_size, dtype=torch.int64)

    for tensor in tensors:
        padded = _pad_to_size(tensor, max_size)
        result += padded.to(torch.int64)

    return _clamp_to_int32(result)


def _compute_degrees_from_indptr(indptr: torch.Tensor) -> torch.Tensor:
    """Compute degrees from CSR row pointers: degree[i] = indptr[i+1] - indptr[i]."""
    return (indptr[1:] - indptr[:-1]).contiguous().to(torch.int32)


def _log_degree_stats(
    degrees: torch.Tensor,
    label: str,
    include_sample: bool = False,
) -> None:
    """Log statistics about computed degrees."""
    logger.info(
        f"Computed degrees for {label}: "
        f"{degrees.size(0)} nodes, max={degrees.max().item()}, min={degrees.min().item()}"
    )
    if include_sample:
        logger.info("Sample of degree tensor: %s", degrees[:100])


# =============================================================================
# Local Dataset (DistDataset) Processing
# =============================================================================


def _compute_degrees_from_local_dataset(
    dataset: "DistDataset",
) -> Union[torch.Tensor, dict[EdgeType, torch.Tensor]]:
    """Compute degrees from a local DistDataset with all-reduce across machines."""
    graph = dataset.graph
    if graph is None:
        raise ValueError("Dataset graph is None. Cannot compute degrees.")

    is_distributed = torch.distributed.is_initialized()
    local_world_size = _get_local_world_size() if is_distributed else 1
    gloo_group = _get_or_create_gloo_group() if is_distributed else None

    def reduce_fn(degrees: torch.Tensor) -> torch.Tensor:
        if not is_distributed:
            return degrees
        return _all_reduce_with_size_sync(degrees, local_world_size, gloo_group)

    if isinstance(graph, dict):
        return _process_graph_dict(graph, reduce_fn, "local")
    else:
        return _process_single_graph(graph, reduce_fn, "homogeneous graph")


def _get_local_world_size() -> int:
    """Get the number of processes per machine (for over-counting correction)."""
    rank = torch.distributed.get_rank()
    rank_ips = gigl.distributed.utils.get_internal_ip_from_all_ranks()
    local_world_size = Counter(rank_ips)[rank_ips[0]]
    logger.info(f"Degree computation: rank={rank}, local_world_size={local_world_size}")
    return local_world_size


def _get_or_create_gloo_group() -> Optional[torch.distributed.ProcessGroup]:
    """Create gloo process group for CPU all-reduce if needed (NCCL doesn't support CPU)."""
    backend = torch.distributed.get_backend()
    if backend != "gloo":
        logger.info(f"Creating gloo subgroup for CPU all-reduce (current: {backend})")
        return torch.distributed.new_group(backend="gloo")
    return None


def _all_reduce_with_size_sync(
    local_degrees: torch.Tensor,
    local_world_size: int,
    gloo_group: Optional[torch.distributed.ProcessGroup],
) -> torch.Tensor:
    """All-reduce degrees across ranks, handling size mismatches and over-counting."""
    # Synchronize max size across ranks
    local_size = torch.tensor([local_degrees.size(0)], dtype=torch.long)
    torch.distributed.all_reduce(local_size, op=torch.distributed.ReduceOp.MAX, group=gloo_group)
    max_size = int(local_size.item())

    # Pad and reduce
    local_degrees = _pad_to_size(local_degrees, max_size)
    torch.distributed.all_reduce(local_degrees, op=torch.distributed.ReduceOp.SUM, group=gloo_group)

    # Correct for over-counting (multiple local processes share same partition)
    return local_degrees // local_world_size


# =============================================================================
# Remote Dataset (RemoteDistDataset) Processing
# =============================================================================


def _compute_degrees_from_remote_dataset(
    dataset: "RemoteDistDataset",
) -> Union[torch.Tensor, dict[EdgeType, torch.Tensor]]:
    """Compute degrees by fetching from remote storage nodes and aggregating."""
    all_local_degrees = dataset.fetch_all_local_degrees()

    if not all_local_degrees:
        raise ValueError("No degree tensors returned from storage nodes.")

    first_result = next(iter(all_local_degrees.values()))

    if isinstance(first_result, dict):
        return _aggregate_remote_heterogeneous(all_local_degrees)
    else:
        return _aggregate_remote_homogeneous(all_local_degrees)


def _aggregate_remote_homogeneous(
    all_local_degrees: dict[int, torch.Tensor],
) -> torch.Tensor:
    """Aggregate homogeneous degrees from multiple storage nodes."""
    tensors = list(all_local_degrees.values())
    result = _sum_tensors_with_padding(tensors)
    _log_degree_stats(result, "remote homogeneous graph", include_sample=True)
    return result


def _aggregate_remote_heterogeneous(
    all_local_degrees: dict[int, dict[EdgeType, torch.Tensor]],
) -> dict[EdgeType, torch.Tensor]:
    """Aggregate heterogeneous degrees from multiple storage nodes."""
    # Collect all edge types
    all_edge_types: set[EdgeType] = set()
    for degrees_by_type in all_local_degrees.values():
        all_edge_types.update(degrees_by_type.keys())

    result: dict[EdgeType, torch.Tensor] = {}
    for edge_type in all_edge_types:
        tensors = [
            degrees_by_type[edge_type]
            for degrees_by_type in all_local_degrees.values()
            if edge_type in degrees_by_type
        ]
        if tensors:
            result[edge_type] = _sum_tensors_with_padding(tensors)
            _log_degree_stats(result[edge_type], f"edge type {edge_type}")
        else:
            logger.warning(f"No degree tensors found for edge type {edge_type}")

    return result


# =============================================================================
# Graph Processing (Shared Logic)
# =============================================================================


def _process_single_graph(
    graph: Graph,
    reduce_fn: Callable[[torch.Tensor], torch.Tensor],
    label: str,
) -> torch.Tensor:
    """Process a single graph to compute degrees."""
    topo = graph.topo
    if topo is None or topo.indptr is None:
        raise ValueError(f"Topology/indptr not available for {label}.")

    degrees = _compute_degrees_from_indptr(topo.indptr)
    degrees = _clamp_to_int32(reduce_fn(degrees))
    _log_degree_stats(degrees, label, include_sample=True)
    return degrees


def _process_graph_dict(
    graph: dict[EdgeType, Graph],
    reduce_fn: Callable[[torch.Tensor], torch.Tensor],
    source: str,
) -> dict[EdgeType, torch.Tensor]:
    """Process a heterogeneous graph dict to compute degrees per edge type."""
    result: dict[EdgeType, torch.Tensor] = {}

    for edge_type, edge_graph in graph.items():
        topo = edge_graph.topo
        if topo is None or topo.indptr is None:
            logger.warning(f"Topology/indptr not available for edge type {edge_type}, skipping.")
            continue

        degrees = _compute_degrees_from_indptr(topo.indptr)
        degrees = _clamp_to_int32(reduce_fn(degrees))
        result[edge_type] = degrees
        _log_degree_stats(degrees, f"edge type {edge_type} ({source})")

    return result


# =============================================================================
# Test Suite
# =============================================================================


if __name__ == "__main__":
    import argparse
    from unittest.mock import MagicMock

    # Mock classes with proper __name__ for type dispatch
    class MockDistDataset:
        def __init__(self):
            self.graph = None

    MockDistDataset.__name__ = "DistDataset"

    class MockRemoteDistDataset:
        def __init__(self):
            self._fetch_result = None

        def fetch_all_local_degrees(self):
            return self._fetch_result

    MockRemoteDistDataset.__name__ = "RemoteDistDataset"

    def create_mock_graph(num_nodes: int, avg_degree: int = 5):
        """Create mock Graph with CSR indptr."""
        mock_graph = MagicMock()
        mock_topo = MagicMock()
        degrees = torch.randint(1, avg_degree * 2, (num_nodes,), dtype=torch.int64)
        indptr = torch.zeros(num_nodes + 1, dtype=torch.int64)
        indptr[1:] = torch.cumsum(degrees, dim=0)
        mock_topo.indptr = indptr
        mock_graph.topo = mock_topo
        return mock_graph, degrees

    def run_test(name: str, test_fn):
        """Run a test with formatted output."""
        print(f"\n{'=' * 60}\nTEST: {name}\n{'=' * 60}")
        test_fn()
        print("PASSED!")

    def test_local_homogeneous():
        num_nodes = 100
        mock_graph, expected = create_mock_graph(num_nodes)
        dataset = MockDistDataset()
        dataset.graph = mock_graph
        result = compute_and_broadcast_degree_tensors(dataset)
        assert result.shape[0] == num_nodes
        assert torch.allclose(result.to(torch.int64), expected)

    def test_local_heterogeneous():
        edge_types = [("user", "follows", "user"), ("user", "likes", "item")]
        expected_by_type = {}
        graphs = {}
        for i, et in enumerate(edge_types):
            graphs[et], expected_by_type[et] = create_mock_graph(50 + i * 50)
        dataset = MockDistDataset()
        dataset.graph = graphs
        result = compute_and_broadcast_degree_tensors(dataset)
        assert isinstance(result, dict) and len(result) == len(edge_types)
        for et in edge_types:
            assert torch.allclose(result[et].to(torch.int64), expected_by_type[et])

    def test_remote_homogeneous():
        num_nodes, num_servers = 100, 3
        partition_degrees = [
            torch.randint(0, 5, (num_nodes,), dtype=torch.int32)
            for _ in range(num_servers)
        ]
        expected = sum(d.to(torch.int64) for d in partition_degrees)
        dataset = MockRemoteDistDataset()
        dataset._fetch_result = {i: d for i, d in enumerate(partition_degrees)}
        result = compute_and_broadcast_degree_tensors(dataset)
        assert torch.allclose(result.to(torch.int64), expected)

    def test_remote_heterogeneous():
        edge_types = [("user", "follows", "user"), ("user", "likes", "item")]
        num_servers = 2
        partition_data = {i: {} for i in range(num_servers)}
        expected_by_type = {}
        for et in edge_types:
            num_nodes = 50
            total = torch.zeros(num_nodes, dtype=torch.int64)
            for i in range(num_servers):
                d = torch.randint(0, 5, (num_nodes,), dtype=torch.int32)
                partition_data[i][et] = d
                total += d.to(torch.int64)
            expected_by_type[et] = total
        dataset = MockRemoteDistDataset()
        dataset._fetch_result = partition_data
        result = compute_and_broadcast_degree_tensors(dataset)
        for et in edge_types:
            assert torch.allclose(result[et].to(torch.int64), expected_by_type[et])

    def main():
        parser = argparse.ArgumentParser(description="Test degree computation")
        parser.add_argument(
            "--test",
            choices=["all", "local_homo", "local_hetero", "remote_homo", "remote_hetero"],
            default="all",
        )
        args = parser.parse_args()

        tests = {
            "local_homo": ("Local Homogeneous", test_local_homogeneous),
            "local_hetero": ("Local Heterogeneous", test_local_heterogeneous),
            "remote_homo": ("Remote Homogeneous", test_remote_homogeneous),
            "remote_hetero": ("Remote Heterogeneous", test_remote_heterogeneous),
        }

        print("=" * 60 + "\nDEGREE COMPUTATION TEST SUITE\n" + "=" * 60)

        to_run = tests.items() if args.test == "all" else [(args.test, tests[args.test])]
        for key, (name, fn) in to_run:
            run_test(name, fn)

        print(f"\n{'=' * 60}\nALL TESTS PASSED!\n{'=' * 60}")

    main()
