"""
Utility functions for computing node degrees in distributed graph settings.

This module provides functions to compute node out-degrees from graph partitions
and aggregate them across distributed machines. Degrees are computed from the
CSR (Compressed Sparse Row) topology stored in GraphLearn-Torch Graph objects.

The main entry point is compute_and_broadcast_degree_tensors(dataset), which dispatches
to the appropriate computation function based on dataset type.

Note: Degree tensors are not moved to shared memory and may be duplicated across
processes on the same machine.

Overview
========

    compute_and_broadcast_degree_tensors(dataset)
        │
        ├─► DistDataset (local graph partition)
        │       └─► _compute_degrees_from_local_dataset
        │               └─► _process_graph
        │                       └─► _compute_degrees_from_indptr
        │                       └─► _all_reduce_with_size_sync (if distributed)
        │
        └─► RemoteDistDataset (graph store mode)
                └─► _compute_degrees_from_remote_dataset
                        └─► _sum_tensors_with_padding
"""

from collections import Counter
from typing import Union

import torch
from graphlearn_torch.data import Graph
from torch_geometric.typing import EdgeType

import gigl.distributed.utils
from gigl.common.logger import Logger
from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.graph_store.remote_dist_dataset import RemoteDistDataset

logger = Logger()


def compute_and_broadcast_degree_tensors(
    dataset: Union[DistDataset, RemoteDistDataset],
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
    if isinstance(dataset, DistDataset):
        return _compute_degrees_from_local_dataset(dataset)
    elif isinstance(dataset, RemoteDistDataset):
        return _compute_degrees_from_remote_dataset(dataset)
    else:
        raise TypeError(
            f"Unsupported dataset type: {type(dataset)}. "
            f"Expected DistDataset or RemoteDistDataset."
        )


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


def _clamp_to_int16(tensor: torch.Tensor) -> torch.Tensor:
    """Clamp tensor values to int16 max and convert dtype."""
    max_int16 = torch.iinfo(torch.int16).max
    return tensor.clamp(max=max_int16).to(torch.int16)


def _compute_degrees_from_indptr(indptr: torch.Tensor) -> torch.Tensor:
    """Compute degrees from CSR row pointers: degree[i] = indptr[i+1] - indptr[i]."""
    return (indptr[1:] - indptr[:-1]).contiguous().to(torch.int16)


# Dist Dataset Degree Computation Utilities


def _compute_degrees_from_local_dataset(
    dataset: DistDataset,
) -> Union[torch.Tensor, dict[EdgeType, torch.Tensor]]:
    """Compute degrees from a local DistDataset with all-reduce across machines."""
    graph = dataset.graph
    if graph is None:
        raise ValueError("Dataset graph is None. Cannot compute degrees.")

    return _process_graph(graph)


def _all_reduce_with_size_sync(local_degrees: torch.Tensor) -> torch.Tensor:
    """All-reduce degrees across ranks, handling size mismatches and over-counting.

    Moves tensors to GPU for the all-reduce if CUDA is available (for NCCL compatibility),
    then moves back to CPU.

    Requires torch.distributed to be initialized.

    Args:
        local_degrees: Local degree tensor to be aggregated across all ranks.

    Returns:
        Aggregated degree tensor after all-reduce and over-counting correction.

    Raises:
        RuntimeError: If torch.distributed is not initialized.
    """
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "_all_reduce_with_size_sync requires torch.distributed to be initialized."
        )

    rank = torch.distributed.get_rank()
    rank_ips = gigl.distributed.utils.get_internal_ip_from_all_ranks()
    local_world_size = Counter(rank_ips)[rank_ips[rank]]
    logger.info(f"Degree computation: rank={rank}, local_world_size={local_world_size}")

    # Use GPU if available (required for NCCL backend)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Synchronize max size across all ranks
    local_size = torch.tensor([local_degrees.size(0)], dtype=torch.long, device=device)
    torch.distributed.all_reduce(local_size, op=torch.distributed.ReduceOp.MAX)
    max_size = int(local_size.item())

    # Pad, convert to int64 (all_reduce doesn't support int16), move to device
    local_degrees = _pad_to_size(local_degrees, max_size).to(torch.int64).to(device)
    torch.distributed.all_reduce(local_degrees, op=torch.distributed.ReduceOp.SUM)

    # Correct for over-counting and move back to CPU
    return (local_degrees // local_world_size).cpu()


def _process_graph(
    graph: Union[Graph, dict[EdgeType, Graph]],
) -> Union[torch.Tensor, dict[EdgeType, torch.Tensor]]:
    """Process graph(s) to compute degrees with optional all-reduce.

    If torch.distributed is initialized, performs all-reduce to aggregate
    degrees across all ranks. Otherwise, returns local degrees only.

    Args:
        graph: Either a single Graph (homogeneous) or dict of Graphs (heterogeneous).

    Returns:
        For homogeneous: A single degree tensor.
        For heterogeneous: A dict mapping EdgeType to degree tensors.
    """
    is_distributed = torch.distributed.is_initialized()

    def compute_and_reduce(indptr: torch.Tensor) -> torch.Tensor:
        degrees = _compute_degrees_from_indptr(indptr)
        if is_distributed:
            degrees = _all_reduce_with_size_sync(degrees)
        return _clamp_to_int16(degrees)

    # Homogeneous case: single Graph
    if isinstance(graph, Graph):
        topo = graph.topo
        if topo is None or topo.indptr is None:
            raise ValueError("Topology/indptr not available for graph.")
        degrees = compute_and_reduce(topo.indptr)
        logger.info(
            f"{degrees.size(0)} nodes, max={degrees.max().item()}, min={degrees.min().item()}"
        )
        return degrees

    # Heterogeneous case: dict of Graphs
    result: dict[EdgeType, torch.Tensor] = {}
    for edge_type, edge_graph in graph.items():
        topo = edge_graph.topo
        if topo is None or topo.indptr is None:
            logger.warning(
                f"Topology/indptr not available for edge type {edge_type}, skipping."
            )
            continue
        degrees = compute_and_reduce(topo.indptr)
        result[edge_type] = degrees
        logger.info(
            f"{degrees.size(0)} nodes, max={degrees.max().item()}, min={degrees.min().item()}"
        )

    return result


# Remote Dist Dataset Degree Computation Utilities


def _sum_tensors_with_padding(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Sum multiple tensors, padding shorter ones to match the longest."""
    if not tensors:
        return torch.tensor([], dtype=torch.int16)

    max_size = max(t.size(0) for t in tensors)
    result = torch.zeros(max_size, dtype=torch.int64)

    for tensor in tensors:
        padded = _pad_to_size(tensor, max_size)
        result += padded.to(torch.int64)

    return _clamp_to_int16(result)


def _compute_degrees_from_remote_dataset(
    dataset: RemoteDistDataset,
) -> Union[torch.Tensor, dict[EdgeType, torch.Tensor]]:
    """Compute degrees by fetching from remote storage nodes and aggregating."""
    all_local_degrees = dataset.fetch_local_degrees()

    if not all_local_degrees:
        raise ValueError("No degree tensors returned from storage nodes.")

    first_result = next(iter(all_local_degrees.values()))

    # Homogeneous case: each server returns a tensor
    if isinstance(first_result, torch.Tensor):
        tensors = [d for d in all_local_degrees.values() if isinstance(d, torch.Tensor)]
        result = _sum_tensors_with_padding(tensors)
        logger.info(
            f"{result.size(0)} nodes, max={result.max().item()}, min={result.min().item()}"
        )
        return result

    # Heterogeneous case: each server returns dict[EdgeType, Tensor]
    all_edge_types: set[EdgeType] = set()
    for degrees in all_local_degrees.values():
        if isinstance(degrees, dict):
            all_edge_types.update(degrees.keys())

    result_dict: dict[EdgeType, torch.Tensor] = {}
    for edge_type in all_edge_types:
        tensors = [
            degrees[edge_type]
            for degrees in all_local_degrees.values()
            if isinstance(degrees, dict) and edge_type in degrees
        ]
        if tensors:
            result_dict[edge_type] = _sum_tensors_with_padding(tensors)
            logger.info(
                f"{result_dict[edge_type].size(0)} nodes, max={result_dict[edge_type].max().item()}, min={result_dict[edge_type].min().item()}"
            )
        else:
            logger.warning(f"No degree tensors found for edge type {edge_type}")

    return result_dict
