"""
Utility functions for computing node degrees in distributed graph settings.

This module provides functions to compute node out-degrees from graph partitions
and aggregate them across distributed machines. Degrees are computed from the
CSR (Compressed Sparse Row) topology stored in GraphLearn-Torch Graph objects.

Note: Degree tensors are not moved to shared memory and may be duplicated across
processes on the same machine.

Usage
=====

Use dataset.compute_degree_tensor() or compute_and_broadcast_degree_tensor(dataset):

    compute_and_broadcast_degree_tensor(dataset, num_shared_data_processes)
        └─► _compute_degrees_from_indptr (for each edge type)
        └─► _all_reduce_degrees (if distributed)

The num_shared_data_processes parameter controls over-counting correction:
- Default is 1 (no correction) - appropriate for graph store mode
- For colocated mode, caller should compute and pass the number of processes
  per machine that share the same data
"""

from typing import Union

import torch
from graphlearn_torch.data import Graph
from torch_geometric.typing import EdgeType

from gigl.common.logger import Logger
from gigl.distributed.dist_dataset import DistDataset

logger = Logger()


def compute_and_broadcast_degree_tensor(
    dataset: DistDataset,
    num_shared_data_processes: int = 1,
) -> Union[torch.Tensor, dict[EdgeType, torch.Tensor]]:
    """
    Compute node degrees from a local graph partition and aggregate across all machines.

    Extracts the graph topology from the dataset, computes degrees from the CSR
    row pointers (indptr), and performs all-reduce if torch.distributed is initialized.

    Args:
        dataset: A DistDataset containing the local graph partition.
        num_shared_data_processes: Number of processes that share the same data (typically
            processes on the same machine in colocated mode). Used to correct for
            over-counting in the all-reduce. Defaults to 1 (no correction).

    Returns:
        Union[torch.Tensor, dict[EdgeType, torch.Tensor]]: The aggregated degree tensors.
            - For homogeneous graphs: A tensor of shape [num_nodes].
            - For heterogeneous graphs: A dict mapping EdgeType to degree tensors.

    Raises:
        ValueError: If the dataset graph is None or topology is unavailable.
    """
    graph = dataset.graph
    if graph is None:
        raise ValueError("Dataset graph is None. Cannot compute degrees.")

    # Compute local degrees from graph topology
    if isinstance(graph, Graph):
        topo = graph.topo
        if topo is None or topo.indptr is None:
            raise ValueError("Topology/indptr not available for graph.")
        local_degrees: Union[
            torch.Tensor, dict[EdgeType, torch.Tensor]
        ] = _compute_degrees_from_indptr(topo.indptr)
    else:
        local_dict: dict[EdgeType, torch.Tensor] = {}
        for edge_type, edge_graph in graph.items():
            topo = edge_graph.topo
            if topo is None or topo.indptr is None:
                logger.warning(
                    f"Topology/indptr not available for edge type {edge_type}, skipping."
                )
                continue
            local_dict[edge_type] = _compute_degrees_from_indptr(topo.indptr)
        local_degrees = local_dict

    # All-reduce if distributed
    if torch.distributed.is_initialized():
        result = _all_reduce_degrees(local_degrees, num_shared_data_processes)
    else:
        if isinstance(local_degrees, torch.Tensor):
            result = _clamp_to_int16(local_degrees)
        else:
            result = {k: _clamp_to_int16(v) for k, v in local_degrees.items()}

    # Log results
    if isinstance(result, torch.Tensor):
        logger.info(
            f"{result.size(0)} nodes, max={result.max().item()}, min={result.min().item()}"
        )
    else:
        for degrees in result.values():
            logger.info(
                f"{degrees.size(0)} nodes, max={degrees.max().item()}, min={degrees.min().item()}"
            )

    return result


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


def _all_reduce_degrees(
    local_degrees: Union[torch.Tensor, dict[EdgeType, torch.Tensor]],
    num_shared_data_processes: int = 1,
) -> Union[torch.Tensor, dict[EdgeType, torch.Tensor]]:
    """All-reduce degree tensors across ranks, handling both homogeneous and heterogeneous cases.

    For heterogeneous graphs, gathers all edge types across ranks first (since different
    ranks may have different edge types), then performs all-reduce for each edge type.

    Moves tensors to GPU for the all-reduce if CUDA is available (for NCCL compatibility),
    then moves back to CPU.

    Args:
        local_degrees: Either a single tensor (homogeneous) or dict mapping EdgeType
            to tensors (heterogeneous).
        num_shared_data_processes: Number of processes that share the same data.
            Used to correct for over-counting. Defaults to 1 (no correction),
            which is correct for graph store mode where each server has distinct data.
            For colocated mode, pass the number of processes sharing data.

    Returns:
        Aggregated degree tensors in the same format as input.

    Raises:
        RuntimeError: If torch.distributed is not initialized.
    """
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "_all_reduce_degrees requires torch.distributed to be initialized."
        )

    # Use GPU if available (required for NCCL backend)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """All-reduce a single tensor with size sync and over-counting correction."""
        # Synchronize max size across all ranks
        local_size = torch.tensor([tensor.size(0)], dtype=torch.long, device=device)
        torch.distributed.all_reduce(local_size, op=torch.distributed.ReduceOp.MAX)
        max_size = int(local_size.item())

        # Pad, convert to int64 (all_reduce doesn't support int16), move to device
        padded = _pad_to_size(tensor, max_size).to(torch.int64).to(device)
        torch.distributed.all_reduce(padded, op=torch.distributed.ReduceOp.SUM)

        # Correct for over-counting, move back to CPU, and clamp to int16
        return _clamp_to_int16((padded // num_shared_data_processes).cpu())

    # Homogeneous case
    if isinstance(local_degrees, torch.Tensor):
        return reduce_tensor(local_degrees)

    # Heterogeneous case: gather all edge types across ranks
    local_edge_types = list(local_degrees.keys())
    world_size = torch.distributed.get_world_size()
    all_edge_types_gathered: list[list[EdgeType]] = [None] * world_size  # type: ignore[list-item]
    torch.distributed.all_gather_object(all_edge_types_gathered, local_edge_types)

    all_edge_types: set[EdgeType] = set()
    for edge_types in all_edge_types_gathered:
        all_edge_types.update(edge_types)

    # All-reduce each edge type
    result: dict[EdgeType, torch.Tensor] = {}
    for edge_type in all_edge_types:
        if edge_type in local_degrees:
            tensor = local_degrees[edge_type]
        else:
            tensor = torch.zeros(1, dtype=torch.int16)
        result[edge_type] = reduce_tensor(tensor)

    return result
