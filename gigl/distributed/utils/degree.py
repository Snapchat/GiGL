"""
Utility functions for computing node degrees in distributed graph settings.

This module provides functions to compute node out-degrees from graph partitions
and aggregate them across distributed machines. Degrees are computed from the
CSR (Compressed Sparse Row) topology stored in GraphLearn-Torch Graph objects.

Note: Degree tensors are not moved to shared memory and may be duplicated across
processes on the same machine.

Requirements
============

torch.distributed must be initialized before calling these functions.

Usage
=====

Access dataset.degree_tensor to lazily compute and cache the degree tensor.

Over-counting correction is handled automatically in _all_reduce_degrees by
detecting how many processes share the same machine (and thus the same data).
"""

from collections import Counter
from typing import Union

import torch
from graphlearn_torch.data import Graph
from torch_geometric.typing import EdgeType

from gigl.common.logger import Logger
from gigl.distributed.utils.device import get_device_from_process_group
from gigl.distributed.utils.networking import get_internal_ip_from_all_ranks

logger = Logger()


def compute_and_broadcast_degree_tensor(
    graph: Union[Graph, dict[EdgeType, Graph]],
) -> Union[torch.Tensor, dict[EdgeType, torch.Tensor]]:
    """
    Compute node degrees from a graph and aggregate across all machines.

    Computes degrees from the CSR row pointers (indptr) and performs all-reduce
    to aggregate across ranks.

    Over-counting correction (for processes sharing the same data) is handled
    automatically by detecting the distributed topology.

    Args:
        graph: A Graph (homogeneous) or dict[EdgeType, Graph] (heterogeneous).

    Returns:
        Union[torch.Tensor, dict[EdgeType, torch.Tensor]]: The aggregated degree tensors.
            - For homogeneous graphs: A tensor of shape [num_nodes].
            - For heterogeneous graphs: A dict mapping EdgeType to degree tensors.

    Raises:
        RuntimeError: If torch.distributed is not initialized.
        ValueError: If topology is unavailable.
    """
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "compute_and_broadcast_degree_tensor requires torch.distributed to be initialized."
        )

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
                    f"Topology/indptr not available for edge type {edge_type}, using empty tensor."
                )
                local_dict[edge_type] = torch.empty(0, dtype=torch.int16)
            else:
                local_dict[edge_type] = _compute_degrees_from_indptr(topo.indptr)
        local_degrees = local_dict

    # All-reduce across ranks (over-counting correction handled internally)
    result = _all_reduce_degrees(local_degrees)

    # Log results
    if isinstance(result, torch.Tensor):
        if result.numel() > 0:
            logger.info(
                f"{result.size(0)} nodes, max={result.max().item()}, min={result.min().item()}"
            )
        else:
            logger.info("Graph contained 0 nodes when computing degrees")
    else:
        for edge_type, degrees in result.items():
            if degrees.numel() > 0:
                logger.info(
                    f"{edge_type}: {degrees.size(0)} nodes, max={degrees.max().item()}, min={degrees.min().item()}"
                )
            else:
                logger.info(
                    f"Graph contained 0 nodes for edge type {edge_type} when computing degrees"
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
    return (indptr[1:] - indptr[:-1]).to(torch.int16)


def _all_reduce_degrees(
    local_degrees: Union[torch.Tensor, dict[EdgeType, torch.Tensor]],
) -> Union[torch.Tensor, dict[EdgeType, torch.Tensor]]:
    """All-reduce degree tensors across ranks, handling both homogeneous and heterogeneous cases.

    For heterogeneous graphs, iterates over the edge types in local_degrees. All partitions
    are expected to have entries for all edge types (even if some have empty tensors).

    Moves tensors to GPU for the all-reduce if using NCCL backend (which requires CUDA),
    otherwise keeps tensors on CPU (for Gloo backend).

    Over-counting correction:
        In distributed training, multiple processes on the same machine often share the
        same graph partition data (via shared memory). When we all-reduce degrees, each
        process contributes its "local" degrees - but if 4 processes on one machine all
        read the same partition, that partition's degrees get summed 4 times instead of 1.

        Example: Machine A has 2 processes sharing partition with degrees [3, 5, 2].
                 Machine B has 2 processes sharing partition with degrees [1, 4, 6].

                 Without correction: all-reduce sums = [3+3+1+1, 5+5+4+4, 2+2+6+6]
                                                     = [8, 18, 16]  (wrong!)

                 With correction: divide by local_world_size (2 per machine)
                                  = [4, 9, 8]  (correct: [3+1, 5+4, 2+6])

        This function detects how many processes share the same machine by comparing
        IP addresses, then divides by that count to correct the over-counting.

    Args:
        local_degrees: Either a single tensor (homogeneous) or dict mapping EdgeType
            to tensors (heterogeneous). For heterogeneous graphs, all partitions must
            have entries for all edge types.

    Returns:
        Aggregated degree tensors in the same format as input.

    Raises:
        RuntimeError: If torch.distributed is not initialized.
    """
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "_all_reduce_degrees requires torch.distributed to be initialized."
        )

    # Compute local_world_size: number of processes on the same machine sharing data
    all_ips = get_internal_ip_from_all_ranks()
    my_rank = torch.distributed.get_rank()
    my_ip = all_ips[my_rank]
    local_world_size = Counter(all_ips)[my_ip]

    # NCCL backend requires CUDA tensors; Gloo works with CPU
    device = get_device_from_process_group()

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
        # TODO (mkolodner-sc): Potentially want to paramaterize this in the future if we want degrees higher than the int16 max.
        return _clamp_to_int16((padded // local_world_size).cpu())

    # Homogeneous case
    if isinstance(local_degrees, torch.Tensor):
        return reduce_tensor(local_degrees)

    # Heterogeneous case: all-reduce each edge type
    # Sort edge types for deterministic ordering across ranks
    result: dict[EdgeType, torch.Tensor] = {}
    for edge_type in sorted(local_degrees.keys()):
        result[edge_type] = reduce_tensor(local_degrees[edge_type])

    return result
