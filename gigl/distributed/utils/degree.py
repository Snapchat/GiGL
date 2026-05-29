"""
Utility functions for computing node degrees in distributed graph settings.

This module provides functions to compute node out-degrees from graph partitions
and aggregate them across distributed machines. Degrees are computed from the
CSR (Compressed Sparse Row) topology stored in GraphLearn-Torch Graph objects.

For homogeneous graphs, callers receive a single ``torch.Tensor``. For
heterogeneous graphs, degrees are accumulated per anchor node type (summing
across all edge types incident to that node type) before the distributed
all-reduce, so callers receive ``dict[NodeType, torch.Tensor]``.

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
from graphlearn_torch.typing import NodeType
from torch_geometric.typing import EdgeType

from gigl.common.logger import Logger
from gigl.distributed.utils.device import get_device_from_process_group
from gigl.distributed.utils.networking import get_internal_ip_from_all_ranks
from gigl.types.graph import is_label_edge_type

logger = Logger()


def compute_and_broadcast_degree_tensor(
    graph: Union[Graph, dict[EdgeType, Graph]],
    edge_dir: str,
) -> Union[torch.Tensor, dict[NodeType, torch.Tensor]]:
    """Compute node degrees from a graph and aggregate across all machines.

    For each non-label edge type, degrees are derived from the CSR row pointers
    (indptr).  For heterogeneous graphs, degrees are summed across all edge types
    incident to each anchor node type **locally** before the all-reduce, so the
    per-edge-type tensor is only a transient intermediate and is never stored,
    returned, or transmitted over RPC.

    Over-counting correction (for processes sharing the same data) is handled
    automatically by detecting the distributed topology.

    Args:
        graph: A Graph (homogeneous) or dict[EdgeType, Graph] (heterogeneous).
            For heterogeneous graphs, label edge types are automatically excluded
            — they are supervision edges and should not contribute to node degree
            for graph traversal algorithms like PPR.
        edge_dir: Sampling direction — ``"in"`` or ``"out"``.  Determines which
            end of each edge is the anchor node type for degree accumulation.

    Returns:
        Union[torch.Tensor, dict[NodeType, torch.Tensor]]: Aggregated degree
            tensors. For homogeneous graphs, returns an int32 tensor of shape
            ``[num_nodes]``. For heterogeneous graphs, returns int32 tensors
            keyed by node type with shape ``[num_nodes_of_that_type]``.

    Raises:
        RuntimeError: If torch.distributed is not initialized.
        ValueError: If topology is unavailable.
    """
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "compute_and_broadcast_degree_tensor requires torch.distributed to be initialized."
        )

    # Compute local degrees from graph topology.
    if isinstance(graph, Graph):
        topo = graph.topo
        if topo is None or topo.indptr is None:
            raise ValueError("Topology/indptr not available for graph.")

        # Homogeneous graphs keep the usual GiGL shape: a single tensor.
        result = _all_reduce_degree_tensor(_compute_degrees_from_indptr(topo.indptr))
        if result.numel() > 0:
            logger.info(
                f"{result.size(0)} nodes, max={result.max().item()}, min={result.min().item()}"
            )
        else:
            logger.info("Graph contained 0 nodes when computing degrees")
        return result

    local_dict: dict[NodeType, torch.Tensor] = {}
    for edge_type, edge_graph in graph.items():
        # Label edge types are supervision edges and should not contribute to
        # node degree for traversal algorithms like PPR.
        if is_label_edge_type(edge_type):
            continue
        anchor_type: NodeType = edge_type[-1] if edge_dir == "in" else edge_type[0]
        topo = edge_graph.topo
        if topo is None or topo.indptr is None:
            logger.warning(
                f"Topology/indptr not available for edge type {edge_type}, using empty tensor."
            )
            degrees = torch.empty(0, dtype=torch.int32)
        else:
            degrees = _compute_degrees_from_indptr(topo.indptr)

        if anchor_type in local_dict:
            existing = local_dict[anchor_type]
            max_len = max(len(existing), len(degrees))
            summed = _pad_to_size(existing, max_len).to(torch.int64)
            summed[: len(degrees)] += degrees.to(torch.int64)
            local_dict[anchor_type] = summed.to(torch.int32)
        else:
            local_dict[anchor_type] = degrees

    # All-reduce across ranks after local per-node-type aggregation.
    result = _all_reduce_degrees(local_dict)
    for node_type, degrees in result.items():
        if degrees.numel() > 0:
            logger.info(
                f"{node_type}: {degrees.size(0)} nodes, "
                f"max={degrees.max().item()}, min={degrees.min().item()}"
            )
        else:
            logger.info(
                f"Graph contained 0 nodes for node type {node_type} when computing degrees"
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


def _compute_degrees_from_indptr(indptr: torch.Tensor) -> torch.Tensor:
    """Compute degrees from CSR row pointers: degree[i] = indptr[i+1] - indptr[i]."""
    return (indptr[1:] - indptr[:-1]).to(torch.int32)


def _get_degree_reduce_context() -> tuple[int, torch.device]:
    """Return local-world-size correction factor and all-reduce device."""
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "_all_reduce_degrees requires torch.distributed to be initialized."
        )

    # Compute local_world_size: number of processes on the same machine sharing data.
    all_ips = get_internal_ip_from_all_ranks()
    my_rank = torch.distributed.get_rank()
    my_ip = all_ips[my_rank]
    local_world_size = Counter(all_ips)[my_ip]

    # NCCL backend requires CUDA tensors; Gloo works with CPU.
    device = get_device_from_process_group()
    return local_world_size, device


def _all_reduce_single_degree_tensor(
    tensor: torch.Tensor,
    local_world_size: int,
    device: torch.device,
) -> torch.Tensor:
    """All-reduce a single tensor with size sync and over-counting correction."""
    # Synchronize max size across all ranks.
    local_size = torch.tensor([tensor.size(0)], dtype=torch.long, device=device)
    torch.distributed.all_reduce(local_size, op=torch.distributed.ReduceOp.MAX)
    max_size = int(local_size.item())

    # Pad, convert to int64 for all_reduce, and move to the process-group device.
    padded = _pad_to_size(tensor, max_size).to(torch.int64).to(device)
    torch.distributed.all_reduce(padded, op=torch.distributed.ReduceOp.SUM)

    # Correct for over-counting and move back to CPU. We keep int32 so high-degree
    # nodes do not saturate at int16.
    return (padded // local_world_size).to(torch.int32).cpu()


def _all_reduce_degree_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce a homogeneous degree tensor across ranks."""
    local_world_size, device = _get_degree_reduce_context()
    return _all_reduce_single_degree_tensor(tensor, local_world_size, device)


def _all_reduce_degrees(
    local_degrees: dict[NodeType, torch.Tensor],
) -> dict[NodeType, torch.Tensor]:
    """All-reduce degree tensors across ranks.

    Moves tensors to GPU for the all-reduce if using NCCL backend (which
    requires CUDA), otherwise keeps tensors on CPU (for Gloo backend).

    Over-counting correction:
        In distributed training, multiple processes on the same machine often
        share the same graph partition data (via shared memory). When we
        all-reduce degrees, each process contributes its "local" degrees — but
        if 4 processes on one machine all read the same partition, that
        partition's degrees get summed 4 times instead of 1.

        Example: Machine A has 2 processes sharing partition with degrees [3, 5, 2].
                 Machine B has 2 processes sharing partition with degrees [1, 4, 6].

                 Without correction: all-reduce sums = [3+3+1+1, 5+5+4+4, 2+2+6+6]
                                                     = [8, 18, 16]  (wrong!)

                 With correction: divide by local_world_size (2 per machine)
                                  = [4, 9, 8]  (correct: [3+1, 5+4, 2+6])

        This function detects how many processes share the same machine by
        comparing IP addresses, then divides by that count to correct the
        over-counting.

    Args:
        local_degrees: Dict mapping NodeType to local degree tensors.

    Returns:
        Aggregated degree tensors keyed by NodeType.

    Raises:
        RuntimeError: If torch.distributed is not initialized.
    """
    local_world_size, device = _get_degree_reduce_context()

    # Heterogeneous case: all-reduce each node type in deterministic order.
    result: dict[NodeType, torch.Tensor] = {}
    for node_type in sorted(local_degrees.keys()):
        result[node_type] = _all_reduce_single_degree_tensor(
            local_degrees[node_type], local_world_size, device
        )
    return result
