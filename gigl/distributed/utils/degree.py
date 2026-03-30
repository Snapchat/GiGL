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
from torch_geometric.typing import EdgeType, NodeType

from gigl.common.logger import Logger
from gigl.distributed.utils.device import get_device_from_process_group
from gigl.distributed.utils.networking import get_internal_ip_from_all_ranks
from gigl.types.graph import is_label_edge_type

logger = Logger()


def compute_and_broadcast_degree_tensor(
    graph: Union[Graph, dict[EdgeType, Graph]],
    edge_dir: str,
) -> Union[torch.Tensor, dict[NodeType, torch.Tensor]]:
    """
    Compute node degrees from a graph and aggregate across all machines.

    For heterogeneous graphs, degrees are summed across all edge types sharing
    the same anchor node type (source for ``edge_dir="out"``, destination for
    ``edge_dir="in"``), then all-reduced across ranks.  The result is one
    ``int16`` tensor per anchor node type rather than one per edge type,
    reducing both stored memory and the number of all-reduce calls.

    Over-counting correction (for processes sharing the same data) is handled
    automatically by detecting the distributed topology.

    Args:
        graph: A Graph (homogeneous) or dict[EdgeType, Graph] (heterogeneous).
        edge_dir: ``"out"`` to group by source node type (out-degree);
            ``"in"`` to group by destination node type (in-degree).

    Returns:
        Union[torch.Tensor, dict[NodeType, torch.Tensor]]: The aggregated degree tensors.
            - For homogeneous graphs: A tensor of shape [num_nodes].
            - For heterogeneous graphs: A dict mapping NodeType to ``int16`` degree tensors,
              where each entry is the total degree across all edge types for that anchor node type.

    Raises:
        RuntimeError: If torch.distributed is not initialized.
        ValueError: If topology is unavailable for a homogeneous graph.
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
            torch.Tensor, dict[NodeType, torch.Tensor]
        ] = _compute_degrees_from_indptr(topo.indptr)
    else:
        local_degrees = _compute_hetero_degrees_by_node_type(graph, edge_dir)

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
        for node_type, degrees in result.items():
            if degrees.numel() > 0:
                logger.info(
                    f"{node_type}: {degrees.size(0)} nodes, max={degrees.max().item()}, min={degrees.min().item()}"
                )
            else:
                logger.info(
                    f"Graph contained 0 nodes for node type {node_type} when computing degrees"
                )

    return result


def _compute_hetero_degrees_by_node_type(
    graph: dict[EdgeType, Graph],
    edge_dir: str,
) -> dict[NodeType, torch.Tensor]:
    """Sum per-edge-type degrees into per-anchor-node-type totals.

    Label edge types (ABLP supervision edges) are excluded — they should not
    contribute to degree counts used in PPR, matching the sampler's own exclusion.

    Uses a two-pass approach to avoid dynamic accumulator resizing:

    - Pass 1: compute ``int32`` degrees per edge type and record the maximum
      node count per anchor type (to pre-size the accumulator).
    - Pass 2: allocate one ``int32`` accumulator per anchor type and sum.

    Returns ``int32`` tensors; the caller is responsible for clamping to
    ``int16`` after the all-reduce (see ``_all_reduce_degrees``).

    All anchor node types derived from non-label ``graph.keys()`` are present in
    the output (with size-0 tensors for types whose ``indptr`` is unavailable),
    ensuring symmetric participation in the subsequent all-reduce.

    Args:
        graph: Heterogeneous graph mapping EdgeType to Graph.
        edge_dir: ``"out"`` to anchor on source node type; ``"in"`` to anchor
            on destination node type.

    Returns:
        dict[NodeType, torch.Tensor]: ``int32`` total-degree tensor per anchor node type.
    """
    # All anchor node types must appear in the output so the all_reduce is
    # symmetric across ranks, even if this rank has no edges for a given type.
    # Label edge types are excluded — they represent ABLP supervision edges, not
    # structural neighbors, and should not contribute to degree counts used in PPR.
    anchor_ntypes: set[NodeType] = {
        etype[-1] if edge_dir == "in" else etype[0]
        for etype in graph.keys()
        if not is_label_edge_type(etype)
    }

    # Pass 1: compute int32 degrees per valid edge type and track max node
    # count per anchor type.  int32 is sufficient: per-node degrees are always
    # well below int32 max for any realistic graph.
    et_degrees_i32: dict[EdgeType, torch.Tensor] = {}
    max_sizes: dict[NodeType, int] = {ntype: 0 for ntype in anchor_ntypes}
    for edge_type, edge_graph in graph.items():
        if is_label_edge_type(edge_type):
            continue
        anchor_ntype: NodeType = edge_type[-1] if edge_dir == "in" else edge_type[0]
        topo = edge_graph.topo
        if topo is None or topo.indptr is None:
            logger.warning(
                f"Topology/indptr not available for edge type {edge_type}, skipping."
            )
            continue
        degrees_i32 = (topo.indptr[1:] - topo.indptr[:-1]).to(torch.int32)
        et_degrees_i32[edge_type] = degrees_i32
        max_sizes[anchor_ntype] = max(max_sizes[anchor_ntype], len(degrees_i32))

    # Pass 2: allocate accumulators once (sized from pass 1) and sum.
    accumulators: dict[NodeType, torch.Tensor] = {
        ntype: torch.zeros(size, dtype=torch.int32)
        for ntype, size in max_sizes.items()
    }
    for edge_type, degrees_i32 in et_degrees_i32.items():
        anchor_ntype = edge_type[-1] if edge_dir == "in" else edge_type[0]
        accumulators[anchor_ntype][: len(degrees_i32)] += degrees_i32

    return accumulators


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
    local_degrees: Union[torch.Tensor, dict[NodeType, torch.Tensor]],
) -> Union[torch.Tensor, dict[NodeType, torch.Tensor]]:
    """All-reduce degree tensors across ranks, handling both homogeneous and heterogeneous cases.

    For heterogeneous graphs, iterates over the node types in local_degrees. All partitions
    are expected to have entries for all node types (even if some have empty tensors).

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
        local_degrees: Either a single tensor (homogeneous) or dict mapping NodeType
            to tensors (heterogeneous). For heterogeneous graphs, all partitions must
            have entries for all node types.

    Returns:
        Aggregated degree tensors in the same format as input, stored as ``int16``.

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

        # Pad and convert to int32 for the all-reduce (int16 is not supported).
        # int32 is sufficient: the raw sum across W ranks is at most W * max_degree,
        # which fits comfortably in int32 for any realistic world size and degree.
        padded = _pad_to_size(tensor, max_size).to(torch.int32).to(device)
        torch.distributed.all_reduce(padded, op=torch.distributed.ReduceOp.SUM)

        # Correct for over-counting, move back to CPU, and clamp to int16.
        # TODO (mkolodner-sc): Potentially want to parameterize this in the future if we want degrees higher than the int16 max.
        return _clamp_to_int16((padded // local_world_size).cpu())

    # Homogeneous case
    if isinstance(local_degrees, torch.Tensor):
        return reduce_tensor(local_degrees)

    # Heterogeneous case: all-reduce each node type.
    # Sort node types for deterministic ordering across ranks.
    result: dict[NodeType, torch.Tensor] = {}
    for node_type in sorted(local_degrees.keys()):
        result[node_type] = reduce_tensor(local_degrees[node_type])

    return result
