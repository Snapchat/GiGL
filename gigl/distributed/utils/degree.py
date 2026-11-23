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

Heterogeneous partitioned graphs are expected to materialize all registered
non-label edge types on every rank, even when a rank has no local edges for a
type. This keeps the per-node-type all-reduce order consistent across ranks.

Degree tensors are stored as int32 to stay aligned with the PPR C++ sampler's
total-degree tensor requirement while keeping memory lower than int64. We avoid
int16 because it has caused compatibility issues in this path: during the C++
PPR sampler migration, ``torch.distributed.all_reduce`` on an int16 tensor
produced ``RuntimeError: Invalid scalar type``. Values above the int32 maximum
are clamped before casting to avoid wraparound.
"""

from collections import Counter
from typing import Final, Union, overload

import torch
from graphlearn_torch.data import Graph, Topology
from graphlearn_torch.typing import NodeType
from torch_geometric.typing import EdgeType

from gigl.common.logger import Logger
from gigl.distributed.utils.device import get_device_from_process_group
from gigl.distributed.utils.networking import get_internal_ip_from_all_ranks
from gigl.types.graph import is_label_edge_type
from gigl.utils.csr import get_compressed_id_offset

logger = Logger()

_INT32_MAX: Final[int] = torch.iinfo(torch.int32).max

# All-reduce the degree tensor one slice at a time. The full padded int64
# tensor for a large graph is several GiB; materializing it (plus the collective
# buffer) on the process-group device OOMs small GPUs. Since the result is
# CPU-bound anyway, only one slice is moved to the device per collective. A
# chunked element-wise SUM is bit-identical to a whole-tensor SUM (every rank
# pads to the same size and uses identical slice bounds). This runs once at
# loader setup, so it adds no per-batch throughput cost.
_DEGREE_ALLREDUCE_CHUNK: Final[int] = 64_000_000

# Host-side slice size for summing degrees across edge types. Every rank on a
# machine runs this at once, so the int64 temporaries (about four per slice)
# are kept small.
_DEGREE_ACCUMULATE_CHUNK: Final[int] = 16_000_000


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
        result = _all_reduce_degrees(_compute_degrees_from_topology(topo))
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
            degrees = _compute_degrees_from_topology(topo)

        if anchor_type in local_dict:
            local_dict[anchor_type] = _accumulate_degrees_in_place(
                local_dict[anchor_type], degrees
            )
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


def _accumulate_degrees_in_place(
    existing: torch.Tensor,
    degrees: torch.Tensor,
    chunk_size: int = _DEGREE_ACCUMULATE_CHUNK,
) -> torch.Tensor:
    """Add ``degrees`` into int32 ``existing`` slice by slice, saturating at int32 max.

    Only ``chunk_size``-sized int64 slices are live at a time, so summing a very
    large node type across edge types needs no full-size int64 temporaries.

    Args:
        existing: int32 running totals. Modified in place unless it is shorter
            than ``degrees``, in which case a zero-padded copy is modified.
        degrees: Degrees to add, indexed from node 0. May be shorter or longer
            than ``existing``.
        chunk_size: Number of nodes widened to int64 at a time.

    Returns:
        torch.Tensor: int32 totals with ``max(len(existing), len(degrees))`` entries.
    """
    if len(degrees) > len(existing):
        existing = _pad_to_size(existing, len(degrees))
    for start in range(0, len(degrees), chunk_size):
        end = min(start + chunk_size, len(degrees))
        segment = existing[start:end]
        segment.copy_(
            _clamp_to_int32(
                segment.to(torch.int64) + degrees[start:end].to(torch.int64)
            )
        )
    return existing


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
    """Clamp degree values to int32 range before converting dtype."""
    return tensor.clamp(max=_INT32_MAX).to(torch.int32)


def _compute_degrees_from_topology(topo: Topology) -> torch.Tensor:
    """Compute degrees indexed by global compressed id.

    A topology with a row offset stores no rows below it, so those ids get degree 0.
    """
    indptr = topo.indptr
    local_degrees = indptr[1:] - indptr[:-1]
    if local_degrees.dtype != torch.int32:
        # int32 indptr differences already fit in int32. Wider ones are clamped in
        # place so no second full-size int64 copy is made before the narrowing cast.
        local_degrees = local_degrees.clamp_(max=_INT32_MAX).to(torch.int32)
    compressed_id_offset = get_compressed_id_offset(topo)
    if not compressed_id_offset:
        return local_degrees
    degrees = torch.zeros(
        compressed_id_offset + local_degrees.numel(), dtype=local_degrees.dtype
    )
    degrees[compressed_id_offset:] = local_degrees
    return degrees


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
    chunk_size: int = _DEGREE_ALLREDUCE_CHUNK,
) -> torch.Tensor:
    """All-reduce a single tensor with size sync and over-counting correction.

    The reduction is chunked so only one ``chunk_size`` slice is resident on
    ``device`` at a time, keeping the (potentially multi-GiB) degree tensor off
    the GPU except for the slice currently being reduced.

    The result is written into an int32 buffer on the host. When ``tensor`` is
    already int32 and needs no padding, that buffer is ``tensor`` itself, so its
    contents are overwritten with the reduced degrees.
    """
    # Synchronize max size across all ranks.
    local_size = torch.tensor([tensor.size(0)], dtype=torch.long, device=device)
    torch.distributed.all_reduce(local_size, op=torch.distributed.ReduceOp.MAX)
    max_size = int(local_size.item())

    # Pad on CPU in int32. Each slice is widened to int64 only while it is being
    # reduced, so the host never holds a full int64 copy of the tensor, which
    # would be multiplied by every rank on the machine.
    padded = _pad_to_size(tensor, max_size).to(torch.int32)
    # Reduce the degree tensor one contiguous slice at a time. Every rank agreed
    # on max_size above, so all ranks iterate identical [start, end) bounds and
    # issue the same number of all_reduce calls -- the collective stays in
    # lockstep and cannot hang on a rank-count mismatch. Only the current slice is
    # copied to the device and each summed slice is written straight back to CPU,
    # so peak device memory is a single chunk rather than the whole (multi-GiB)
    # tensor. Summing per-slice is exact: each element belongs to exactly one
    # slice and is reduced across all ranks there.
    for start in range(0, max_size, chunk_size):
        end = min(start + chunk_size, max_size)
        # int64 on the device: the pre-correction sum counts each machine's
        # degrees local_world_size times and can exceed int32.
        chunk = padded[start:end].to(device=device, dtype=torch.int64)
        torch.distributed.all_reduce(chunk, op=torch.distributed.ReduceOp.SUM)
        # Correct for over-counting, then clamp before casting so high-degree
        # nodes saturate instead of wrapping. Written back into the int32 slice.
        padded[start:end] = _clamp_to_int32(
            torch.div(chunk, local_world_size, rounding_mode="floor")
        ).cpu()

    return padded


@overload
def _all_reduce_degrees(
    local_degrees: dict[NodeType, torch.Tensor],
) -> dict[NodeType, torch.Tensor]: ...


@overload
def _all_reduce_degrees(local_degrees: torch.Tensor) -> torch.Tensor: ...


def _all_reduce_degrees(
    local_degrees: Union[torch.Tensor, dict[NodeType, torch.Tensor]],
) -> Union[torch.Tensor, dict[NodeType, torch.Tensor]]:
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
        local_degrees: Either a homogeneous degree tensor or a dict mapping
            NodeType to local degree tensors.

    Returns:
        Aggregated degree tensors matching the input shape.

    Raises:
        RuntimeError: If torch.distributed is not initialized.
    """
    local_world_size, device = _get_degree_reduce_context()

    if isinstance(local_degrees, torch.Tensor):
        return _all_reduce_single_degree_tensor(local_degrees, local_world_size, device)

    # Heterogeneous case: all-reduce each node type in deterministic order.
    result: dict[NodeType, torch.Tensor] = {}
    for node_type in sorted(local_degrees.keys()):
        result[node_type] = _all_reduce_single_degree_tensor(
            local_degrees[node_type], local_world_size, device
        )
    return result
