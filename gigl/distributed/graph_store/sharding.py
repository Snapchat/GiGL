"""Graph-store-specific sharding helpers."""

from dataclasses import dataclass
from enum import Enum

import torch


class ShardStrategy(Enum):
    """Strategies for splitting remote graph-store inputs across compute nodes."""

    ROUND_ROBIN = "round_robin"
    CONTIGUOUS = "contiguous"


@dataclass(frozen=True)
class ServerSlice:
    """The fraction of a storage server owned by one compute node."""

    server_rank: int
    start_num: int
    start_den: int
    end_num: int
    end_den: int

    def slice_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Slice a tensor according to this server assignment."""
        total = len(tensor)
        start_idx = total * self.start_num // self.start_den
        end_idx = total * self.end_num // self.end_den
        if start_idx == 0 and end_idx == total:
            return tensor
        return tensor[start_idx:end_idx].clone()


def compute_server_assignments(
    num_servers: int,
    num_compute_nodes: int,
    compute_rank: int,
) -> dict[int, ServerSlice]:
    """Compute which servers, and which fractions of them, belong to one compute node."""
    if num_servers <= 0:
        raise ValueError(f"num_servers must be positive, got {num_servers}")
    if num_compute_nodes <= 0:
        raise ValueError(f"num_compute_nodes must be positive, got {num_compute_nodes}")
    if compute_rank < 0 or compute_rank >= num_compute_nodes:
        raise ValueError(
            f"compute_rank must be in [0, {num_compute_nodes}), got {compute_rank}"
        )

    seg_start = compute_rank * num_servers
    seg_end = (compute_rank + 1) * num_servers

    assignments: dict[int, ServerSlice] = {}
    for server_rank in range(num_servers):
        server_start = server_rank * num_compute_nodes
        server_end = (server_rank + 1) * num_compute_nodes

        overlap_start = max(seg_start, server_start)
        overlap_end = min(seg_end, server_end)
        if overlap_start >= overlap_end:
            continue

        assignments[server_rank] = ServerSlice(
            server_rank=server_rank,
            start_num=overlap_start - server_start,
            start_den=num_compute_nodes,
            end_num=overlap_end - server_start,
            end_den=num_compute_nodes,
        )

    return assignments
