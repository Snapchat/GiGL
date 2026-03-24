"""Graph-store-specific sharding helpers."""

from dataclasses import dataclass
from enum import Enum

import torch


class ShardStrategy(Enum):
    """Strategies for splitting remote graph-store inputs across compute nodes.

    When fetching node IDs (or ABLP input) from storage servers, the shard
    strategy controls how data is divided among compute nodes.

    Suppose we have 2 storage nodes and 2 compute nodes, with 16 total nodes.
    Nodes are partitioned across storage nodes, with splits defined as::

        Storage rank 0: [0, 1, 2, 3, 4, 5, 6, 7]
            train=[0, 1, 2, 3], val=[4, 5], test=[6, 7]
        Storage rank 1: [8, 9, 10, 11, 12, 13, 14, 15]
            train=[8, 9, 10, 11], val=[12, 13], test=[14, 15]

    **ROUND_ROBIN** — Each storage server independently shards its own nodes
    across the requested ``world_size``. Every compute node contacts every
    storage server and receives an interleaved slice::

        # All training nodes, sharded across 2 compute nodes (round-robin):
        >>> dataset.fetch_node_ids(rank=0, world_size=2, split="train")
        {
            0: tensor([0, 2]),   # Even-indexed training nodes from storage 0
            1: tensor([8, 10])   # Even-indexed training nodes from storage 1
        }
        >>> dataset.fetch_node_ids(rank=1, world_size=2, split="train")
        {
            0: tensor([1, 3]),   # Odd-indexed training nodes from storage 0
            1: tensor([9, 11])   # Odd-indexed training nodes from storage 1
        }

    Both strategies also support ``split=None``, which returns all nodes
    (train + val + test) from each storage server::

        # Round-robin, all nodes (no split), sharded across 2 compute nodes:
        >>> dataset.fetch_node_ids(rank=0, world_size=2)
        {
            0: tensor([0, 2, 4, 6]),  # Even-indexed nodes from storage 0
            1: tensor([8, 10, 12, 14])  # Even-indexed nodes from storage 1
        }

        # Contiguous, all nodes (no split), sharded across 2 compute nodes:
        >>> dataset.fetch_node_ids(rank=0, world_size=2,
        ...     shard_strategy=ShardStrategy.CONTIGUOUS)
        {
            0: tensor([0, 1, 2, 3, 4, 5, 6, 7]),  # All nodes from storage 0
            1: tensor([])                            # Nothing from storage 1
        }

    **CONTIGUOUS** — Storage servers are assigned to compute nodes in
    contiguous blocks. Each compute node fetches *all* data from its
    assigned server(s) and receives empty tensors for unassigned ones.
    When servers outnumber compute nodes a server's data is fractionally
    split; when compute nodes outnumber servers a compute node may own a
    fraction of one server::

        # All training nodes, sharded across 2 compute nodes (contiguous):
        >>> dataset.fetch_node_ids(rank=0, world_size=2, split="train",
        ...     shard_strategy=ShardStrategy.CONTIGUOUS)
        {
            0: tensor([0, 1, 2, 3]),  # All training nodes from storage 0
            1: tensor([])              # Nothing from storage 1
        }
        >>> dataset.fetch_node_ids(rank=1, world_size=2, split="train",
        ...     shard_strategy=ShardStrategy.CONTIGUOUS)
        {
            0: tensor([]),             # Nothing from storage 0
            1: tensor([8, 9, 10, 11]) # All training nodes from storage 1
        }

        # With 3 storage nodes and 2 compute nodes, server 1 is fractionally split:
        >>> dataset.fetch_node_ids(rank=0, world_size=2, split="train",
        ...     shard_strategy=ShardStrategy.CONTIGUOUS)
        {
            0: tensor([0, 1, 2, 3]),       # All of storage 0
            1: tensor([8, 9]),             # First half of storage 1
            2: tensor([])                  # Nothing from storage 2
        }
    """

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
        return tensor[start_idx:end_idx]


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
