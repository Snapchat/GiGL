"""Graph-store-specific sharding helpers."""

from dataclasses import dataclass
from enum import Enum

import torch


class ShardStrategy(Enum):
    """Strategies for splitting remote graph-store inputs across compute nodes.

    When fetching node IDs (or ABLP input) from storage servers, the shard
    strategy controls how data is divided among compute nodes.

    ``CONTIGUOUS`` is generally recommended as it reduces per-server load and
    increases cluster stability.
    ``ROUND_ROBIN`` is the default as it supports ``None`` for rank/world_size
    (i.e. fetching all data unsharded).

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

    **CONTIGUOUS** — Storage servers are assigned to compute nodes in
    contiguous blocks. Each compute node fetches *all* data from its
    assigned server(s) and receives empty tensors for unassigned ones.

    If there are more servers than compute nodes, the extra servers are
    divided among compute nodes and each server's data is sliced
    proportionally (e.g. with 3 servers and 2 compute nodes, one compute
    node receives the first half of the middle server's data and the other
    receives the second half).
    If there are more compute nodes than servers, multiple compute nodes
    share a single server, each receiving a proportional slice of that
    server's data::

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

        # Contiguous, all nodes (no split), sharded across 2 compute nodes:
        >>> dataset.fetch_node_ids(rank=0, world_size=2,
        ...     shard_strategy=ShardStrategy.CONTIGUOUS)
        {
            0: tensor([0, 1, 2, 3, 4, 5, 6, 7]),  # All nodes from storage 0
            1: tensor([])                            # Nothing from storage 1
        }
    """

    ROUND_ROBIN = "round_robin"
    CONTIGUOUS = "contiguous"


@dataclass(frozen=True)
class ServerSlice:
    """The fraction of a storage server's data owned by one compute node.

    Fractions are stored as numerator/denominator pairs rather than concrete
    indices because the actual tensor length is unknown at assignment time
    (assignments are computed before data is fetched).
    The concrete start/end indices are resolved lazily in :meth:`slice_tensor`.

    The fraction ``[start_numerator/start_denominator, end_numerator/end_denominator)``
    describes the half-open interval of the server's data that this compute node owns.

    Args:
        server_rank: The rank of the storage server this slice refers to.
        start_numerator: Numerator of the start fraction.
        start_denominator: Denominator of the start fraction.
        end_numerator: Numerator of the end fraction.
        end_denominator: Denominator of the end fraction.

    Examples:
        A slice covering the full server (fraction 0/1 to 1/1):

        >>> ServerSlice(server_rank=0, start_numerator=0, start_denominator=1,
        ...             end_numerator=1, end_denominator=1)

        A slice covering the first half of a server (fraction 0/2 to 1/2):

        >>> ServerSlice(server_rank=1, start_numerator=0, start_denominator=2,
        ...             end_numerator=1, end_denominator=2)
    """

    server_rank: int
    start_numerator: int
    start_denominator: int
    end_numerator: int
    end_denominator: int

    def slice_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Slice a tensor according to this server assignment.

        Converts the stored fractions to concrete indices using the tensor's
        length, then returns ``tensor[start_idx:end_idx]``.

        Args:
            tensor: The full data tensor from the server identified by
                :attr:`server_rank`.

        Returns:
            The sub-tensor belonging to this compute node.
        """
        total = len(tensor)
        start_idx = total * self.start_numerator // self.start_denominator
        end_idx = total * self.end_numerator // self.end_denominator
        if start_idx == 0 and end_idx == total:
            return tensor
        return tensor[start_idx:end_idx]


def compute_server_assignments(
    num_servers: int,
    num_compute_nodes: int,
    compute_rank: int,
) -> dict[int, ServerSlice]:
    """Compute which servers, and which fractions of them, belong to one compute node.

    Maps the range of servers onto the range of compute nodes using a
    segment-overlap algorithm.
    Each compute node "owns" a contiguous segment of length ``num_servers``
    on a number line of total length ``num_servers * num_compute_nodes``.
    Each server occupies a segment of length ``num_compute_nodes`` on that
    same line.
    The overlap between a compute node's segment and a server's segment
    determines the fraction of that server assigned to the compute node.

    Args:
        num_servers: Total number of storage servers.
        num_compute_nodes: Total number of compute nodes.
        compute_rank: The rank of the compute node to compute assignments for,
            in ``[0, num_compute_nodes)``.

    Returns:
        A dict mapping server rank to a :class:`ServerSlice` describing the
        fraction of that server's data owned by ``compute_rank``.
        Servers with no overlap are omitted from the dict.

    Raises:
        ValueError: If any argument is out of its valid range.

    Examples:
        With 2 servers and 2 compute nodes, each compute node gets one full server:

        >>> compute_server_assignments(num_servers=2, num_compute_nodes=2, compute_rank=0)
        {0: ServerSlice(server_rank=0, start_numerator=0, start_denominator=2,
                        end_numerator=2, end_denominator=2)}

        With 3 servers and 2 compute nodes, compute rank 0 gets all of server 0
        and the first half of server 1:

        >>> compute_server_assignments(num_servers=3, num_compute_nodes=2, compute_rank=0)
        {0: ServerSlice(..., 0/2..2/2), 1: ServerSlice(..., 0/2..1/2)}
    """
    if num_servers <= 0:
        raise ValueError(f"num_servers must be positive, got {num_servers}")
    if num_compute_nodes <= 0:
        raise ValueError(f"num_compute_nodes must be positive, got {num_compute_nodes}")
    if compute_rank < 0 or compute_rank >= num_compute_nodes:
        raise ValueError(
            f"compute_rank must be in [0, {num_compute_nodes}), got {compute_rank}"
        )

    # Each compute node owns a segment of length num_servers on a number line
    # of total length num_servers * num_compute_nodes.
    seg_start = compute_rank * num_servers
    seg_end = (compute_rank + 1) * num_servers

    assignments: dict[int, ServerSlice] = {}
    for server_rank in range(num_servers):
        # Each server occupies a segment of length num_compute_nodes.
        server_start = server_rank * num_compute_nodes
        server_end = (server_rank + 1) * num_compute_nodes

        # The overlap determines the fraction of this server for this compute node.
        overlap_start = max(seg_start, server_start)
        overlap_end = min(seg_end, server_end)
        if overlap_start >= overlap_end:
            continue

        assignments[server_rank] = ServerSlice(
            server_rank=server_rank,
            start_numerator=overlap_start - server_start,
            start_denominator=num_compute_nodes,
            end_numerator=overlap_end - server_start,
            end_denominator=num_compute_nodes,
        )

    return assignments
