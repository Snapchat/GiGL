"""Routing of ids to the partitions that own them, for distributed sampling and feature lookups.

GLT routes ids with one mask and two ``masked_select`` calls per partition. Each of those torch ops releases and
reacquires the GIL, so with many partitions the routing dominates the sampler's event-loop thread. Grouping the ids
once with a stable sort does the same routing in a handful of ops and yields identical ids in identical order.

``group_by_partition`` replaces the per-partition mask loops of two GLT functions, which GiGL copies and edits:

- ``graphlearn_torch.distributed.dist_neighbor_sampler.DistNeighborSampler._sample_one_hop``, copied as
  ``gigl.distributed.base_sampler.BaseDistNeighborSampler._sample_one_hop``.
- ``graphlearn_torch.distributed.dist_feature.DistFeature._remote_selecting_get``, copied as
  ``gigl.distributed.utils.fused_feature.FusedDistFeature._remote_selecting_get``.

The other mask loops in GLT's ``DistFeature`` (``async_get``, its all-to-all path) are not replaced.
"""

import torch
from jaxtyping import Int64, Integer


def group_by_partition(
    ids: Integer[torch.Tensor, "num_ids"],
    partition_ids: Integer[torch.Tensor, "num_ids"],
    num_partitions: int,
) -> tuple[
    list[Integer[torch.Tensor, "num_selected"]],
    list[Int64[torch.Tensor, "num_selected"]],
]:
    """Splits ``ids`` by owning partition with one stable sort.

    For every partition ``p`` the result equals ``masked_select(ids, partition_ids == p)`` and
    ``masked_select(arange(len(ids)), partition_ids == p)``, in the same order. The ids of a partition have a storage
    of their own, of the same size as ``masked_select`` gives, whereas the positions are views of one sorted
    permutation. A partition that owns none of the ids gets empty tensors.

    Args:
        ids: Ids to route.
        partition_ids: Owning partition of each id. A value that is negative or is ``num_partitions`` or more
            matches no partition, so the id is dropped, as ``partition_ids == p`` is never true for it. Dropped ids
            cost the same as any other id, whatever their value.
        num_partitions: Number of partitions.

    Returns:
        The ids of each partition, and the positions of those ids in ``ids``. Both lists are indexed by partition.
    """
    # Every id that no partition owns is sent to the extra partition ``num_partitions``, which is dropped below.
    # This bounds the work by ``num_partitions`` instead of by the largest id.
    partition_ids = partition_ids.to(torch.long)
    partition_ids = partition_ids.masked_fill(
        (partition_ids < 0) | (partition_ids > num_partitions), num_partitions
    )
    perm: Int64[torch.Tensor, "num_ids"] = torch.argsort(partition_ids, stable=True)
    counts: list[int] = torch.bincount(
        partition_ids, minlength=num_partitions + 1
    ).tolist()
    # Torch RPC pickles the whole storage of a view, and TensorPipe only re-copies a storage that is at least
    # twice the bytes in use. Without a storage of their own, the ids of a partition could put up to twice their
    # bytes on the wire, e.g. all of the sorted ids when there are two partitions.
    ids_by_partition: list[Integer[torch.Tensor, "num_selected"]] = []
    for chunk in ids[perm.to(ids.device)].split(counts)[:num_partitions]:
        ids_by_partition.append(chunk.clone())
    index_by_partition: list[Int64[torch.Tensor, "num_selected"]] = list(
        perm.split(counts)
    )[:num_partitions]
    return ids_by_partition, index_by_partition
