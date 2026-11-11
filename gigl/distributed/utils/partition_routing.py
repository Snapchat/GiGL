"""Routing of ids to the partitions that own them, for distributed sampling and feature lookups.

GLT routes ids with one mask and two ``masked_select`` calls per partition. Each of those torch ops releases and
reacquires the GIL, so with many partitions the routing dominates the sampler's event-loop thread. Grouping the ids
once with a stable sort does the same routing in a handful of ops and yields identical ids in identical order.
"""

import os
from typing import Final

import torch

from gigl.common.logger import Logger

logger = Logger()

GROUPED_ROUTING_ENV: Final[str] = "GIGL_SAMPLING_GROUPED_ROUTING"

_ENABLED_VALUES: Final[frozenset[str]] = frozenset({"1", "true"})
_DISABLED_VALUES: Final[frozenset[str]] = frozenset({"", "0", "false"})

# Samplers and feature stores read the variable once each, in every worker, so only the first invalid value warns.
_invalid_value_warned: bool = False


def grouped_routing_enabled() -> bool:
    """Whether ``GIGL_SAMPLING_GROUPED_ROUTING`` asks for sort-based routing.

    ``1`` and ``true`` (any case) enable it, and unset, empty, ``0`` and ``false`` disable it. Any other value
    disables it and logs a warning the first time it is seen in this process.
    """
    global _invalid_value_warned
    raw = os.environ.get(GROUPED_ROUTING_ENV, "")
    value = raw.strip().lower()
    if value in _ENABLED_VALUES:
        return True
    if value not in _DISABLED_VALUES and not _invalid_value_warned:
        _invalid_value_warned = True
        logger.warning(
            f"{GROUPED_ROUTING_ENV}={raw!r} is not 1, true, 0 or false; grouped routing is off"
        )
    return False


def group_by_partition(
    ids: torch.Tensor, partition_ids: torch.Tensor, num_partitions: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Splits ``ids`` by owning partition with one stable sort.

    For every partition ``p`` the result equals ``masked_select(ids, partition_ids == p)`` and
    ``masked_select(arange(len(ids)), partition_ids == p)``, in the same order. The ids of a partition have a storage
    of their own, of the same size as ``masked_select`` gives, whereas the positions are views of one sorted
    permutation. A partition that owns none of the ids gets empty tensors.

    Args:
        ids: 1-D ids to route.
        partition_ids: 1-D owning partition of each id, with values in ``[0, num_partitions)``. A negative value
            raises, whereas GLT's mask routing silently drops such an id. A value of ``num_partitions`` or more is
            dropped by both, at a cost that does not grow with its size. A float that is not a whole number is
            dropped, as ``partition_ids == p`` is never true for it; a whole-number float routes like the integer.
        num_partitions: Number of partitions.

    Returns:
        The ids of each partition, and the positions of those ids in ``ids``. Both lists are indexed by partition.
    """
    # Partition books may hold floats, which bincount rejects.
    if partition_ids.is_floating_point():
        whole = partition_ids == partition_ids.trunc()
        partition_ids = torch.where(
            whole, partition_ids, torch.full_like(partition_ids, num_partitions)
        ).clamp(-1, num_partitions)
    # Every id that no partition owns is sent to the extra partition ``num_partitions``, which is dropped below.
    # This bounds the work by ``num_partitions`` instead of by the largest id.
    partition_ids = partition_ids.to(torch.long).clamp_max(num_partitions)
    perm = torch.argsort(partition_ids, stable=True)
    counts = torch.bincount(partition_ids, minlength=num_partitions + 1).tolist()
    # Torch RPC pickles the whole storage of a view, and TensorPipe only re-copies a storage that is at least
    # twice the bytes in use. Without a storage of their own, the ids of a partition could put up to twice their
    # bytes on the wire, e.g. all of the sorted ids when there are two partitions.
    ids_by_partition = []
    for chunk in ids[perm.to(ids.device)].split(counts)[:num_partitions]:
        ids_by_partition.append(chunk.clone())
    return ids_by_partition, list(perm.split(counts))[:num_partitions]
