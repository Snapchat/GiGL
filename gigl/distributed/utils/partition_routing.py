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


def grouped_routing_enabled() -> bool:
    """Whether ``GIGL_SAMPLING_GROUPED_ROUTING`` asks for sort-based routing.

    ``1`` and ``true`` (any case) enable it, and unset, empty, ``0`` and ``false`` disable it. Any other value
    disables it with a warning.
    """
    raw = os.environ.get(GROUPED_ROUTING_ENV, "")
    value = raw.strip().lower()
    if value in _ENABLED_VALUES:
        return True
    if value not in _DISABLED_VALUES:
        logger.warning(
            f"{GROUPED_ROUTING_ENV}={raw!r} is not 1, true, 0 or false; grouped routing is off"
        )
    return False


def group_by_partition(
    ids: torch.Tensor, partition_ids: torch.Tensor, num_partitions: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Splits ``ids`` by owning partition with one stable sort.

    For every partition ``p`` the result equals ``masked_select(ids, partition_ids == p)`` and
    ``masked_select(arange(len(ids)), partition_ids == p)``, in the same order and with storage of the same size. A
    partition that owns none of the ids gets empty tensors.

    Args:
        ids: 1-D ids to route.
        partition_ids: 1-D owning partition of each id, with values in ``[0, num_partitions)``. A negative value
            raises, whereas GLT's mask routing silently drops such an id. A value of ``num_partitions`` or more is
            dropped by both.
        num_partitions: Number of partitions.

    Returns:
        The ids of each partition, and the positions of those ids in ``ids``. Both lists are indexed by partition.
    """
    # Partition books may hold floats, which bincount rejects.
    partition_ids = partition_ids.to(torch.long)
    perm = torch.argsort(partition_ids, stable=True)
    counts = torch.bincount(partition_ids, minlength=num_partitions).tolist()
    # Torch RPC pickles the whole storage of a view, and TensorPipe only re-copies a storage that is at least
    # twice the bytes in use. Without a storage of their own, the ids of a partition could put up to twice their
    # bytes on the wire, e.g. all of the sorted ids when there are two partitions.
    ids_by_partition = []
    for chunk in ids[perm.to(ids.device)].split(counts):
        ids_by_partition.append(chunk.clone())
    return ids_by_partition, list(perm.split(counts))
