"""Opt-in CPU affinity for the rank processes and sampling workers of one machine.

``GIGL_SAMPLING_CPU_PINNING`` selects the mode:

- ``none`` (default): no affinity calls.
- ``disjoint``: the CPUs this container may use are grouped by physical core (hyperthread
  siblings stay together) and divided into disjoint sets, one for every rank process and one for
  every sampling worker on the machine. A rank process takes a smaller share than a sampling
  worker; the cores left over go to sampling workers, one extra each, spread across ranks.

A rank process pins itself before it creates the sampling workers. The workers are spawned
processes, so they start with the rank's mask and re-pin themselves from the context the rank
process left in the environment. Threads inherit the mask of the thread that creates them, so the
pin has to happen before RPC and OpenMP threads are created.

Pinning never raises: when it cannot be applied it logs one warning and the process stays unpinned.
"""

import json
import os
from collections.abc import Sequence
from typing import Optional

from gigl.common.logger import Logger

logger = Logger()

CPU_PINNING_ENV = "GIGL_SAMPLING_CPU_PINNING"
CPU_PINNING_CONTEXT_ENV = "GIGL_SAMPLING_CPU_PINNING_CONTEXT"
_MODE_NONE = "none"
_MODE_DISJOINT = "disjoint"
_MODES = (_MODE_NONE, _MODE_DISJOINT)
_TOPOLOGY_DIR = "/sys/devices/system/cpu"


def cpu_pinning_mode() -> str:
    """The pinning mode from the environment (``none`` when unset)."""
    mode = os.environ.get(CPU_PINNING_ENV, _MODE_NONE).strip().lower() or _MODE_NONE
    if mode not in _MODES:
        raise ValueError(f"{CPU_PINNING_ENV} must be one of {_MODES}, got {mode!r}")
    return mode


def _parse_cpu_list(text: str) -> frozenset[int]:
    """Parses a kernel CPU list such as ``0-3,8,10-11``."""
    cpus: set[int] = set()
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            low, high = part.split("-")
            cpus.update(range(int(low), int(high) + 1))
        else:
            cpus.add(int(part))
    return frozenset(cpus)


def read_sibling_groups(allowed_cpus: frozenset[int]) -> list[frozenset[int]]:
    """Groups ``allowed_cpus`` by physical core using ``thread_siblings_list``.

    Raises:
        OSError: If a topology file cannot be read.
    """
    groups: set[frozenset[int]] = set()
    for cpu in sorted(allowed_cpus):
        path = f"{_TOPOLOGY_DIR}/cpu{cpu}/topology/thread_siblings_list"
        with open(path) as f:
            siblings = _parse_cpu_list(f.read())
        groups.add(siblings & allowed_cpus)
    return sorted(groups, key=min)


def split_cpus_for_process(
    allowed_cpus: frozenset[int],
    sibling_groups: Sequence[frozenset[int]],
    local_rank: int,
    num_local_ranks: int,
    worker_index: Optional[int],
    workers_per_rank: int,
) -> Optional[frozenset[int]]:
    """The CPUs one process gets under ``disjoint`` pinning.

    The cores (sibling groups inside ``allowed_cpus``, ordered by lowest CPU) are handed out in
    process order: rank 0, rank 0's workers, rank 1, rank 1's workers, and so on. Each rank
    process gets ``max(1, cores // (2 * processes))`` cores and every sampling worker an equal
    share of the rest. The cores that do not divide evenly go one each to the first workers in
    worker-index-major order (worker 0 of every rank, then worker 1 of every rank), which spreads
    them across ranks.

    Args:
        allowed_cpus: CPUs the container may use.
        sibling_groups: Hyperthread sibling groups (see ``read_sibling_groups``).
        local_rank: Rank of this process among the ranks on the machine.
        num_local_ranks: Number of rank processes on the machine.
        worker_index: Index of the sampling worker within its rank, or ``None`` for the rank
            process itself.
        workers_per_rank: Sampling workers per rank process.

    Returns:
        The CPU set for the process, or ``None`` when there are fewer cores than processes.
    """
    cores = [
        group for group in sorted(sibling_groups, key=min) if group <= allowed_cpus
    ]
    num_processes = num_local_ranks * (1 + workers_per_rank)
    if len(cores) < num_processes:
        return None
    num_workers = num_local_ranks * workers_per_rank
    if num_workers == 0:
        rank_cores = len(cores) // num_local_ranks
        extra_workers = 0
        worker_cores = 0
    else:
        rank_cores = max(1, len(cores) // (2 * num_processes))
        remaining = len(cores) - rank_cores * num_local_ranks
        worker_cores, extra_workers = divmod(remaining, num_workers)

    start = 0
    for rank in range(num_local_ranks):
        sizes = [rank_cores]
        for index in range(workers_per_rank):
            global_order = index * num_local_ranks + rank
            sizes.append(worker_cores + int(global_order < extra_workers))
        slot = 0
        if worker_index is not None:
            slot = 1 + worker_index
        if rank == local_rank:
            begin = start + sum(sizes[:slot])
            chosen = cores[begin : begin + sizes[slot]]
            return frozenset(cpu for group in chosen for cpu in group)
        start += sum(sizes)
    return None


def _format_cpus(cpus: Sequence[int]) -> str:
    return ",".join(str(cpu) for cpu in sorted(cpus))


def apply_cpu_pinning(
    mode: str,
    local_rank: int,
    worker_index: Optional[int],
    num_local_ranks: int,
    workers_per_rank: int,
    allowed_cpus: Optional[frozenset[int]] = None,
) -> Optional[frozenset[int]]:
    """Pins the calling process (``os.sched_setaffinity(0, ...)``) according to ``mode``.

    A rank process (``worker_index=None``) also records the original allowed CPUs and the
    process counts in the environment so the sampling workers it spawns can pin themselves with
    ``apply_worker_cpu_pinning``. Repeated calls in a rank process keep the first recording.

    Args:
        mode: ``none`` or ``disjoint``.
        local_rank: Rank of this process among the ranks on the machine.
        worker_index: Sampling worker index within the rank, or ``None`` for the rank process.
        num_local_ranks: Number of rank processes on the machine.
        workers_per_rank: Sampling workers per rank process.
        allowed_cpus: CPUs the container may use; defaults to the current affinity mask.

    Returns:
        The CPU set the process is now pinned to, or ``None`` if it was left unpinned.
    """
    if mode not in _MODES:
        raise ValueError(f"CPU pinning mode must be one of {_MODES}, got {mode!r}")
    if mode == _MODE_NONE:
        return None
    try:
        if allowed_cpus is None:
            allowed_cpus = frozenset(os.sched_getaffinity(0))
        if worker_index is None:
            context = os.environ.get(CPU_PINNING_CONTEXT_ENV)
            if context is not None:
                allowed_cpus = frozenset(json.loads(context)["allowed"])
            else:
                os.environ[CPU_PINNING_CONTEXT_ENV] = json.dumps(
                    {
                        "allowed": sorted(allowed_cpus),
                        "local_rank": local_rank,
                        "num_local_ranks": num_local_ranks,
                        "workers_per_rank": workers_per_rank,
                    }
                )
        groups = read_sibling_groups(allowed_cpus)
        cpus = split_cpus_for_process(
            allowed_cpus,
            groups,
            local_rank,
            num_local_ranks,
            worker_index,
            workers_per_rank,
        )
        if cpus is None:
            logger.warning(
                f"CPU_PINNING skipped reason=fewer_cores_than_processes mode={mode} "
                f"pid={os.getpid()} allowed={len(allowed_cpus)} cores={len(groups)}"
            )
            return None
        os.sched_setaffinity(0, cpus)
    except (OSError, ValueError, KeyError, AttributeError) as e:
        logger.warning(
            f"CPU_PINNING skipped reason={type(e).__name__}: {e} mode={mode} pid={os.getpid()}"
        )
        return None
    logger.info(
        f"CPU_PINNING mode={mode} pid={os.getpid()} cpus={_format_cpus(sorted(cpus))}"
    )
    return cpus


def apply_worker_cpu_pinning(worker_index: int) -> Optional[frozenset[int]]:
    """Pins a sampling worker from the context its rank process recorded.

    A no-op when the mode is ``none`` or the rank process did not pin.
    """
    mode = cpu_pinning_mode()
    context = os.environ.get(CPU_PINNING_CONTEXT_ENV)
    if mode == _MODE_NONE or context is None:
        return None
    parsed = json.loads(context)
    return apply_cpu_pinning(
        mode,
        local_rank=parsed["local_rank"],
        worker_index=worker_index,
        num_local_ranks=parsed["num_local_ranks"],
        workers_per_rank=parsed["workers_per_rank"],
        allowed_cpus=frozenset(parsed["allowed"]),
    )
