"""How much host memory this process may still allocate.

``psutil.virtual_memory().available`` reads ``/proc/meminfo``, which inside a container reports the
HOST's memory rather than the container's limit, so a budget check trusting it alone can pass
immediately before the container is killed. The cgroup knows the real limit; read both and believe
the smaller.

Most callers want :func:`available_memory_bytes`. The public API:

- :func:`available_memory_bytes` -- bytes this process can still allocate; the number to budget against.
- :func:`cgroup_limit_and_usage` -- the binding cgroup ``(limit, current)`` in bytes, or None when unlimited.
- :func:`cgroup_memory_breakdown` -- current usage split by reclaimability (anon, shmem, file, dirty, writeback).
- :func:`log_stage_memory` -- log the memory position after a pipeline stage.
- :func:`describe_memory` -- one line covering both views, for logs where a budget is decided.

Everything else in the module is the ``/proc`` and cgroup plumbing behind those five.
"""

import os
from typing import Optional

import psutil

from gigl.common.logger import Logger

logger = Logger()

# cgroup v1 spells "unlimited" as a sentinel near the top of the int64/page range rather than with a
# word, so anything at or above this is treated as no limit.
_CGROUP_V1_UNLIMITED_THRESHOLD = 1 << 62


def _read_int(path: str) -> Optional[int]:
    try:
        with open(path) as handle:
            text = handle.read().strip()
    except OSError:
        return None
    if text == "max":
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    return None if value >= _CGROUP_V1_UNLIMITED_THRESHOLD else value


def _cgroup_paths() -> list[str]:
    """This process's cgroup path per controller, from ``/proc/self/cgroup``.

    Reading ``/sys/fs/cgroup/memory.max`` directly is wrong: that is the hierarchy root, which has no
    limit, while the process lives below it.

    Returns the paths to try, most specific first, each expanded up its ancestors, ending at the
    root.
    """
    paths: list[str] = []
    try:
        with open("/proc/self/cgroup") as handle:
            for line in handle:
                # Each line is `hierarchy-id:controllers:path`. v2 lines are `0::<path>`; v1 lines
                # name their controllers, and only the one carrying `memory` is relevant here:
                #   0::/kubepods/burstable/pod1234/5678
                #   9:memory:/docker/5678
                fields = line.strip().split(":", 2)
                if len(fields) != 3:
                    continue
                hierarchy_id, controllers, path = fields
                if hierarchy_id == "0" or "memory" in controllers.split(","):
                    paths.append(path or "/")
    except OSError:
        pass
    paths.append("/")
    # A container may be limited at an ancestor rather than at its own leaf, so walk upward.
    expanded: list[str] = []
    for path in paths:
        parts = [part for part in path.split("/") if part]
        while True:
            candidate = "/" + "/".join(parts)
            if candidate not in expanded:
                expanded.append(candidate)
            if not parts:
                break
            parts.pop()
    return expanded


def _cgroup_mounts() -> list[tuple[str, str, str]]:
    """``(mount root, mount point, type)`` for every cgroup mount, from ``/proc/self/mountinfo``.

    The mount root is not always ``/``: a container often bind-mounts its own subtree, so the path
    from ``/proc/self/cgroup`` is relative to that root and joining the two naively yields a
    directory that does not exist.
    """
    mounts: list[tuple[str, str, str]] = []
    try:
        with open("/proc/self/mountinfo") as handle:
            for line in handle:
                fields = line.split(" - ")
                if len(fields) != 2:
                    continue
                before = fields[0].split()
                if len(before) < 5:
                    continue
                mount_root, mount_point = before[3], before[4]
                filesystem = fields[1].split()[0]
                if filesystem in ("cgroup2", "cgroup"):
                    mounts.append((mount_root, mount_point, filesystem))
    except OSError:
        pass
    if not mounts:
        mounts = [("/", "/sys/fs/cgroup", "cgroup2")]
    return mounts


def _relative_to_mount_root(path: str, mount_root: str) -> Optional[str]:
    """Translate a cgroup path into one relative to ``mount_root``, or None if it is outside it."""
    if mount_root in ("", "/"):
        return path
    if path == mount_root:
        return "/"
    if path.startswith(mount_root.rstrip("/") + "/"):
        return path[len(mount_root.rstrip("/")) :]
    return None


def _tightest_cgroup() -> Optional[tuple[int, int, str, str]]:
    """``(limit, current, directory, filesystem)`` for the cgroup with the least headroom.

    ``None`` when no limit can be read, the normal case on an unconstrained host.

    The directory is returned so ``memory.stat`` can be read from the same cgroup as the limit;
    mixing levels yields a breakdown that does not add up to the usage. The filesystem is returned
    because a v1 ancestor's ``memory.usage_in_bytes`` includes descendants while its bare
    ``rss``/``cache``/``dirty`` fields do not -- the hierarchical figures live under ``total_*``.
    """
    mounts = _cgroup_mounts()
    tightest: Optional[tuple[int, int, str, str]] = None
    for path in _cgroup_paths():
        for mount_root, mount_point, filesystem in mounts:
            relative = _relative_to_mount_root(path, mount_root)
            if relative is None:
                continue
            base = f"{mount_point}{relative}".rstrip("/") or mount_point
            if filesystem == "cgroup2":
                candidates = [(f"{base}/memory.max", f"{base}/memory.current", base)]
            else:
                # v1 puts each controller in its own directory under the mount point.
                controller_base = f"{mount_point}/memory{relative}".replace("//", "/")
                candidates = [
                    (
                        f"{controller_base}/memory.limit_in_bytes",
                        f"{controller_base}/memory.usage_in_bytes",
                        controller_base,
                    ),
                    (
                        f"{base}/memory.limit_in_bytes",
                        f"{base}/memory.usage_in_bytes",
                        base,
                    ),
                ]
            for max_path, current_path, directory in candidates:
                limit = _read_int(max_path)
                if limit is None:
                    continue
                current = _read_int(current_path)
                if current is None:
                    continue
                if tightest is None or (limit - current) < (tightest[0] - tightest[1]):
                    tightest = (limit, current, directory, filesystem)
                break
    return tightest


def cgroup_limit_and_usage() -> Optional[tuple[int, int]]:
    """The tightest ``(limit, current)`` in bytes over every cgroup constraining this process.

    Every ancestor constrains the process, so the first finite limit found walking upward is not
    necessarily the binding one: a leaf with 63 GiB of headroom under a parent with 1 GiB is really
    limited to 1 GiB. All applicable levels are read and the least headroom wins.

    Returns None when no limit can be read.
    """
    tightest = _tightest_cgroup()
    return None if tightest is None else (tightest[0], tightest[1])


# v2 name -> v1 name. Only the fields that decide whether a peak is survivable are read. The v1
# hierarchical `total_*` variants are preferred, because `_tightest_cgroup` can select an ancestor
# whose usage includes descendants that its bare fields exclude. v2's memory.stat is already
# hierarchical.
_BREAKDOWN_FIELDS = {
    "anon": "rss",
    "file": "cache",
    "shmem": "shmem",
    "file_dirty": "dirty",
    "file_writeback": "writeback",
}


def cgroup_memory_breakdown() -> dict[str, int]:
    """``memory.stat`` split into the categories that behave differently under pressure.

    ``memory.current`` alone cannot say whether a peak is survivable: ``file_dirty`` bytes are
    reclaimable once written back, so the kernel throttles the writer, while the same fraction held
    as ``anon`` is fatal.

    Returns an empty dict when no cgroup or no ``memory.stat`` can be read.
    """
    tightest = _tightest_cgroup()
    if tightest is None:
        return {}
    try:
        with open(f"{tightest[2]}/memory.stat") as handle:
            raw = dict(
                (fields[0], int(fields[1]))
                for fields in (line.split() for line in handle)
                if len(fields) == 2 and fields[1].isdigit()
            )
    except OSError:
        return {}
    filesystem = tightest[3]
    breakdown: dict[str, int] = {}
    for v2_name, v1_name in _BREAKDOWN_FIELDS.items():
        if filesystem == "cgroup2":
            if v2_name in raw:
                breakdown[v2_name] = raw[v2_name]
            continue
        # v1: the hierarchical figure first, because the usage beside it is hierarchical too.
        if f"total_{v1_name}" in raw:
            breakdown[v2_name] = raw[f"total_{v1_name}"]
        elif v1_name in raw:
            breakdown[v2_name] = raw[v1_name]
    return breakdown


def log_stage_memory(stage: str) -> None:
    """Log the memory position after a pipeline stage, split by reclaimability.

    Reads three small proc files, so call it at phase boundaries rather than in loops.

    Args:
        stage: What just finished, e.g. ``"assembled node features"``.
    """
    parts = [f"[mem] after {stage}:"]
    limits = cgroup_limit_and_usage()
    if limits is None:
        parts.append("cgroup=unlimited")
    else:
        limit, current = limits
        # A cgroup can report a limit of 0; skip the percentage rather than divide by it.
        percent = f"({100.0 * current / limit:.0f}%) " if limit else ""
        parts.append(
            f"cgroup {current / 2**30:.1f}/{limit / 2**30:.1f} GiB "
            f"{percent}headroom {(limit - current) / 2**30:.1f} GiB"
        )
    breakdown = cgroup_memory_breakdown()
    if breakdown:
        # anon is the number that kills; dirty is the number that merely slows things down.
        parts.append(
            "| anon {:.1f} shmem {:.1f} file {:.1f} (dirty {:.1f}, writeback {:.1f}) GiB".format(
                breakdown.get("anon", 0) / 2**30,
                breakdown.get("shmem", 0) / 2**30,
                breakdown.get("file", 0) / 2**30,
                breakdown.get("file_dirty", 0) / 2**30,
                breakdown.get("file_writeback", 0) / 2**30,
            )
        )
    parts.append(
        f"| rss {psutil.Process(os.getpid()).memory_info().rss / 2**30:.1f} GiB"
    )
    logger.info(" ".join(parts))


def available_memory_bytes() -> int:
    """Bytes this process can still allocate, as the minimum of OS-free and cgroup-remaining.

    Both are needed: the OS figure misses a container limit below the machine's RAM, and the cgroup
    figure misses pressure from other processes on a shared host.
    """
    available = int(psutil.virtual_memory().available)
    limits = cgroup_limit_and_usage()
    if limits is None:
        return available
    limit, current = limits
    return min(available, max(limit - current, 0))


def describe_memory() -> str:
    """One line covering both views, for logging where a budget is being decided."""
    virtual = psutil.virtual_memory()
    parts = [
        f"meminfo total={virtual.total / 2**30:.1f} GiB "
        f"available={virtual.available / 2**30:.1f} GiB",
        f"rss={psutil.Process(os.getpid()).memory_info().rss / 2**30:.1f} GiB",
    ]
    limits = cgroup_limit_and_usage()
    if limits is None:
        parts.append("cgroup=unlimited")
    else:
        limit, current = limits
        parts.append(
            f"cgroup limit={limit / 2**30:.1f} GiB current={current / 2**30:.1f} GiB "
            f"headroom={(limit - current) / 2**30:.1f} GiB"
        )
    parts.append(f"effective available={available_memory_bytes() / 2**30:.1f} GiB")
    return " | ".join(parts)
