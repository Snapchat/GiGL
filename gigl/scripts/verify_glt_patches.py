"""Assert that the patches in ``gigl/scripts/patches/`` are live in the INSTALLED graphlearn_torch.

Run by ``install_glt.sh`` immediately after ``uv pip install dist/*.whl``, and exits non-zero if
any patched behaviour is missing.

WHY THIS EXISTS
    ``install_glt.sh`` already fails loudly if a patch file does not APPLY. That is a weaker
    guarantee than it looks: applying is a property of the source tree, while what ships is a
    compiled ``.so``. A patch can apply to a file the build then excludes, a stale build directory
    can be reused, or a wheel can be installed from somewhere other than the tree that was
    patched -- and every one of those failures is silent, because the unpatched code paths work.
    They just work at 3x the memory (patch 0001) or reject the int32 topology the trainer is
    about to build (patch 0002), several hours into a 16-GPU job.

    The unit tests in ``tests/unit/utils/glt_int32_indices_test.py`` cover the same ground in far
    more detail, but they SKIP on an unpatched wheel by design -- GiGL's CI runs against the
    released graphlearn_torch, where failing would be wrong. So the test suite cannot be the gate
    for the image, and this script cannot be the detailed test. Both exist, deliberately.

    Deliberately dependency-free (torch and graphlearn_torch only): it runs inside the image build
    before the rest of the repo's test dependencies are necessarily importable.

Usage:
    python gigl/scripts/verify_glt_patches.py
"""

import os
import sys

import torch
from graphlearn_torch.data import Graph, Topology

# 0001 replaces at::_unique with a bitmap count; a column id beyond the id domain the bitmap can
# describe must fall back rather than allocate, and a negative id must raise rather than write out
# of bounds. Both are patch-only behaviours: upstream accepts them silently.
_NEGATIVE_COLUMN_ID = -1


def _cpu_graph(indptr: torch.Tensor, indices: torch.Tensor) -> Graph:
    topology = Topology.__new__(Topology)
    topology._layout = "CSR"
    topology._indptr = indptr
    topology._indices = indices
    topology._edge_ids = None
    topology._edge_weights = None
    graph = Graph.__new__(Graph)
    graph.topo = topology
    graph.mode = "CPU"
    graph.device = None
    graph._graph = None
    graph.lazy_init()
    return graph


def _check(name: str, passed: bool, detail: str = "") -> bool:
    print(f"  [{'OK  ' if passed else 'FAIL'}] {name}{': ' + detail if detail else ''}")
    return passed


def verify_0001_bitmap_col_count() -> bool:
    """The hardened distinct-count: exact on valid input, loud on invalid input."""
    indptr = torch.tensor([0, 2, 3, 5], dtype=torch.int64)
    indices = torch.tensor([7, 3, 7, 1, 4], dtype=torch.int64)
    graph = _cpu_graph(indptr, indices)
    exact = _check(
        "0001 col_count is exact",
        graph.col_count == int(torch.unique(indices).numel()),
        f"{graph.col_count} vs {int(torch.unique(indices).numel())}",
    )

    # Upstream's _unique accepts a negative id; the patched count must reject it, because casting
    # it to an unsigned bitmap offset would write outside the allocation.
    rejects_negative = False
    try:
        _cpu_graph(
            torch.tensor([0, 1], dtype=torch.int64),
            torch.tensor([_NEGATIVE_COLUMN_ID], dtype=torch.int64),
        )
    except RuntimeError as error:
        rejects_negative = "non-negative" in str(error)
    rejected = _check(
        "0001 rejects a negative column id",
        rejects_negative,
        "an unpatched wheel accepts this",
    )

    # Same reasoning for a strided view: every sampler reads the column array as flat storage.
    strided = torch.arange(8, dtype=torch.int64)[::2]
    rejects_strided = False
    try:
        _cpu_graph(torch.tensor([0, 2, 4], dtype=torch.int64), strided)
    except RuntimeError as error:
        rejects_strided = "contiguous" in str(error)
    contiguous = _check(
        "0001 rejects non-contiguous indices",
        rejects_strided,
        "an unpatched wheel accepts this",
    )
    return exact and rejected and contiguous


def verify_0002_int32_indices() -> bool:
    """int32 columns must be accepted AND sample identically to int64."""
    indptr = torch.tensor([0, 3, 3, 6], dtype=torch.int64)
    columns = [5, 2, 9, 1, 7, 4]
    indices64 = torch.tensor(columns, dtype=torch.int64)
    indices32 = torch.tensor(columns, dtype=torch.int32)

    try:
        graph32 = _cpu_graph(indptr, indices32)
    except RuntimeError as error:
        return _check("0002 accepts int32 indices", False, str(error).splitlines()[0])
    accepted = _check("0002 accepts int32 indices", True)

    graph64 = _cpu_graph(indptr, indices64)
    counts_match = _check(
        "0002 col_count matches the int64 graph",
        graph32.col_count == graph64.col_count,
        f"{graph32.col_count} vs {graph64.col_count}",
    )

    # Full fanout (req_num > max degree) makes UniformSample copy rather than draw, so the two
    # graphs are comparable id-by-id. This is the property whose failure is SILENT.
    from graphlearn_torch import py_graphlearn_torch as pywrap

    seeds = torch.tensor([0, 2], dtype=torch.int64)
    neighbors64, degrees64 = pywrap.CPURandomSampler(graph64.graph_handler).sample(
        seeds, 8
    )
    neighbors32, degrees32 = pywrap.CPURandomSampler(graph32.graph_handler).sample(
        seeds, 8
    )
    identical = _check(
        "0002 samples identically to the int64 graph",
        bool(torch.equal(neighbors64, neighbors32))
        and bool(torch.equal(degrees64, degrees32)),
        f"{neighbors32.tolist()} vs {neighbors64.tolist()}",
    )
    int64_out = _check(
        "0002 sampled ids stay int64",
        neighbors32.dtype == torch.int64,
        str(neighbors32.dtype),
    )

    # A sampler that was not taught the layout must raise, not read the nullptr col_idx_.
    rejects = False
    try:
        pywrap.CPUWeightedSampler(graph32.graph_handler).sample(seeds, 2)
    except RuntimeError as error:
        rejects = "int32" in str(error)
    guarded = _check(
        "0002 untaught samplers reject int32 loudly",
        rejects,
        "a silent nullptr read would be the alternative",
    )
    return accepted and counts_match and identical and int64_out and guarded


def _sysv_segments_created_by_this_process() -> int:
    # /proc/sysvipc/shm is system-wide; filtering on the creator pid keeps the count immune to
    # whatever else the build host is doing.
    own_pid = str(os.getpid())
    count = 0
    with open("/proc/sysvipc/shm") as shm_table:
        next(shm_table)
        for line in shm_table:
            fields = line.split()
            if len(fields) > 4 and fields[4] == own_pid:
                count += 1
    return count


def verify_0003_queue_teardown() -> bool:
    """The teardown-unpin patch must be COMPILED IN, and teardown must stay leak-free.

    The runtime payload -- the deleter calling cudaHostUnregister for a mapping this process
    pinned -- cannot be exercised without a CUDA device, and image builds have none. Worse,
    ``pin_memory()`` on a driverless host does not raise: GLT's ``CUDACheckError`` calls
    ``exit(EXIT_FAILURE)``, which would kill this verifier and the build with it. So this
    check NEVER pins unless a device is actually available. Presence of the patched BINARY is
    proven by a compiled capability marker instead (``supports_unpin_on_teardown``, added by
    the same patch to the same .so as the deleter); the unregister BEHAVIOUR is proven by the
    GPU canary probe (pin -> destroy -> repin cycles), which fails on an unpatched wheel.

    What runs everywhere: the deleter still detaches and removes the SysV segment across
    repeated cycles, and the zero-copy ShmData path still holds the mapping open while a
    dequeued message is in flight.
    """
    import pickle

    from graphlearn_torch import py_graphlearn_torch as pywrap

    marker = getattr(pywrap.SampleQueue, "supports_unpin_on_teardown", None)
    compiled_in = _check(
        "0003 teardown-unpin capability is compiled in",
        marker is not None and bool(marker()),
        "an unpatched wheel lacks this binding",
    )
    if not compiled_in:
        return False

    cuda_usable = torch.cuda.is_available()

    def one_cycle() -> None:
        queue = pywrap.SampleQueue(8, 1 << 20)
        if cuda_usable:
            queue.pin_memory()
            queue.pin_memory()  # second call must be a no-op, not a re-register error
        queue.send({"ids": torch.arange(64, dtype=torch.int64)})
        consumer = pickle.loads(pickle.dumps(queue))
        message = consumer.receive(5000)
        assert torch.equal(message["ids"], torch.arange(64, dtype=torch.int64))
        del message, consumer, queue

    # The leak check is only meaningful if the pid-filtered /proc view can SEE this
    # process's segments at all (an unusual /proc mount from another pid namespace would
    # count zero forever and report any leak as clean). Measure the DELTA the sentinel
    # causes, not an absolute count -- a pre-existing segment attributed to the same
    # numeric pid must not vouch for an invisible sentinel.
    count_before_sentinel = _sysv_segments_created_by_this_process()
    sentinel = pywrap.SampleQueue(8, 1 << 20)
    count_with_sentinel = _sysv_segments_created_by_this_process()
    del sentinel
    count_after_sentinel = _sysv_segments_created_by_this_process()
    observable = _check(
        "0003 pid filter observes a live segment",
        count_with_sentinel == count_before_sentinel + 1
        and count_after_sentinel == count_before_sentinel,
        f"{count_before_sentinel} -> {count_with_sentinel} -> {count_after_sentinel}",
    )
    if not observable:
        return False

    one_cycle()  # torch lazily allocates a process-level shm singleton on first send
    baseline = _sysv_segments_created_by_this_process()
    for _ in range(20):
        one_cycle()
    leaked = _sysv_segments_created_by_this_process() - baseline
    no_leak = _check(
        "0003 teardown cycles leak no shm segments",
        leaked <= 0,
        f"{leaked} leaked over 20 cycles"
        if leaked > 0
        else f"20 cycles clean ({'pinned' if cuda_usable else 'unpinned; no CUDA device'})",
    )

    # Dequeue is zero-copy: the received tensors view the segment through ShmData. Destroying
    # BOTH queue objects while the message is still held must leave the data readable -- this
    # ordering is exactly why the unpin lives in the shared_ptr deleter, not in ~ShmQueue.
    producer = pywrap.SampleQueue(8, 1 << 20)
    producer.send({"ids": torch.arange(64, dtype=torch.int64)})
    consumer = pickle.loads(pickle.dumps(producer))
    in_flight = consumer.receive(5000)
    del producer, consumer
    survives = _check(
        "0003 in-flight message survives queue teardown",
        bool(torch.equal(in_flight["ids"], torch.arange(64, dtype=torch.int64))),
    )
    del in_flight
    return no_leak and survives


def main() -> int:
    print(f"Verifying GLT patches against {Graph.__module__}")
    results = {
        "0001-glt-cpu-graph-col-count-bitmap": verify_0001_bitmap_col_count(),
        "0002-glt-int32-csr-indices": verify_0002_int32_indices(),
        "0003-glt-unpin-shm-queue-on-teardown": verify_0003_queue_teardown(),
    }
    failed = [name for name, passed in results.items() if not passed]
    if failed:
        print(
            f"\nFATAL: {failed} did not take effect in the installed graphlearn_torch. The patch "
            f"files applied to the source tree, so the wheel that got INSTALLED is not the one "
            f"that was built from it -- check for a stale build directory or a second wheel on "
            f"the path. Shipping this image would OOM (0001) or reject the int32 topology the "
            f"trainer builds (0002), hours into a multi-GPU job."
        )
        return 1
    print("\nAll GLT patches verified live in the installed graphlearn_torch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
