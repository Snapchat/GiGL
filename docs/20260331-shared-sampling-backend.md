# Shared Sampling Backend for Graph-Store Mode

## Context

On `origin/main`, graph-store sampling uses a **per-producer model**: each `(worker_key, sampler_input)` pair creates a
separate `DistSamplingProducer` on each storage node. This is wasteful — each producer spawns its own worker processes,
even when all compute ranks participating in one logical dataloader instantiation share the same sampling configuration.
The expensive producer init (~20-28s) is repeated per worker_key.

This plan replaces the per-producer model with a **shared backend + per-channel model**:

- A `SharedDistSamplingBackend` is created once per dataloader instantiation (`backend_key`)
- All compute ranks participating in that dataloader register their inputs as lightweight channels on the shared backend
- The backend's worker processes handle all channels concurrently via a fair-queued scheduler

This is implemented through a **two-phase initialization**:

1. `init_sampling_backend(backend_key)` — create/reuse expensive backend (one per dataloader instantiation)
2. `register_sampling_input(backend_id, worker_key, input)` — register lightweight channel

### What exists on `origin/main` today

| File                            | Lines   | What it has                                                                                                           |
| ------------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------- |
| `dist_server.py`                | 671     | `DistServer` with per-producer model: `_producer_pool`, `_worker_key2producer_id`, `create_sampling_producer()`, etc. |
| `dist_sampling_producer.py`     | 275     | `DistSamplingProducer(DistMpSamplingProducer)` — single-input producer, `_sampling_worker_loop()`                     |
| `base_dist_loader.py`           | 844     | Single-phase graph-store init: leader election → `create_sampling_producer` RPCs → `RemoteReceivingChannel`           |
| `remote_channel.py`             | **n/a** | Does not exist. Uses GLT's `RemoteReceivingChannel`                                                                   |
| `distributed_neighborloader.py` | ~580    | Passes `DistServer.create_sampling_producer` as callable to `BaseDistLoader`                                          |
| `dist_ablp_neighborloader.py`   | ~640    | Same pattern as DistNeighborLoader for ABLP                                                                           |
| `compute.py`                    | ~160    | `async_request_server`, `request_server`, RPC init                                                                    |
| `utils/neighborloader.py`       | ~267    | `SamplingClusterSetup`, `DatasetSchema`, `patch_fanout_for_sampling`, etc.                                            |

### What needs to be created

1. **`SharedDistSamplingBackend`** + **`_shared_sampling_worker_loop()`** — multi-channel backend (in
   `dist_sampling_producer.py`)
2. **`RemoteReceivingChannel`** — GiGL's own receiving channel (new file `remote_channel.py`)
3. **Two-phase DistServer** — `init_sampling_backend` + `register_sampling_input` with `SamplingBackendState` (in
   `dist_server.py`)
4. **Two-phase BaseDistLoader** — backend init phase + input registration phase with shared helper (in
   `base_dist_loader.py`)
5. **Tests** for all new components

______________________________________________________________________

## Session 1: `gigl/distributed/dist_sampling_producer.py` — Shared Backend + Worker Loop

This is the largest new component. On `origin/main` this file is 275 lines with just `DistSamplingProducer` and
`_sampling_worker_loop`. We add ~900 lines of new code.

### 1A. Add new imports and constants

Add to imports:

```python
import datetime
import queue
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum, auto
from multiprocessing.process import BaseProcess
from threading import Barrier
from typing import Callable, Optional, Union, cast

from graphlearn_torch.channel import ChannelBase, SampleMessage
from graphlearn_torch.distributed import RemoteDistSamplingWorkerOptions, get_context
from gigl.distributed.sampler import ABLPNodeSamplerInput
```

Add module-level constants:

```python
EPOCH_DONE_EVENT = "EPOCH_DONE"
SCHEDULER_TICK_SECS = 0.05
SCHEDULER_STATE_LOG_INTERVAL_SECS = 10.0
SCHEDULER_STATE_MAX_CHANNELS = 6
SCHEDULER_SLOW_SUBMIT_SECS = 1.0
```

**Note**: No `TASK_DEQUEUED_EVENT` — omitted by design.

### 1B. Add command types and data structures

```python
class SharedMpCommand(Enum):
    REGISTER_INPUT = auto()
    UNREGISTER_INPUT = auto()
    START_EPOCH = auto()
    STOP = auto()

@dataclass(frozen=True)
class RegisterInputCmd:
    channel_id: int
    worker_key: str
    sampler_input: Union[NodeSamplerInput, EdgeSamplerInput, ABLPNodeSamplerInput]
    sampling_config: SamplingConfig
    channel: ChannelBase

@dataclass(frozen=True)
class StartEpochCmd:
    channel_id: int
    epoch: int
    seeds_index: Optional[torch.Tensor]

@dataclass
class ActiveEpochState:
    channel_id: int
    epoch: int
    input_len: int
    batch_size: int
    drop_last: bool
    seeds_index: Optional[torch.Tensor]
    total_batches: int
    submitted_batches: int = 0
    completed_batches: int = 0
    cancelled: bool = False
```

### 1C. Add helper functions

```python
def _command_channel_id(command: SharedMpCommand, payload: object) -> Optional[int]:
    """Extract channel_id from a command payload, if present."""
    ...

def _compute_num_batches(input_len: int, batch_size: int, drop_last: bool) -> int:
    """Calculate total batches for a given input length."""
    ...

def _epoch_batch_indices(state: ActiveEpochState) -> Optional[torch.Tensor]:
    """Get batch indices for the next submission."""
    ...

def _compute_worker_seeds_ranges(input_len: int, batch_size: int, num_workers: int) -> list[tuple[int, int]]:
    """Distribute complete batches evenly across workers."""
    ...

def _slice_sampler_input(sampler_input, start: int, end: int):
    """Slice a sampler input by index range. Supports Node, Edge, and ABLP inputs."""
    ...
```

These are pure functions — reference the branch implementation at `dist_sampling_producer.py:288-409`.

### 1D. Add `_shared_sampling_worker_loop()` function

This is the core scheduler (~450 lines). It runs in a subprocess and manages multiple channels concurrently.

**Worker-local state:**

```python
channels: dict[int, ChannelBase]
inputs: dict[int, SamplerInput]
cfgs: dict[int, SamplingConfig]
route_key_by_channel: dict[int, str]
started_epoch: dict[int, int]
active_epochs_by_channel: dict[int, ActiveEpochState]
runnable_channels: deque[int]  # fair-queue for batch submission
runnable_set: set[int]
removing: set[int]
```

**Nested functions:**

- `_enqueue_channel_if_runnable_locked(channel_id)` — add to runnable queue if has pending batches
- `_clear_registered_input_locked(channel_id)` — remove all per-channel state
- `_format_scheduler_state_locked()` — compact state string for logging
- `_maybe_log_scheduler_state(reason, force)` — rate-limited state logging
- `_on_batch_done(channel_id, epoch)` — completion callback; emits `EPOCH_DONE_EVENT` when all batches done
- `_submit_one_batch(channel_id)` — submit next batch to `dist_sampler`
- `_pump_runnable_channels()` — inner loop submitting batches up to concurrency limit

**Command handler** (`_handle_command`):

- `REGISTER_INPUT` — store channel/input/config, register output with sampler
- `START_EPOCH` — create `ActiveEpochState`, enqueue channel; if 0 batches, emit `EPOCH_DONE_EVENT` immediately
- `UNREGISTER_INPUT` — cancel active epoch, wait for inflight to drain
- `STOP` — call `dist_sampler.wait_all()`, exit

**Main loop:**

```python
while keep_running:
    # drain commands
    while keep_running:
        command, payload = task_queue.get_nowait()
        keep_running = _handle_command(command, payload)
    # submit batches
    made_progress = _pump_runnable_channels()
    _maybe_log_scheduler_state("steady_state")
    if not (processed_command or made_progress):
        command, payload = task_queue.get(timeout=SCHEDULER_TICK_SECS)
        keep_running = _handle_command(command, payload)
```

**Key difference from branch**: No `TASK_DEQUEUED_EVENT` emission in `_handle_command`.

Reference: branch `dist_sampling_producer.py:412-868`.

### 1E. Add `SharedDistSamplingBackend` class

**Instance variables:**

```python
self.data: DistDataset
self.worker_options: RemoteDistSamplingWorkerOptions
self.num_workers: int
self._backend_sampling_config: SamplingConfig
self._sampler_options: SamplerOptions
self._task_queues: list[mp.Queue]
self._workers: list[BaseProcess]
self._event_queue: Optional[mp.Queue]
self._shutdown: bool
self._initialized: bool
self._lock: threading.RLock
self._channel_sampling_config: dict[int, SamplingConfig]
self._channel_input_sizes: dict[int, list[int]]
self._channel_worker_seeds_ranges: dict[int, list[tuple[int, int]]]
self._channel_shuffle_generators: dict[int, Optional[torch.Generator]]
self._channel_epoch: dict[int, int]
self._completed_workers: dict[tuple[int, int], set[int]]  # (channel_id, epoch) -> worker ranks
```

**No `_queued_task_counts_by_worker`** — omitted by design.

**Methods:**

1. `init_backend()` — before spawning worker processes, prepare a backend-local copy of `worker_options` exactly like
   `DistMpSamplingProducer` does today:
   - call `_assign_worker_devices()`
   - call `_set_worker_ranks(get_context())`
   - validate that the current distributed context is initialized as a server worker context Then spawn worker processes
     via `_shared_sampling_worker_loop`, barrier sync.
2. `register_input(channel_id, worker_key, sampler_input, sampling_config, channel)` — share_memory on input, compute
   worker ranges once, initialize per-channel shuffle state, enqueue `REGISTER_INPUT` to each worker
3. `start_new_epoch_sampling(channel_id, epoch)` — drain events, update epoch, and preserve current `shuffle=True`
   semantics:
   - if `sampling_config.shuffle` is false, use deterministic worker-local slices from the precomputed ranges
   - if `sampling_config.shuffle` is true, generate a fresh per-channel permutation for this epoch, slice it by the
     stored worker ranges, and enqueue worker-specific `START_EPOCH` payloads
   - keep shuffle state per channel so one channel's epoch schedule does not perturb another's RNG stream **No
     `time.sleep(0.1)`**, no `describe_channel()` call.
4. `unregister_input(channel_id)` — clean up tracking, enqueue `UNREGISTER_INPUT`
5. `is_channel_epoch_done(channel_id, epoch)` — drain events, check completed_workers count
6. `describe_channel(channel_id)` — diagnostic method returning epoch, input sizes, completed workers. **No queue sizes
   or task counts**.
7. `shutdown()` — enqueue STOP, join workers, close queues

**`_enqueue_worker_command` (simplified):**

```python
def _enqueue_worker_command(self, worker_rank, command, payload):
    queue_ = self._task_queues[worker_rank]
    enqueue_start = time.monotonic()
    queue_.put((command, payload))
    elapsed = time.monotonic() - enqueue_start
    if elapsed >= SCHEDULER_SLOW_SUBMIT_SECS:
        logger.warning(f"task_queue enqueue_slow worker_rank={worker_rank} ...")
```

**`_drain_events` (simplified — EPOCH_DONE only):**

```python
def _drain_events(self):
    while True:
        event = self._event_queue.get_nowait()
        if event[0] == EPOCH_DONE_EVENT:
            _, channel_id, epoch, worker_rank = event
            self._completed_workers[(channel_id, epoch)].add(worker_rank)
```

Reference: branch `dist_sampling_producer.py:871-1194`.

### 1F. Keep existing `DistSamplingProducer` and `_sampling_worker_loop`

These are still needed for colocated mode. No changes to existing code.

______________________________________________________________________

## Session 2: `gigl/distributed/graph_store/remote_channel.py` — From `kmonte/remote-channel`

**Already implemented** in `origin/kmonte/remote-channel` (3 commits). Cherry-pick or merge this branch first. It
provides:

- `gigl/distributed/graph_store/remote_channel.py` (~244 lines) — GiGL's `RemoteReceivingChannel(ChannelBase)` with
  `active_mask` and `pin_memory` support
- `tests/unit/distributed/graph_store/remote_channel_test.py` (~161 lines) — unit tests
- `gigl/distributed/base_dist_loader.py` changes: import swap (GLT → GiGL channel) and `active_mask` / `pin_memory`
  params on channel creation

**Key features of the RemoteReceivingChannel:**

- `active_mask` — skip servers whose inputs produce 0 batches
- `pin_memory` — pin tensors to CUDA host memory in the `on_done` callback for faster GPU DMA
- Routes fetch RPCs through `gigl.distributed.graph_store.compute.async_request_server` →
  `DistServer.fetch_one_sampled_message`
- Prefetch queue with per-server tracking, rate-limited recv logging

**Prerequisite**: Merge `kmonte/remote-channel` into the working branch before starting Sessions 3-4. The
`RemoteReceivingChannel` implementation and the `base_dist_loader.py` channel wiring from that branch are the foundation
for our two-phase init rewrite.

______________________________________________________________________

## Session 3: `gigl/distributed/graph_store/dist_server.py` — Two-Phase Server

Replace the per-producer model with the shared backend + per-channel model. The file grows from ~671 to ~700 lines (the
simplified version is more compact than the branch's ~989 lines).

### 3A. Add new dataclasses and update imports

Replace `from gigl.distributed.dist_sampling_producer import DistSamplingProducer` with `SharedDistSamplingBackend`.

Add:

```python
from dataclasses import dataclass, field
```

Keep imports of `FetchABLPInputRequest`, `FetchNodesRequest`. The shared-backend rewrite should not change the existing
request-object RPC surface for metadata fetches.

Add dataclasses:

```python
@dataclass(frozen=True)
class InitSamplingBackendOpts:
    backend_key: str
    worker_options: RemoteDistSamplingWorkerOptions
    sampler_options: SamplerOptions
    sampling_config: SamplingConfig

@dataclass(frozen=True)
class RegisterBackendOpts:
    backend_id: int
    worker_key: str
    sampler_input: Union[NodeSamplerInput, EdgeSamplerInput, RemoteSamplerInput, ABLPNodeSamplerInput]
    sampling_config: SamplingConfig
    buffer_capacity: int
    buffer_size: Union[int, str]

@dataclass
class ChannelState:
    backend_id: int
    worker_key: str
    channel: ShmChannel
    epoch: int = -1
    lock: threading.RLock = field(default_factory=threading.RLock)

@dataclass
class SamplingBackendState:
    backend_id: int
    backend_key: str
    runtime: SharedDistSamplingBackend
    active_channels: set[int] = field(default_factory=set)
```

**No `FetchStats`** — omitted by design.

Add constant:

```python
FETCH_SLOW_LOG_SECS = 1.0
```

### 3B. Rewrite `DistServer.__init__`

Replace the per-producer state with backend+channel state:

```python
def __init__(self, dataset: DistDataset) -> None:
    self.dataset = dataset
    self._lock = threading.RLock()
    self._exit = False
    self._next_backend_id = 0
    self._next_channel_id = 0
    self._backend_key_to_id: dict[str, int] = {}
    self._backend_state_by_id: dict[int, SamplingBackendState] = {}
    self._channel_state: dict[int, ChannelState] = {}
```

**Removed**: `_cur_producer_idx`, `_worker_key2producer_id`, `_producer_pool`, `_msg_buffer_pool`, `_epoch`,
`_producer_lock`. Also no `_backend_id_to_key`, `_backend_refcount`, `_fetch_inflight`, etc. —
`SamplingBackendState.active_channels` handles refcounting.

### 3C. Add new public methods

**`init_sampling_backend(opts: InitSamplingBackendOpts) -> int`**

- Dedup by `backend_key`: if key exists, return existing `backend_id`
- Otherwise: allocate new id, create `SamplingBackendState` with `SharedDistSamplingBackend`
- Call `backend_state.runtime.init_backend()`
- Log creation + elapsed time
- Return `backend_id`

**`register_sampling_input(opts: RegisterBackendOpts) -> int`**

- Look up `SamplingBackendState` by `backend_id`
- Convert `RemoteSamplerInput` to local if needed
- Create `ShmChannel`, allocate `channel_id`, create `ChannelState`
- Add `channel_id` to `backend_state.active_channels`
- Call `backend_state.runtime.register_input(...)`
- On error: rollback channel and active_channels
- Log registration + total connections
- Return `channel_id`

**`destroy_sampling_input(channel_id: int) -> None`**

- Pop `ChannelState`
- Under channel lock: call `backend_state.runtime.unregister_input(channel_id)`
- Remove from `active_channels`; if empty, shut down backend and clean maps

### 3D. Rewrite existing methods

**`start_new_epoch_sampling(channel_id, epoch)`** — simplified:

- Look up `ChannelState`, check epoch idempotence
- Look up `SamplingBackendState`, call `runtime.start_new_epoch_sampling()`
- **2 log lines** (not 7)

**`fetch_one_sampled_message(channel_id)`** — simplified:

- Look up `ChannelState` and `SamplingBackendState`
- Acquire `channel_state.lock`
- Poll `channel.recv(timeout_ms=100)` (not 500)
- On timeout: check `runtime.is_channel_epoch_done()` → return `(None, True)` if done and empty
- **No fetch accounting** — only log if elapsed >= `FETCH_SLOW_LOG_SECS`

**`shutdown()`** — iterate `_backend_state_by_id` values

**`create_sampling_producer(...)` / `destroy_sampling_producer(...)`** — **Delete entirely**. All callers
(`DistNeighborLoader`, `DistABLPLoader`, integration test) are updated in Sessions 4-5 to use the two-phase API
directly. No external consumers exist.

### 3E. Keep `get_node_ids` and `get_ablp_input` request-object signatures

Do **not** change these methods away from `FetchNodesRequest` / `FetchABLPInputRequest`.

Reasons:

- `RemoteDistDataset` already calls them with request dataclasses on `origin/main`
- `messages.py` and its unit tests already define and validate those request objects
- this backend rewrite does not benefit from widening the RPC surface here

So the shared-backend change should leave:

- `DistServer.get_node_ids(request: FetchNodesRequest)`
- `DistServer.get_ablp_input(request: FetchABLPInputRequest)`

and their existing downstream callers/tests intact.

### 3F. Simplify `_call_func_on_server`

Keep simple — no `start_new_epoch_sampling`-specific tracing (it doesn't have any on `origin/main` either, so this is
already correct).

______________________________________________________________________

## Session 4: `gigl/distributed/base_dist_loader.py` — Two-Phase Loader Init

Replace the single-phase `_init_graph_store_connections()` with two-phase init using a shared helper.

### 4A. Update imports

```python
# Remove
from graphlearn_torch.channel import RemoteReceivingChannel
from graphlearn_torch.distributed.dist_client import async_request_server

# Add
from gigl.distributed.graph_store.remote_channel import RemoteReceivingChannel
from gigl.distributed.graph_store.compute import async_request_server
from gigl.distributed.graph_store.dist_server import (
    DistServer,
    InitSamplingBackendOpts,
    RegisterBackendOpts,
)
```

Also rewrite `BaseDistLoader.__init__` so graph-store mode is no longer inferred from `producer` being a callable. The
constructor should branch on the actual mode-defining types that already exist in the code:

- colocated mode: `DistDataset` + `MpDistSamplingWorkerOptions` + a concrete `DistSamplingProducer`
- graph-store mode: `RemoteDistDataset` + `RemoteDistSamplingWorkerOptions`, with the two-phase RPC flow handled
  internally

This keeps the constructor aligned with the real invariants and removes the old callable sentinel API.

### 4B. Add `GroupLeaderInfo` dataclass and `_compute_group_leader()` static method

Extract leader election logic from the existing `_init_graph_store_connections()` monolith into a reusable helper.
Return a named dataclass instead of an opaque tuple:

```python
@dataclass(frozen=True)
class GroupLeaderInfo:
    leader_rank: int
    is_leader: bool
    my_batch: int
    num_batches: int
    stagger_sleep: float
    group_size: int

@staticmethod
def _compute_group_leader(
    my_key: str,
    all_keys: list[Optional[str]],
    rank: int,
    process_start_gap_seconds: float,
    max_concurrent_producer_inits: int,
) -> GroupLeaderInfo:
    """Compute leader election, batching, and stagger for a group of ranks sharing the same key."""
```

This consolidates the leader election + batching + stagger logic that currently lives inline in
`_init_graph_store_connections()` (around line 700-720 on `origin/main`).

### 4C. Add `_dispatch_grouped_graph_store_phase()` helper

DRY the all-gather → leader-elect → stagger → dispatch → all-gather pattern:

```python
T = TypeVar("T")

@staticmethod
def _dispatch_grouped_graph_store_phase(
    *,
    my_key: str,
    runtime: DistributedRuntimeInfo,
    process_start_gap_seconds: float,
    max_concurrent_producer_inits: int,
    issue_phase_rpcs: Callable[[], list[T]],
) -> list[T]:
    """Shared leader-election, stagger, and all-gather for graph-store RPC phases."""
    all_keys = [None] * runtime.world_size
    torch.distributed.all_gather_object(all_keys, my_key)
    group_info = BaseDistLoader._compute_group_leader(...)
    if group_info.is_leader and group_info.stagger_sleep > 0:
        time.sleep(group_info.stagger_sleep)
    results = issue_phase_rpcs() if group_info.is_leader else []
    all_results = [[] for _ in range(runtime.world_size)]
    torch.distributed.all_gather_object(all_results, results)
    return all_results[group_info.leader_rank]
```

### 4D. Add `_build_graph_store_backend_key()` method

Generates a deterministic key from loader-scoped worker options + sampling_config:

```python
def _build_graph_store_backend_key(self) -> str:
    """Construct backend_key from worker options + sampling config."""
    # Pipe-separated string of all params that affect loader-scoped backend identity
```

The key should intentionally include `master_port`, so concurrent dataloader instantiations do **not** share a backend
even if their sampling configs match. The dedup scope is:

- same `backend_key` across the compute ranks participating in one dataloader instantiation -> share one backend
- different loader instantiations (for example one `DistNeighborLoader` and one `DistABLPLoader`) -> different backends

### 4E. Add phase methods

**`_init_graph_store_sampling_backends(runtime, ...) -> list[int]`**

- Compute `backend_key`
- Use `_dispatch_grouped_graph_store_phase` with `issue_rpcs` that dispatches `init_sampling_backend` to each server
- Returns `list[backend_id]`

**`_register_graph_store_sampling_inputs(runtime, backend_id_list, ...) -> list[int]`**

- Use `worker_key` as the grouping key
- Use `_dispatch_grouped_graph_store_phase` with `issue_rpcs` that dispatches `register_sampling_input` to each server
- Returns `list[channel_id]`

### 4F. Rewrite `_init_graph_store_connections()`

Replace the monolithic method (~150 lines) with:

1. Validate context, move inputs to CPU
2. `backend_id_list = self._init_graph_store_sampling_backends(...)`
3. `channel_id_list = self._register_graph_store_sampling_inputs(...)`
4. Create `RemoteReceivingChannel` with `active_mask=self._remote_input_has_batches` and `pin_memory`

**Remove**: `create_producer_fn` parameter — the two-phase flow is internal. If needed, keep an optional colocated-only
`producer` parameter, but graph-store mode should no longer accept or require a callable producer factory.

### 4G. Add `_sampler_input_has_batches()` + `_remote_input_has_batches` attribute

```python
def _sampler_input_has_batches(self, sampler_input: NodeSamplerInput) -> bool:
    input_len = len(sampler_input)
    return input_len > 0 and not (self.drop_last and input_len < self.batch_size)
```

Set `self._remote_input_has_batches = [self._sampler_input_has_batches(inp) for inp in self._input_data_list]` during
init.

### 4H. Update `shutdown()` and `__iter__()`

**`shutdown()`**: Change `destroy_sampling_producer` → `destroy_sampling_input`, `_producer_id_list` →
`_channel_id_list`

**`__iter__()`**: Change `start_new_epoch_sampling(producer_id, epoch)` → `start_new_epoch_sampling(channel_id, epoch)`.
Only dispatch for channels where `_remote_input_has_batches[i]` is True.

### 4I. Add `_collate_fn()` override for remote mode

No new base-class `_collate_fn()` override is required for this backend change.

Keep the existing loader-specific `_collate_fn()` implementations in:

- `distributed_neighborloader.py`
- `dist_ablp_neighborloader.py`

The remote-channel `pin_memory` support is sufficient for this plan; moving device-transfer logic into `BaseDistLoader`
is not required to make the shared-backend design correct.

______________________________________________________________________

## Session 5: `gigl/distributed/distributed_neighborloader.py` — Wire up two-phase init

### 5A. Update `_setup_for_graph_store()`

The method currently returns `(list[NodeSamplerInput], RemoteDistSamplingWorkerOptions, DatasetSchema)` and the caller
passes `DistServer.create_sampling_producer` to `BaseDistLoader.__init__`.

Update to no longer pass `create_producer_fn`. The `BaseDistLoader.__init__` handles the two-phase flow internally based
on `RemoteDistDataset` + `RemoteDistSamplingWorkerOptions`.

### 5B. Similar update for `dist_ablp_neighborloader.py`

Same pattern change.

______________________________________________________________________

## Session 6: Tests

### 6A. `tests/unit/distributed/dist_sampling_producer_test.py` — New file

Test the shared worker loop and helper functions:

- `test_compute_num_batches` — edge cases (0 input, drop_last)
- `test_epoch_batch_indices` — correct slicing with seeds_index
- `test_compute_worker_seeds_ranges` — even distribution across workers
- `test_init_backend_prepares_worker_options` — worker ranks/devices are initialized before subprocess spawn
- `test_start_new_epoch_sampling_shuffle_refreshes_per_epoch` — shuffled channels get a fresh permutation each epoch
  without sharing RNG state across channels
- `test_shared_worker_loop_round_robin` — fair submission across channels
- `test_shared_worker_loop_epoch_done` — EPOCH_DONE_EVENT emitted correctly
- `test_shared_worker_loop_unregister_drains` — inflight batches drain before cleanup

### 6B. `tests/unit/distributed/dist_server_test.py` — Add sampling tests

Add `TestDistServerSampling` class (mock `SharedDistSamplingBackend` and `ShmChannel`):

- `test_init_sampling_backend_idempotent` — same key within one loader group → same id, constructor called once
- `test_register_creates_channel` — returns channel_id, backend `register_input` called
- `test_destroy_last_channel_shuts_down_backend` — `runtime.shutdown()` called
- `test_destroy_unknown_channel_noop` — no error
- `test_start_epoch_idempotent` — duplicate call is no-op
- `test_shutdown_cleans_all_backends` — multiple backends cleaned
- `test_create_sampling_producer_removed` — verify old methods no longer exist on `DistServer`

### 6C. `tests/unit/distributed/graph_store/remote_channel_test.py` — Already in `kmonte/remote-channel`

Comes from the cherry-picked branch. No additional work needed.

### 6D. `tests/unit/distributed/distributed_neighborloader_test.py` + `tests/unit/distributed/dist_ablp_neighborloader_test.py`

Add graph-store-specific regression coverage for the constructor contract change:

- graph-store loaders enter the remote path based on dataset/worker-option types, not a callable producer
- colocated loaders still use the existing concrete producer path

### 6E. `tests/integration/distributed/graph_store/graph_store_integration_test.py` — Update for shared backend

Update the existing integration test to cover the new shared-backend model end-to-end:

- **Multi-rank loader test**: For one logical dataloader instantiation spanning multiple compute ranks, verify all ranks
  connect to the same backend on each storage node and complete epochs independently via separate channels.
- **Distinct-loader test**: Create one `DistNeighborLoader` and one `DistABLPLoader` at the same time. Verify they use
  different backends even if their sampling configs overlap.
- **Repeated epoch test**: Run multiple epochs on the same loader/channel to confirm no stale completion leakage across
  epochs.
- **ABLP test**: Verify `DistABLPLoader` works with the shared backend model.

Reference the existing test structure on `origin/main` at
`tests/integration/distributed/graph_store/graph_store_integration_test.py` (~1025 lines of existing compute-side
tests).

______________________________________________________________________

## Improvements Found During Research

1. **`_call_func_on_server` on `origin/main` is already simple** — no special-casing to remove (the branch added tracing
   that the plan originally removed). Keep as-is.

2. **`get_node_ids` / `get_ablp_input` signatures** — keep the `FetchNodesRequest` / `FetchABLPInputRequest` dataclasses
   on `origin/main`. The backend rewrite should not refactor this RPC surface.

3. **`_init_graph_store_connections` on `origin/main` already has leader election and staggering** — but it's a
   ~150-line monolith. The two-phase split + `_dispatch_grouped_graph_store_phase` helper is genuinely cleaner.

4. **Worker-option prep and shuffle semantics must be preserved explicitly** — the shared backend replaces
   `DistSamplingProducer`, so it must take over:

   - worker device/rank initialization that GLT currently performs before worker spawn
   - per-epoch shuffle regeneration for `shuffle=True` The plan above now makes those steps explicit.

5. **`RemoteReceivingChannel` on `origin/main` is GLT's version** — it lacks `active_mask` and `pin_memory` support.
   GiGL's version adds both, which are needed for the new channel model (inactive channels) and GPU perf.

6. **The `time.sleep(0.1)` in `start_new_epoch_sampling`** — this was added on the branch and is NOT on `origin/main`.
   Since we're building clean, we simply don't add it.

______________________________________________________________________

## Execution Order

```
Session 2 (merge kmonte/remote-channel) — prerequisite
    ↓
Session 1 (dist_sampling_producer.py) — SharedDistSamplingBackend + worker loop
    ↓
Session 3 (dist_server.py) — two-phase server (depends on Session 1)
    ↓
Session 4 (base_dist_loader.py) — two-phase loader (depends on Sessions 2 + 3)
    ↓
Session 5 (neighborloader.py, ablp_loader.py) — wire up (depends on Session 4)
    ↓
Session 6 (tests) — verify everything
```

Session 2 is a merge, not new code. Session 1 is independent of it.

______________________________________________________________________

## Correctness Constraints

- `create_sampling_producer` / `destroy_sampling_producer` are deleted — all callers migrate to two-phase API
- Epoch idempotence: duplicate `start_new_epoch_sampling(channel_id, same_epoch)` is a no-op
- `shuffle=True` semantics must match today's graph-store producer behavior: fresh epoch-local shuffles, isolated per
  channel
- Destroy during active sampling: mark cancelled, drain inflight, then cleanup
- Backend dedup: identical `backend_key` shares one `SharedDistSamplingBackend` only across the compute ranks
  participating in the same dataloader instantiation
- ABLP inputs and label tensors must preserve current semantics
- `RemoteSamplerInput -> local sampler input` conversion must remain in `register_sampling_input`
- `FetchNodesRequest` / `FetchABLPInputRequest` remain the RPC contract for metadata fetches

______________________________________________________________________

## Files Modified/Created

| File                                                                        | Action                                                                                             |
| --------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `gigl/distributed/dist_sampling_producer.py`                                | Add ~900 lines: `SharedDistSamplingBackend`, `_shared_sampling_worker_loop`, helper types          |
| `gigl/distributed/graph_store/remote_channel.py`                            | **From `kmonte/remote-channel`**: merge, not new code                                              |
| `gigl/distributed/graph_store/dist_server.py`                               | Major rewrite: two-phase model with `SamplingBackendState`                                         |
| `gigl/distributed/base_dist_loader.py`                                      | Rewrite graph-store init: two-phase with shared helper (builds on `kmonte/remote-channel` changes) |
| `gigl/distributed/distributed_neighborloader.py`                            | Minor: remove `create_producer_fn` passing                                                         |
| `gigl/distributed/dist_ablp_neighborloader.py`                              | Minor: same as above                                                                               |
| `tests/unit/distributed/dist_sampling_producer_test.py`                     | **New file**: shared worker loop tests                                                             |
| `tests/unit/distributed/dist_server_test.py`                                | Add sampling lifecycle tests                                                                       |
| `tests/unit/distributed/graph_store/remote_channel_test.py`                 | **From `kmonte/remote-channel`**: merge, not new code                                              |
| `tests/integration/distributed/graph_store/graph_store_integration_test.py` | Update: add multi-rank sharing, distinct-loader isolation, repeated-epoch, ABLP scenarios          |

______________________________________________________________________

## Verification

1. `make type_check` — all modified/new files pass mypy
2. `make unit_test_py PY_TEST_FILES="dist_server_test.py"`
3. `make unit_test_py PY_TEST_FILES="dist_sampling_producer_test.py"`
4. `make unit_test_py PY_TEST_FILES="remote_channel_test.py"`
5. `make unit_test_py PY_TEST_FILES="distributed_neighborloader_test.py"`
6. `make unit_test_py PY_TEST_FILES="dist_ablp_neighborloader_test.py"`
7. `make format_py` — formatting pass
8. Integration: `make integration_test PY_TEST_FILES="graph_store_integration_test.py"` (if env supports it)
