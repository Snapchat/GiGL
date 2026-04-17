# Plan: Decompose `kmonte/shared-backend` into 4 PRs

## Context

The `kmonte/shared-backend` branch has ~1,865 insertions / ~471 deletions across 10 files, implementing a shared backend
\+ per-channel model that replaces the per-producer model in graph-store mode. The commit history is messy (merges,
squashed work), so we'll create fresh branches from `main` and selectively apply changes from the current branch's final
state.

**Dependency chain:** PR 1 \<- PR 2 \<- PR 3 \<- PR 4 (strict linear, each stacks on the previous)

______________________________________________________________________

## PR 1: "Extract sampler factory helpers"

**Branch:** `kmonte/shared-backend-decomp-1`\
**Base:** `main`\
**~100 lines changed**

### Files

1. **`gigl/distributed/utils/dist_sampler.py`** (new file) — public helpers extracted from `dist_sampling_producer.py`:

   - Type aliases `SamplerInput`, `SamplerRuntime`
   - `create_dist_sampler()` — sampler factory (public, no leading underscore)

2. **`gigl/distributed/dist_sampling_producer.py`** — update to use the new utils module:

   - Import `create_dist_sampler`, `SamplerInput`, `SamplerRuntime` from `gigl.distributed.utils.dist_sampler`
   - Update `_sampling_worker_loop()` to call the imported `create_dist_sampler()`
   - Minor: rename `w` -> `worker` in `DistSamplingProducer.init()` (at `dist_sampling_producer.py:267`)

### How to create

```bash
# Create branch from main
git checkout main && git checkout -b kmonte/shared-backend-decomp-1

# Create new utils/dist_sampler.py with the extracted public functions
# Update dist_sampling_producer.py to import from the new module

# Verify: make type_check
```

### Verification

- `make type_check` — no new type errors
- Manual review: `_sampling_worker_loop()` behavior is identical (just calls imported function)

______________________________________________________________________

## PR 2: "Add SharedDistSamplingBackend — multi-channel sampling backend"

**Branch:** `kmonte/shared-backend-decomp-2`\
**Base:** PR 1 branch\
**~1,000 lines added**

### Files

1. **`gigl/distributed/graph_store/shared_dist_sampling_producer.py`** (new file, ~793 lines):

   - **Update imports**: change
     `from gigl.distributed.dist_sampling_producer import _create_dist_sampler, SamplerInput, SamplerRuntime` to
     `from gigl.distributed.utils.dist_sampler import create_dist_sampler, SamplerInput, SamplerRuntime` (public names,
     new module path)
   - `SharedMpCommand` enum (REGISTER_INPUT, UNREGISTER_INPUT, START_EPOCH, STOP)
   - `RegisterInputCmd`, `StartEpochCmd`, `ActiveEpochState` dataclasses
   - `_shared_sampling_worker_loop()` — fair-queued round-robin scheduler
   - `SharedDistSamplingBackend` class — public interface (init_backend, register_input, start_new_epoch_sampling,
     unregister_input, is_channel_epoch_done, shutdown)
   - Helper functions: `_compute_num_batches`, `_epoch_batch_indices`, `_compute_worker_seeds_ranges`

2. **`tests/unit/distributed/dist_sampling_producer_test.py`** (new file, ~214 lines):

   - Tests for pure business logic helpers: `_compute_num_batches`, `_epoch_batch_indices`,
     `_compute_worker_seeds_ranges`
   - Tests for `start_new_epoch_sampling` shuffle behavior
   - Tests for `describe_channel` completion reporting

### How to create

```bash
# Stack on PR 1
git checkout kmonte/shared-backend-decomp-1
git checkout -b kmonte/shared-backend-decomp-2

# Copy the new file from the current branch
git show kmonte/shared-backend:gigl/distributed/graph_store/shared_dist_sampling_producer.py > gigl/distributed/graph_store/shared_dist_sampling_producer.py

# Copy the new test file from the current branch
git show kmonte/shared-backend:tests/unit/distributed/dist_sampling_producer_test.py > tests/unit/distributed/dist_sampling_producer_test.py
```

### Verification

- `make type_check` — no new type errors

______________________________________________________________________

## PR 3: "Add two-phase sampling API to DistServer"

**Branch:** `kmonte/shared-backend-decomp-3`\
**Base:** PR 2 branch\
**~500 lines changed**

### Files

1. **`gigl/distributed/graph_store/messages.py`** — additive (moved from PR 1):

   - Add `InitSamplingBackendRequest` frozen dataclass (backend_key, worker_options, sampler_options, sampling_config)
   - Add `RegisterBackendRequest` frozen dataclass (backend_id, worker_key, sampler_input, sampling_config,
     buffer_capacity, buffer_size)

2. **`gigl/distributed/graph_store/dist_server.py`** (~394 lines changed):

   - Add `ChannelState`, `SamplingBackendState`, `_ChannelFetchStats` dataclasses
   - Add new state: `_backend_state_by_id`, `_channel_state`, `_backend_key_to_id`
   - Add `init_sampling_backend(InitSamplingBackendRequest)` — creates/reuses backend
   - Add `register_sampling_input(RegisterBackendRequest)` — registers lightweight channel on existing backend
   - Add `destroy_sampling_input(channel_id)` — unregisters channel, destroys backend when last channel removed
   - Update `start_new_epoch_sampling()` to accept `channel_id`
   - Update `fetch_one_sampled_message()` to accept `channel_id`, add `_log_fetch_stats_if_due()`
   - Update `shutdown()` to iterate backends
   - **Bridge**: Refactor existing `create_sampling_producer()` to internally delegate to `init_sampling_backend` +
     `register_sampling_input`, returning a `channel_id` as the `producer_id`. Similarly bridge
     `destroy_sampling_producer()` → `destroy_sampling_input`. This keeps existing loaders working without changes.

3. **`tests/unit/distributed/dist_server_test.py`** (existing file — append new class):

   - Add `TestDistServerSampling` class — tests for business logic only (no heavy mocking / `cls.__new__`)

### How to create

```bash
# Stack on PR 2
git checkout kmonte/shared-backend-decomp-2
git checkout -b kmonte/shared-backend-decomp-3

# Add new dataclasses to messages.py
# Apply dist_server.py changes from current branch, adding bridge methods
# Add server tests
```

### Verification

- `make type_check` — no new type errors

______________________________________________________________________

## PR 4: "Switch loaders to two-phase init"

**Branch:** `kmonte/shared-backend-decomp-4`\
**Base:** PR 3 branch\
**~600 lines changed**

### Files

1. **`gigl/distributed/base_dist_loader.py`** (~474 lines changed):

   - Add `GroupLeaderInfo` frozen dataclass
   - Extract `_compute_group_leader()` helper function
   - Extract `_dispatch_grouped_graph_store_phase()` generic RPC helper (uses `all_gather_object` on `_backend_key` for
     leader election, `base_dist_loader.py:128-159`)
   - Add `_init_graph_store_sampling_backends()` method
   - Add `_register_graph_store_sampling_inputs()` method
   - Add `_sampler_input_has_batches()` helper
   - Refactor `_init_graph_store_connections()` to use two-phase init
   - Change `producer` parameter type from `Union[..., Callable]` to `Optional[DistSamplingProducer]`
   - Add `_backend_key`, `_backend_id_list`, `_channel_id_list` instance attributes
   - **Revert `_global_loader_counter` to per-class counters** (see "Global counter risk" below)
   - Update `shutdown()` and `__iter__()` for channel-based IDs (note: epoch start is in `__iter__` at
     `base_dist_loader.py:942`, not a standalone `produce_all()`)
   - Simplify `create_graph_store_worker_options()` (remove `compute_rank` param)

2. **`gigl/distributed/dist_ablp_neighborloader.py`** (~26 lines changed):

   - Use per-class counter (not `BaseDistLoader._global_loader_counter`) for `_backend_key`
   - Set `self._backend_key` for two-tier key naming (e.g. `dist_ablp_loader_0`)
   - Remove `DistServer.create_sampling_producer` callable; pass `None` for graph-store producer
   - Remove `compute_rank` from `create_graph_store_worker_options()` call

3. **`gigl/distributed/distributed_neighborloader.py`** (~24 lines changed):

   - Same adaptations as ABLP loader (per-class counter, backend_key, no callable, no compute_rank)

4. **`gigl/distributed/graph_store/dist_server.py`** (~50 lines removed):

   - Remove `create_sampling_producer()` / `destroy_sampling_producer()` bridge methods added in PR 3
   - Remove old `_producer_pool` / `_worker_key2producer_id` state

5. **`tests/integration/distributed/graph_store/graph_store_integration_test.py`** (~48 lines changed):

   - Rename `_producer_id_list` references to `_backend_id_list`/`_channel_id_list`
   - Add assertions for backend sharing across ranks and isolation across loaders
   - Update comments and logging
   - **Fix multi-rank gap**: `test_multiple_loaders_in_graph_store` currently uses
     `num_compute_nodes=1, num_processes_per_compute=1` (line 1017), making the `all_gather_object` assertions at lines
     406-430 trivially pass with `world_size=1`. Update to at least `num_compute_nodes=2` or
     `num_processes_per_compute=2` so backend-sharing and leader-election are actually exercised across ranks.

### How to create

```bash
# Stack on PR 3
git checkout kmonte/shared-backend-decomp-3
git checkout -b kmonte/shared-backend-decomp-4

# Apply remaining changes from the current branch, with these manual adjustments:
# 1. Revert _global_loader_counter to per-class counters in base_dist_loader.py,
#    dist_ablp_neighborloader.py, and distributed_neighborloader.py
# 2. Remove bridge methods from dist_server.py
# 3. Update test_multiple_loaders_in_graph_store to use num_compute_nodes=2
```

### Verification

- `make type_check` — no new type errors

______________________________________________________________________

## Execution strategy

Since the commit history is messy (merge commits, "update" messages), **don't cherry-pick**. Instead:

1. For each PR branch, start fresh from the base (main or previous PR branch)
2. Use `git diff main...kmonte/shared-backend -- <file>` to understand what changed in each file
3. Apply changes selectively — either by copying files from the current branch state or by manually applying the
   relevant subset of diffs
4. For PR 1: create `utils/dist_sampler.py` as a new file with the extracted public helpers, then update
   `dist_sampling_producer.py` to import from it. The branch currently keeps these in `dist_sampling_producer.py` — this
   is a divergence from the branch.
5. For PR 2: copy the new files from the current branch, then update imports to point at
   `gigl.distributed.utils.dist_sampler`
6. For PR 3: apply `dist_server.py` changes, add message dataclasses, and wire `create_sampling_producer` to delegate to
   the new two-phase API as a bridge
7. For PR 4: apply loader diffs, remove bridge methods, revert per-class counters, fix multi-rank test

Each PR gets its own clean, squashed commit with a descriptive message.

______________________________________________________________________

## Codex review findings addressed

### Global counter risk (Issue 1 — High)

The branch uses `BaseDistLoader._global_loader_counter = count(0)` (`base_dist_loader.py:202`) shared across all loader
subclasses. Both loaders already namespace keys by type prefix (`dist_ablp_loader_` vs `dist_neighbor_loader_`), so a
global counter is unnecessary to prevent cross-type collisions. Worse, it creates a fragile invariant: all ranks must
construct every loader instance in identical global order across both classes, or `_backend_key` values diverge and
`_dispatch_grouped_graph_store_phase` (line 128) groups ranks incorrectly.

**Resolution**: Revert to per-class counters in **PR 4**. Each loader class keeps its own `_counter = count(0)` as it
was on `main`. The type-prefixed `_backend_key` (e.g. `dist_ablp_loader_0`) already prevents collisions. Per-class
counters only require same-type loaders to be constructed in the same order per rank, which is a much weaker invariant.

### Vacuous multi-rank integration test (Issue 2 — High)

`test_multiple_loaders_in_graph_store` (`graph_store_integration_test.py:1016`) runs with
`num_compute_nodes=1, num_processes_per_compute=1`, so `world_size=1`. The `all_gather_object` backend-sharing
assertions (lines 406-430) trivially pass — they can't detect cross-rank key mismatch or leader-election bugs.

**Resolution**: In **PR 4**, update `test_multiple_loaders_in_graph_store` to use `num_compute_nodes=2` (or
`num_processes_per_compute=2`). This is the same pattern as `test_homogeneous_training`
(`graph_store_integration_test.py:992`) which already uses 2 compute nodes.
