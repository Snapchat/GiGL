# Shared Backend PR Restructure Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the 3-PR decomposition of `kmonte/shared-backend` into 4 cleaner PRs, fixing issues found during
the first pass.

**Architecture:** Linear chain PR 1 \<- PR 2 \<- PR 3 \<- PR 4. PR 1 extracts sampler factory only (no unused code). PR
2 adds the shared backend. PR 3 adds server-side switchover. PR 4 adds client-side switchover.

**Tech Stack:** Python 3.11, GraphLearn-for-PyTorch, torch.distributed

______________________________________________________________________

## Changes from the current 3-PR decomposition

### 1. Move message dataclasses out of PR 1 into where they're consumed

`InitSamplingBackendRequest` and `RegisterBackendRequest` in `messages.py` are not used until PR 3 (server) and PR 4
(client). Adding them in PR 1 is premature.

**Action:** Remove them from PR 1 (where nothing uses them). Add them to `messages.py` in PR 3 when the server first
needs them. `messages.py` remains the correct shared location since both `dist_server.py` and `base_dist_loader.py`
import from it.

### 2. Remove `prepare_degree_tensors` from PR 1's `dist_sampler.py`

`prepare_degree_tensors` is only called from `shared_dist_sampling_producer.py` (PR 2). It is NOT called from
`dist_sampling_producer.py` — the original code inlines that logic. Extracting it to a public utility in PR 1 is
premature.

**Action:** Remove `prepare_degree_tensors` from `gigl/distributed/utils/dist_sampler.py`. Move it into
`shared_dist_sampling_producer.py` as a module-private function `_prepare_degree_tensors` in PR 2. Update the test mock
path to match.

### 3. Also remove `SamplerInput` and `SamplerRuntime` type aliases from PR 1

These type aliases are only used in `shared_dist_sampling_producer.py` (PR 2). They are not used by
`dist_sampling_producer.py`.

**Action:** Remove from `gigl/distributed/utils/dist_sampler.py`. Define them locally in
`shared_dist_sampling_producer.py` in PR 2.

### 4. Split PR 3 into server-side (PR 3) and client-side (PR 4)

Current PR 3 is ~730 lines across 6 files touching both the server (`dist_server.py`) and clients
(`base_dist_loader.py`, both loaders). These are independently reviewable.

**Server PR 3** (~390 lines): `dist_server.py`, `dist_server_test.py`, `messages.py` (add the new dataclasses here)
**Client PR 4** (~470 lines): `base_dist_loader.py`, `dist_ablp_neighborloader.py`, `distributed_neighborloader.py`,
`graph_store_integration_test.py`

______________________________________________________________________

## Revised PR chain

### PR 1: `kmonte/shared-backend-pr1-helpers` (base: `main`)

**Files:**

- Create: `gigl/distributed/utils/dist_sampler.py` — contains only `create_dist_sampler()`
- Modify: `gigl/distributed/dist_sampling_producer.py` — import `create_dist_sampler` from utils, rename `w` -> `worker`

`dist_sampler.py` exports:

- `create_dist_sampler()` — public sampler factory

That's it. No type aliases, no `prepare_degree_tensors`, no message dataclasses.

**Verify:** `make type_check && make format_py`

______________________________________________________________________

### PR 2: `kmonte/shared-backend-pr2-backend` (base: PR 1)

**Files:**

- Create: `gigl/distributed/graph_store/shared_dist_sampling_producer.py`
- Create: `tests/unit/distributed/dist_sampling_producer_test.py`

`shared_dist_sampling_producer.py` defines locally:

- `SamplerInput` type alias (was in `dist_sampler.py`)
- `SamplerRuntime` type alias (was in `dist_sampler.py`)
- `_prepare_degree_tensors()` as a module-private function (was in `dist_sampler.py`)
- Imports `create_dist_sampler` from `gigl.distributed.utils.dist_sampler`

The test file mocks `gigl.distributed.graph_store.shared_dist_sampling_producer._prepare_degree_tensors` (with
underscore — matches module-private name).

**Verify:** `make type_check && make unit_test_py PY_TEST_FILES="dist_sampling_producer_test.py"`

______________________________________________________________________

### PR 3: `kmonte/shared-backend-pr3-server` (base: PR 2)

**Files:**

- Modify: `gigl/distributed/graph_store/messages.py` — add `InitSamplingBackendRequest` and `RegisterBackendRequest`
- Modify: `gigl/distributed/graph_store/dist_server.py` — replace per-producer model with shared backend
- Modify: `tests/unit/distributed/dist_server_test.py` — append `TestDistServerSampling` class

Server-side changes only. The old `create_sampling_producer()` / `destroy_sampling_producer()` methods are replaced with
`init_sampling_backend()` / `register_sampling_input()` / `destroy_sampling_input()`. New dataclasses: `ChannelState`,
`SamplingBackendState`, `_ChannelFetchStats`.

**Verify:** `make type_check && make unit_test_py PY_TEST_FILES="dist_server_test.py"`

______________________________________________________________________

### PR 4: `kmonte/shared-backend-pr4-loaders` (base: PR 3)

**Files:**

- Modify: `gigl/distributed/base_dist_loader.py` — two-phase graph-store init, `GroupLeaderInfo`,
  `_dispatch_grouped_graph_store_phase()`
- Modify: `gigl/distributed/dist_ablp_neighborloader.py` — per-class `_counter`, `_backend_key`, remove callable
  producer
- Modify: `gigl/distributed/distributed_neighborloader.py` — same adaptations
- Modify: `tests/integration/distributed/graph_store/graph_store_integration_test.py` — rename `_producer_id_list` refs,
  `num_compute_nodes=2`

Client-side changes. Both loaders use per-class `_counter = count(0)` (NOT `BaseDistLoader._global_loader_counter`). The
`producer` parameter changes from `Union[..., Callable]` to `Optional[DistSamplingProducer]`.

**Verify:** `make type_check && make unit_test_py`

______________________________________________________________________

## Execution plan

We are currently on `kmonte/shared-backend-pr3-switchover` which has all 3 original PRs. We need to rebuild 4 branches.

### Task 1: Rebuild PR 1

- [ ] **Step 1:** Check out `main`, create `kmonte/shared-backend-pr1-helpers` (force, overwriting existing)
- [ ] **Step 2:** Create `gigl/distributed/utils/dist_sampler.py` with ONLY `create_dist_sampler()` — no
  `prepare_degree_tensors`, no `SamplerInput`, no `SamplerRuntime`
- [ ] **Step 3:** Update `gigl/distributed/dist_sampling_producer.py` to import `create_dist_sampler` from the new
  module, replace inline sampler creation in `_sampling_worker_loop()`, rename `w` -> `worker`
- [ ] **Step 4:** Run `make format_py && make type_check`
- [ ] **Step 5:** Commit and force push

### Task 2: Rebuild PR 2

- [ ] **Step 1:** From PR 1 branch, create `kmonte/shared-backend-pr2-backend` (force)
- [ ] **Step 2:** Copy `shared_dist_sampling_producer.py` from `kmonte/shared-backend`. Then:
  - Change import to `from gigl.distributed.utils.dist_sampler import create_dist_sampler` (just the factory)
  - Define `SamplerInput` and `SamplerRuntime` type aliases locally in the file
  - Define `_prepare_degree_tensors()` as a module-private function in the file (move from `dist_sampler.py`)
  - Update call sites to use `_prepare_degree_tensors` (with underscore)
- [ ] **Step 3:** Copy `dist_sampling_producer_test.py` from `kmonte/shared-backend`. The mock path should target
  `_prepare_degree_tensors` (with underscore) — which matches the module-private name. No change needed from the
  original branch version.
- [ ] **Step 4:** Run `make format_py && make type_check`
- [ ] **Step 5:** Commit and force push

### Task 3: Build PR 3 (server-side, new branch name)

- [ ] **Step 1:** From PR 2 branch, create `kmonte/shared-backend-pr3-server`
- [ ] **Step 2:** Add `InitSamplingBackendRequest` and `RegisterBackendRequest` to
  `gigl/distributed/graph_store/messages.py` (copy from current `kmonte/shared-backend-pr3-switchover`)
- [ ] **Step 3:** Copy `dist_server.py` from `kmonte/shared-backend-pr3-switchover`
- [ ] **Step 4:** Append `TestDistServerSampling` to `tests/unit/distributed/dist_server_test.py` (copy from current
  `kmonte/shared-backend-pr3-switchover`), plus imports and `_make_sampling_config` helper
- [ ] **Step 5:** Run `make format_py && make type_check`
- [ ] **Step 6:** Commit and push

### Task 4: Build PR 4 (client-side, new branch name)

- [ ] **Step 1:** From PR 3 branch, create `kmonte/shared-backend-pr4-loaders`
- [ ] **Step 2:** Copy `base_dist_loader.py`, `dist_ablp_neighborloader.py`, `distributed_neighborloader.py` from
  `kmonte/shared-backend-pr3-switchover`
- [ ] **Step 3:** Apply per-class counter fix: replace `BaseDistLoader._global_loader_counter` with per-class
  `_counter = count(0)` + `next(self._counter)` in both loaders (add `from itertools import count` to both). Remove
  `_global_loader_counter` from `base_dist_loader.py`.
- [ ] **Step 4:** Copy `graph_store_integration_test.py` from `kmonte/shared-backend-pr3-switchover`. Update
  `test_multiple_loaders_in_graph_store` to `num_compute_nodes=2`.
- [ ] **Step 5:** Run `make format_py && make type_check`
- [ ] **Step 6:** Commit and push

### Task 5: Clean up old branches

- [ ] **Step 1:** Delete remote branch `kmonte/shared-backend-pr3-switchover` (replaced by pr3-server + pr4-loaders)
