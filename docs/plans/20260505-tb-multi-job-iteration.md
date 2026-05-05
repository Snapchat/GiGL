# Multi-Job TensorBoard: Local Iteration & Final Design Plan

Date: 2026-05-05
Branch: `kmonte/add-tb-for-glt`

This plan supersedes the earlier branch plan at `docs/plans/20260504-tb-experiment-name-proto.md`. It incorporates findings from two Codex plan reviews — round 1 at `.claude/tmp/codex-verify/20260505-155740-plan-crystalline-giggling-backus/review.md` and round 2 at `.claude/tmp/codex-verify/20260505-161326-plan-crystalline-giggling-backus/review.md`. Round-2 deltas (e.g. uniqueness via timestamp suffix, returning the `CustomJob` from `launch_single_pool_job`, `--container-uri` required, no commit of experiment name into the e2e CORA config) are applied during implementation, not via plan edits.

## Context

Across three full-pipeline iterations on this branch we've cycled through three TB integration designs, each broken in a different way:

1. **`submit(tensorboard=…)`** — auto-uploader runs, but the destination `TensorboardExperiment` is named after the (numeric) `CustomJob` ID. Per-job page works (R1 ✓), but multiple jobs cannot share one TB page (R2 ✗).
2. **`submit(experiment=…)`** — never streams events. The SDK's `experiment=` is for Vertex AI Experiments parameter/metric tracking; Vertex's TB auto-uploader is gated on `jobSpec.tensorboard` being set, which `experiment=` is mutually exclusive with. Result: events written to `AIP_TENSORBOARD_LOG_DIR` sit in GCS un-uploaded. Job 6570151780682825728 confirmed this empirically.
3. **Custom uploader from chief rank, no `tensorboard=`** — events stream to the chosen experiment (R2 ✓), but the VAI job page no longer shows the "Open TensorBoard" link because that link is keyed on `jobSpec.tensorboard` (R1 ✗). Job 4543918976459079680 confirmed this.

R1 (TB link from job page) and R2 (multi-job comparison) are not mutually exclusive — they just can't be satisfied by a single mechanism. The right approach combines both: server-side auto-uploader for the job-page link, plus a chief-rank uploader for the cross-job comparison experiment, pointing at two different `TensorboardExperiment`s under the same `Tensorboard` instance. Implementation is small; the risk is verifying behavior end-to-end. The fix for that is a tight local iteration loop.

## Success criteria

| ID | Criterion | How verified |
|----|-----------|--------------|
| R1 | The Vertex AI job UI shows "Open TensorBoard" for a successful job, and clicking it loads the per-job experiment with this job's scalar runs. | Manual: open the job in the cloud console; click the link. |
| R2 | Two jobs submitted with the same `tensorboardExperimentName` show **two distinct runs** on one TB page (the user-named experiment), each carrying its own scalars. | Manual: open the named experiment URL; toggle both runs in the scalars dashboard. Smoke script also asserts run count + ≥1 `TensorboardTimeSeries` per run. |
| R3 | Jobs without `tensorboardExperimentName` keep working: events flow to a per-job auto-named experiment. No regression. | Existing `tests/unit/src/common/vertex_ai_test.py::test_submit_job_passes_tensorboard_and_base_output_dir` plus a smoke run with the field unset. |
| R4 | `make unit_test_py` and `make type_check` pass on the branch. | CI / local. |
| R5 (process) | A new dev script lets us submit a tiny CustomJob from a laptop and verify R1+R2 in <2 minutes, end-to-end. | Run it twice; time both invocations. |
| R6 | Trainer process exits cleanly even when training fails — the chief-rank uploader does not hang the worker. | Inspected via the `try/finally` (or `with`) wrapping in all four training entrypoints; `make unit_test_py` covers the writer's idempotent close. |

## Final design

**(A) Set `jobSpec.tensorboard=<resource>` on every job that has a TB resource configured (even when an experiment name is also set).** This restores the VAI job-page TB link unconditionally and continues to populate `AIP_TENSORBOARD_RESOURCE_NAME` and `AIP_TENSORBOARD_LOG_DIR` in the worker. Vertex's auto-uploader streams events to a per-job experiment named after the job's numeric ID — that's R1.

**(B) When `tensorboard_experiment_name` is set, the launcher injects three env vars:**

- `GIGL_TENSORBOARD_RESOURCE_NAME` — full Tensorboard resource name (already injected at HEAD).
- `GIGL_TENSORBOARD_EXPERIMENT_NAME` — the user-chosen experiment name (already injected at HEAD).
- `GIGL_TENSORBOARD_RUN_NAME` — **new**: derived from the launcher's `job_name`, with `_` → `-` (so the GCS subdir name matches what the SDK's `reformat_run_name` will produce). Codex Issue 1 fix.

**(C) `TensorBoardWriter.from_env()` (chief rank only):**

- If `GIGL_TENSORBOARD_RUN_NAME` is set: write events to `<AIP_TENSORBOARD_LOG_DIR>/<run_name>/` (a *subdirectory*), not to the parent. This makes the run name visible to both the server-side auto-uploader and our chief-rank uploader as a `relpath` of the parent logdir, instead of the SDK's hardcoded `DEFAULT_RUN_NAME = "default"` (`.venv/lib/python3.11/site-packages/google/cloud/aiplatform/tensorboard/uploader_utils.py:44`). Two jobs with different run names → two distinct runs in the named experiment. Codex Issue 1 fix.
- If `GIGL_TENSORBOARD_RUN_NAME` is unset: write to `AIP_TENSORBOARD_LOG_DIR` directly (today's behavior, R3 path).
- If both `GIGL_TENSORBOARD_RESOURCE_NAME` and `GIGL_TENSORBOARD_EXPERIMENT_NAME` are also set, additionally `aiplatform.start_upload_tb_log(tensorboard_id=…, tensorboard_experiment_name=…, logdir=AIP_TENSORBOARD_LOG_DIR)` — the parent logdir, not the subdir, so the uploader's `LogdirLoader` discovers the subdir as a run via `os.path.relpath`. **Do not pass `run_name_prefix`** — the subdir already gives us the run identity, and a non-empty prefix would concatenate awkwardly with the discovered run name.
- `close()` already pairs with `aiplatform.end_upload_tb_log()` (`gigl/utils/tensorboard_writer.py:149`).

**(D) Always use `with TensorBoardWriter.from_env(...)` in trainer entrypoints.** The SDK uploader thread is **not** a daemon (`.venv/lib/python3.11/site-packages/google/cloud/aiplatform/tensorboard/uploader_tracker.py:162` — `threading.Thread(...).start()` without `daemon=True`); the SDK's docstring explicitly says to call `end_upload_tb_log()` in `finally` (`uploader_tracker.py:109`). Today's example trainers call `close()` only on the happy path. Codex Issue 3 fix: switch all four trainers to context-manager use.

The `submit(experiment=…)` SDK path and the `_ensure_experiment_with_backing_tb` helper are not needed for either requirement; both are gone as of HEAD `e19f1050`.

## Files to modify

- `gigl/common/services/vertex_ai.py` — `_submit_job`: drop the experiment-name early branch; always set `tensorboard=<resource>` whenever `job_config.tensorboard_resource_name` is non-empty. Keep the experiment-name regex validation (fail-fast). Update the `VertexAiJobConfig` docstring around `gigl/common/services/vertex_ai.py:150` (Codex Issue 6).
- `gigl/src/common/vertex_ai_launcher.py` — `_build_job_config`: keep the existing `GIGL_TENSORBOARD_RESOURCE_NAME` / `GIGL_TENSORBOARD_EXPERIMENT_NAME` injection; **add** `GIGL_TENSORBOARD_RUN_NAME` (sanitized job name). Update the comment block at `gigl/src/common/vertex_ai_launcher.py:300` describing what `_submit_job` does (Codex Issue 6).
- `gigl/utils/tensorboard_writer.py` — `from_env()` reads `GIGL_TENSORBOARD_RUN_NAME` and uses it as a subdir of `AIP_TENSORBOARD_LOG_DIR` for the `tf.summary.create_file_writer` log_dir; `_maybe_start_uploader` still watches the parent logdir.
- `proto/snapchat/research/gbml/gbml_config.proto:204` — update the `tensorboard_experiment_name` comment to describe the dual-uploader behavior, not the dropped `experiment=`-backed design (Codex Issue 6). Run `make compile_protos` to regenerate Python + Scala stubs.
- `examples/link_prediction/configs/e2e_hom_cora_sup_task_config.yaml:26` — change `tensorboardExperimentName` from the personal `kmonte-test-experiment` to `homogeneous-link-prediction-comparison` (Codex Issue 5).
- `examples/link_prediction/homogeneous_training.py`, `examples/link_prediction/heterogeneous_training.py`, `examples/link_prediction/graph_store/homogeneous_training.py`, `examples/link_prediction/graph_store/heterogeneous_training.py` — replace the existing `tensorboard_writer = TensorBoardWriter.from_env(...)` + later `.close()` pattern with a `with` block. (Codex Issue 3 + Impact Analysis.)
- `tests/unit/src/common/vertex_ai_test.py` — rename `test_submit_job_skips_experiment_and_tensorboard_when_experiment_name_set` to `test_submit_job_passes_tensorboard_with_or_without_experiment_name` and assert `tensorboard=` is set in both branches.
- `tests/unit/src/common/vertex_ai_launcher_test.py` — assert `GIGL_TENSORBOARD_RUN_NAME` is injected when an experiment name is set; not injected otherwise.
- `tests/unit/utils/tensorboard_writer_test.py` — assert the writer's effective `log_dir` is the subdir (`<parent>/<run_name>/`) when `GIGL_TENSORBOARD_RUN_NAME` is set; assert `start_upload_tb_log` is called with `logdir=<parent>` (NOT the subdir) and no `run_name_prefix`.
- `tools/dev_submit_tb_smoke_job.py` — **new** local iteration tool. The `tools/` directory already exists in the repo (Codex correction).

## Local iteration tool

A standalone Python script that bypasses ConfigPopulator and the full pipeline. Goal: <2 min from "I changed code" to "I see whether TB shows up."

Path: `tools/dev_submit_tb_smoke_job.py`.

What it does:

1. **Use the production launcher path** (`gigl.src.common.vertex_ai_launcher.launch_single_pool_job`) — *not* `VertexAIService.launch_job` directly — so the same `_build_job_config` env-var injection runs as in production. Codex Issue 2 fix.
2. Constructs a small `VertexAiResourceConfig` proto inline:
   - `machine_type="n1-standard-2"`, `gpu_type="ACCELERATOR_TYPE_UNSPECIFIED"`, `gpu_limit=0`, `num_replicas=1`, `tensorboard_resource_name=<from CLI>`.
3. Constructs a small `GiglResourceConfig` proto with that trainer config + `shared_resource_config.common_compute_config` populated from CLI flags.
4. Calls `launch_single_pool_job(...)` with:
   - `process_command="python -m gigl.utils.dev.tb_smoke_main"` — a tiny module added in the same commit; reads env vars, instantiates `TensorBoardWriter.from_env(enabled=True)`, writes 3 scalar events at steps 0/1/2, sleeps ~30s, exits.
   - `tensorboard_logs_uri = GcsUri("gs://<bucket>/tb-smoke/<timestamp>/logs/")` — drives `base_output_dir` via the existing helper at `gigl/src/common/vertex_ai_launcher.py:_get_base_output_dir_from_tensorboard_logs_uri`.
   - `tensorboard_experiment_name` from a CLI flag (or `None`).
5. After completion, queries the Vertex AI APIs:
   - `aiplatform.TensorboardExperiment.list(tensorboard_name=<TB resource>)` (`tensorboard_resource.py:518`) to confirm both expected experiments exist (the per-job auto-named one always; the user-named one only when the experiment-name flag was passed).
   - For each expected run, `aiplatform.TensorboardTimeSeries.list(tensorboard_run_name=<run resource>)` (`tensorboard_resource.py:1264`) to confirm at least one scalar tag exists. Codex Issue 4 fix — `TensorboardRun.list` alone only confirms run *existence*, not that scalars were ingested.
6. Prints both TB UI URLs (per-job and named) for manual inspection.

Required CLI flags: `--project`, `--region`, `--service-account`, `--staging-bucket`, `--tensorboard` (full resource name), and optional `--experiment-name`, `--container-uri` (defaults to `DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU` from `gigl/common/constants.py:69`), `--dry-run`.

Existing infrastructure to reuse:
- `gigl/src/common/vertex_ai_launcher.py:launch_single_pool_job` — production entry; running through this exercises env-var injection.
- `gigl/common/services/vertex_ai.py:VertexAiJobConfig` — config dataclass.
- `gigl/utils/tensorboard_writer.py:TensorBoardWriter` — same writer the trainers use.
- `aiplatform.TensorboardExperiment.list` / `aiplatform.TensorboardRun.list` / `aiplatform.TensorboardTimeSeries.list` — verification surfaces.
- `DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU` from `gigl/common/constants.py:69` — default container image.

## Step-by-step plan

Each step ends with a verification.

### Step 1: revert `_submit_job` to always pass `tensorboard=` and refresh stale comments

Production code:
- `gigl/common/services/vertex_ai.py:_submit_job` — set `submit_kwargs["tensorboard"] = job_config.tensorboard_resource_name` whenever `job_config.tensorboard_resource_name` is non-empty, regardless of `tensorboard_experiment_name`. Keep the experiment-name regex validation gate.
- `gigl/common/services/vertex_ai.py:150` — update the `VertexAiJobConfig.tensorboard_experiment_name` docstring to describe "auxiliary chief-rank uploader streams events to this experiment in addition to the per-job auto-named one."
- `gigl/src/common/vertex_ai_launcher.py:300` — update the comment block describing `_submit_job` behavior.
- `proto/snapchat/research/gbml/gbml_config.proto:204` — replace the `experiment=`-backed description with the new dual-uploader description; run `make compile_protos`.

Tests:
- `tests/unit/src/common/vertex_ai_test.py` — rename `test_submit_job_skips_experiment_and_tensorboard_when_experiment_name_set` → `test_submit_job_passes_tensorboard_with_or_without_experiment_name`; assert `tensorboard=` is set in both branches.

Verify: `make unit_test_py PY_TEST_FILES="vertex_ai_test.py"` passes; `make type_check` is clean.

Commit: `vertex_ai: always pass tensorboard= so VAI job page links to TB`.

### Step 2: inject `GIGL_TENSORBOARD_RUN_NAME` and consume it in the writer

Production code:
- `gigl/src/common/vertex_ai_launcher.py:_build_job_config` — when `tensorboard_experiment_name` is set, also append `env_var.EnvVar(name="GIGL_TENSORBOARD_RUN_NAME", value=job_name.replace("_", "-"))` next to the existing two GIGL_TENSORBOARD_* env vars. (We pre-sanitize so the GCS subdir name and the SDK-derived run name agree.)
- `gigl/utils/tensorboard_writer.py:from_env` — if `GIGL_TENSORBOARD_RUN_NAME` is set, compute `effective_log_dir = os.path.join(AIP_TENSORBOARD_LOG_DIR, run_name)` and pass that to `tf.summary.create_file_writer`. Otherwise pass `AIP_TENSORBOARD_LOG_DIR` (today's behavior).
- `gigl/utils/tensorboard_writer.py:_maybe_start_uploader` — keep watching the **parent** `AIP_TENSORBOARD_LOG_DIR` (so the SDK's `LogdirLoader` discovers the run via `os.path.relpath(subdir, logdir)` as the subdir name). No `run_name_prefix`.

Tests:
- `tests/unit/src/common/vertex_ai_launcher_test.py` — assert the GIGL_TENSORBOARD_RUN_NAME env var is injected when an experiment name is set; underscores in the job name become hyphens; not injected when experiment name is unset.
- `tests/unit/utils/tensorboard_writer_test.py` — when `GIGL_TENSORBOARD_RUN_NAME=my-run`: assert the writer's underlying file-writer was created for `<parent>/my-run/`; assert `start_upload_tb_log` called with `logdir=<parent>` and no `run_name_prefix`. When unset: writer uses parent dir directly (regression coverage for R3).

Verify: `make unit_test_py PY_TEST_FILES="vertex_ai_launcher_test.py"`; `make unit_test_py PY_TEST_FILES="tensorboard_writer_test.py"`.

Commit: `tensorboard: emit unique run names so multi-job comparison shows two runs`.

### Step 3: harden trainer uploader lifecycle

For each of:
- `examples/link_prediction/homogeneous_training.py` (`tensorboard_writer = TensorBoardWriter.from_env(...)` at line 364, `.close()` at line 621)
- `examples/link_prediction/heterogeneous_training.py`
- `examples/link_prediction/graph_store/homogeneous_training.py`
- `examples/link_prediction/graph_store/heterogeneous_training.py`

Replace the assignment + later `.close()` pattern with `with TensorBoardWriter.from_env(enabled=is_chief_process) as tensorboard_writer:` wrapping the body. The writer already supports `__enter__`/`__exit__`; this just guarantees `end_upload_tb_log` runs even when training raises.

If the writer is used at module scope across many functions (and a single `with` block would force a large diff), wrap the function that owns the training loop in `try/finally` and call `tensorboard_writer.close()` in `finally`.

Tests: existing `make unit_test_py PY_TEST_FILES="tensorboard_writer_test.py"` already covers idempotent close. No new unit tests required (these example scripts are not unit-tested today).

Verify: `make type_check`; manually re-read each modified entrypoint to confirm the writer's lifetime spans the entire training-loop scope.

Commit: `examples: scope TensorBoardWriter to a context manager in all training entrypoints`.

### Step 4: write `tools/dev_submit_tb_smoke_job.py` + `gigl/utils/dev/tb_smoke_main.py`

- `gigl/utils/dev/tb_smoke_main.py`: new module. ~25 lines. Uses `TensorBoardWriter.from_env(enabled=True)` to write 3 scalar events (`{"smoke/value": float(step)}` at steps 0, 1, 2) inside a `with` block, then `time.sleep(30)` to let both uploaders flush. Module-level entry so it can be invoked with `python -m gigl.utils.dev.tb_smoke_main`.
- `tools/dev_submit_tb_smoke_job.py`: new top-level script.
  - argparse for `--project`, `--region`, `--service-account`, `--staging-bucket`, `--tensorboard`, optional `--experiment-name`, `--container-uri`, `--dry-run`.
  - Builds `VertexAiResourceConfig` and `GiglResourceConfig` protos inline (mirror the patterns in `tests/unit/src/common/vertex_ai_launcher_test.py:_create_gigl_resource_config_with_single_pool_inference` for shape).
  - Calls `launch_single_pool_job(... vertex_ai_region=<region>, tensorboard_logs_uri=GcsUri("gs://<staging>/tb-smoke/<timestamp>/logs/"), tensorboard_experiment_name=<flag>)`.
  - On `--dry-run`: print the resulting `VertexAiJobConfig` and exit 0.
  - On real run: wait via `service.launch_job` (synchronous), then poll the verification APIs:
    - `aiplatform.TensorboardExperiment.list(tensorboard_name=<TB resource>)` — assert per-job experiment with the job's numeric ID exists; assert user-experiment exists iff flag passed.
    - For each expected experiment: `aiplatform.TensorboardRun.list(tensorboard_experiment_name=<full exp resource>)` — assert at least one run, and (for `--experiment-name` mode) that the run name matches the sanitized job name.
    - For each expected run: `aiplatform.TensorboardTimeSeries.list(tensorboard_run_name=<full run resource>)` — assert at least one time series with at least one tag (Codex Issue 4 fix).
  - Print both UI URLs.

Verify (offline): `python tools/dev_submit_tb_smoke_job.py --dry-run --project=… --region=… --service-account=… --staging-bucket=gs://… --tensorboard=projects/…/tensorboards/… --experiment-name=tb-smoke-multi` prints the `VertexAiJobConfig` and exits 0 without touching GCP.

Commit: `tools: add dev_submit_tb_smoke_job + tb_smoke_main for fast TB iteration`.

### Step 5: smoke-validate R1 + R3 (no experiment name)

Run the smoke script without `--experiment-name`. After completion (≤2 min):
- The Vertex AI job UI for the run shows "Open TensorBoard"; clicking it loads the per-job experiment (R1).
- The per-job experiment exists with one run named `default` (R3 — no `GIGL_TENSORBOARD_RUN_NAME` injected, the writer falls back to writing to the parent logdir).
- No experiment with the user-named slug exists.

If R3 fails, suspect Step 1's submit-kwargs change. The smoke loop iteration is the diagnostic surface.

### Step 6: smoke-validate R1 + R2 (with experiment name)

Run twice with the same flag: `--experiment-name=tb-smoke-multi`. After both complete:
- Both job pages still show working "Open TensorBoard" links (R1).
- Two per-job experiments exist (one per job, auto-named).
- The `tb-smoke-multi` experiment exists with **two runs**, named after each sanitized job name.
- Each of those runs has at least one `TensorboardTimeSeries` for the `smoke/value` tag.

If R2 fails (e.g., one merged run instead of two), suspect Step 2's run-name plumbing — iterate within the smoke loop, not the full pipeline.

### Step 7: full-pipeline regression test

With R1 + R2 verified at the smoke layer, kick off one real homogeneous training run with `tensorboardExperimentName: "homogeneous-link-prediction-comparison"` (the value updated in Step 1's config edit, Codex Issue 5). Verify:
- "Open TensorBoard" link works on the job page (R1).
- The named experiment shows the run with all trainer scalar tags (R2).

### Step 8: shipping checklist

- `make unit_test_py` and `make type_check` clean.
- The original branch plan's Task 11 manual smoke test gate is now satisfied by Steps 5–7.
- `make format`.
- Optionally request final code review on the post-step-1 diff via `superpowers:code-reviewer`.
- Open the PR.

### Step 0 (close-out, runs after exit-plan-mode): relocate this plan to `docs/plans/`

`mv /home/kmontemayor/.claude/plans/crystalline-giggling-backus.md docs/plans/20260505-tb-multi-job-iteration.md` — and add a note in the new file's header pointing at the supersedence relationship with `docs/plans/20260504-tb-experiment-name-proto.md`. Per CLAUDE.md plan conventions (`CLAUDE.md:252`, Codex Issue 7).

## Verification summary

| Step | Type | Cost | What it proves |
|------|------|------|----------------|
| 1, 2 | Unit tests + `type_check` | seconds | Code paths aren't broken; env-var injection + writer subdir wiring correct |
| 3 | Read-through + `type_check` | seconds | Lifecycle hardening compiles |
| 4 | `--dry-run` of smoke script | seconds | Script wires correctly without submitting |
| 5 | One smoke run (no experiment-name) | ~1–2 min | R1 + R3 |
| 6 | Two smoke runs (same experiment-name) | ~3–4 min | R1 + R2 (run identity, scalar ingestion) |
| 7 | One real homogeneous training run | ~5–15 min | Full pipeline + R1 + R2 |

Total budget for design-and-verify: ~30 minutes of cluster time.

## Risks & open questions

- **The chief-rank uploader thread is not a daemon** (`uploader_tracker.py:162`). Process exit will not reap it; `end_upload_tb_log()` MUST be called. Step 3 enforces this via `with` blocks in all four trainer entrypoints. Codex Issue 3 fix — the original plan's claim that "the SDK's uploader thread is daemon" was wrong.
- **Race between two uploaders on the same logdir.** Both uploaders read events from GCS; neither writes. Each maintains its own `LogdirLoader` state. No conflict observed in the SDK source. Step 5 + 6 confirm in practice.
- **Quota.** Two uploaders ≈ 2× ingestion request rate per opt-in job. Acceptable; revisit only on 429s.
- **GCS subdir vs logdir parent.** The chief-rank uploader watches `AIP_TENSORBOARD_LOG_DIR` (parent) and discovers the run as the subdir name. The server-side auto-uploader does the same. If we ever switch to writing events directly at the parent (no subdir), R2 collapses back to a single `default` run. Step 2's tests pin both ends.
- **`make compile_protos` regenerates Scala stubs as well.** The proto comment update in Step 1 will create a noisy diff in `scala/...` and `scala_spark35/...`. Acceptable.

## Roll-back

If Steps 5 or 6 fail and the chief-rank uploader is the cause, set just `tensorboard=<resource>` on submit and stop injecting any `GIGL_TENSORBOARD_*` env vars. Falls back to R1-only (per-job TB), losing R2 — back to the state before this branch, with no regression.

## Codex review traceability

Issues 1–7 from `.claude/tmp/codex-verify/20260505-155740-plan-crystalline-giggling-backus/review.md`:

| Issue | Severity | Addressed in |
|-------|----------|--------------|
| 1 — Run identity collapse | High | Step 2 (subdir-based run names, no `run_name_prefix`) |
| 2 — Smoke script bypasses env injection | High | Step 4 (smoke script uses `launch_single_pool_job`) |
| 3 — Uploader thread not daemon | High | Step 3 (`with` wrapping in all four trainers) |
| 4 — TimeSeries verification | Medium | Step 4 (smoke script asserts `TensorboardTimeSeries.list`) |
| 5 — Wrong experiment-name in Step 5 | Medium | Step 1 (config update from `kmonte-test-experiment` → `homogeneous-link-prediction-comparison`) |
| 6 — Stale comments / proto doc | Low | Step 1 (vertex_ai.py:150, vertex_ai_launcher.py:300, gbml_config.proto:204) |
| 7 — Plan-file location convention | Low | Step 0 (move to `docs/plans/20260505-tb-multi-job-iteration.md`) |
